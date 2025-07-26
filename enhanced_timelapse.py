#!/usr/bin/env python3
"""
Enhanced Timelapse Generator with AI Frame Interpolation
========================================================

A sophisticated timelapse generator using TensorFlow Hub's FILM model for smooth
frame interpolation with extensive customization options.

Features:
- AI-powered frame interpolation for smooth transitions
- Configurable frame rates, durations, and effects
- Multiple output formats and resolutions
- Image alignment and stabilization
- Configuration file support
- Preview functionality
- Comprehensive logging

Author: Your Name
License: MIT
"""

import os
import sys
import json
import argparse
import signal
import logging
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('timelapse.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Handle Ctrl+C gracefully
def signal_handler(sig, frame):
    logger.info("Interrupted! Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Environment tweaks for stability and reduced logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Configured {len(gpus)} GPU(s) for memory growth")
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")

@dataclass
class TimelapseConfig:
    """Configuration class for timelapse generation parameters."""
    
    # Input/Output
    input_folder: str = "images"
    output_file: str = "timelapse.mp4"
    
    # Video settings
    fps: int = 30
    width: int = 1920
    height: int = 1080
    codec: str = "mp4v"
    quality: int = 95
    
    # Timing settings (in frames)
    fade_in_frames: int = 30
    fade_out_frames: int = 30
    hold_frames: int = 30
    
    # Interpolation settings
    interpolation_recursions: int = 4  # 2^4-1 = 15 intermediate frames
    enable_interpolation: bool = True
    
    # Alignment settings
    enable_alignment: bool = False
    alignment_method: str = "ecc"  # "ecc" or "orb"
    alignment_reference_mode: str = "first"  # "first", "most_zoomed", "middle"
    alignment_scale: float = 0.25
    
    # Effects settings
    enable_fade_effects: bool = True
    enable_stabilization: bool = False
    crop_to_stable: bool = False
    
    # Processing settings
    batch_size: int = 1
    preview_mode: bool = False
    preview_frames: int = 100
    
    # Model caching settings
    model_cache_dir: str = None  # Will default to ~/.cache/tfhub
    force_model_download: bool = False
    clear_model_cache: bool = False
    
    # File filters
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TimelapseConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return cls()
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Configuration saved to {config_path}")

class Interpolator:
    """TensorFlow Hub FILM interpolator with caching."""
    
    def __init__(self, config: TimelapseConfig = None, align: int = 64) -> None:
        self._align = align
        self._setup_model_cache(config)
        logger.info("Loading FILM model from TensorFlow Hub...")
        self._model = hub.load("https://tfhub.dev/google/film/1")
        logger.info("FILM model loaded successfully")
    
    def _setup_model_cache(self, config: TimelapseConfig) -> None:
        """Setup TensorFlow Hub model caching."""
        import shutil
        
        # Determine cache directory
        if config and config.model_cache_dir:
            cache_dir = Path(config.model_cache_dir).expanduser().resolve()
        else:
            # Default to user cache directory
            cache_dir = Path.home() / ".cache" / "tfhub"
        
        # Create cache directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear cache if requested
        if config and config.clear_model_cache and cache_dir.exists():
            logger.info(f"Clearing model cache at {cache_dir}")
            try:
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Model cache cleared successfully")
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")
        
        # Set TensorFlow Hub cache directory
        os.environ['TFHUB_CACHE_DIR'] = str(cache_dir)
        
        # Check if model is already cached
        model_cache_path = cache_dir / "google_film_1"
        if model_cache_path.exists() and not (config and config.force_model_download):
            logger.info(f"Using cached FILM model from {cache_dir}")
        else:
            if config and config.force_model_download:
                logger.info("Force download enabled - will re-download model")
            else:
                logger.info(f"Model will be cached to {cache_dir} after first download")
        
        # Log cache information
        try:
            if cache_dir.exists():
                cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                cache_size_mb = cache_size / (1024 * 1024)
                logger.info(f"Cache directory: {cache_dir} ({cache_size_mb:.1f} MB)")
        except Exception as e:
            logger.debug(f"Could not calculate cache size: {e}")

    def __call__(self, x0: np.ndarray, x1: np.ndarray, dt: np.ndarray) -> np.ndarray:
        # Ensure inputs are float32
        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        dt = tf.cast(dt, tf.float32)
        
        if self._align is not None:
            x0, bbox_to_crop = self._pad_to_align(x0, self._align)
            x1, _ = self._pad_to_align(x1, self._align)
        
        inputs = {'x0': x0, 'x1': x1, 'time': dt[..., np.newaxis]}
        result = self._model(inputs, training=False)
        image = result['image']
        
        if self._align is not None:
            image = tf.image.crop_to_bounding_box(image, **bbox_to_crop)
        
        return image.numpy()

    def _pad_to_align(self, x: np.ndarray, align: int) -> Tuple[np.ndarray, Dict[str, int]]:
        """Pads image to the nearest multiple of align pixels."""
        assert np.ndim(x) == 4
        height, width = x.shape[-3:-1]
        height_to_pad = (align - height % align) if height % align != 0 else 0
        width_to_pad = (align - width % align) if width % align != 0 else 0
        
        bbox_to_pad = {
            'offset_height': height_to_pad // 2,
            'offset_width': width_to_pad // 2,
            'target_height': height + height_to_pad,
            'target_width': width + width_to_pad
        }
        
        padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)
        
        bbox_to_crop = {
            'offset_height': height_to_pad // 2,
            'offset_width': width_to_pad // 2,
            'target_height': height,
            'target_width': width
        }
        
        return padded_x, bbox_to_crop

class TimelapseGenerator:
    """Main timelapse generator class."""
    
    def __init__(self, config: TimelapseConfig):
        self.config = config
        self.interpolator = None
        self.image_files = []
        self.images = []
        
    def load_images(self) -> List[str]:
        """Load and validate image files from input folder."""
        input_path = Path(self.config.input_folder)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder {input_path} does not exist")
        
        # Get image files
        image_files = []
        for ext in self.config.supported_formats:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No supported image files found in {input_path}")
        
        # Sort by filename (assumes date format allows alphabetical sort)
        self.image_files = sorted([str(f) for f in image_files])
        logger.info(f"Found {len(self.image_files)} images")
        
        return self.image_files
    
    def load_and_resize_image(self, file_path: str) -> np.ndarray:
        """Load and resize image to target resolution."""
        try:
            image_data = tf.io.read_file(file_path)
            image = tf.io.decode_image(image_data, channels=3)
            image = tf.image.resize(image, [self.config.height, self.config.width])
            image_numpy = tf.cast(image, dtype=tf.float32).numpy()
            return image_numpy / 255.0
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            raise
    
    def find_most_zoomed_image(self, images: List[np.ndarray]) -> int:
        """Find the most zoomed image using geometric analysis."""
        if len(images) <= 1:
            return 0
        
        logger.info("Analyzing images to find most zoomed reference...")
        
        # Initialize ORB detector for geometric feature analysis
        orb = cv2.ORB_create(nfeatures=2000)
        
        # Calculate geometric zoom scores for all images
        zoom_scores = []
        
        for i, img in enumerate(tqdm(images, desc="Analyzing zoom levels")):
            try:
                # Convert to grayscale
                gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                
                # Detect keypoints and descriptors
                kp, des = orb.detectAndCompute(gray, None)
                
                if des is None or len(kp) < 10:
                    zoom_scores.append(0.0)
                    continue
                
                # Calculate geometric zoom score based on multiple factors
                zoom_score = self._calculate_zoom_score(kp, gray)
                zoom_scores.append(zoom_score)
                
            except Exception as e:
                logger.warning(f"Error analyzing image {i}: {e}")
                zoom_scores.append(0.0)
        
        # Find image with highest zoom score
        most_zoomed_idx = np.argmax(zoom_scores)
        
        logger.info(f"Most zoomed image found at index {most_zoomed_idx} "
                   f"with score {zoom_scores[most_zoomed_idx]:.3f}")
        
        return most_zoomed_idx
    
    def _calculate_zoom_score(self, keypoints: List, gray_image: np.ndarray) -> float:
        """Calculate zoom score based on geometric features."""
        if len(keypoints) < 10:
            return 0.0
        
        # Factor 1: Keypoint density (more keypoints = more detail = more zoomed)
        keypoint_density = len(keypoints) / (gray_image.shape[0] * gray_image.shape[1])
        
        # Factor 2: Keypoint spatial distribution (concentrated = more zoomed)
        keypoint_positions = np.array([kp.pt for kp in keypoints])
        spatial_variance = np.var(keypoint_positions, axis=0).sum()
        
        # Factor 3: Average keypoint response (stronger responses = more detail)
        avg_response = np.mean([kp.response for kp in keypoints])
        
        # Factor 4: Edge density analysis
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Combine factors with empirical weights
        zoom_score = (
            keypoint_density * 1000 +      # Normalize keypoint density
            (1.0 / (spatial_variance + 1)) * 0.5 +  # Inverse spatial variance
            avg_response * 10 +            # Response strength
            edge_density * 2               # Edge density
        )
        
        return zoom_score
    
    def find_reference_image_by_homography(self, images: List[np.ndarray]) -> int:
        """Find most zoomed image using homography scale analysis."""
        if len(images) <= 1:
            return 0
        
        logger.info("Using homography analysis to find optimal reference...")
        
        # Initialize feature detector
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Calculate relative scales between image pairs
        scale_scores = []
        
        for i, img in enumerate(images):
            try:
                gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                kp, des = orb.detectAndCompute(gray, None)
                
                if des is None:
                    scale_scores.append(0.0)
                    continue
                
                # Compare with other images to find relative scale
                relative_scales = []
                
                for j, other_img in enumerate(images):
                    if i == j:
                        continue
                    
                    other_gray = cv2.cvtColor((other_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    other_kp, other_des = orb.detectAndCompute(other_gray, None)
                    
                    if other_des is None:
                        continue
                    
                    # Match features
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des, other_des)
                    
                    if len(matches) < 10:
                        continue
                    
                    # Calculate homography and extract scale
                    src_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([other_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    try:
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if M is not None:
                            # Extract scale from homography matrix
                            scale_x = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
                            scale_y = np.sqrt(M[1, 0]**2 + M[1, 1]**2)
                            avg_scale = (scale_x + scale_y) / 2
                            relative_scales.append(avg_scale)
                    except:
                        continue
                
                # Image with consistently smaller scale relative to others = more zoomed
                if relative_scales:
                    avg_relative_scale = np.mean(relative_scales)
                    scale_scores.append(1.0 / (avg_relative_scale + 0.1))  # Inverse scale
                else:
                    scale_scores.append(0.0)
                    
            except Exception as e:
                logger.warning(f"Error in homography analysis for image {i}: {e}")
                scale_scores.append(0.0)
        
        most_zoomed_idx = np.argmax(scale_scores)
        logger.info(f"Homography analysis: most zoomed image at index {most_zoomed_idx}")
        
        return most_zoomed_idx
    
    def select_reference_image(self, images: List[np.ndarray]) -> int:
        """Select reference image based on configuration."""
        if self.config.alignment_reference_mode == "first":
            return 0
        elif self.config.alignment_reference_mode == "middle":
            return len(images) // 2
        elif self.config.alignment_reference_mode == "most_zoomed":
            # Use the simpler geometric analysis by default
            return self.find_most_zoomed_image(images)
        else:
            logger.warning(f"Unknown reference mode: {self.config.alignment_reference_mode}")
            return 0
    
    def align_image_to_reference(self, image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Align image to reference using selected method."""
        if self.config.alignment_method == "ecc":
            return self._align_ecc(image, reference)
        elif self.config.alignment_method == "orb":
            return self._align_orb(image, reference)
        else:
            logger.warning(f"Unknown alignment method: {self.config.alignment_method}")
            return image
    
    def _align_ecc(self, image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Align using Enhanced Correlation Coefficient."""
        scale_factor = self.config.alignment_scale
        small_size = (int(self.config.width * scale_factor), int(self.config.height * scale_factor))
        
        # Convert to grayscale and resize
        image_gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        ref_gray = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        image_gray_small = cv2.resize(image_gray, small_size)
        ref_gray_small = cv2.resize(ref_gray, small_size)
        
        # Initialize warp matrix
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # ECC alignment
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
        try:
            _, warp_matrix = cv2.findTransformECC(
                ref_gray_small, image_gray_small, warp_matrix, 
                cv2.MOTION_AFFINE, criteria
            )
            # Scale translation back to original size
            warp_matrix[:, 2] /= scale_factor
        except Exception as e:
            logger.warning(f"ECC alignment failed: {e}")
            return image
        
        # Apply warp
        aligned = cv2.warpAffine(
            (image * 255).astype(np.uint8), warp_matrix, 
            (self.config.width, self.config.height),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
        
        return aligned / 255.0
    
    def _align_orb(self, image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Align using ORB feature matching."""
        # Convert to grayscale
        image_gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        ref_gray = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(ref_gray, None)
        kp2, des2 = orb.detectAndCompute(image_gray, None)
        
        if des1 is None or des2 is None:
            logger.warning("ORB alignment failed: no descriptors found")
            return image
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 4:
            logger.warning("ORB alignment failed: insufficient matches")
            return image
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography
        try:
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if M is None:
                logger.warning("ORB alignment failed: no homography found")
                return image
            
            # Apply transformation
            aligned = cv2.warpPerspective(
                (image * 255).astype(np.uint8), M,
                (self.config.width, self.config.height)
            )
            
            return aligned / 255.0
            
        except Exception as e:
            logger.warning(f"ORB alignment failed: {e}")
            return image
    
    def recursive_generator(self, frame1: np.ndarray, frame2: np.ndarray, 
                          num_recursions: int, interpolator: Interpolator) -> Generator[np.ndarray, None, None]:
        """Recursive generator for midpoint interpolation."""
        if num_recursions == 0:
            yield frame1
        else:
            dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
            mid_frame = interpolator(
                np.expand_dims(frame1, axis=0), 
                np.expand_dims(frame2, axis=0), 
                dt
            )[0]
            yield from self.recursive_generator(frame1, mid_frame, num_recursions - 1, interpolator)
            yield from self.recursive_generator(mid_frame, frame2, num_recursions - 1, interpolator)
        yield frame2
    
    def generate_timelapse(self):
        """Generate the timelapse video."""
        logger.info("Starting timelapse generation...")
        
        # Load images
        self.load_images()
        
        # Load and resize images
        logger.info("Loading and resizing images...")
        raw_images = []
        for file_path in tqdm(self.image_files, desc="Loading images"):
            raw_images.append(self.load_and_resize_image(file_path))
        
        # Align images if enabled
        if self.config.enable_alignment:
            logger.info(f"Aligning images using {self.config.alignment_method} method...")
            logger.info(f"Reference selection mode: {self.config.alignment_reference_mode}")
            
            # Select reference image
            reference_idx = self.select_reference_image(raw_images)
            reference = raw_images[reference_idx]
            
            logger.info(f"Using image {reference_idx} as alignment reference")
            
            # Align all images to the reference
            self.images = []
            for i, img in enumerate(tqdm(raw_images, desc="Aligning images")):
                if i == reference_idx:
                    # Reference image doesn't need alignment
                    self.images.append(img)
                else:
                    aligned = self.align_image_to_reference(img, reference)
                    self.images.append(aligned)
        else:
            self.images = raw_images
            logger.info("Skipping image alignment")
        
        # Generate frames
        all_frames = []
        black = np.zeros((self.config.height, self.config.width, 3), dtype=np.float32)
        
        # Fade in
        if self.config.enable_fade_effects and self.config.fade_in_frames > 0:
            logger.info("Generating fade-in effect...")
            for i in range(self.config.fade_in_frames):
                alpha = i / (self.config.fade_in_frames - 1)
                frame = black * (1 - alpha) + self.images[0] * alpha
                all_frames.append(frame)
        
        # Hold first image
        if self.config.hold_frames > 0:
            logger.info(f"Adding {self.config.hold_frames} hold frames for first image...")
            for _ in range(self.config.hold_frames):
                all_frames.append(self.images[0])
        
        # Process transitions
        if self.config.enable_interpolation:
            logger.info("Initializing interpolator...")
            self.interpolator = Interpolator(self.config)
            
            logger.info("Processing image transitions with interpolation...")
            for idx in tqdm(range(len(self.images) - 1), desc="Processing transitions"):
                img1 = self.images[idx]
                img2 = self.images[idx + 1]
                
                # Generate interpolated frames
                gen = self.recursive_generator(
                    img1, img2, self.config.interpolation_recursions, self.interpolator
                )
                transition_frames = list(gen)
                
                # Add intermediate frames (exclude start/end)
                intermediates = transition_frames[1:-1]
                all_frames.extend(intermediates)
                
                # Hold next image
                if self.config.hold_frames > 0:
                    for _ in range(self.config.hold_frames):
                        all_frames.append(img2)
        else:
            logger.info("Processing without interpolation...")
            for idx in range(1, len(self.images)):
                # Add image directly
                if self.config.hold_frames > 0:
                    for _ in range(self.config.hold_frames):
                        all_frames.append(self.images[idx])
        
        # Fade out
        if self.config.enable_fade_effects and self.config.fade_out_frames > 0:
            logger.info("Generating fade-out effect...")
            last_img = self.images[-1]
            for i in range(self.config.fade_out_frames):
                alpha = 1 - (i / (self.config.fade_out_frames - 1))
                frame = last_img * alpha + black * (1 - alpha)
                all_frames.append(frame)
        
        # Preview mode
        if self.config.preview_mode:
            total_frames = min(len(all_frames), self.config.preview_frames)
            all_frames = all_frames[:total_frames]
            logger.info(f"Preview mode: using first {total_frames} frames")
        
        # Write video
        logger.info(f"Writing {len(all_frames)} frames to video...")
        self._write_video(all_frames)
        
        logger.info(f"Timelapse generation completed! Output: {self.config.output_file}")
        logger.info(f"Total frames: {len(all_frames)}")
        logger.info(f"Duration: {len(all_frames) / self.config.fps:.2f} seconds")
    
    def _write_video(self, frames: List[np.ndarray]):
        """Write frames to video file."""
        # Get codec
        if self.config.codec == 'h264':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif self.config.codec == 'xvid':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        
        # Create video writer
        video_writer = cv2.VideoWriter(
            self.config.output_file, fourcc, self.config.fps,
            (self.config.width, self.config.height)
        )
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {self.config.output_file}")
        
        # Write frames
        for frame in tqdm(frames, desc="Writing video"):
            frame_uint8 = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()

def create_sample_config():
    """Create a sample configuration file."""
    config = TimelapseConfig()
    config.save_to_file("timelapse_config.json")
    logger.info("Sample configuration created: timelapse_config.json")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Timelapse Generator with AI Frame Interpolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_timelapse.py images/                    # Basic usage
  python enhanced_timelapse.py images/ --fps 60          # Custom frame rate
  python enhanced_timelapse.py images/ --config my.json  # Use config file
  python enhanced_timelapse.py --create-config           # Create sample config
  python enhanced_timelapse.py images/ --preview         # Preview mode
  python enhanced_timelapse.py images/ --clear-cache     # Clear model cache first
  python enhanced_timelapse.py images/ --force-download  # Force re-download model
        """
    )
    
    parser.add_argument('input_folder', nargs='?', help="Path to images folder")
    parser.add_argument('--output', '-o', default='timelapse.mp4', help="Output video file")
    parser.add_argument('--config', '-c', help="Configuration file path")
    parser.add_argument('--create-config', action='store_true', help="Create sample config file")
    
    # Video settings
    parser.add_argument('--fps', type=int, default=30, help="Output frame rate")
    parser.add_argument('--width', type=int, default=1920, help="Output width")
    parser.add_argument('--height', type=int, default=1080, help="Output height")
    parser.add_argument('--codec', default='mp4v', help="Video codec")
    
    # Timing settings
    parser.add_argument('--fade-in', type=int, default=30, help="Fade-in duration (frames)")
    parser.add_argument('--fade-out', type=int, default=30, help="Fade-out duration (frames)")
    parser.add_argument('--hold', type=int, default=30, help="Hold duration per image (frames)")
    
    # Effects
    parser.add_argument('--no-fade', action='store_true', help="Disable fade effects")
    parser.add_argument('--no-interpolation', action='store_true', help="Disable frame interpolation")
    parser.add_argument('--interpolation-level', type=int, default=4, help="Interpolation recursion level")
    
    # Alignment
    parser.add_argument('--align', action='store_true', help="Enable image alignment")
    parser.add_argument('--alignment-method', choices=['ecc', 'orb'], default='ecc', help="Alignment method")
    parser.add_argument('--alignment-reference', choices=['first', 'most_zoomed', 'middle'], default='first', help="Reference image selection for alignment")
    
    # Model caching
    parser.add_argument('--model-cache-dir', help="Custom model cache directory")
    parser.add_argument('--force-download', action='store_true', help="Force re-download model even if cached")
    parser.add_argument('--clear-cache', action='store_true', help="Clear model cache before running")
    
    # Preview and debug
    parser.add_argument('--preview', action='store_true', help="Preview mode (limited frames)")
    parser.add_argument('--preview-frames', type=int, default=100, help="Number of preview frames")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create sample config
    if args.create_config:
        create_sample_config()
        return
    
    # Validate input
    if not args.input_folder:
        parser.error("Input folder is required (unless using --create-config)")
    
    # Load configuration
    if args.config:
        config = TimelapseConfig.from_file(args.config)
    else:
        config = TimelapseConfig()
    
    # Override config with command line arguments
    config.input_folder = args.input_folder
    config.output_file = args.output
    config.fps = args.fps
    config.width = args.width
    config.height = args.height
    config.codec = args.codec
    config.fade_in_frames = args.fade_in
    config.fade_out_frames = args.fade_out
    config.hold_frames = args.hold
    config.enable_fade_effects = not args.no_fade
    config.enable_interpolation = not args.no_interpolation
    config.interpolation_recursions = args.interpolation_level
    config.enable_alignment = args.align
    config.alignment_method = args.alignment_method
    config.alignment_reference_mode = args.alignment_reference
    config.preview_mode = args.preview
    config.preview_frames = args.preview_frames
    
    # Model caching settings
    if args.model_cache_dir:
        config.model_cache_dir = args.model_cache_dir
    config.force_model_download = args.force_download
    config.clear_model_cache = args.clear_cache
    
    # Log configuration
    logger.info("Configuration:")
    for key, value in asdict(config).items():
        logger.info(f"  {key}: {value}")
    
    # Generate timelapse
    try:
        generator = TimelapseGenerator(config)
        generator.generate_timelapse()
    except KeyboardInterrupt:
        logger.info("Generation cancelled by user")
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise

if __name__ == "__main__":
    main()
