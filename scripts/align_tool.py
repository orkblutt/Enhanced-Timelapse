#!/usr/bin/env python3
"""
Image Alignment Tool for Timelapse Preparation
================================================

Two-phase alignment workflow:
1. Automatic SIFT-based alignment with confidence scoring
2. Interactive overlay UI for manual corrections

Usage:
    python scripts/align_tool.py --input images/ --output aligned/
    python scripts/align_tool.py --input images/ --output aligned/ --reference 5
    python scripts/align_tool.py --input images/ --output aligned/ --auto-only
    python scripts/align_tool.py --input images/ --output aligned/ --review-all
"""

import os
import sys
import json
import argparse
import logging
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


# ---------------------------------------------------------------------------
# Auto-alignment (SIFT + RANSAC)
# ---------------------------------------------------------------------------

def compute_sift_alignment(image: np.ndarray, reference: np.ndarray,
                           min_matches: int = 10) -> Tuple[Optional[np.ndarray], float]:
    """Align image to reference using SIFT features + RANSAC homography.

    Returns (homography_matrix, confidence_score).
    Returns (None, 0.0) on failure.
    """
    # Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # SIFT detection
    sift = cv2.SIFT_create(nfeatures=3000)
    kp_ref, des_ref = sift.detectAndCompute(gray_ref, None)
    kp_img, des_img = sift.detectAndCompute(gray_img, None)

    if des_ref is None or des_img is None:
        return None, 0.0

    # FLANN matcher with Lowe's ratio test
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if len(des_ref) < 2 or len(des_img) < 2:
        return None, 0.0

    raw_matches = flann.knnMatch(des_img, des_ref, k=2)

    # Lowe's ratio test
    good_matches = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    if len(good_matches) < min_matches:
        return None, 0.0

    # Extract matched points
    src_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None or mask is None:
        return None, 0.0

    # Confidence scoring
    inliers = int(mask.sum())
    inlier_ratio = inliers / len(good_matches)

    # Reprojection error
    src_transformed = cv2.perspectiveTransform(src_pts, H)
    reproj_errors = np.sqrt(((src_transformed - dst_pts) ** 2).sum(axis=2)).flatten()
    mean_reproj_error = float(reproj_errors[mask.flatten() == 1].mean()) if inliers > 0 else 999.0

    # Homography condition: check it's not degenerate
    det = abs(np.linalg.det(H[:2, :2]))
    det_score = min(det, 1.0 / max(det, 1e-6))  # Close to 1.0 = good

    # Combined confidence
    confidence = (
        min(inlier_ratio, 1.0) * 0.4 +
        max(0, 1.0 - mean_reproj_error / 10.0) * 0.3 +
        min(det_score, 1.0) * 0.2 +
        min(inliers / 50.0, 1.0) * 0.1
    )

    return H, float(np.clip(confidence, 0.0, 1.0))


def auto_align_all(images: List[np.ndarray], filenames: List[str],
                   ref_idx: int) -> List[dict]:
    """Auto-align all images using chain alignment.

    Instead of aligning every image directly to the reference (which fails
    when images are far apart in time), align each image to its neighbor
    and compose the transforms to get a path back to the reference.

    Falls back to direct alignment when chain confidence is too low.
    """
    n = len(images)

    # Step 1: Compute pairwise alignments between consecutive images
    logger.info("Computing pairwise alignments between consecutive images...")
    pairwise = []  # pairwise[i] = (H from image i+1 to image i, confidence)
    for i in tqdm(range(n - 1), desc="Pairwise alignment"):
        H, conf = compute_sift_alignment(images[i + 1], images[i])
        pairwise.append((H, conf))

    # Step 2: Compose transforms to build path from each image to the reference
    # H_to_ref[i] = transform that maps image i into the reference frame
    logger.info("Composing chain transforms to reference frame...")
    H_to_ref = [None] * n
    conf_to_ref = [0.0] * n
    H_to_ref[ref_idx] = np.eye(3)
    conf_to_ref[ref_idx] = 1.0

    # Forward chain: ref_idx -> ref_idx-1 -> ... -> 0
    for i in range(ref_idx - 1, -1, -1):
        H_pair, c_pair = pairwise[i]  # maps image i+1 -> image i, we need i -> i+1
        if H_pair is not None and c_pair > 0.1:
            try:
                H_inv = np.linalg.inv(H_pair)
                H_to_ref[i] = H_to_ref[i + 1] @ H_inv
                conf_to_ref[i] = min(conf_to_ref[i + 1], c_pair)
            except np.linalg.LinAlgError:
                H_to_ref[i] = np.eye(3)
                conf_to_ref[i] = 0.0
        else:
            H_to_ref[i] = np.eye(3)
            conf_to_ref[i] = 0.0

    # Backward chain: ref_idx -> ref_idx+1 -> ... -> n-1
    for i in range(ref_idx + 1, n):
        H_pair, c_pair = pairwise[i - 1]  # maps image i to image i-1
        if H_pair is not None and c_pair > 0.1:
            H_to_ref[i] = H_to_ref[i - 1] @ H_pair
            conf_to_ref[i] = min(conf_to_ref[i - 1], c_pair)
        else:
            H_to_ref[i] = np.eye(3)
            conf_to_ref[i] = 0.0

    # Step 3: For low-confidence chain results, try direct alignment as fallback
    logger.info("Trying direct alignment for low-confidence images...")
    reference = images[ref_idx]
    for i in tqdm(range(n), desc="Direct fallback"):
        if i == ref_idx:
            continue
        if conf_to_ref[i] < 0.3:
            H_direct, c_direct = compute_sift_alignment(images[i], reference)
            if H_direct is not None and c_direct > conf_to_ref[i]:
                H_to_ref[i] = H_direct
                conf_to_ref[i] = c_direct

    # Build results
    results = []
    for i in range(n):
        is_ref = (i == ref_idx)
        status = "REF" if is_ref else ("OK" if conf_to_ref[i] >= 0.7 else "LOW")
        if not is_ref:
            logger.info(f"  {filenames[i]}: confidence={conf_to_ref[i]:.2f} [{status}]")
        results.append({
            'filename': filenames[i],
            'transform': H_to_ref[i].tolist(),
            'confidence': conf_to_ref[i],
            'manual_adjusted': False,
            'is_reference': is_ref,
        })

    return results


# ---------------------------------------------------------------------------
# Interactive overlay UI
# ---------------------------------------------------------------------------

class AlignmentUI:
    """Interactive overlay UI for manual alignment adjustment."""

    WINDOW_NAME = "Alignment Tool"

    def __init__(self, images: List[np.ndarray], filenames: List[str],
                 alignment_data: List[dict], ref_idx: int,
                 display_width: int = 1280, display_height: int = 720):
        self.images = images
        self.filenames = filenames
        self.alignment_data = alignment_data
        self.ref_idx = ref_idx
        self.display_w = display_width
        self.display_h = display_height

        # Current state
        self.current_idx = self._find_first_review_idx()
        self.opacity = 0.5
        self.side_by_side = False
        self.dragging = False
        self.rotating = False
        self.drag_start = (0, 0)
        self.modified = False

        # Per-image manual transform (translation, rotation, scale)
        # Applied on top of the auto homography
        self.manual_offsets = {}
        for i in range(len(images)):
            self.manual_offsets[i] = {'tx': 0.0, 'ty': 0.0, 'angle': 0.0, 'scale': 1.0}

    def _find_first_review_idx(self) -> int:
        """Find first non-reference image that needs review."""
        for i, data in enumerate(self.alignment_data):
            if not data.get('is_reference', False):
                return i
        return 0

    def _get_composite_transform(self, idx: int) -> np.ndarray:
        """Get the combined auto + manual transform for an image."""
        H_auto = np.array(self.alignment_data[idx]['transform'], dtype=np.float64)
        m = self.manual_offsets[idx]

        # Build manual correction matrix
        cx, cy = self.images[idx].shape[1] / 2, self.images[idx].shape[0] / 2
        cos_a = math.cos(m['angle'])
        sin_a = math.sin(m['angle'])
        s = m['scale']

        # Translate to center, rotate+scale, translate back, then offset
        M_manual = np.array([
            [s * cos_a, -s * sin_a, m['tx'] + cx * (1 - s * cos_a) + cy * s * sin_a],
            [s * sin_a,  s * cos_a, m['ty'] + cy * (1 - s * cos_a) - cx * s * sin_a],
            [0, 0, 1]
        ], dtype=np.float64)

        return M_manual @ H_auto

    def _warp_image(self, idx: int) -> np.ndarray:
        """Warp an image using its composite transform."""
        H = self._get_composite_transform(idx)
        h, w = self.images[self.ref_idx].shape[:2]
        return cv2.warpPerspective(self.images[idx], H, (w, h),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))

    def _render(self) -> np.ndarray:
        """Render the current view."""
        ref = self.images[self.ref_idx]
        h, w = ref.shape[:2]

        if self.current_idx == self.ref_idx:
            display = ref.copy()
            label = "REFERENCE IMAGE"
        else:
            warped = self._warp_image(self.current_idx)

            if self.side_by_side:
                # Side by side: reference left, warped right
                half_w = w // 2
                display = np.zeros_like(ref)
                display[:, :half_w] = ref[:, :half_w]
                display[:, half_w:] = warped[:, half_w:]
                cv2.line(display, (half_w, 0), (half_w, h), (0, 255, 0), 2)
                label = "SIDE-BY-SIDE (Tab to toggle)"
            else:
                # Blended overlay
                alpha = self.opacity
                display = cv2.addWeighted(ref, 1 - alpha, warped, alpha, 0)
                label = f"OVERLAY opacity={alpha:.0%}"

        # Resize for display
        display = cv2.resize(display, (self.display_w, self.display_h))

        # Draw UI overlay
        self._draw_hud(display, label)

        return display

    def _draw_hud(self, display: np.ndarray, mode_label: str):
        """Draw heads-up display info."""
        data = self.alignment_data[self.current_idx]
        m = self.manual_offsets[self.current_idx]

        # Background bar
        cv2.rectangle(display, (0, 0), (self.display_w, 80), (0, 0, 0), -1)

        # Image info
        conf = data['confidence']
        conf_color = (0, 255, 0) if conf >= 0.7 else (0, 165, 255) if conf >= 0.4 else (0, 0, 255)
        cv2.putText(display, f"[{self.current_idx + 1}/{len(self.images)}] {data['filename']}",
                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"Confidence: {conf:.2f}", (10, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
        cv2.putText(display, mode_label, (10, 75),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Manual offset info
        info = f"tx={m['tx']:.1f} ty={m['ty']:.1f} rot={math.degrees(m['angle']):.1f}° scale={m['scale']:.3f}"
        cv2.putText(display, info, (self.display_w - 500, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if data.get('manual_adjusted'):
            cv2.putText(display, "MANUALLY ADJUSTED", (self.display_w - 250, 50),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Help bar at bottom
        cv2.rectangle(display, (0, self.display_h - 30), (self.display_w, self.display_h), (0, 0, 0), -1)
        help_text = "LMB:move  RMB:rot/scale  Wheel:opacity  Tab:split  WASD:fine  QE:rot  +/-:scale  Enter:accept  R:reset  N/P:nav  Esc:save&quit"
        cv2.putText(display, help_text, (10, self.display_h - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    def _on_mouse(self, event, x, y, flags, param):
        """Handle mouse events."""
        if self.current_idx == self.ref_idx:
            return

        # Scale mouse coordinates to image space
        img_h, img_w = self.images[self.ref_idx].shape[:2]
        sx = img_w / self.display_w
        sy = img_h / self.display_h

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.rotating = True
            self.drag_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                dx = (x - self.drag_start[0]) * sx
                dy = (y - self.drag_start[1]) * sy
                self.manual_offsets[self.current_idx]['tx'] += dx
                self.manual_offsets[self.current_idx]['ty'] += dy
                self.drag_start = (x, y)
                self.modified = True
            elif self.rotating:
                dx = (x - self.drag_start[0])
                dy = (y - self.drag_start[1])
                self.manual_offsets[self.current_idx]['angle'] += dx * 0.002
                self.manual_offsets[self.current_idx]['scale'] *= (1.0 + dy * 0.002)
                self.manual_offsets[self.current_idx]['scale'] = max(0.5, min(2.0, self.manual_offsets[self.current_idx]['scale']))
                self.drag_start = (x, y)
                self.modified = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_RBUTTONUP:
            self.rotating = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.opacity = min(1.0, self.opacity + 0.05)
            else:
                self.opacity = max(0.0, self.opacity - 0.05)

    def _navigate(self, direction: int):
        """Move to next/previous image."""
        self.current_idx = (self.current_idx + direction) % len(self.images)

    def run(self, review_indices: List[int]):
        """Run the alignment UI for the specified image indices."""
        if not review_indices:
            logger.info("No images to review.")
            return

        self.current_idx = review_indices[0]

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

        logger.info(f"Opening alignment UI for {len(review_indices)} images...")
        logger.info("Controls: LMB=drag, RMB=rotate/scale, Wheel=opacity, Esc=save&quit")

        while True:
            display = self._render()
            cv2.imshow(self.WINDOW_NAME, display)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # Esc
                break
            elif key == 13 or key == 10:  # Enter — accept and go to next review image
                if self.modified:
                    self.alignment_data[self.current_idx]['manual_adjusted'] = True
                    self.modified = False
                # Find next review image
                cur_pos = review_indices.index(self.current_idx) if self.current_idx in review_indices else -1
                if cur_pos < len(review_indices) - 1:
                    self.current_idx = review_indices[cur_pos + 1]
                else:
                    logger.info("All review images processed.")
                    break
            elif key == ord('n'):
                self._navigate(1)
            elif key == ord('p'):
                self._navigate(-1)
            elif key == ord('r'):  # Reset to auto
                self.manual_offsets[self.current_idx] = {'tx': 0.0, 'ty': 0.0, 'angle': 0.0, 'scale': 1.0}
                self.alignment_data[self.current_idx]['manual_adjusted'] = False
                self.modified = False
            elif key == 8:  # Backspace — reset to identity (no alignment)
                self.alignment_data[self.current_idx]['transform'] = np.eye(3).tolist()
                self.manual_offsets[self.current_idx] = {'tx': 0.0, 'ty': 0.0, 'angle': 0.0, 'scale': 1.0}
                self.alignment_data[self.current_idx]['manual_adjusted'] = True
                self.modified = False
            elif key == 9:  # Tab
                self.side_by_side = not self.side_by_side
            elif key == ord('a'):
                self.manual_offsets[self.current_idx]['tx'] -= 2.0
                self.modified = True
            elif key == ord('d'):
                self.manual_offsets[self.current_idx]['tx'] += 2.0
                self.modified = True
            elif key == ord('w'):
                self.manual_offsets[self.current_idx]['ty'] -= 2.0
                self.modified = True
            elif key == ord('s'):
                self.manual_offsets[self.current_idx]['ty'] += 2.0
                self.modified = True
            elif key == ord('q'):
                self.manual_offsets[self.current_idx]['angle'] -= 0.005
                self.modified = True
            elif key == ord('e'):
                self.manual_offsets[self.current_idx]['angle'] += 0.005
                self.modified = True
            elif key == ord('+') or key == ord('='):
                self.manual_offsets[self.current_idx]['scale'] = min(2.0, self.manual_offsets[self.current_idx]['scale'] + 0.005)
                self.modified = True
            elif key == ord('-'):
                self.manual_offsets[self.current_idx]['scale'] = max(0.5, self.manual_offsets[self.current_idx]['scale'] - 0.005)
                self.modified = True

        cv2.destroyAllWindows()

        # Update transforms with manual corrections
        for i in range(len(self.images)):
            if i == self.ref_idx:
                continue
            m = self.manual_offsets[i]
            if m['tx'] != 0 or m['ty'] != 0 or m['angle'] != 0 or m['scale'] != 1.0:
                H_composite = self._get_composite_transform(i)
                self.alignment_data[i]['transform'] = H_composite.tolist()
                self.alignment_data[i]['manual_adjusted'] = True


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_images(input_dir: Path, width: int, height: int) -> Tuple[List[np.ndarray], List[str]]:
    """Load and resize all images from a directory."""
    files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ])

    if not files:
        raise ValueError(f"No supported image files found in {input_dir}")

    images = []
    filenames = []

    for f in tqdm(files, desc="Loading images"):
        img = cv2.imread(str(f))
        if img is None:
            logger.warning(f"Could not load {f.name}, skipping")
            continue
        img = cv2.resize(img, (width, height))
        images.append(img)
        filenames.append(f.name)

    logger.info(f"Loaded {len(images)} images at {width}x{height}")
    return images, filenames


def save_aligned_images(images: List[np.ndarray], filenames: List[str],
                        alignment_data: List[dict], ref_idx: int,
                        output_dir: Path, width: int, height: int):
    """Apply transforms and save aligned images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, fname, data) in enumerate(tqdm(
            zip(images, filenames, alignment_data),
            total=len(images), desc="Saving aligned images")):

        H = np.array(data['transform'], dtype=np.float64)

        if i == ref_idx:
            aligned = img
        else:
            aligned = cv2.warpPerspective(img, H, (width, height),
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0))

        cv2.imwrite(str(output_dir / fname), aligned)

    # Save alignment data
    json_path = output_dir / 'alignment_data.json'
    save_data = {
        'reference_image': filenames[ref_idx],
        'width': width,
        'height': height,
        'images': alignment_data,
    }
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    logger.info(f"Saved {len(images)} aligned images to {output_dir}")
    logger.info(f"Alignment data saved to {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Image Alignment Tool for Timelapse Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/align_tool.py --input images/ --output aligned/
  python scripts/align_tool.py --input images/ --output aligned/ --reference 5
  python scripts/align_tool.py --input images/ --output aligned/ --auto-only
  python scripts/align_tool.py --input images/ --output aligned/ --review-all
  python scripts/align_tool.py --input images/ --output aligned/ --threshold 0.5
        """
    )

    parser.add_argument('--input', '-i', required=True, help="Input images directory")
    parser.add_argument('--output', '-o', required=True, help="Output directory for aligned images")
    parser.add_argument('--reference', type=int, default=0, help="Reference image index (default: 0)")
    parser.add_argument('--width', type=int, default=1920, help="Output width (default: 1920)")
    parser.add_argument('--height', type=int, default=1080, help="Output height (default: 1080)")
    parser.add_argument('--threshold', type=float, default=0.7, help="Confidence threshold for manual review (default: 0.7)")
    parser.add_argument('--auto-only', action='store_true', help="Skip manual review UI")
    parser.add_argument('--review-all', action='store_true', help="Show UI for all images, not just low-confidence")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Load images
    images, filenames = load_images(input_dir, args.width, args.height)

    if args.reference >= len(images):
        logger.error(f"Reference index {args.reference} out of range (0-{len(images) - 1})")
        sys.exit(1)

    ref_idx = args.reference
    logger.info(f"Using image {ref_idx} ({filenames[ref_idx]}) as reference")

    # Phase 1: Auto-alignment
    logger.info("Phase 1: Auto-aligning images with SIFT...")
    alignment_data = auto_align_all(images, filenames, ref_idx)

    # Report results
    low_conf = [d for d in alignment_data if d['confidence'] < args.threshold and not d.get('is_reference')]
    high_conf = [d for d in alignment_data if d['confidence'] >= args.threshold or d.get('is_reference')]
    logger.info(f"Auto-alignment complete: {len(high_conf)} good, {len(low_conf)} need review")

    # Phase 2: Manual review
    if not args.auto_only:
        if args.review_all:
            review_indices = [i for i in range(len(images)) if i != ref_idx]
        else:
            review_indices = [i for i, d in enumerate(alignment_data)
                              if d['confidence'] < args.threshold and not d.get('is_reference')]

        if review_indices:
            logger.info(f"Phase 2: Opening manual review for {len(review_indices)} images...")
            ui = AlignmentUI(images, filenames, alignment_data, ref_idx)
            ui.run(review_indices)
            alignment_data = ui.alignment_data
        else:
            logger.info("Phase 2: All images aligned with high confidence, skipping manual review")
    else:
        logger.info("Phase 2: Skipped (--auto-only)")

    # Phase 3: Save results
    logger.info("Phase 3: Saving aligned images...")
    save_aligned_images(images, filenames, alignment_data, ref_idx,
                        output_dir, args.width, args.height)

    # Summary
    manual_count = sum(1 for d in alignment_data if d.get('manual_adjusted'))
    logger.info(f"Done! {manual_count} images were manually adjusted.")
    logger.info(f"Run: python enhanced_timelapse.py {output_dir}/ --output timelapse.mp4")


if __name__ == "__main__":
    main()
