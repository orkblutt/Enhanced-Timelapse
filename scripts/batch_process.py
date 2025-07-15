#!/usr/bin/env python3
"""
Batch Processing Script for Enhanced Timelapse Generator
=======================================================

Process multiple image folders to create timelapse videos automatically.
Useful for processing large numbers of timelapse sequences.

Usage:
    python batch_process.py --input-dir /path/to/folders --config config.json
    python batch_process.py --folders folder1 folder2 folder3
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

# Add parent directory to path to import enhanced_timelapse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_timelapse import TimelapseConfig, TimelapseGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_process.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def process_folder(folder_path: str, config: TimelapseConfig, output_dir: str) -> Dict:
    """Process a single folder to create timelapse."""
    folder_name = Path(folder_path).name
    result = {
        'folder': folder_path,
        'folder_name': folder_name,
        'success': False,
        'error': None,
        'output_file': None
    }
    
    try:
        # Create folder-specific config
        folder_config = TimelapseConfig(**config.__dict__)
        folder_config.input_folder = folder_path
        
        # Generate output filename
        output_file = os.path.join(output_dir, f"{folder_name}_timelapse.mp4")
        folder_config.output_file = output_file
        
        logger.info(f"Processing folder: {folder_path}")
        
        # Generate timelapse
        generator = TimelapseGenerator(folder_config)
        generator.generate_timelapse()
        
        result['success'] = True
        result['output_file'] = output_file
        logger.info(f"Successfully processed: {folder_path} -> {output_file}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error processing {folder_path}: {e}")
    
    return result

def find_image_folders(base_dir: str, min_images: int = 5) -> List[str]:
    """Find folders containing images."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    folders = []
    
    base_path = Path(base_dir)
    
    for item in base_path.iterdir():
        if item.is_dir():
            # Count images in folder
            image_count = sum(1 for f in item.iterdir() 
                            if f.is_file() and f.suffix.lower() in image_extensions)
            
            if image_count >= min_images:
                folders.append(str(item))
                logger.info(f"Found folder with {image_count} images: {item}")
    
    return sorted(folders)

def create_summary_report(results: List[Dict], output_dir: str):
    """Create a summary report of batch processing results."""
    report_file = os.path.join(output_dir, "batch_processing_report.txt")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    with open(report_file, 'w') as f:
        f.write("Batch Processing Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total folders processed: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n\n")
        
        if successful:
            f.write("SUCCESSFUL PROCESSING:\n")
            f.write("-" * 30 + "\n")
            for result in successful:
                f.write(f"✓ {result['folder_name']}\n")
                f.write(f"  Input: {result['folder']}\n")
                f.write(f"  Output: {result['output_file']}\n\n")
        
        if failed:
            f.write("FAILED PROCESSING:\n")
            f.write("-" * 30 + "\n")
            for result in failed:
                f.write(f"✗ {result['folder_name']}\n")
                f.write(f"  Input: {result['folder']}\n")
                f.write(f"  Error: {result['error']}\n\n")
    
    logger.info(f"Summary report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Batch process multiple folders for timelapse generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all folders in a directory
  python batch_process.py --input-dir /path/to/timelapse/folders

  # Process specific folders
  python batch_process.py --folders folder1 folder2 folder3

  # Use custom config and output directory
  python batch_process.py --input-dir /path/to/folders --config config.json --output-dir /path/to/output

  # Parallel processing with 4 workers
  python batch_process.py --input-dir /path/to/folders --workers 4
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-dir', help="Directory containing image folders")
    input_group.add_argument('--folders', nargs='+', help="Specific folders to process")
    
    # Configuration options
    parser.add_argument('--config', help="Configuration file path")
    parser.add_argument('--output-dir', default='./batch_output', help="Output directory for videos")
    
    # Processing options
    parser.add_argument('--workers', type=int, default=1, help="Number of parallel workers")
    parser.add_argument('--min-images', type=int, default=5, help="Minimum images per folder")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be processed without actually processing")
    
    # Video settings (override config)
    parser.add_argument('--fps', type=int, help="Output frame rate")
    parser.add_argument('--width', type=int, help="Output width")
    parser.add_argument('--height', type=int, help="Output height")
    parser.add_argument('--codec', help="Video codec")
    
    # Processing flags
    parser.add_argument('--no-interpolation', action='store_true', help="Disable frame interpolation")
    parser.add_argument('--align', action='store_true', help="Enable image alignment")
    parser.add_argument('--preview', action='store_true', help="Enable preview mode")
    
    args = parser.parse_args()
    
    # Load base configuration
    if args.config:
        config = TimelapseConfig.from_file(args.config)
    else:
        config = TimelapseConfig()
    
    # Override config with command line arguments
    if args.fps:
        config.fps = args.fps
    if args.width:
        config.width = args.width
    if args.height:
        config.height = args.height
    if args.codec:
        config.codec = args.codec
    if args.no_interpolation:
        config.enable_interpolation = False
    if args.align:
        config.enable_alignment = True
    if args.preview:
        config.preview_mode = True
    
    # Get folders to process
    if args.input_dir:
        folders = find_image_folders(args.input_dir, args.min_images)
    else:
        folders = [os.path.abspath(f) for f in args.folders]
    
    if not folders:
        logger.error("No folders found to process")
        return
    
    logger.info(f"Found {len(folders)} folders to process")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual processing will occur")
        for folder in folders:
            folder_name = Path(folder).name
            output_file = output_dir / f"{folder_name}_timelapse.mp4"
            logger.info(f"Would process: {folder} -> {output_file}")
        return
    
    # Process folders
    results = []
    
    if args.workers == 1:
        # Sequential processing
        for folder in folders:
            result = process_folder(folder, config, str(output_dir))
            results.append(result)
    else:
        # Parallel processing
        logger.info(f"Using {args.workers} parallel workers")
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_folder = {
                executor.submit(process_folder, folder, config, str(output_dir)): folder
                for folder in folders
            }
            
            # Collect results
            for future in as_completed(future_to_folder):
                folder = future_to_folder[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Unexpected error processing {folder}: {e}")
                    results.append({
                        'folder': folder,
                        'folder_name': Path(folder).name,
                        'success': False,
                        'error': str(e),
                        'output_file': None
                    })
    
    # Create summary report
    create_summary_report(results, str(output_dir))
    
    # Final summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    logger.info(f"Batch processing completed!")
    logger.info(f"Total: {len(results)}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logger.warning(f"{failed} folders failed to process. Check the log and report for details.")

if __name__ == "__main__":
    main()
