#!/usr/bin/env python3
"""
Script to combine result images into a video.
Reads result_view_*.png files from a folder and creates a video.

Usage:
    python images_to_video.py <input_folder> [--fps 20] [--pattern result_view] [--output output.mp4]
"""

import os
import re
import argparse
import cv2
import numpy as np
from pathlib import Path


def natural_sort_key(filename):
    """
    Sort key for natural sorting of filenames like result_view_1_0.png, result_view_2_0.png, etc.
    Extracts batch_idx and sample_idx from the filename.
    """
    # Match pattern like result_view_1_0.png or gt_view_1_0.png
    match = re.search(r'_(\d+)_(\d+)\.png$', filename)
    if match:
        batch_idx = int(match.group(1))
        sample_idx = int(match.group(2))
        return (batch_idx, sample_idx)
    return (float('inf'), float('inf'))


def get_image_files(folder, pattern='result_view'):
    """
    Get all image files matching the pattern from the folder.
    
    Args:
        folder: Path to the folder containing images
        pattern: Prefix pattern to match (e.g., 'result_view', 'gt_view')
    
    Returns:
        List of sorted image file paths
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    # Find all matching PNG files
    image_files = list(folder.glob(f'{pattern}_*.png'))
    
    if not image_files:
        raise ValueError(f"No images found matching pattern '{pattern}_*.png' in {folder}")
    
    # Sort naturally by batch_idx and sample_idx
    image_files.sort(key=lambda x: natural_sort_key(x.name))
    
    return image_files


def images_to_video(image_files, output_path, fps=20):
    """
    Combine images into a video.
    
    Args:
        image_files: List of image file paths (sorted)
        output_path: Path to save the output video
        fps: Frames per second for the video
    """
    if not image_files:
        raise ValueError("No image files provided")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]), cv2.IMREAD_UNCHANGED)
    if first_image is None:
        raise ValueError(f"Could not read image: {image_files[0]}")
    
    height, width = first_image.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    print(f"Total frames: {len(image_files)}")
    print(f"FPS: {fps}")
    print(f"Expected duration: {len(image_files) / fps:.2f} seconds")
    
    # Create video writer
    # Use mp4v codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_path}")
    
    print(f"Writing video to: {output_path}")
    
    for i, image_path in enumerate(image_files):
        # Read image (might be 16-bit)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        
        if image is None:
            print(f"Warning: Could not read image {image_path}, skipping...")
            continue
        
        # Convert 16-bit to 8-bit if necessary
        if image.dtype == np.uint16:
            # Scale from 16-bit (0-65535) to 8-bit (0-255)
            image = (image / 256).astype(np.uint8)
        
        # Ensure image is BGR (3 channels) for video
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        video_writer.write(image)
        
        if (i + 1) % 50 == 0 or i == len(image_files) - 1:
            print(f"Progress: {i + 1}/{len(image_files)} frames")
    
    video_writer.release()
    print(f"Video saved successfully: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Combine result images into a video.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - convert result_view images to video
    python images_to_video.py /path/to/images/
    
    # Specify custom fps
    python images_to_video.py /path/to/images/ --fps 30
    
    # Convert ground truth images instead
    python images_to_video.py /path/to/images/ --pattern gt_view
    
    # Specify custom output filename
    python images_to_video.py /path/to/images/ --output my_video.mp4
        """
    )
    
    parser.add_argument(
        'input_folder',
        type=str,
        help='Path to the folder containing result images'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='Frames per second for the output video (default: 20)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='result_view',
        help='Image filename pattern to match (default: result_view)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video filename (default: {pattern}_video.mp4 in the same folder)'
    )
    
    args = parser.parse_args()
    
    # Get image files
    image_files = get_image_files(args.input_folder, args.pattern)
    print(f"Found {len(image_files)} images matching pattern '{args.pattern}_*.png'")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
        # If output is just a filename (no directory), save in input folder
        if output_path.parent == Path('.'):
            output_path = Path(args.input_folder) / output_path
    else:
        output_path = Path(args.input_folder) / f'{args.pattern}_video.mp4'
    
    # Create video
    images_to_video(image_files, output_path, args.fps)


if __name__ == '__main__':
    main()
