#!/usr/bin/env python3
"""
Debayer and color correction script for raw 16-bit PNG images.
Uses the same parameters as ImageSaver_Mask class.
"""

import argparse
import os
import cv2
import numpy as np


# Parameters from ImageSaver_Mask class
BLACK_LEVEL = {"R": 68.0, "G1": 64.0, "G2": 64.0, "B": 116.0}
QE_R, QE_G, QE_B = 1.58056, 1, 1.06588

# CCM from config/data/base.yaml
CCM = np.array([
    [3.6724617, -0.94800931, 0.08428962],
    [-0.44629176, 2.96095854, -1.17898539],
    [-0.47909694, -0.39991418, 2.10705124]
], dtype=np.float64)


def debayer_with_wb(raw16: np.ndarray) -> np.ndarray:
    """
    Apply black level subtraction, debayer, and rough white balance.
    Same logic as ImageSaver_Mask.run() method.
    
    Args:
        raw16: Raw 16-bit Bayer image (HxW, uint16)
    
    Returns:
        img16: Debayered and white-balanced RGB image (HxWx3, uint16)
    """
    # Black level subtraction
    tmp = raw16.astype(np.int32)
    tmp[0::2, 0::2] -= int(round(BLACK_LEVEL["R"]))
    tmp[0::2, 1::2] -= int(round(BLACK_LEVEL["G1"]))
    tmp[1::2, 0::2] -= int(round(BLACK_LEVEL["G2"]))
    tmp[1::2, 1::2] -= int(round(BLACK_LEVEL["B"]))
    np.clip(tmp, 0, 65535, out=tmp)
    raw16 = tmp.astype(np.uint16)
    
    # Demosaic (debayer)
    img16 = cv2.cvtColor(raw16, cv2.COLOR_BAYER_RG2RGB)
    
    # Rough white balance
    imgf = img16.astype(np.float32)
    imgf[..., 0] *= QE_R
    imgf[..., 1] *= QE_G
    imgf[..., 2] *= QE_B
    np.clip(imgf, 0, 65535, out=imgf)
    img16 = imgf.astype(np.uint16)
    
    return img16


def apply_ccm(img16: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    """
    Apply color correction matrix to RGB image.
    corrected_rgb = raw_rgb @ CCM
    
    Args:
        img16: RGB image (HxWx3, uint16)
        ccm: Color correction matrix (3x3)
    
    Returns:
        corrected: Color-corrected RGB image (HxWx3, uint16)
    """
    h, w, c = img16.shape
    imgf = img16.astype(np.float64)
    
    # Reshape to (H*W, 3), apply CCM, reshape back
    pixels = imgf.reshape(-1, 3)
    corrected = pixels @ ccm
    corrected = corrected.reshape(h, w, c)
    
    # Clip to valid range and convert back to uint16
    np.clip(corrected, 0, 65535, out=corrected)
    corrected = corrected.astype(np.uint16)
    
    return corrected


def process_image(input_path: str, output_folder: str):
    """
    Process a raw 16-bit PNG image: debayer, white balance, and color correction.
    
    Args:
        input_path: Path to input raw 16-bit PNG file
        output_folder: Output folder for processed images
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load raw image
    raw16 = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if raw16 is None:
        raise ValueError(f"Failed to load image: {input_path}")
    
    print(f"[INFO] Loaded raw image: {input_path}")
    print(f"[INFO] Shape: {raw16.shape}, dtype: {raw16.dtype}")
    
    # Check if it's a raw Bayer image (single channel)
    if len(raw16.shape) == 3:
        print("[WARN] Input image has 3 channels, expected single-channel Bayer image")
        print("[WARN] Converting to grayscale for processing")
        raw16 = cv2.cvtColor(raw16, cv2.COLOR_BGR2GRAY)
    
    # Debayer with white balance
    rgb16 = debayer_with_wb(raw16)
    print(f"[INFO] Debayered image shape: {rgb16.shape}")
    
    # Apply color correction matrix
    corrected16 = apply_ccm(rgb16, CCM)
    print(f"[INFO] Color-corrected image shape: {corrected16.shape}")
    
    # Generate output filenames
    basename = os.path.splitext(os.path.basename(input_path))[0]
    rgb_path = os.path.join(output_folder, f"{basename}_rgb.png")
    ccm_path = os.path.join(output_folder, f"{basename}_ccm.png")
    
    # Save images (OpenCV uses BGR, so convert from RGB to BGR for saving)
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR))
    cv2.imwrite(ccm_path, cv2.cvtColor(corrected16, cv2.COLOR_RGB2BGR))
    
    print(f"[INFO] Saved RGB image: {rgb_path}")
    print(f"[INFO] Saved color-corrected image: {ccm_path}")
    
    return rgb16, corrected16


def main():
    parser = argparse.ArgumentParser(
        description="Debayer and color correct raw 16-bit PNG images"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input raw 16-bit PNG file"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Output folder for processed images"
    )
    
    args = parser.parse_args()
    
    process_image(args.input_path, args.output_folder)


if __name__ == "__main__":
    main()

