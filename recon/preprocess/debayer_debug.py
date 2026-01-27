#!/usr/bin/env python3
import argparse
import os
import cv2
import numpy as np

BLACK_LEVEL = {"R": 68.0, "G1": 64.0, "G2": 64.0, "B": 116.0}
QE_R, QE_G, QE_B = 1.58056, 1, 1.06588

CCM = np.array([
    [3.6724617, -0.94800931, 0.08428962],
    [-0.44629176, 2.96095854, -1.17898539],
    [-0.47909694, -0.39991418, 2.10705124]
], dtype=np.float64)

def subtract_black_level(raw16: np.ndarray) -> np.ndarray:
    tmp = raw16.astype(np.int32)
    tmp[0::2, 0::2] -= int(round(BLACK_LEVEL["R"]))   # R
    tmp[0::2, 1::2] -= int(round(BLACK_LEVEL["G1"]))  # G1
    tmp[1::2, 0::2] -= int(round(BLACK_LEVEL["G2"]))  # G2
    tmp[1::2, 1::2] -= int(round(BLACK_LEVEL["B"]))   # B
    np.clip(tmp, 0, 65535, out=tmp)
    return tmp.astype(np.uint16)

def demosaic_best(raw16: np.ndarray, pattern: str, method: str) -> np.ndarray:
    """
    Returns RGB float32 in [0, 65535] (still linear).
    pattern: one of {"RGGB","BGGR","GRBG","GBRG"}
    method: one of {"menon2007","malvar2004","opencv_ea","opencv_vng","opencv_bilinear"}
    """
    method = method.lower()
    pattern = pattern.upper()

    # ---- Colour-Demosaicing path (recommended) ----
    if method in {"menon2007", "malvar2004"}:
        try:
            from colour_demosaicing import (
                demosaicing_CFA_Bayer_Menon2007,
                demosaicing_CFA_Bayer_Malvar2004,
            )
        except ImportError as e:
            raise ImportError(
                "Please install colour-demosaicing:\n"
                "  pip install colour-demosaicing\n"
                "Then re-run with --demosaic menon2007 (recommended)."
            ) from e

        rawf = raw16.astype(np.float32) / 65535.0  # normalize to [0,1]

        if method == "menon2007":
            rgb = demosaicing_CFA_Bayer_Menon2007(rawf, pattern)
        else:
            rgb = demosaicing_CFA_Bayer_Malvar2004(rawf, pattern)

        rgb = np.clip(rgb, 0.0, 1.0)
        rgb16f = rgb * 65535.0
        return rgb16f.astype(np.float32)

    # ---- OpenCV path (fallback) ----
    # Map Bayer pattern to OpenCV conversion code.
    # Note: OpenCV codes are in BGR/RGB variants; we want RGB.
    cv_code_map = {
        ("RGGB", "opencv_bilinear"): cv2.COLOR_BayerRG2RGB,
        ("RGGB", "opencv_vng"):      cv2.COLOR_BayerRG2RGB_VNG,
        ("RGGB", "opencv_ea"):       cv2.COLOR_BayerRG2RGB_EA,

        ("BGGR", "opencv_bilinear"): cv2.COLOR_BayerBG2RGB,
        ("BGGR", "opencv_vng"):      cv2.COLOR_BayerBG2RGB_VNG,
        ("BGGR", "opencv_ea"):       cv2.COLOR_BayerBG2RGB_EA,

        ("GRBG", "opencv_bilinear"): cv2.COLOR_BayerGR2RGB,
        ("GRBG", "opencv_vng"):      cv2.COLOR_BayerGR2RGB_VNG,
        ("GRBG", "opencv_ea"):       cv2.COLOR_BayerGR2RGB_EA,

        ("GBRG", "opencv_bilinear"): cv2.COLOR_BayerGB2RGB,
        ("GBRG", "opencv_vng"):      cv2.COLOR_BayerGB2RGB_VNG,
        ("GBRG", "opencv_ea"):       cv2.COLOR_BayerGB2RGB_EA,
    }
    key = (pattern, method)
    if key not in cv_code_map:
        raise ValueError(f"Unsupported pattern/method: {pattern}, {method}")

    rgb16 = cv2.cvtColor(raw16, cv_code_map[key])
    return rgb16.astype(np.float32)

def apply_white_balance(rgb16f: np.ndarray) -> np.ndarray:
    rgb16f = rgb16f.copy()
    rgb16f[..., 0] *= QE_R
    rgb16f[..., 1] *= QE_G
    rgb16f[..., 2] *= QE_B
    np.clip(rgb16f, 0, 65535, out=rgb16f)
    return rgb16f

def apply_ccm(img16: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    h, w, _ = img16.shape
    pixels = img16.reshape(-1, 3).astype(np.float64)
    corrected = (pixels @ ccm).reshape(h, w, 3)
    np.clip(corrected, 0, 65535, out=corrected)
    return corrected.astype(np.uint16)

DEMOSAIC_METHODS = ["menon2007", "malvar2004", "opencv_ea", "opencv_vng", "opencv_bilinear"]

def process_image(input_path: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    pattern = "RGGB"

    raw16 = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if raw16 is None:
        raise ValueError(f"Failed to load image: {input_path}")
    if raw16.ndim == 3:
        raise ValueError("Expected single-channel Bayer mosaic PNG. Got 3-channel image.")

    raw16 = subtract_black_level(raw16)
    basename = os.path.splitext(os.path.basename(input_path))[0]

    for demosaic in DEMOSAIC_METHODS:
        print(f"[INFO] Processing with {demosaic}...")
        
        rgb16f = demosaic_best(raw16, pattern=pattern, method=demosaic)

        # WB in linear
        rgb16f = apply_white_balance(rgb16f)

        rgb16 = rgb16f.astype(np.uint16)
        corrected16 = apply_ccm(rgb16, CCM)

        rgb_path = os.path.join(output_folder, f"{basename}_rgb_{demosaic}.png")
        ccm_path = os.path.join(output_folder, f"{basename}_ccm_{demosaic}.png")

        cv2.imwrite(rgb_path, cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR))
        cv2.imwrite(ccm_path, cv2.cvtColor(corrected16, cv2.COLOR_RGB2BGR))

        print(f"[OK] Saved: {rgb_path}")
        print(f"[OK] Saved: {ccm_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_folder", type=str)
    args = parser.parse_args()
    process_image(args.input_path, args.output_folder)

if __name__ == "__main__":
    main()
