#!/usr/bin/env python3
"""
Batch debayer PNGs (mosaic stored as 1ch or 3/4ch grayscale-RGB(A)) to RGB PNGs.

- Input: PNGs containing a Bayer mosaic:
    * 1-channel uint8/uint16, OR
    * 3/4-channel where all channels are identical copies of the mosaic
- Output: RGB PNGs (same filenames) to output folder
- Bit depth preserved: uint8 stays 8-bit; uint16 stays 16-bit
- No white balance or color correction
- Multithreaded

Usage:
  python debayer_folder.py \
      --input /path/to/in \
      --output /path/to/out \
      --pattern RGGB \
      --jobs 8 \
      [--recursive]
"""

import argparse
import concurrent.futures as cf
import os
from pathlib import Path
import sys
import cv2
import numpy as np

BAYER_MAP = {
    "RGGB": cv2.COLOR_BayerRG2RGB,
    "BGGR": cv2.COLOR_BayerBG2RGB,
    "GRBG": cv2.COLOR_BayerGR2RGB,
    "GBRG": cv2.COLOR_BayerGB2RGB,
    # If you want VNG (edge-aware) versions, swap to *_VNG constants.
}

def parse_args():
    ap = argparse.ArgumentParser(description="Batch debayer PNG mosaics to RGB PNGs.")
    ap.add_argument("--input", "-i", required=True, type=Path, help="Input folder with PNGs")
    ap.add_argument("--output", "-o", required=True, type=Path, help="Output folder for RGB PNGs")
    ap.add_argument("--pattern", "-p", required=True, choices=BAYER_MAP.keys(),
                    help="Bayer pattern of input mosaics (e.g., RGGB)")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count(),
                    help="Number of worker threads (default: CPU count)")
    ap.add_argument("--recursive", "-r", action="store_true", help="Recurse into subfolders")
    return ap.parse_args()
def find_pngs(root: Path):
    """Find PNG files in the root folder, optionally recursively."""
    pngs = []
    for png_file in root.glob("*.png"):
        pngs.append(png_file)
    for png_file in root.glob("*.PNG"):
        pngs.append(png_file)
    return pngs

def ensure_out_path(out_root: Path, src_file: Path, in_root: Path, recursive: bool) -> Path:
    if recursive:
        rel = src_file.relative_to(in_root)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        return dst.with_suffix(".png")
    out_root.mkdir(parents=True, exist_ok=True)
    return (out_root / src_file.name).with_suffix(".png")

def collapse_to_single_channel(img: np.ndarray) -> (np.ndarray, str):
    """
    Accepts 1ch, 3ch, or 4ch images.
    - If 1ch: return as-is.
    - If 3/4ch with identical channels: return the first channel.
    - If 3/4ch but not identical: average channels (warn).
    Preserves dtype.
    """
    if img.ndim == 2:
        return img, ""
    if img.ndim == 3:
        if img.shape[2] == 1:
            return img[:, :, 0], ""
        # Use only color channels (ignore alpha if present)
        chans = img[:, :, :3] if img.shape[2] >= 3 else img
        # Check if channels are identical
        c0 = chans[:, :, 0]
        if np.all(chans == chans[:, :, [0]]):
            return c0, ""
        # Not identical—fallback to mean (stays integer for uint8/uint16 via rounding)
        msg = " [WARN] Channels not identical; averaging to single channel."
        if np.issubdtype(img.dtype, np.integer):
            single = np.rint(np.mean(chans.astype(np.float64), axis=2)).astype(img.dtype)
        else:
            single = np.mean(chans, axis=2)
        return single, msg
    return None, " [SKIP] Unsupported image rank."

def debayer_one(src: Path, dst: Path, cvt_code: int) -> str:
    mosaic = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if mosaic is None:
        return f"[SKIP] Failed to read: {src}"

    mono, note = collapse_to_single_channel(mosaic)
    if mono is None:
        return f"[SKIP] Not a valid PNG (rank={mosaic.ndim}): {src}"

    # OpenCV expects HxW single-channel, dtype uint8 or uint16
    if mono.dtype not in (np.uint8, np.uint16):
        # try casting unusual integer types to nearest supported
        if np.issubdtype(mono.dtype, np.integer):
            mono = mono.astype(np.uint16 if mono.max() > 255 else np.uint8)
        else:
            return f"[SKIP] Unsupported dtype: {mono.dtype} in {src}"

    try:
        rgb = cv2.cvtColor(mono, cvt_code)
    except cv2.error as e:
        return f"[SKIP] cvtColor failed for {src}: {e}"

    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), rgb)
    if not ok:
        return f"[FAIL] Write failed: {dst}"
    return f"[OK] {src.name} -> {dst.name} ({rgb.shape[1]}x{rgb.shape[0]}, {rgb.dtype}){note}"

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    pattern = "RGGB"
    num_workers = min(32, os.cpu_count() or 32)
    recursive = True
    image_folder = os.path.join(cfg.exp_folder, "BRDF_recon/masks", "masked_images")
    output_folder = os.path.join(cfg.exp_folder, "BRDF_recon/masks", "debayered_masked_images")
    if not os.path.exists(image_folder):
        print(f"Input path does not exist: {image_folder}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(image_folder):
        print(f"Input is not a directory: {image_folder}", file=sys.stderr)
        sys.exit(1)

    files = find_pngs(Path(image_folder))
    if not files:
        print("No PNG files found.", file=sys.stderr)
        sys.exit(1)

    code = BAYER_MAP[pattern]
    print(f"Found {len(files)} PNG(s). Pattern={pattern}, Jobs={num_workers}")

    with cf.ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = []
        for f in files:
            dst = ensure_out_path(output_folder, f, image_folder, recursive)
            futures.append(ex.submit(debayer_one, f, dst, code))
        done = 0
        for fut in cf.as_completed(futures):
            done += 1
            print(f"[{done}/{len(files)}] {fut.result()}")

if __name__ == "__main__":
    main()
