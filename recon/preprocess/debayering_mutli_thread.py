from pathlib import Path
import argparse
import os
import sys
import concurrent.futures as cf

import numpy as np
import cv2
import imageio.v3 as iio
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

BAYER_MAP = {
    "RGGB": cv2.COLOR_BayerRG2RGB_EA,
    "BGGR": cv2.COLOR_BayerBG2RGB_EA,
    "GRBG": cv2.COLOR_BayerGR2RGB_EA,
    "GBRG": cv2.COLOR_BayerGB2RGB_EA,
}

def parse_args():
    ap = argparse.ArgumentParser(description="Debayer EXR mosaics → RGB; save HDR(EXR) and LDR(PNG).")
    ap.add_argument("--input", "-i", required=True, type=Path, help="Input folder with EXR mosaics")
    ap.add_argument("--output", "-o", required=True, type=Path, help="Output folder (HDR EXR here, LDR PNG in sibling 'ldr')")
    ap.add_argument("--pattern", "-p", required=True, choices=BAYER_MAP.keys(), help="Bayer pattern (e.g., RGGB)")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count(), help="Threads")
    return ap.parse_args()

def debayer_and_save(src: Path, dst_exr: Path, dst_png: Path, cvt_code: int) -> str:
    # input: EXR H×W float32 in [0,1]
    mosaic   = cv2.imread(str(src), cv2.IMREAD_UNCHANGED).astype(np.float32)
    mono_u16 = np.rint(mosaic * 65535.0).astype(np.uint16)
    rgb_u16  = cv2.cvtColor(mono_u16, cvt_code)
    rgb_f32  = rgb_u16.astype(np.float32) / 65535.0   # HDR linear [0,1]

    # HDR EXR
    cv2.imwrite(str(dst_exr), rgb_f32)
    # LDR PNG (clip HDR to [0,255])
    rgb_u8 = np.clip(rgb_f32 * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(dst_png), rgb_u8)


    h, w = rgb_f32.shape[:2]
    return f"[OK] {src.name} -> {dst_exr.name} & {dst_png.name} ({w}x{h})"


def find_exrs(root: Path):
    # flat folder; only EXR
    return sorted([*root.glob("*.exr"), *root.glob("*.EXR")])

def main():
    args = parse_args()
    in_root = args.input
    out_root = args.output
    ldr_out_root = out_root.parent / "ldr"

    out_root.mkdir(parents=True, exist_ok=True)
    ldr_out_root.mkdir(parents=True, exist_ok=True)

    files = find_exrs(in_root)
    if not files:
        print("No EXR files found.", file=sys.stderr)
        sys.exit(1)

    code = BAYER_MAP[args.pattern]
    num_workers = min(32, args.jobs or (os.cpu_count() or 32))
    print(f"Found {len(files)} EXR file(s). Pattern={args.pattern}, Jobs={num_workers}")

    with cf.ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = []
        for f in files:
            dst_exr = out_root / f.name.replace(f.suffix, ".exr")
            dst_png = ldr_out_root / f.name.replace(f.suffix, ".png")
            futs.append(ex.submit(debayer_and_save, f, dst_exr, dst_png, code))
        for i, fut in enumerate(cf.as_completed(futs), 1):
            print(f"[{i}/{len(futs)}] {fut.result()}")

if __name__ == "__main__":
    main()
