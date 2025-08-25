#!/usr/bin/env python3
"""
Strip a COLMAP images.txt to poses-only (one header line per image + blank line).

Usage:
  python make_images_poses_only.py --in /path/to/model_txt/images.txt \
                                   --out /path/to/known_model_txt/images.txt
Notes:
  - Keeps IMAGE_ID, (qw qx qy qz tx ty tz), CAMERA_ID, and NAME exactly as-is.
  - Drops the long POINTS2D lines.
  - If you only have BIN files, first convert to TXT:
      colmap model_converter --input_path /path/to/model_bin \
                             --output_path /path/to/model_txt --output_type TXT
"""

import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_path",  required=True, help="input images.txt from COLMAP (TXT)")
    ap.add_argument("--out", dest="out_path", required=True, help="output images.txt (poses only)")
    args = ap.parse_args()

    in_p  = Path(args.in_path)
    out_p = Path(args.out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    lines_out = []
    with in_p.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if not line or line.lstrip().startswith("#"):
            # preserve initial comments (optional)
            if i < 2:  # keep the first two comment lines, like COLMAP's default header
                lines_out.append(line + "\n")
            i += 1
            continue

        toks = line.strip().split()
        # Header lines have at least 10 tokens: IMAGE_ID qw qx qy qz tx ty tz CAMERA_ID NAME...
        if len(toks) >= 10:
            # Write the image header line exactly as-is
            lines_out.append(line + "\n")
            # Write an empty POINTS2D line (required by COLMAP format)
            lines_out.append("\n")

            # Skip the actual POINTS2D line in the input (if present)
            if i + 1 < len(lines):
                nxt = lines[i+1].strip()
                # In standard COLMAP TXT, the next line is points2D "(x, y, point3D_id) ..."
                # which may be empty; either way, advance past one line.
                i += 2
            else:
                i += 1
        else:
            # Unexpected short line; just skip
            i += 1

    with out_p.open("w", encoding="utf-8") as f:
        f.writelines(lines_out)

    print(f"Wrote poses-only images.txt to: {out_p}")

if __name__ == "__main__":
    main()
