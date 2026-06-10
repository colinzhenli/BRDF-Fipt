"""Crop a material image to 3:2 and overlay an N×M thin grid (latent-grid look).

Usage:
    python add_latent_grid.py --image /path/to/material_crop.png \
        --out /path/to/material_496_latentgrid.png \
        --cols 6 --rows 4

Produces a 3:2 center-cropped version of the input with a thin light grid drawn
on top (default 6 columns × 4 rows), matching the reference tile-grid style used
to represent the dense latent texture T_z.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Input material crop image")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--cols", type=int, default=6, help="Number of grid columns")
    p.add_argument("--rows", type=int, default=4, help="Number of grid rows")
    p.add_argument("--target_w", type=int, default=1200, help="Output width px (height = 2/3 * width)")
    p.add_argument("--line_color", type=int, nargs=3, default=[235, 235, 230],
                   help="Grid line BGR color (default light cream)")
    p.add_argument("--line_frac", type=float, default=0.004,
                   help="Grid line thickness as a fraction of the image width")
    p.add_argument("--gap_frac", type=float, default=0.0,
                   help="Optional gap between tiles as a fraction of cell size (0 = single line)")
    p.add_argument("--border_color", type=int, nargs=3, default=[0, 0, 0],
                   help="Outer boundary BGR color (default black)")
    p.add_argument("--border_frac", type=float, default=0.012,
                   help="Outer boundary thickness as a fraction of image width (0 = no border)")
    args = p.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read {args.image}")
    h, w = img.shape[:2]

    # ----- center-crop to 3:2 (w:h) -----
    target_ar = 3.0 / 2.0
    cur_ar = w / h
    if cur_ar > target_ar:
        # too wide → crop width
        new_w = int(round(h * target_ar))
        x0 = (w - new_w) // 2
        crop = img[:, x0:x0 + new_w]
    else:
        # too tall → crop height
        new_h = int(round(w / target_ar))
        y0 = (h - new_h) // 2
        crop = img[y0:y0 + new_h, :]

    # ----- resize to target -----
    out_w = args.target_w
    out_h = int(round(out_w / target_ar))
    crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)

    # ----- draw N×M grid -----
    color = tuple(int(c) for c in args.line_color)
    thickness = max(1, int(round(out_w * args.line_frac)))

    cols, rows = args.cols, args.rows
    # internal + border lines
    for c in range(cols + 1):
        x = int(round(c * out_w / cols))
        x = min(max(x, thickness // 2), out_w - 1 - thickness // 2)
        cv2.line(crop, (x, 0), (x, out_h - 1), color, thickness, lineType=cv2.LINE_AA)
    for r in range(rows + 1):
        y = int(round(r * out_h / rows))
        y = min(max(y, thickness // 2), out_h - 1 - thickness // 2)
        cv2.line(crop, (0, y), (out_w - 1, y), color, thickness, lineType=cv2.LINE_AA)

    # ----- outer black boundary -----
    if args.border_frac > 0:
        bt = max(1, int(round(out_w * args.border_frac)))
        bcolor = tuple(int(c) for c in args.border_color)
        cv2.rectangle(crop, (bt // 2, bt // 2),
                      (out_w - 1 - bt // 2, out_h - 1 - bt // 2), bcolor, bt, lineType=cv2.LINE_AA)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    print(f"wrote {out_path}  ({out_w}×{out_h}, {cols}×{rows} grid, line {thickness}px, "
          f"border {int(out_w*args.border_frac)}px)")


if __name__ == "__main__":
    main()
