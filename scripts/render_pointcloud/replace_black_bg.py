"""Replace the black background of a masked fabric image with white (or any color).

Robustly isolates the BACKGROUND (not dark pixels inside the fabric) by
flood-filling from the image corners over near-black pixels, so dark threads
within the fabric are preserved.

Usage:
    python replace_black_bg.py --image fabric_on_black.png --out fabric_on_white.png
    python replace_black_bg.py --image in.png --out out.png --bg 255 255 255 --thresh 30
"""
import argparse
from pathlib import Path
import numpy as np
import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bg", type=int, nargs=3, default=[255, 255, 255],
                   help="Replacement background BGR (default white)")
    p.add_argument("--thresh", type=int, default=30,
                   help="Pixels with max channel <= thresh count as 'black'")
    p.add_argument("--from_corners", action="store_true", default=True,
                   help="Only replace black connected to the image border (preserves dark fabric)")
    args = p.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)
    h, w = img.shape[:2]

    # near-black mask
    blackish = (img.max(axis=2) <= args.thresh).astype(np.uint8)

    if args.from_corners:
        # flood fill from the border: keep only black regions touching the edge
        # build a mask that marks border-connected black via connected components
        num, labels = cv2.connectedComponents(blackish, connectivity=8)
        border_labels = set()
        border_labels.update(labels[0, :].tolist())
        border_labels.update(labels[-1, :].tolist())
        border_labels.update(labels[:, 0].tolist())
        border_labels.update(labels[:, -1].tolist())
        border_labels.discard(0)  # 0 = non-black
        bg_mask = np.isin(labels, list(border_labels))
    else:
        bg_mask = blackish.astype(bool)

    out = img.copy()
    out[bg_mask] = np.array(args.bg, dtype=np.uint8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    pct = 100 * bg_mask.sum() / (h * w)
    print(f"wrote {out_path}  (replaced {pct:.1f}% of pixels, thresh={args.thresh})")


if __name__ == "__main__":
    main()
