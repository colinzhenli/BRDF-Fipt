"""Produce a clean fabric crop from a capture, two ways:

  A: direct center-crop of the near-top-down view (keeps slight perspective)
  B: homography rectification of the fabric quad → perfectly flat rectangle

Both brighten the dark HDR capture with gain + gamma.

Usage:
  python make_material_crop.py --material 496 --view 564 --mode A --out figures/material_496_cropA.png
  python make_material_crop.py --material 496 --view 564 --mode B --out figures/material_496_cropB.png \
      --corners "x0,y0 x1,y1 x2,y2 x3,y3"   # TL TR BR BL in full-res px
"""
import argparse
import glob
from pathlib import Path
import numpy as np
import cv2


def load_hdr(material, view):
    base = f"/media/raid/cloth/capture_data/Dataset_Nov11/{material}/hdr"
    f = glob.glob(f"{base}/scan-{view:04d}_*.png")
    if not f:
        raise FileNotFoundError(f"No HDR for view {view} in {base}")
    return cv2.imread(f[0], cv2.IMREAD_UNCHANGED).astype(np.float32)  # BGR 16-bit


def expose(img16, target_median=18000.0, gamma=2.2):
    """Gain so the median maps near target, then gamma to 8-bit."""
    med = np.median(img16[img16 > 0])
    gain = target_median / max(med, 1.0)
    lin = np.clip(img16 * gain, 0, 65535) / 65535.0
    out8 = (np.clip(lin ** (1.0 / gamma), 0, 1) * 255).astype(np.uint8)
    return out8


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--material", type=int, default=496)
    p.add_argument("--view", type=int, default=564)
    p.add_argument("--mode", choices=["A", "B"], default="A")
    p.add_argument("--out", required=True)
    p.add_argument("--crop", type=int, nargs=4, default=[800, 2300, 580, 1560],
                   help="A-mode crop x0 x1 y0 y1 (full-res px)")
    p.add_argument("--corners", type=str, default=None,
                   help="B-mode fabric quad 'x0,y0 x1,y1 x2,y2 x3,y3' (TL TR BR BL)")
    p.add_argument("--rect_w", type=int, default=1500, help="B-mode output width")
    p.add_argument("--target_median", type=float, default=18000.0)
    args = p.parse_args()

    img16 = load_hdr(args.material, args.view)

    if args.mode == "A":
        x0, x1, y0, y1 = args.crop
        crop16 = img16[y0:y1, x0:x1]
        out8 = expose(crop16, args.target_median)
    else:  # B — homography rectify
        if args.corners is None:
            raise ValueError("--corners required for mode B")
        pts = []
        for tok in args.corners.split():
            xx, yy = tok.split(",")
            pts.append([float(xx), float(yy)])
        src = np.array(pts, dtype=np.float32)        # TL TR BR BL
        rect_w = args.rect_w
        rect_h = int(round(rect_w * 2 / 3))          # 3:2
        dst = np.array([[0, 0], [rect_w - 1, 0],
                        [rect_w - 1, rect_h - 1], [0, rect_h - 1]], dtype=np.float32)
        Hmat = cv2.getPerspectiveTransform(src, dst)
        warped16 = cv2.warpPerspective(img16, Hmat, (rect_w, rect_h))
        out8 = expose(warped16, args.target_median)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out8)
    print(f"wrote {out_path}  ({out8.shape[1]}×{out8.shape[0]}, mode {args.mode})")


if __name__ == "__main__":
    main()
