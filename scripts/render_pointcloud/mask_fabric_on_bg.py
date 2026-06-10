"""Mask the fabric out of a full capture and composite it on a solid background.

Detects the fabric (purple) region, exposure-matches it, sets everything else
to a chosen background colour (white by default), and crops to the fabric
bounding box (keeping the natural tilt).

Usage:
    python mask_fabric_on_bg.py --material 496 --view 170 --out figures/fabric_on_white.png
    python mask_fabric_on_bg.py --material 496 --view 170 --bg 0 0 0 --out fabric_on_black.png
"""
import argparse
import glob
from pathlib import Path
import numpy as np
import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--material", type=int, default=496)
    p.add_argument("--view", type=int, default=170)
    p.add_argument("--out", required=True)
    p.add_argument("--bg", type=int, nargs=3, default=[255, 255, 255], help="background BGR")
    p.add_argument("--target_median", type=float, default=18000.0)
    p.add_argument("--gamma", type=float, default=2.2)
    p.add_argument("--margin", type=int, default=30, help="crop margin around fabric (px)")
    p.add_argument("--hue_lo", type=int, default=120)
    p.add_argument("--hue_hi", type=int, default=170)
    p.add_argument("--mask_mode", choices=["quad", "contour"], default="quad",
                   help="quad = clean quadrilateral boundary; contour = raw jagged edge")
    p.add_argument("--quad_shrink", type=float, default=0.08,
                   help="fraction to shrink the quad toward center (clips corner magnets)")
    args = p.parse_args()

    base = f"/media/raid/cloth/capture_data/Dataset_Nov11/{args.material}/hdr"
    f = glob.glob(f"{base}/scan-{args.view:04d}_*.png")
    if not f:
        raise FileNotFoundError(f"No HDR for view {args.view} in {base}")
    img16 = cv2.imread(f[0], cv2.IMREAD_UNCHANGED).astype(np.float32)  # BGR 16-bit

    # --- rough exposure just for detection ---
    rough = (np.clip(img16 / (img16.max() or 1) * 5, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(rough, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((H > args.hue_lo) & (H < args.hue_hi) & (S > 30) & (V > 25) & (V < 235)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))

    # largest contour → fabric region
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    fabric_mask = np.zeros(mask.shape, np.uint8)

    if args.mask_mode == "quad":
        # TRUE perspective quadrilateral (NOT a rectangle): the fabric viewed at
        # an angle projects to a general 4-gon. Use the convex hull's 4 extreme
        # corners (min/max of x+y and x-y) which are robust for a convex quad.
        # Try adaptive approxPolyDP for an exact 4-point fit first; fall back to
        # the extreme-corner method. (Never minAreaRect — that forces a rectangle.)
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
                break
        if quad is None:
            pts = c.reshape(-1, 2).astype(np.float32)
            ssum = pts[:, 0] + pts[:, 1]
            sdif = pts[:, 0] - pts[:, 1]
            quad = np.array([pts[ssum.argmin()],   # top-left
                             pts[sdif.argmax()],   # top-right
                             pts[ssum.argmax()],   # bottom-right
                             pts[sdif.argmin()]],  # bottom-left
                            dtype=np.float32)
        # shrink toward centroid to clip the corner magnets (keeps the quad shape)
        centroid = quad.mean(axis=0)
        quad = centroid + (1.0 - args.quad_shrink) * (quad - centroid)
        cv2.fillConvexPoly(fabric_mask, quad.astype(np.int32), 255)
    else:  # 'contour' — raw filled contour (jagged thread-level edge)
        cv2.drawContours(fabric_mask, [c], -1, 255, cv2.FILLED)

    fabric_bool = fabric_mask > 0

    # --- proper exposure from fabric-region median ---
    fabric_med = np.median(img16[fabric_bool])
    gain = args.target_median / max(fabric_med, 1.0)
    exposed = (np.clip(img16 * gain, 0, 65535) / 65535.0) ** (1 / args.gamma)
    exposed = (np.clip(exposed, 0, 1) * 255).astype(np.uint8)

    # --- composite on background ---
    out = exposed.copy()
    out[~fabric_bool] = np.array(args.bg, dtype=np.uint8)

    # --- crop to fabric bbox + margin ---
    ys, xs = np.where(fabric_bool)
    y0 = max(0, ys.min() - args.margin); y1 = min(out.shape[0], ys.max() + args.margin)
    x0 = max(0, xs.min() - args.margin); x1 = min(out.shape[1], xs.max() + args.margin)
    out = out[y0:y1, x0:x1]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    pct = 100 * fabric_bool.sum() / fabric_bool.size
    print(f"wrote {out_path}  ({out.shape[1]}×{out.shape[0]}, fabric={pct:.1f}% of frame)")


if __name__ == "__main__":
    main()
