"""Emphasize the DENSE latent grid in the stage-2 pipeline figure.

Two operations:
  1. image:  overlay a perspective-projected 6x4 grid on the tilted fabric
             image (figures/image.png). The quad is auto-detected from the
             white background; grid lines are mapped through the homography
             so they follow the perspective.
  2. grid:   draw a query dot at EVERY cell center of the front-facing
             latent grid (figures/latent_grid.png) — 6x4 = 24 dots, same
             style as the pipeline's query dots — showing each texel has
             its own latent.

Usage:
  python annotate_dense_latent.py --mode image \
      --in figures/image.png --out figures/image_v2.png --cols 6 --rows 4
  python annotate_dense_latent.py --mode grid \
      --in figures/latent_grid.png --out figures/latent_grid_v2.png --cols 6 --rows 4
"""
import argparse
from pathlib import Path
import numpy as np
import cv2


def detect_quad(img, white_thresh=245):
    """Fabric quad corners on a white background. Returns TL, TR, BR, BL."""
    non_white = (img.min(axis=2) < white_thresh).astype(np.uint8) * 255
    non_white = cv2.morphologyEx(non_white, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    non_white = cv2.morphologyEx(non_white, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    pts = c.reshape(-1, 2).astype(np.float32)
    ssum = pts[:, 0] + pts[:, 1]
    sdif = pts[:, 0] - pts[:, 1]
    return np.array([pts[ssum.argmin()],   # TL
                     pts[sdif.argmax()],   # TR
                     pts[ssum.argmax()],   # BR
                     pts[sdif.argmin()]],  # BL
                    dtype=np.float32)


def mode_image(args):
    img = cv2.imread(args.inp, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    quad = detect_quad(img)

    # unit rect -> quad homography (lines stay straight under homography,
    # so mapping endpoints is exact)
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(unit, quad)

    def map_pt(u, v):
        p = H @ np.array([u, v, 1.0])
        return (p[0] / p[2], p[1] / p[2])

    # draw on an overlay, then alpha-blend so the grid reads as a projection
    overlay = img.copy()
    color = tuple(int(c) for c in args.line_color)          # BGR
    thickness = max(1, int(round(w * args.line_frac)))
    for c in range(args.cols + 1):
        u = c / args.cols
        p0 = map_pt(u, 0.0); p1 = map_pt(u, 1.0)
        cv2.line(overlay, (int(round(p0[0])), int(round(p0[1]))),
                 (int(round(p1[0])), int(round(p1[1]))), color, thickness, cv2.LINE_AA)
    for r in range(args.rows + 1):
        v = r / args.rows
        p0 = map_pt(0.0, v); p1 = map_pt(1.0, v)
        cv2.line(overlay, (int(round(p0[0])), int(round(p0[1]))),
                 (int(round(p1[0])), int(round(p1[1]))), color, thickness, cv2.LINE_AA)

    out = cv2.addWeighted(overlay, args.alpha, img, 1 - args.alpha, 0)
    return out


def mode_grid(args):
    img = cv2.imread(args.inp, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    r_dot = max(3, int(round(w * args.dot_frac)))
    ring = max(2, int(round(r_dot * 0.38)))
    fill = tuple(int(c) for c in args.dot_fill)      # BGR
    edge = tuple(int(c) for c in args.dot_edge)      # BGR

    for row in range(args.rows):
        for col in range(args.cols):
            cx = int(round((col + 0.5) / args.cols * w))
            cy = int(round((row + 0.5) / args.rows * h))
            cv2.circle(img, (cx, cy), r_dot, fill, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), r_dot, edge, ring, cv2.LINE_AA)
    return img


def mode_glyph(args):
    """Draw a small latent glyph inside every cell: one large bar (z),
    one middle bar (g), and two small squares (n, t) — black outline,
    white fill — showing each texel stores its own structured latent."""
    img = cv2.imread(args.inp, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    cell_w = w / args.cols
    cell_h = h / args.rows

    # glyph proportions in units of bar height: z=4, g=2, n=1, t=1, gap=0.3
    widths_u = [4.0, 2.0, 1.0, 1.0]
    gap_u = 0.3
    total_u = sum(widths_u) + gap_u * (len(widths_u) - 1)

    bar_h = cell_w * args.glyph_frac / total_u   # so total glyph width = glyph_frac * cell_w
    bar_h = max(6, bar_h)
    fill = tuple(int(c) for c in args.dot_fill)
    edge = (0, 0, 0)
    ring = max(2, int(round(bar_h * 0.14)))

    for row in range(args.rows):
        for col in range(args.cols):
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            total_w = total_u * bar_h
            x = cx - total_w / 2
            y0 = int(round(cy - bar_h / 2))
            y1 = int(round(cy + bar_h / 2))
            for wu in widths_u:
                x0 = int(round(x))
                x1 = int(round(x + wu * bar_h))
                cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
                cv2.rectangle(img, (x0, y0), (x1, y1), edge, ring, cv2.LINE_AA)
                x += (wu + gap_u) * bar_h
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["image", "grid", "glyph"], required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--rows", type=int, default=4)
    # image-mode style
    p.add_argument("--line_color", type=int, nargs=3, default=[235, 235, 230],
                   help="grid line BGR (matches latent grid lines)")
    p.add_argument("--line_frac", type=float, default=0.004)
    p.add_argument("--alpha", type=float, default=0.75,
                   help="grid opacity on the image (1 = solid)")
    # grid-mode style
    p.add_argument("--dot_frac", type=float, default=0.014,
                   help="dot radius as fraction of width")
    p.add_argument("--dot_fill", type=int, nargs=3, default=[250, 248, 245],
                   help="dot fill BGR (near-white)")
    p.add_argument("--dot_edge", type=int, nargs=3, default=[100, 40, 45],
                   help="dot ring BGR (dark navy)")
    # glyph-mode style
    p.add_argument("--glyph_frac", type=float, default=0.62,
                   help="glyph total width as fraction of cell width")
    args = p.parse_args()

    out = {"image": mode_image, "grid": mode_grid, "glyph": mode_glyph}[args.mode](args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print(f"wrote {out_path} ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
