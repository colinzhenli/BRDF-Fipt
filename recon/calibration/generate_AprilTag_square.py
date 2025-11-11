#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate an AprilTag 36h11 grid inside a largest-possible square on A3 paper,
keeping TAG_SIZE_MM and GAP_MM EXACTLY as specified (no scaling).
- A3: 420 x 297 mm (we use landscape)
- Square side defaults to 297 mm (max that fits on A3). Change SQUARE_MM if you switch paper size.

Outputs PNG (300 DPI) and PDF.
Requires: opencv-contrib-python, Pillow
"""

import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    import cv2
    from cv2 import aruco
except Exception as e:
    raise SystemExit("OpenCV contrib is required. Install: pip install --upgrade opencv-contrib-python") from e

# ----------------- CONFIG -----------------
DPI = 300
A3_W_MM, A3_H_MM = 420.0, 297.0   # A3 landscape
SQUARE_MM = 297.0                 # max square that fits on A3 short edge (29.7 cm)

# Keep these UNCHANGED to preserve your previous physical tag & gap size
TAG_SIZE_MM = 16.0                # <-- set to your previous tag size (mm)
GAP_MM      = 3.0                 # <-- set to your previous gap (mm)

DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
FIRST_TAG_ID = 0                  # starting ID for the grid
BORDER_MM = 3.0                   # small white border to avoid printer crop

PNG_PATH = "aprilgrid_A3_square.png"
PDF_PATH = "aprilgrid_A3_square.pdf"
# ------------------------------------------

def mm2px(mm): return int(round(mm * DPI / 25.4))

def main():
    # Sanity check: square must fit on A3
    if SQUARE_MM > min(A3_W_MM, A3_H_MM):
        print(f"[WARN] SQUARE_MM={SQUARE_MM} exceeds A3 short edge. Clamping to {min(A3_W_MM, A3_H_MM)} mm.")
    square_mm = min(SQUARE_MM, min(A3_W_MM, A3_H_MM))

    # Compute how many tags fit per side with given size & gap
    pitch_mm = TAG_SIZE_MM + GAP_MM
    # total = n * TAG_SIZE + (n-1) * GAP <= square_mm  ->  n = floor((square_mm + GAP) / (TAG_SIZE + GAP))
    n = int(math.floor((square_mm + GAP_MM) / pitch_mm))
    if n < 1:
        raise ValueError("Given TAG_SIZE_MM and GAP_MM are too large to fit any tag into the requested square.")

    total_side_mm = n * TAG_SIZE_MM + (n - 1) * GAP_MM
    # Layout on full A3 (landscape)
    page_w_px, page_h_px = mm2px(A3_W_MM), mm2px(A3_H_MM)
    canvas = Image.new("L", (page_w_px, page_h_px), color=255)  # grayscale white background

    # Compute top-left of the square to center it on page
    square_px = mm2px(total_side_mm)
    offset_x = (page_w_px - square_px) // 2
    offset_y = (page_h_px - square_px) // 2

    # Also ensure outer border exists
    border_px = mm2px(BORDER_MM)
    if offset_x < border_px or offset_y < border_px:
        print("[WARN] Margins are tight; consider reducing SQUARE_MM or adding margins.")

    tag_px = mm2px(TAG_SIZE_MM)
    gap_px = mm2px(GAP_MM)

    # Verify dictionary capacity
    max_ids = 587  # Tag36h11 has 587 unique codes
    total_tags = n * n
    if FIRST_TAG_ID + total_tags > max_ids:
        raise ValueError(f"Need {total_tags} tags but Tag36h11 has only {max_ids} IDs. Reduce grid size or start ID.")

    # Draw grid
    for r in range(n):
        for c in range(n):
            tag_id = FIRST_TAG_ID + (r * n + c)
            # Render tag (OpenCV returns white background, black marker; we keep it as-is)
            marker = aruco.generateImageMarker(DICT, tag_id, tag_px)
            tag_img = Image.fromarray(marker)

            x = offset_x + c * (tag_px + gap_px)
            y = offset_y + r * (tag_px + gap_px)
            canvas.paste(tag_img, (x, y))

    # Convert to RGB for nicer PDF/PIL handling
    rgb = canvas.convert("RGB")
    rgb.save(PNG_PATH, dpi=(DPI, DPI))
    rgb.save(PDF_PATH, "PDF", resolution=DPI)

    print(f"Done. Saved:\n- {PNG_PATH}\n- {PDF_PATH}")
    print(f"Grid: {n} x {n} tags (total {total_tags})  |  tag: {TAG_SIZE_MM} mm  gap: {GAP_MM} mm")
    print(f"Square side (actual): {total_side_mm:.2f} mm, centered on A3 ({A3_W_MM} x {A3_H_MM} mm)")

if __name__ == "__main__":
    main()
