#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate TWO AprilTag 36h11 markers on an A4 page.
Each tag is 60 mm x 60 mm (6 cm).
Outputs: PNG (300 DPI) and PDF.

Dependencies:
    pip install --upgrade opencv-contrib-python pillow numpy
"""

import sys
import numpy as np
from PIL import Image, ImageDraw
try:
    import cv2
except Exception as e:
    print("ERROR: OpenCV is required. Install with: pip install --upgrade opencv-contrib-python", file=sys.stderr)
    raise

# =================== CONFIG (millimeters) ===================
# A4 size (landscape or portrait doesn't matter; we use 297 x 210 here)
BOARD_W_MM = 210.0   # width  (short edge, typical "portrait")
BOARD_H_MM = 297.0   # height (long edge)

TAG_SIZE_MM = 40.0   # 6 cm x 6 cm
TAG_GAP_MM  = 30.0   # gap between the two tags

DPI = 300
# ============================================================

def mm_to_px(mm, dpi=DPI):
    return int(round(mm * dpi / 25.4))

def ensure_apriltag_dict():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV is installed but the 'aruco' module is missing. "
            "Install: pip install --upgrade opencv-contrib-python"
        )
    if not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
        raise RuntimeError(
            "Your OpenCV build does not include DICT_APRILTAG_36h11.\n"
            "Try: pip install --upgrade opencv-contrib-python (>=4.7)"
        )
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

def generate_marker(dict36h11, tag_id, size_px):
    # OpenCV 4.7+ aruco.generateImageMarker returns a numpy array
    m = cv2.aruco.generateImageMarker(dict36h11, int(tag_id), int(size_px))
    pil = Image.fromarray(m).convert("RGB")
    return pil

def main():
    # 1. AprilTag dictionary
    aruco_dict = ensure_apriltag_dict()

    # 2. Canvas size in pixels (A4 @ 300 dpi)
    BW = mm_to_px(BOARD_W_MM)
    BH = mm_to_px(BOARD_H_MM)

    # 3. Tag size and gap in pixels
    tag_px = mm_to_px(TAG_SIZE_MM)
    gap_px = mm_to_px(TAG_GAP_MM)

    # 4. Create white canvas
    canvas = Image.new("RGB", (BW, BH), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 5. Layout: two tags centered horizontally with equal margins
    #    total occupied width = 2 * tag_px + gap_px
    total_tags_w = 2 * tag_px + gap_px
    margin_x = (BW - total_tags_w) // 2
    center_y = BH // 2

    # Centers of the two tags
    center1 = (margin_x + tag_px // 2, center_y)
    center2 = (margin_x + tag_px // 2 + tag_px + gap_px, center_y)

    # 6. Generate the two tags (IDs 0 and 1)
    tag_ids = [0, 1]

    def paste_tag(tag_id, center):
        tag_img = generate_marker(aruco_dict, tag_id, tag_px)
        w, h = tag_img.size
        cx, cy = center
        x = int(round(cx - w / 2))
        y = int(round(cy - h / 2))
        canvas.paste(tag_img, (x, y))

    paste_tag(tag_ids[0], center1)
    paste_tag(tag_ids[1], center2)

    # (Optional) draw a thin border around each tag to visually verify size
    # for cx, cy in [center1, center2]:
    #     x0 = cx - tag_px // 2
    #     y0 = cy - tag_px // 2
    #     x1 = x0 + tag_px
    #     y1 = y0 + tag_px
    #     draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=2)

    # 7. Save PNG & PDF
    png_path = "apriltag_two_4cm_A4.png"
    pdf_path = "apriltag_two_4cm_A4.pdf"
    canvas.save(png_path, "PNG", dpi=(DPI, DPI))
    canvas.save(pdf_path, "PDF", resolution=DPI)

    print("Saved:", png_path, "and", pdf_path)
    print("Each tag is 60 mm x 60 mm (6 cm) on A4 at 300 DPI. "
          "Print with 100% scale / no 'fit to page'.")

if __name__ == "__main__":
    main()
