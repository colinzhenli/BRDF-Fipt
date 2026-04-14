"""Create side-by-side BRDF lobe comparison figures (GT vs Ours vs From-Bonn)."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

BASE = "/media/raid/cloth/output/BRDF/Bonn-Theia2"
GT_DIR   = os.path.join(BASE, "Stage-2_UBO_fabric11_from-Bonn_Logrel-Learnable-factor_run_1/images/brdf_lobes_gt")
BONN_DIR = os.path.join(BASE, "Stage-2_UBO_fabric11_from-Bonn_Logrel-Learnable-factor_run_1/images/brdf_lobes")
REAL_DIR = os.path.join(BASE, "Stage-2_UBO_fabric11_from-Real_Logrel-Learnable-factor_run_1/images/brdf_lobes")
OUT_DIR  = "/home/zla247/projects/BRDF-Fipt/scripts"

# Crop box: remove title (top ~210px), legend (right, starts ~x=920),
# and bottom whitespace (plot ends ~y=880).
# Original image is 1200x1200.
CROP_BOX = (10, 210, 920, 890)  # (left, top, right, bottom)


def load_and_crop(path):
    img = Image.open(path)
    return img.crop(CROP_BOX)


def try_get_font(size):
    """Try to get a nice font, fall back to default."""
    for name in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                 "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"]:
        if os.path.exists(name):
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def make_comparison_figure(vary_type, point_ids, output_name, title=None):
    """
    vary_type: 'wo' or 'wi'
    point_ids: list of point IDs for rows
    title: figure title string
    """
    cols = [
        ("Ground Truth", GT_DIR),
        ("Ours (from Real)", REAL_DIR),
        ("From Bonn", BONN_DIR),
    ]

    # Load one image to get cropped size
    sample_fname = f"polar_brdf_vary_{vary_type}_pt_{point_ids[0]}.png"
    sample = load_and_crop(os.path.join(GT_DIR, sample_fname))
    cw, ch = sample.size  # cropped width, height

    n_rows = len(point_ids)
    n_cols = len(cols)

    # Layout params
    title_h = 60 if title else 0
    col_label_h = 50
    row_label_w = 140
    pad = 10
    legend_w = 200  # space for shared legend on the right

    fig_w = row_label_w + n_cols * cw + (n_cols - 1) * pad + legend_w
    fig_h = title_h + col_label_h + n_rows * ch + (n_rows - 1) * pad

    canvas = Image.new("RGB", (fig_w, fig_h), "white")
    draw = ImageDraw.Draw(canvas)

    font_title = try_get_font(34)
    font_col = try_get_font(28)
    font_row = try_get_font(22)

    # Figure title
    if title:
        bbox = draw.textbbox((0, 0), title, font=font_title)
        tw = bbox[2] - bbox[0]
        draw.text((fig_w // 2 - tw // 2, 12), title, fill="black", font=font_title)

    # Column headers
    for j, (label, _) in enumerate(cols):
        x = row_label_w + j * (cw + pad) + cw // 2
        bbox = draw.textbbox((0, 0), label, font=font_col)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw // 2, title_h + 10), label, fill="black", font=font_col)

    # Rows
    for i, pid in enumerate(point_ids):
        y_top = title_h + col_label_h + i * (ch + pad)

        # Row label
        row_label = f"Point {pid}"
        bbox = draw.textbbox((0, 0), row_label, font=font_row)
        th = bbox[3] - bbox[1]
        draw.text((10, y_top + ch // 2 - th // 2), row_label, fill="black", font=font_row)

        for j, (_, src_dir) in enumerate(cols):
            fname = f"polar_brdf_vary_{vary_type}_pt_{pid}.png"
            fpath = os.path.join(src_dir, fname)
            crop = load_and_crop(fpath)
            x_left = row_label_w + j * (cw + pad)
            canvas.paste(crop, (x_left, y_top))

    # Add shared legend from one of the source images (uncropped, extract legend region)
    sample_full = Image.open(os.path.join(GT_DIR, f"polar_brdf_vary_{vary_type}_pt_{point_ids[0]}.png"))
    # Legend is on the right side of the original 1200x1200 image
    legend_crop = sample_full.crop((970, 130, 1195, 550))
    # Place it on the right side of the canvas, vertically centered
    legend_y = title_h + col_label_h + (n_rows * ch + (n_rows - 1) * pad) // 2 - legend_crop.size[1] // 2
    canvas.paste(legend_crop, (row_label_w + n_cols * (cw + pad) + 10, legend_y))

    out_path = os.path.join(OUT_DIR, output_name)
    canvas.save(out_path, dpi=(150, 150))
    print(f"Saved: {out_path}  ({fig_w}x{fig_h})")


# Figure 1: Fixed wi, vary wo — points 16908, 46047, 2643
make_comparison_figure("wo", [16908, 46047, 2643], "comparison_vary_wo.png",
                       title="BRDF Polar Plot (Fixed wi, Vary wo)")

# Figure 2: Fixed wo, vary wi — points 46047, 78142, 120593
make_comparison_figure("wi", [46047, 78142, 120593], "comparison_vary_wi.png",
                       title="BRDF x cos(theta_i) Polar Plot (Fixed wo, Vary wi)")
