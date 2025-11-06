import cv2
import numpy as np
import os
import glob

def overlay_nonblack(fg_path, bg_path, output_path):
    fg = cv2.imread(fg_path, cv2.IMREAD_COLOR)
    bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)

    if fg is None or bg is None:
        raise ValueError(f"Could not load image(s):\n  FG: {fg_path}\n  BG: {bg_path}")
    if fg.shape != bg.shape:
        raise ValueError(f"Image size mismatch:\n  FG: {fg.shape}\n  BG: {bg.shape}")

    mask = np.any(fg > 10, axis=2)  # non-black pixel mask
    merged = bg.copy()
    merged[mask] = fg[mask]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, merged)
    print(f"Saved merged image to {output_path}")


def batch_overlay_nonblack(image_dir, output_dir):
    # Find all result_view and gt_view images
    result_images = sorted(glob.glob(os.path.join(image_dir, "result_view*.png")))
    gt_images = sorted(glob.glob(os.path.join(image_dir, "gt_view*.png")))

    # Map images by view suffix
    result_map = {os.path.basename(path).replace("result_view", ""): path for path in result_images}
    gt_map = {os.path.basename(path).replace("gt_view", ""): path for path in gt_images}

    # Process only matched views
    for view_suffix in sorted(result_map.keys()):
        if view_suffix in gt_map:
            fg_path = result_map[view_suffix]
            bg_path = gt_map[view_suffix]
            output_path = os.path.join(output_dir, f"merged_view{view_suffix}")
            overlay_nonblack(fg_path, bg_path, output_path)
        else:
            print(f"Warning: No matching gt_view for result_view{view_suffix}")


# ===== Run this block =====
image_folder = "/media/raid/cloth/capture_data/output_h/moe_lantent_light_normal/real/moe_lantent_light_normal/images"
output_folder = image_folder  # You can change this if you want to save elsewhere
batch_overlay_nonblack(image_folder, output_folder)
