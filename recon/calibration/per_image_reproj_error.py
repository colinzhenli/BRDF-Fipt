import os
import numpy as np
from collections import defaultdict
from read_write_model import read_cameras_text, read_images_text, read_points3D_text

# Path to your sparse model (TXT export)
sparse_path = "/media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22/sparse"

# Load model
cameras = read_cameras_text(os.path.join(sparse_path, "cameras.txt"))
images = read_images_text(os.path.join(sparse_path, "images.txt"))
points3D = read_points3D_text(os.path.join(sparse_path, "points3D.txt"))

# Accumulate reprojection errors per image
err_sum = defaultdict(float)
counts = defaultdict(int)

for pt in points3D.values():
    for img_id in pt.image_ids:
        err_sum[img_id] += pt.error
        counts[img_id] += 1

# Print per-image statistics
for img_id, img in images.items():
    if counts[img_id] > 0:
        mean_err = err_sum[img_id] / counts[img_id]
        print(f"{img_id:4d} {img.name:40s} obs={counts[img_id]:6d} mean_err={mean_err:.4f}px")
    else:
        print(f"{img_id:4d} {img.name:40s} obs=     0 mean_err=NaN")
