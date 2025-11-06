import os
import cv2
import numpy as np
import re

def is_too_purple(image, id, red_blue_threshold=80, green_ratio=0.6):
    """
    Decide whether an image is too purple.
    """
    avg_b = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_r = np.mean(image[:, :, 2])

    rb_avg = (avg_r + avg_b) / 2
    g_ratio = avg_g / (rb_avg + 1e-6)

    if g_ratio < 0.6:
        print("green_ratio", id, g_ratio)

    return g_ratio < green_ratio


def find_purple_image_ids(folder_path, output_file="purple_ids.txt"):
    purple_ids = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(folder_path, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue

            if is_too_purple(img, filename):
                # Extract the number after "scan-"
                match = re.search(r"scan-(\d+)_", filename)
                if match:
                    scan_id = match.group(1)
                    purple_ids.append(scan_id)
                    print(f"Found purple image with scan id: {scan_id}")

    # Save only scan ids to a text file
    output_path = os.path.join(folder_path, output_file)
    with open(output_path, "w") as f:
        for sid in purple_ids:
            f.write(sid + "\n")

    print(f"\nTotal purple images: {len(purple_ids)}")
    print(f"Saved scan IDs to: {output_path}")


if __name__ == "__main__":
    folder = "/media/raid/cloth/No_cable_capture_Sep22/BRDF_recon/images"
    find_purple_image_ids(folder)
