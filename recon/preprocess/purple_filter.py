import os
import cv2
import numpy as np
from tqdm import tqdm

def is_too_purple(image,id, red_blue_threshold=80, green_ratio=0.6):
    """
    Decide whether an image is too purple.
    
    Args:
        image: BGR image (from cv2).
        red_blue_threshold: minimum average of (R + B).
        green_ratio: maximum allowed ratio (G / (R+B)).
    """
    # Compute average values
    avg_b = np.mean(image[:, :, 0])  # Blue
    avg_g = np.mean(image[:, :, 1])  # Green
    avg_r = np.mean(image[:, :, 2])  # Red

    # Purple detection rule: R and B strong, G relatively weak
    rb_avg = (avg_r + avg_b) / 2
    if avg_g / (rb_avg + 1e-6)<0.6:
        print("green_ratio",id,avg_g / (rb_avg + 1e-6))
    #return False
    if avg_g / (rb_avg + 1e-6) < green_ratio:
        return True
    return False

def delete_purple_images(folder_path):
    from tqdm import tqdm
    
    # Get list of image files first to show progress
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    deleted_count = 0
    for filename in tqdm(image_files, desc="Processing images"):
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath)

        if img is None:
            continue

        if is_too_purple(img,filename):
            os.remove(filepath)
            print(f"Deleted: {filename}")
            deleted_count += 1

    print(f"\nTotal deleted: {deleted_count}")

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    delete_purple_images(os.path.join(cfg.exp_folder, "BRDF_recon/masks", "filtered_purple"))

if __name__ == "__main__":
    main()