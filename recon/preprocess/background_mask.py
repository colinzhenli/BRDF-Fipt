import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.io import load_camera_light_metadata

def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T

def get_c2w_from_robot_pose(camera_info, R_c2g, t_c2g):
    """ get camera to world matrix from robot pose """
    g2w = build_4x4(camera_info["rotation_matrix"], camera_info["position"])
    c2g = build_4x4(R_c2g, t_c2g)
    c2w = g2w @ c2g
    return c2w

def create_circular_mask_from_turntable(width, height, K, T_g2c, 
                                        turntable_center, turntable_radius):
    """
    Create a mask of the projected turntable (a circle in 3D) as an ellipse in image space.

    Args:
        width (int): Image width (pixels)
        height (int): Image height (pixels)
        K (np.ndarray): Intrinsic matrix (3x3)
        T_g2c (np.ndarray): World/Global to camera transform (4x4) with [R|t] in top-left 3x4
        turntable_center (tuple or array): (x, y, z) center in world coords
        turntable_radius (float): Circle radius in world units

    Returns:
        np.ndarray: Binary mask (height x width), dtype=uint8 (1 inside, 0 outside)
    """
    # --- Unpack extrinsics
    R = T_g2c[:3, :3]
    t = T_g2c[:3, 3]

    c0 = np.asarray(turntable_center, dtype=float).reshape(3)  # (x0, y0, z0)
    r = float(turntable_radius)

    # --- Build plane->image homography H = K [ r1 r2 p0_c ]
    r1 = R[:, 0]
    r2 = R[:, 1]
    p0_c = R @ c0 + t  # plane origin in camera coordinates
    H = K @ np.column_stack([r1, r2, p0_c])  # 3x3

    # --- Plane conic of circle: u^2 + v^2 - r^2 = 0
    Q_plane = np.diag([1.0, 1.0, -r * r])

    # --- Image conic: C_img = H^{-T} Q_plane H^{-1}
    H_inv = np.linalg.inv(H)
    C_img = H_inv.T @ Q_plane @ H_inv

    # For efficient evaluation of x^T C x on a grid, expand to Ax^2 + Bxy + Cy^2 + Dx + Ey + F
    A = C_img[0, 0]
    B = C_img[0, 1] + C_img[1, 0]
    Cc = C_img[1, 1]
    D = C_img[0, 2] + C_img[2, 0]
    E = C_img[1, 2] + C_img[2, 1]
    F = C_img[2, 2]

    # Pixel centers (x,y) with half-pixel offset
    xs = np.arange(width, dtype=float) + 0.5
    ys = np.arange(height, dtype=float) + 0.5
    X, Y = np.meshgrid(xs, ys)  # shape (H,W)

    # Evaluate the quadratic form at each pixel center
    val = (A * X * X) + (B * X * Y) + (Cc * Y * Y) + (D * X) + (E * Y) + F

    # Inside the ellipse corresponds to val <= 0
    mask = (val <= 0).astype(np.uint8)

    return mask

def create_turntable_mask(image_folder, output_folder, json_path, config, 
                         turntable_radius=0.1, turntable_center=(0.0, 0.0, 0.0)):
    """
    Create masks for camera images based on turntable projection.
    
    Args:
        image_folder (str): Path to folder containing input images
        output_folder (str): Path to output folder for masks and masked images
        json_path (str): Path to JSON file containing camera metadata
        config (dict): Configuration containing camera intrinsics
        turntable_radius (float): Radius of the turntable in meters
        turntable_center (tuple): Center position of turntable (x, y, z) in robotic base space
    
    Returns:
        list: List of processed filenames
    """
    # Create output directories
    mask_dir = os.path.join(output_folder, 'masks')
    masked_images_dir = os.path.join(output_folder, 'masked_images')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(masked_images_dir, exist_ok=True)
    
    # Load camera metadata
    metadata, camera_metadata, emitter_metadata = load_camera_light_metadata(json_path)
    
    # Extract camera intrinsics from config
    intrinsics = config['intrinsics']
    width = intrinsics['width']
    height = intrinsics['height']
    focal_length = intrinsics['focal_length']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    
    # Camera intrinsic matrix
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])
    
    # Camera to global transformation from config
    R_c2g = np.array(config['R_c2g'])
    t_c2g = np.array(config['t_c2g'])
    processed_files = []
    
    for img_data in tqdm(metadata, desc="Processing image masks"):
        overall_id = img_data["id"]
        camera_info = camera_metadata[str(overall_id)]
        camera_dict = {
            "position": camera_info["position"],
            "rotation_matrix": camera_info["rotation_matrix"],
            "euler": camera_info["euler"]
        }
        c2w = get_c2w_from_robot_pose(camera_dict, R_c2g, t_c2g)
        w2c = np.linalg.inv(c2w)
        filename = img_data['filename']
        # Load the image
        image_path = os.path.join(image_folder, img_data['filename'])
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        # Create turntable mask
        mask = create_circular_mask_from_turntable(width, height, K, w2c, turntable_center, turntable_radius)
        
        # Apply mask to image
        masked_image = image.copy()
        masked_image[mask == 0] = 0  # Set background to black
        
        # Save mask
        mask_filename = f"mask_{filename}"
        mask_path = os.path.join(mask_dir, mask_filename)
        cv2.imwrite(mask_path, mask * 255)  # Convert to 0-255 range
        
        # Save masked image
        masked_filename = f"masked_{filename}"
        masked_path = os.path.join(masked_images_dir, masked_filename)
        cv2.imwrite(masked_path, masked_image)
        
        processed_files.append({
            'original': filename,
            'mask': mask_filename,
            'masked': masked_filename
        })
        
        print(f"Processed: {filename}")
    
    return processed_files

def process_images_with_turntable_mask(image_folder, output_folder, json_path, 
                                     config, turntable_radius=0.1, 
                                     turntable_center=(0.0, 0.0, 0.0),
                                     ):
    """
    Process all images in a folder with turntable masking.
    
    Args:
        image_folder (str): Path to input images
        output_folder (str): Path to output directory
        json_path (str): Path to camera metadata JSON
        config (dict): Configuration containing camera intrinsics
        turntable_radius (float): Turntable radius in meters
        turntable_center (tuple): Turntable center coordinates
    
    Returns:
        list: List of processed files information
    """
    
    # Process images
    processed_files = create_turntable_mask(
        image_folder, output_folder, json_path, config,
        turntable_radius, turntable_center
    )
    
    # Save processing summary
    summary_path = os.path.join(output_folder, 'processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'processed_files': processed_files,
            'turntable_radius': turntable_radius,
            'turntable_center': turntable_center,
            'total_processed': len(processed_files)
        }, f, indent=2)
    
    print(f"Processed {len(processed_files)} images")
    print(f"Results saved to: {output_folder}")
    
    return processed_files


import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    image_folder = os.path.join(cfg.exp_folder, "BRDF_recon", "original_images")
    output_folder = os.path.join(cfg.exp_folder, "BRDF_recon", "masks")
    json_path = os.path.join(cfg.exp_folder, "scan_log_0901.json")
    turntable_radius = 0.16
    turntable_center = (0.2115, -0.1961, -0.06)
    process_images_with_turntable_mask(image_folder, output_folder, json_path, cfg.renderer.camera, turntable_radius, turntable_center)

if __name__ == "__main__":
    main()