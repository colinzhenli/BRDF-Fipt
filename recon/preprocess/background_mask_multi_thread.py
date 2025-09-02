import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.io import load_camera_light_metadata

# ----------------------- math & geometry -----------------------

def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T

def get_c2w_from_robot_pose(camera_info, R_c2g, t_c2g):
    """ get camera to world matrix from robot pose """
    g2w = build_4x4(np.asarray(camera_info["rotation_matrix"], float),
                    np.asarray(camera_info["position"], float))
    c2g = build_4x4(np.asarray(R_c2g, float), np.asarray(t_c2g, float))
    c2w = g2w @ c2g
    return c2w

def _H_from_extrinsics(K, T_g2c, center_w):
    """Homography from plane z=z0 (world normal [0,0,1]) to image."""
    R = T_g2c[:3, :3]
    t = T_g2c[:3, 3]
    r1 = R[:, 0]
    r2 = R[:, 1]
    p0_c = R @ center_w + t  # plane origin in camera coords
    H = K @ np.column_stack([r1, r2, p0_c])  # 3x3
    return H

def _conic_from_H(H, radius):
    """C_img = H^{-T} diag(1,1,-r^2) H^{-1}; return A,B,C,D,E,F of Ax^2 + Bxy + Cy^2 + Dx + Ey + F."""
    Q_plane = np.diag([1.0, 1.0, -float(radius) ** 2])
    H_inv = np.linalg.inv(H)
    C_img = H_inv.T @ Q_plane @ H_inv
    A  = C_img[0, 0]
    B  = C_img[0, 1] + C_img[1, 0]
    Cc = C_img[1, 1]
    D  = C_img[0, 2] + C_img[2, 0]
    E  = C_img[1, 2] + C_img[2, 1]
    F  = C_img[2, 2]
    return A, B, Cc, D, E, F

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
    center_w = np.asarray(turntable_center, dtype=np.float64).reshape(3)
    H = _H_from_extrinsics(np.asarray(K, dtype=np.float64),
                           np.asarray(T_g2c, dtype=np.float64),
                           center_w)
    A, B, Cc, D, E, F = _conic_from_H(H, turntable_radius)

    xs = np.arange(width, dtype=np.float32) + 0.5
    ys = np.arange(height, dtype=np.float32) + 0.5
    X, Y = np.meshgrid(xs, ys)  # (H,W)

    val = (A * X * X) + (B * X * Y) + (Cc * Y * Y) + (D * X) + (E * Y) + F
    mask = (val <= 0).astype(np.uint8)
    return mask

# ----------------------- threaded worker -----------------------

def _process_one_image_task(task):
    (
        filename, width, height, K, R_c2g, t_c2g,
        camera_info, turntable_center, turntable_radius,
        image_folder, mask_dir, masked_images_dir
    ) = task

    try:
        # Build per-frame extrinsics
        c2w = get_c2w_from_robot_pose(camera_info, R_c2g, t_c2g)
        w2c = np.linalg.inv(c2w)

        # Load image
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None

        # Create mask (exact conic)
        mask = create_circular_mask_from_turntable(
            width, height, K, w2c, turntable_center, turntable_radius
        )

        # Apply mask
        masked_image = image.copy()
        masked_image[mask == 0] = 0

        # Save outputs
        mask_filename = f"mask_{filename}"
        masked_filename = f"masked_{filename}"
        cv2.imwrite(os.path.join(mask_dir, mask_filename), (mask * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(masked_images_dir, masked_filename), masked_image)

        return {'original': filename, 'mask': mask_filename, 'masked': masked_filename}
    except Exception as e:
        # If something goes wrong, return a marker to keep the pipeline going
        return {'original': filename, 'error': str(e)}

# ----------------------- public API (threaded) -----------------------

def create_turntable_mask(image_folder, output_folder, json_path, config, 
                         turntable_radius=0.1, turntable_center=(0.0, 0.0, 0.0),
                         num_workers=None):
    """
    Create masks for camera images based on turntable projection using multithreading.

    Args:
        image_folder (str): Path to folder containing input images
        output_folder (str): Path to output folder for masks and masked images
        json_path (str): Path to JSON file containing camera metadata
        config (dict): Configuration containing camera intrinsics (cfg.renderer.camera)
        turntable_radius (float): Radius of the turntable in meters
        turntable_center (tuple): Center position of turntable (x, y, z) in world/robot base space
        num_workers (int or None): Number of threads (default: min(8, os.cpu_count()))

    Returns:
        list: List of processed file info dicts
    """
    # Output dirs
    mask_dir = os.path.join(output_folder, 'masks')
    masked_images_dir = os.path.join(output_folder, 'masked_images')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(masked_images_dir, exist_ok=True)
    
    # Load camera metadata
    metadata, camera_metadata, emitter_metadata = load_camera_light_metadata(json_path)
    
    # Intrinsics
    intrinsics = config['intrinsics']
    width  = int(intrinsics['width'])
    height = int(intrinsics['height'])
    focal_length = float(intrinsics['focal_length'])
    cx = float(intrinsics['cx'])
    cy = float(intrinsics['cy'])
    K = np.array([[focal_length, 0, cx],
                  [0, focal_length, cy],
                  [0, 0, 1]], dtype=np.float64)
    
    # Camera-to-global from config
    R_c2g = np.array(config['R_c2g'], dtype=np.float64)
    t_c2g = np.array(config['t_c2g'], dtype=np.float64)

    # Build tasks
    tasks = []
    for img_data in metadata:
        overall_id = img_data["id"]
        camera_info = camera_metadata[str(overall_id)]
        camera_dict = {
            "position": camera_info["position"],
            "rotation_matrix": camera_info["rotation_matrix"],
            "euler": camera_info["euler"]
        }
        tasks.append((
            img_data['filename'], width, height, K, R_c2g, t_c2g,
            camera_dict, np.asarray(turntable_center, dtype=np.float64),
            float(turntable_radius), image_folder, mask_dir, masked_images_dir
        ))

    processed_files = []
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 8)

    # Threaded execution
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_process_one_image_task, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing image masks"):
            res = fut.result()
            if res is not None:
                processed_files.append(res)

    return processed_files

def process_images_with_turntable_mask(image_folder, output_folder, json_path, 
                                     config, turntable_radius=0.1, 
                                     turntable_center=(0.0, 0.0, 0.0),
                                     num_workers=None):
    """
    Process all images in a folder with turntable masking using multiple threads.
    """
    processed_files = create_turntable_mask(
        image_folder, output_folder, json_path, config,
        turntable_radius, turntable_center, num_workers=num_workers
    )
    
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

# ----------------------- hydra entry -----------------------

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    image_folder = os.path.join(cfg.exp_folder, "BRDF_recon", "original_images")
    output_folder = os.path.join(cfg.exp_folder, "BRDF_recon", "masks")
    json_path = os.path.join(cfg.exp_folder, "scan_log_0901.json")
    turntable_radius = 0.16
    turntable_center = (0.2115, -0.1961, -0.06)

    # threads only
    num_workers = min(32, os.cpu_count() or 32)

    process_images_with_turntable_mask(
        image_folder, output_folder, json_path, cfg.renderer.camera,
        turntable_radius, turntable_center,
        num_workers=num_workers
    )

if __name__ == "__main__":
    main()
