import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.io import load_camera_turntable_light_metadata, read_light_transforms
import hydra
from omegaconf import DictConfig
from utils.transform import build_rot_about_point, rodrigues_axis_angle
import torch
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

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

def _compute_light_gripper_pose(cfg, json_path):
    """
    Compute the world transform for each light.
    """
    # Compute light transformation matrix
    R_l2g = np.array(cfg.get('R_l2g'), dtype=np.float32)
    t_l2g = np.array(cfg.get('t_l2g'), dtype=np.float32)
    base2_to_base1 = np.array(cfg.get('base2_to_base1'), dtype=np.float32)
    base2_to_base1 = torch.from_numpy(base2_to_base1).float().cuda()
    g2b0 = read_light_transforms(json_path, cfg.turntable.center, cfg.turntable.axis, base2_to_base1).cpu().numpy() # [N, 4, 4]
    l2g = np.eye(4, dtype=np.float32)
    l2g[:3, :3] = R_l2g
    l2g[:3, 3] = t_l2g
    l2w = g2b0 @ l2g
    return l2w, g2b0

def _project_position_to_image(position, w2c, K):
    position_h = np.array([position[0], position[1], position[2], 1.0])
    position_cam = w2c @ position_h
    position_img = K @ position_cam[:3]
    position_px = (int(position_img[0] / position_img[2]), int(position_img[1] / position_img[2]))
    return position_px

def _draw_rectangle_edges(image, center_world, width_world, height_world, w2c, K, Tw2w0, color=(255, 255, 255), thickness=2):
    """
    Draw rectangle edges on the image at the given center position.
    
    Args:
        image: Image to draw on
        center_world: Center position (x, y, z) in world coordinates
        width_world: Width of rectangle in world units
        height_world: Height of rectangle in world units
        w2c: World to camera transformation matrix
        K: Camera intrinsic matrix
        color: Color of the rectangle edges (B, G, R)
        thickness: Thickness of the lines
    """
    
    # Calculate rectangle corners in world space (assuming rectangle is in XY plane)
    half_width = width_world / 2
    half_height = height_world / 2
    
    # Define corners in world space relative to center
    corners_world = [
        [center_world[0] - half_width, center_world[1] - half_height, center_world[2]],  # top_left
        [center_world[0] + half_width, center_world[1] - half_height, center_world[2]],  # top_right
        [center_world[0] - half_width, center_world[1] + half_height, center_world[2]],  # bottom_left
        [center_world[0] + half_width, center_world[1] + half_height, center_world[2]]   # bottom_right
    ]
    
    # Project corners to pixel coordinates
    corners_px = []
    for corner in corners_world:
        corner_px = _project_position_to_image(corner, w2c, K)
        corners_px.append(corner_px)
    
    top_left_px, top_right_px, bottom_left_px, bottom_right_px = corners_px

    # Draw rectangle edges
    cv2.line(image, top_left_px, top_right_px, color, thickness)      # Top edge
    cv2.line(image, top_right_px, bottom_right_px, color, thickness)  # Right edge
    cv2.line(image, bottom_right_px, bottom_left_px, color, thickness) # Bottom edge
    cv2.line(image, bottom_left_px, top_left_px, color, thickness)    # Left edge

    return image

def tone_mapping(x):
    """
    Apply tone mapping to convert HDR image to LDR.
    
    Args:
        x (torch.Tensor): HDR image with values in range [0, 1]
        
    Returns:
        torch.Tensor: LDR image with tone mapping applied
    """
    # Simple Reinhard tone mapping: x / (1 + x)
    return x / (1 + x)

def open_exr(file,img_hw):
    """ open image exr file """
    img = cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
    assert img.shape[0] == img_hw[0]
    assert img.shape[1] == img_hw[1]
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = img[...,[2,1,0]]
    img = torch.from_numpy(img.astype(np.float32))
    return img

def _process_one_image(
        filename, K, w2c, l2w0, g2w0, Tw2w0,
        image_folder, masked_images_dir, rectangle_config
    ):
    
    # Compute w02c transformation matrix
    Tw0w = np.linalg.inv(Tw2w0)  # Inverse of world-to-rotated-world transform
    w02c = w2c @ Tw0w  # Transform from rotated world (w0) to camera
    # Paths & ext
    image_path = os.path.join(image_folder, filename)
    # ---- Read image (use OpenCV for EXR as requested) ----
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = image / 65535.0
    image = image * 255.0
    if image is None:
        return {'original': filename, 'error': 'Image read failed'}

    # ---- Apply mask ----
    marked_image = image.copy()
    # ---- Project light position to image coordinates ----
    # Get light position in world coordinates from l2w transform
    light_position_world = l2w0[:3, 3]  # Extract translation from 4x4 matrix [3]
    
    # Project light position to image coordinates using projection matrix
    light_pos_h = np.array([light_position_world[0], light_position_world[1], light_position_world[2], 1.0])
    light_px = _project_position_to_image(light_pos_h, w02c, K)
    
    # Get gripper position in world coordinates from g2w transform
    gripper_position_world = g2w0[:3, 3]  # Extract translation from 4x4 matrix [3]
    
    # Project gripper position to image coordinates using projection matrix
    gripper_pos_h = np.array([gripper_position_world[0], gripper_position_world[1], gripper_position_world[2], 1.0])
    gripper_px = _project_position_to_image(gripper_pos_h, w02c, K)

    # Draw rectangle edges
    marked_image = _draw_rectangle_edges(marked_image, rectangle_config.center, rectangle_config.width, rectangle_config.length, w02c, K, Tw2w0, color=(0, 255, 0), thickness=10)

    # ---- Save masked image per simple policy ----
    masked_filename = f"marked_{os.path.splitext(filename)[0]}.png"
    masked_out_path = os.path.join(masked_images_dir, masked_filename)
    # Keep PNG values as-is (no normalization), preserve original dtype
    masked_vis = marked_image.copy()
    cv2.circle(masked_vis, light_px, 100, (0, 255, 0), 2)  # Green circle for light position
    
    # Draw gripper position as red circle
    # cv2.circle(masked_vis, gripper_px, 10, (0, 255, 0), 2)  # Red circle for gripper position
    
    ok = cv2.imwrite(masked_out_path, masked_vis)
    if not ok:
        print(f"Failed to save image: {masked_out_path}")
    
def draw_light_gripper_mesh(image_folder, output_folder, json_path, config, emitter_config, rectangle_config):
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
    metadata, camera_metadata, _ =  load_camera_turntable_light_metadata(json_path)
    
    # Intrinsics
    intrinsics = config['intrinsics']
    focal_length = float(intrinsics['focal_length'])
    cx = float(intrinsics['cx'])
    cy = float(intrinsics['cy'])
    distortion = float(intrinsics['distortion'])
    K = np.array([[focal_length, 0, cx],
                  [0, focal_length, cy],
                  [0, 0, 1]], dtype=np.float64)
    
    # Camera-to-global from config
    R_c2g = np.array(config['R_c2g'], dtype=np.float64)
    t_c2g = np.array(config['t_c2g'], dtype=np.float64)

    # ---- Create binary mask (0/1) ----
    l2w0, g2w0 = _compute_light_gripper_pose(emitter_config, json_path)  # [N, 4, 4]

    for img_data in metadata:
        overall_id = img_data["overall_id"]
        camera_id = img_data["camera_id"]
        emitter_id = img_data["emitter_id"]
        camera_info = camera_metadata[str(camera_id)]
        turn_angle = img_data['turn_angle']

        # Build per-frame extrinsics
        c2w = get_c2w_from_robot_pose(camera_info, R_c2g, t_c2g)
        w2c = np.linalg.inv(c2w)
        opengl_to_opencv = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=np.float64)

        w2c = opengl_to_opencv @ w2c
        Tw2w0 = build_rot_about_point(rodrigues_axis_angle(emitter_config.turntable.axis, -turn_angle), emitter_config.turntable.center)
        # Tw2w0 = build_rot_about_point(rodrigues_axis_angle(emitter_config.turntable.axis, -turn_angle), emitter_config.turntable.center)
        _process_one_image(
            img_data['filename'], K, w2c, l2w0[emitter_id], g2w0[emitter_id], Tw2w0, image_folder, masked_images_dir, rectangle_config
        )
        
@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig):
    image_folder = os.path.join(cfg.exp_folder, "Lower_exposure_BRDF_recon/images")
    output_folder = os.path.join(cfg.exp_folder, "Lower_exposure_BRDF_recon/masks/marked_light")
    json_path = os.path.join(cfg.exp_folder, "Lower_exposure_BRDF_recon", "scan_log_0918_reindexed.json")

    draw_light_gripper_mesh(
        image_folder, output_folder, json_path, cfg.renderer.camera, cfg.renderer.emitter, cfg.renderer.mesh.rectangle)

if __name__ == "__main__":
    main()
