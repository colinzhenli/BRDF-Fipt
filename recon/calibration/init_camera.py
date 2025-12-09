#!/usr/bin/env python3
import os
import re
import json
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

# COLMAP helpers (same ones you already use)
from read_write_model import read_model, qvec2rotmat, rotmat2qvec, read_points3D_binary, read_points3D_text, detect_model_format, write_images_text, Image

# -------------------- small linear-algebra helpers --------------------

def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3]  = np.asarray(t, float).reshape(3)
    return T

def sim3_umeyama(P, Q, with_scale=True):
    """
    P (N,3): COLMAP camera centres
    Q (N,3): robot (base) camera centres (or targets)
    Returns: s, R, t  such that  Q ≈ s R P + t
    """
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    mu_P, mu_Q = P.mean(0), Q.mean(0)
    P0, Q0 = P - mu_P, Q - mu_Q
    Sigma = (Q0.T @ P0) / len(P)
    U, D, VT = np.linalg.svd(Sigma)
    S = np.diag([1, 1, np.sign(np.linalg.det(U) * np.linalg.det(VT))])
    R = U @ S @ VT
    s = (np.trace(np.diag(D) @ S) / (P0**2).sum() * len(P)) if with_scale else 1.0
    t = mu_Q - s * (R @ mu_P)
    return s, R, t

def rodrigues_axis_angle(n: np.ndarray, degrees: float | np.ndarray) -> np.ndarray:
    """
    Right-handed CCW rotation about axis n by angle_rad (rad).
    Supports scalar or vector of angles → returns (..,3,3).
    """
    th = np.deg2rad(degrees)
    n = np.asarray(n, float)
    n = n / (np.linalg.norm(n) + 1e-15)
    nx, ny, nz = n
    K = np.array([[0.0, -nz,  ny],
                  [nz,  0.0, -nx],
                  [-ny,  nx,  0.0]], dtype=float)
    I = np.eye(3)
    angle = np.asarray(th, float)[..., None, None]
    Sa = np.sin(angle); Ca = np.cos(angle)
    return I + Sa * K + (1.0 - Ca) * (K @ K)

def T_about_point(R, p):
    """Homogeneous transform that rotates by R about world point p (3,)."""
    p = np.asarray(p, float).reshape(3)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = (np.eye(3) - R) @ p
    return T

# -------------------- your existing helpers (kept) --------------------

def parse_colmap_images_txt(images):
    """
    images (read_model output) → dict[name] = 4x4 c2w
    """
    cam_c2w_dict = {}
    for img in images.values():
        rotation = qvec2rotmat(img.qvec)
        translation = img.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([0, 0, 0, 1])[None]], 0)
        opencv_to_opengl = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=np.float64)
        w2c = opencv_to_opengl @ w2c
        c2w = np.linalg.inv(w2c)
        cam_c2w_dict[img.name] = c2w
    return cam_c2w_dict

# -------------------- rotation undo and pose building --------------------

def rotated_c2w(json_entry, R_c2g, t_c2g, rotation_center, rotation_axis):
    """
    Compute T_{c->w0}: camera pose in the 0-angle world by undoing table rotation.
    """
    # gripper->base from robot
    R_g2b = np.asarray(json_entry["rotation_matrix"], dtype=float)
    t_g2b = np.asarray(json_entry["position"], dtype=float) / 1000.0  # mm -> m
    T_g2b = build_4x4(R_g2b, t_g2b)

    # camera->gripper (given)
    T_c2g = build_4x4(R_c2g, t_c2g)

    # camera->base at this frame
    T_c2b = T_g2b @ T_c2g

    theta = float(json_entry.get("turn_angle", 0.0))

    R_undo = rodrigues_axis_angle(rotation_axis, -theta)
    T_b2w0 = T_about_point(R_undo, rotation_center)

    # camera->0-angle world
    return T_b2w0 @ T_c2b

# -------------------- your matching function (unchanged) --------------------

def load_robot_poses_c2w0(scan_log_path, R_c2g, t_c2g, rotation_center, rotation_axis):
    """
    Load robot poses and generate c2w for all entries in scan_log.
    Converts from OpenGL to OpenCV convention for COLMAP.
    """
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)

    # OpenGL to OpenCV conversion matrix
    opengl_to_opencv = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float64)

    robot_poses_opencv = []
    scan_ids = []
    
    for idx, entry in enumerate(scan_log):
        # Robot camera pose in 0-angle world (OpenGL convention)
        c2w_opengl = rotated_c2w(entry, R_c2g, t_c2g, rotation_center, rotation_axis)
        
        # Convert to OpenCV convention for COLMAP
        # c2w (OpenGL) -> w2c (OpenGL) -> w2c (OpenCV) -> c2w (OpenCV)
        w2c_opengl = np.linalg.inv(c2w_opengl)
        w2c_opencv = opengl_to_opencv @ w2c_opengl
        c2w_opencv = np.linalg.inv(w2c_opencv)
        
        robot_poses_opencv.append(c2w_opencv)
        scan_ids.append(idx)
    
    print(f"Generated {len(robot_poses_opencv)} camera poses from scan_log.")
    return robot_poses_opencv, scan_ids

def write_c2w_to_colmap_images(c2w_list, scan_ids, scan_log, output_path, camera_id=1):
    """
    Convert c2w poses to COLMAP images.txt format.
    
    Args:
        c2w_list: List of 4x4 c2w matrices in OpenCV convention
        scan_ids: List of scan_log indices corresponding to each c2w
        scan_log: The scan_log JSON data (list of entries)
        output_path: Path to write images.txt
        camera_id: COLMAP camera ID to use (default 1)
    """
    images_dict = {}
    
    for image_id, (c2w, scan_idx) in enumerate(zip(c2w_list, scan_ids), start=1):
        # Convert c2w to w2c (COLMAP format)
        w2c = np.linalg.inv(c2w)
        
        # Extract rotation and translation
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        
        # Convert rotation matrix to quaternion
        qvec = rotmat2qvec(R)
        
        # Get image filename from scan_log
        entry = scan_log[scan_idx]
        filename = entry.get("filename", f"scan-{scan_idx:04d}.exr")
        
        # Create Image object (with empty 2D points for initialization)
        img = Image(
            id=image_id,
            qvec=qvec,
            tvec=t,
            camera_id=camera_id,
            name=filename,
            xys=np.zeros((0, 2)),  # Empty 2D points
            point3D_ids=np.full(0, -1, dtype=int)  # Empty 3D point associations
        )
        
        images_dict[image_id] = img
    
    # Write to images.txt
    write_images_text(images_dict, output_path)
    print(f"Wrote {len(images_dict)} images to {output_path}")

@hydra.main(version_base=None, config_path="../../config/renderer", config_name="realcapture_area_emitter")
def main(cfg: DictConfig) -> None:
    # Get folder path from config
    folder_path = cfg.shape_matching.folder_path
    print(f"Processing folder: {folder_path}")
    
    # Build paths
    scan_log_path = os.path.join(folder_path, "scan_log.json")
    init_model_path = os.path.join(folder_path, "init_model")
    
    # Create init_model directory if it doesn't exist
    os.makedirs(init_model_path, exist_ok=True)
    
    # Extract camera and turntable parameters from config
    R_c2g = np.array(cfg.camera.R_c2g)
    t_c2g = np.array(cfg.camera.t_c2g)
    rotation_center = np.array(cfg.emitter.turntable.center)
    rotation_axis = np.array(cfg.emitter.turntable.axis)
    
    # Load scan_log
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)
    
    # Generate c2w poses from scan_log
    c2w_list, scan_ids = load_robot_poses_c2w0(
        scan_log_path, R_c2g, t_c2g, rotation_center, rotation_axis
    )
    
    # Write to images.txt
    images_txt_path = os.path.join(init_model_path, "images.txt")
    write_c2w_to_colmap_images(c2w_list, scan_ids, scan_log, images_txt_path)
    
    print(f"Successfully initialized COLMAP model in {init_model_path}")
    print(f"Generated {len(c2w_list)} camera poses")

if __name__ == "__main__":
    main()
