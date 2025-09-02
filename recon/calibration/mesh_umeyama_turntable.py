#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand–eye calibration with metric upgrade.

*   Matches robot‐arm poses (scan_log.json) to camera poses (transforms.json)
*   Estimates global **similarity** (scale + R + t) that maps COLMAP's world to
    the robot base (Umeyama 1991)
*   Runs OpenCV Tsai hand‐eye to obtain camera-to-gripper transform
"""

import os
import json
import argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from read_write_model import read_model, qvec2rotmat
from utils.transform import build_4x4, build_cw_rotz_from_deg, build_ccw_rotz_from_deg, build_rot_about_point
import re
from pathlib import Path

TURNTABLE_CENTER = np.array([0.2115, -0.1961, -0.06])
R_CAMERA2GRIPPER = np.array([[-0.00369406,  0.99992083,  0.01202885],
                      [-0.00272167,  0.01201883, -0.99992407],
                      [-0.99998947, -0.00372652,  0.00267706]])
# t_CAMERA2GRIPPER = np.array([0.03668630125, -0.02461733549, 0.02764501449])
t_CAMERA2GRIPPER = np.array([0.02634460753, -0.01919117879, 0.03014509088])

CLOCKWISE_ROTATION = True

def sim3_umeyama(P, Q, with_scale=True):
    """
    P (N,3): COLMAP camera centres
    Q (N,3): robot gripper positions
    Returns R (3,3), t (3,), s (scalar)
    """
    P, Q = np.asarray(P), np.asarray(Q)
    mu_P, mu_Q = P.mean(0), Q.mean(0)
    P0, Q0 = P - mu_P, Q - mu_Q

    # equation 12 in Umeyama
    Sigma = Q0.T @ P0 / len(P)
    U, D, VT = np.linalg.svd(Sigma)
    S = np.diag([1, 1, np.sign(np.linalg.det(U) * np.linalg.det(VT))])
    R = U @ S @ VT

    if with_scale:
        var_P = (P0**2).sum() / len(P)
        s = np.trace(np.diag(D) @ S) / var_P
    else:
        s = 1.0

    t = mu_Q - s * R @ mu_P
    return s, R, t

def find_matching_robot_pose(fname, scan_log):
    """Return index in scan_log that exactly matches camera_id and light_id."""
    # Remove file extension
    base = fname.replace(".png", "").replace(".jpg", "")
    pattern = r'scan-(\d+)-(\d+)'
    match = re.search(pattern, base)
    if not match:
        raise ValueError(f"No 'scan-<id>-<id>' pattern found in filename: {fname}")
    
    # First ID is light_id, second ID is camera_id
    light_id = int(match.group(1))
    camera_id = int(match.group(2))
    for i, e in enumerate(scan_log):
        if e["id"] == camera_id and e["light_id"] == light_id:
            return i
    error_msg = f"No exact match found for camera_id={camera_id}, light_id={light_id}. Available pairs: {[(e['id'], e['light_id']) for e in scan_log]}"
    raise ValueError(error_msg)

def parse_colmap_images_txt(images):
    """
    Parse COLMAP images.txt and return a dictionary of image_name → 4x4 cam-to-world transform.
    """
    cam_c2w_dict = {}
    for img in images.values():
        rotation = qvec2rotmat(img.qvec)
        translation = img.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([0, 0, 0, 1])[None]], 0)
        c2w = np.linalg.inv(w2c)
        cam_c2w_dict[img.name] = c2w
    return cam_c2w_dict

def rotated_c2w(json_entry, R_c2g, t_c2g, turntable_center):
    """
    Compute T_{c->w0}: camera-to-0-angle-world pose.

    Args:
        json_entry: dict with keys:
            - "rotation_matrix": 3x3 gripper->base rotation
            - "position": 3-vector gripper->base translation
            - "turn_angle": degrees (CCW about +z)
        R_c2g, t_c2g: camera->gripper extrinsics
        turntable_center: (x,y,z) center in base/world coords

    Returns:
        4x4 numpy array: T_c2w0
    """
    # T_g2b from robot state
    R_g2b = np.asarray(json_entry["rotation_matrix"], dtype=float)
    t_g2b = np.asarray(json_entry["position"], dtype=float)
    T_g2b = build_4x4(R_g2b, t_g2b)

    # T_c2g (given)
    T_c2g = build_4x4(R_c2g, t_c2g)

    # Camera -> base at this frame
    T_c2b = T_g2b @ T_c2g

    # Rotate the whole world back by -turn_angle about the turntable axis at center
    theta = float(json_entry.get("turn_angle", 0.0))
    if CLOCKWISE_ROTATION:
        T_b2w0 = build_rot_about_point(build_cw_rotz_from_deg(-theta))
    else:
        T_b2w0 = build_rot_about_point(build_ccw_rotz_from_deg(-theta))

    # Camera -> 0-angle world
    T_c2w0 = T_b2w0 @ T_c2b
    return T_c2w0
    
def load_robot_poses_c2w0(scan_log_path, images):
    """
    Load robot poses and match to COLMAP poses from images.txt via φ/θ in the filename.
    """
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)

    colmap_c2w_dict = parse_colmap_images_txt(images)

    robot_poses, cam_c2w, log_idx = [], [], []

    for fname, c2w in colmap_c2w_dict.items():
        if "theta" not in fname:
            continue

        idx = find_matching_robot_pose(fname, scan_log)
        if idx is None:
            continue
        # COLMAP camera pose
        cam_c2w.append(c2w)
        # Robot gripper → base transform
        entry = scan_log[idx]
        T = rotated_c2w(entry, R_CAMERA2GRIPPER, t_CAMERA2GRIPPER, TURNTABLE_CENTER)
        robot_poses.append(T)
        log_idx.append(idx)

    print(f"Matched {len(robot_poses)}/{len(colmap_c2w_dict)} frames.")
    return robot_poses, cam_c2w, log_idx

def save_camera_log_from_colmap(camera_c2w, T_BW, log_idx, output_path):
    """
    Convert COLMAP camera-to-world poses to camera-to-base poses and save to camera log.
    
    Parameters
    ----------
    camera_c2w : list of (4,4) arrays
        COLMAP camera-to-world transformation matrices
    T_BW : (4,4) array
        Transformation matrix from COLMAP world to robot base coordinates
    output_path : str | Path
        Path to save the camera log JSON file
    """
    # Build world-to-base transformation matrix 
    camera_log = []
    
    for i, C2W in enumerate(camera_c2w):
        C2W = np.asarray(C2W)
        
        # Apply world-to-base transformation to get camera-to-base
        C2B = T_BW @ C2W
        
        # Extract position and rotation matrix
        position = C2B[:3, 3] * 1000.0  # Convert to mm to match scan_log format
        rotation_matrix = C2B[:3, :3]
        
        # Create camera log entry
        camera_entry = {
            "overall_id": log_idx[i],
            "position": position.tolist(),
            "rotation_matrix": rotation_matrix.tolist()
        }
        
        camera_log.append(camera_entry)
    
    # Save camera log to JSON file
    with open(output_path, 'w') as f:
        json.dump(camera_log, f, indent=2)
    
    print(f"Camera log saved to: {output_path}")
    print(f"Saved {len(camera_log)} camera poses")
    
    return camera_log

def estimate_world2base(scan_log_path, images):
    """
    Parameters
    ----------
    scan_log_path          : str | Path   (robot log with gripper→base poses)
    images                 : dict   (COLMAP / NeRF style transforms.json)

    Returns
    -------
    T_BW : (4,4)  homogeneous matrix [  s·R   t ]
                                    [   0     1 ]world → base
           that maps COLMAP‑world coordinates to robot‑base coordinates.
    """
    robot_T, cam_c2w, log_idx = load_robot_poses_c2w0(
        scan_log_path, images
    )
    # ---------- per‑frame camera→base -----------------------------------------
    cam_centres_base, cam_centres_world = [], []
    for T_c2b, c2w in zip(robot_T, cam_c2w):
        cam_centres_base .append(T_c2b[:3, 3])
        cam_centres_world.append(c2w[:3, 3])    # still arbitrary units

    cam_centres_base  = np.vstack(cam_centres_base)
    cam_centres_world = np.vstack(cam_centres_world)

    # ---------- similarity (scale,R,t)  world → base --------------------------
    s, R, t = sim3_umeyama(cam_centres_world, cam_centres_base)
    T_BW = build_4x4(s * R, t)
    
    return T_BW, cam_c2w, log_idx

def transform_mesh_to_base(mesh_path, T_BW, output_path=None):
    """
    Load a mesh and transform it from COLMAP world coordinates to robot base coordinates.
    
    Parameters
    ----------
    mesh_path : str | Path
        Path to the input mesh file (e.g., .ply, .obj)
    T_BW : (4,4) array
        Transformation matrix from COLMAP world to robot base coordinates
    output_path : str | Path, optional
        Path to save the transformed mesh. If None, returns the transformed mesh object.
    
    Returns
    -------
    mesh : open3d.geometry.TriangleMesh
        The transformed mesh object
    """
    import open3d as o3d
    
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh from {mesh_path}")
    
    # Apply the transformation
    mesh.transform(T_BW)
    
    # Save if output path is provided
    if output_path is not None:
        success = o3d.io.write_triangle_mesh(str(output_path), mesh)
        if not success:
            raise RuntimeError(f"Failed to save mesh to {output_path}")
        print(f"Transformed mesh saved to: {output_path}")
    
    return mesh

# -----------------------------------------------------------------------------#
#  Main (unchanged except for the test call at the end)
# -----------------------------------------------------------------------------#
def main(scan_log_path, images, mesh_path, camera_log_path):
    T_BW, cam_c2w, log_idx = estimate_world2base(scan_log_path, images)
    transform_mesh_to_base(mesh_path, T_BW, output_path=mesh_path.replace(".ply", "_transformed.ply"))
    save_camera_log_from_colmap(cam_c2w, T_BW, log_idx, camera_log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform mesh from COLMAP world to robot base coordinates")
    parser.add_argument("--scan_log_path", type=str, required=True,
                        help="Path to robot scan log JSON file")
    parser.add_argument("--mesh_path", type=str, required=True,
                        help="Path to input mesh file (.ply)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to COLMAP sparse model directory")
    args = parser.parse_args()
    # Set camera log path to be in the same folder as scan log with name "rotated_camera.json"
    scan_log_dir = Path(args.scan_log_path).parent
    camera_log_path = scan_log_dir / "rotated_camera.json"
    cameras, images, points3D = read_model(args.model_path, ext=".bin")
    main(args.scan_log_path, images, args.mesh_path, camera_log_path)
 