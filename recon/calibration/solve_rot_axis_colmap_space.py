#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turntable axis & center estimation (COLMAP world ↔ robot base), NO global delta.

- Angles are expected CCW (right-hand rule). If your log is CW, pass --flip_cw.
- Alternating refinement:
    1) Umeyama Sim(3) from W -> expected centers (given n, p)
    2) Solve p from stacked (I - R_i) p = s R W_i + t - R_i C_i
    3) Small LM step on the axis n (unit vector on S^2)
- Optional polish: Sim(3) frozen; optimize (n, p) only with robust loss.

Outputs: dict(s, R, t, n, p, rmse, Wmap, Cexp) and printed metrics.
"""

import os
import re
import json
import argparse
import numpy as np
from dataclasses import dataclass
from numpy.linalg import svd, norm
from scipy.optimize import least_squares
import cv2
import hydra
from omegaconf import DictConfig

# -------------------- External (from your repo) --------------------
from read_write_model import read_model, qvec2rotmat

# ================================================================
# Math / Utility
# ================================================================

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

def _rotated_c2w(json_entry, R_c2g, t_c2g):
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

    return T_c2b

def _rotated_c2w0(json_entry, R_c2g, t_c2g, rotation_center, rotation_axis):
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

def _parse_cam_id(name: str) -> int | None:
    m = re.search(r'camera-(\d+)', name)
    return int(m.group(1)) if m else None

def _find_matching_entry(fname, scan_log):
    """
    Expect filenames like ...scan-<id>....*  → match scan_log entry with e['id']==<id>
    """
    base = fname.replace(".png", "").replace(".jpg", "")
    m = re.search(r'scan-(\d+)', base)
    if not m:
        raise ValueError(f"No 'scan-<id>' in filename: {fname}")
    scan_id = int(m.group(1))
    for i, e in enumerate(scan_log):
        if int(e["scan_id"]) == scan_id:
            return i
    raise ValueError(f"No match for scan_id={scan_id}")

def _filter_camera_c2w_by_id(cam_c2w_dict: dict[str, np.ndarray], camera_id: int = 0) -> dict[str, np.ndarray]:
    """
    Returns a filtered dictionary containing only entries for the specified camera ID.
    """
    filtered_dict = {}
    for name, c2w in cam_c2w_dict.items():
        cam_id = _parse_cam_id(name)
        if cam_id is not None and cam_id == camera_id:
            filtered_dict[name] = c2w
    return filtered_dict

def _filter_colmap_c2w_by_id_range(colmap_c2w_dict, camera_id_range):
    """
    Filter COLMAP c2w dictionary to only include entries for cameras in the specified range.
    
    Args:
        colmap_c2w_dict: Dictionary mapping image names to camera-to-world transforms
        camera_id_range: Tuple of (min_id, max_id) specifying the camera ID range to include
        
    Returns:
        Filtered dictionary containing only entries for camera IDs in the specified range
    """
    min_id, max_id = camera_id_range
    filtered_dict = {}
    for name, c2w in colmap_c2w_dict.items():
        cam_id = _parse_cam_id(name)
        if cam_id is not None and min_id <= cam_id <= max_id:
            filtered_dict[name] = c2w
    return filtered_dict

def load_pose_c2w_pairs(scan_log_path, colmap_c2w_dict, R_c2g, t_c2g):
    """
    Load robot poses and match to COLMAP poses from images.txt via φ/θ in the filename.
    (Matching uses your scan-(light)-(camera) scheme; not θ.)
    """
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)

    robot_poses, cam_c2w = [], []
    for fname, c2w in colmap_c2w_dict.items():

        idx = _find_matching_entry(fname, scan_log)
        if idx is None:
            continue

        # COLMAP camera pose
        cam_c2w.append(c2w)

        # Robot camera pose in 0-angle world (using current TURNTABLE_CENTER)
        entry = scan_log[idx]
        T = _rotated_c2w(entry, R_c2g, t_c2g)
        robot_poses.append(T)

    print(f"Matched {len(robot_poses)}/{len(colmap_c2w_dict)} frames.")
    # collect centres
    cam_centres_base, cam_centres_world = [], []
    for T_c2b, c2w in zip(robot_poses, cam_c2w):
        cam_centres_base.append(T_c2b[:3, 3])
        cam_centres_world.append(c2w[:3, 3])
    cam_centres_base  = np.vstack(cam_centres_base)
    cam_centres_world = np.vstack(cam_centres_world)
    return cam_centres_base, cam_centres_world

def load_pose_c2w0_pairs(scan_log_path, colmap_c2w_dict, R_c2g, t_c2g, rotation_center, rotation_axis):
    """
    Load robot poses and match to COLMAP poses from images.txt via φ/θ in the filename.
    (Matching uses your scan-(light)-(camera) scheme; not θ.)
    """
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)

    robot_poses, cam_c2w = [], []
    for fname, c2w in colmap_c2w_dict.items():

        idx = _find_matching_entry(fname, scan_log)
        if idx is None:
            continue

        # COLMAP camera pose
        cam_c2w.append(c2w)

        # Robot camera pose in 0-angle world (using current TURNTABLE_CENTER)
        entry = scan_log[idx]
        T = _rotated_c2w0(entry, R_c2g, t_c2g, rotation_center, rotation_axis)
        robot_poses.append(T)

    print(f"Matched {len(robot_poses)}/{len(colmap_c2w_dict)} frames.")
    # collect centres
    cam_centres_base, cam_centres_world = [], []
    for T_c2b, c2w in zip(robot_poses, cam_c2w):
        cam_centres_base.append(T_c2b[:3, 3])
        cam_centres_world.append(c2w[:3, 3])
    cam_centres_base  = np.vstack(cam_centres_base)
    cam_centres_world = np.vstack(cam_centres_world)
    return cam_centres_base, cam_centres_world

def _axis_from_R(R: np.ndarray) -> np.ndarray:
    """
    Get (approx) rotation axis from a 3x3 rotation by using the skew part:
    For small angles this is stable; for large angles it's fine as direction.
    """
    w = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * 0.5
    nrm = norm(w)
    return w / (nrm + 1e-15)

def _pca_plane_normal(points: np.ndarray) -> np.ndarray:
    """
    Smallest singular vector of centered points.
    """
    P = points - points.mean(0, keepdims=True)
    _, _, Vt = svd(P, full_matrices=False)
    n = Vt[-1]
    return n / (norm(n) + 1e-15)

def _solve_center_from_poses(Rlist, clist):
    """
    Solve rotation center from camera poses using linear least squares.
    
    Args:
        Rlist: List of 3x3 rotation matrices
        clist: List of 3D camera centers
    
    Returns:
        tuple: (center_p, conditioning_number)
    """
    C = np.asarray(clist, float)
    N = len(Rlist)
    
    # Reference pose (index 0)
    R0 = Rlist[0]
    c0 = C[0]

    # Build stacked linear system for center p:
    # (I - R_i0) p = c_i - R_i0 c0, where R_i0 = R_i R0^T
    A_rows = []
    b_rows = []
    for i in range(N):
        Ri0 = Rlist[i] @ R0.T
        Ai = np.eye(3) - Ri0
        bi = C[i] - (Ri0 @ c0)
        A_rows.append(Ai)
        b_rows.append(bi)
    A = np.vstack(A_rows)     # (3N,3)
    b = np.hstack(b_rows)     # (3N,)

    # Solve with tiny Tikhonov to stabilize near-singular direction
    lam = 1e-10
    AtA = A.T @ A + lam * np.eye(3)
    Atb = A.T @ b
    p = np.linalg.solve(AtA, Atb)

    # Conditioning info
    svals = svd(A, compute_uv=False)
    condA = (svals[0] / max(svals[-1], 1e-15))
    
    return p, condA

def solve_rotation(items: list[tuple[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve rotation axis and center from camera poses.
    
    Args:
        items: List of (name, c2w_4x4) tuples
    
    Returns:
        tuple: (axis_n, center_p, conditioning_number)
    """
    if len(items) < 3:
        raise ValueError("Need at least 3 poses to solve.")

    names, Tlist = zip(*items.items())
    Rlist = [T[:3,:3] for T in Tlist]
    clist = [T[:3, 3] for T in Tlist]           # camera centers in world
    C = np.asarray(clist, float)
    N = len(Rlist)

    # Reference pose (index 0)
    R0 = Rlist[0]

    # Solve center using linear least squares
    p, condA = _solve_center_from_poses(Rlist, clist)

    # Estimate axis in two ways and reconcile direction:
    # 1) average of axes from relative rotations
    axes_from_rel = []
    for i in range(1, N):  # Skip reference pose
        Ri0 = Rlist[i] @ R0.T
        axes_from_rel.append(_axis_from_R(Ri0))
    
    if len(axes_from_rel) > 0:
        n1 = np.mean(np.stack(axes_from_rel, axis=0), axis=0)
        n1 = n1 / (norm(n1) + 1e-15)
    else:
        n1 = np.array([0,0,1.0], float)

    # 2) PCA plane normal from camera centers
    n2 = _pca_plane_normal(C)

    # Make them consistent in sign and average
    if np.dot(n1, n2) < 0:
        n2 = -n2
    n = n1 + n2
    n = n / (norm(n) + 1e-15)

    return n.astype(float), p.astype(float), float(condA)

def ransac_fuse_axis_center(
    n_list,
    p_list,
):
    
    """
    Fuse multiple (axis, center) estimates into a single robust result with RANSAC.
    Invariances handled:
      - Axis is a direction: n and -n are equivalent (uses |dot| for angle).
      - Center is only meaningful up to translation along the axis: we compare
        planar projections onto the plane perpendicular to the candidate axis.

    Args:
      n_list: list/array of shape (N,3) axis estimates (need not be unit, sign arbitrary)
      p_list: list/array of shape (N,3) center estimates corresponding to n_list
      angle_thresh_deg: angular inlier threshold for axis similarity (deg)
      dist_thresh_m: planar distance inlier threshold for center similarity (m)
      max_trials: RANSAC iterations
      refine_iters: number of post-RANSAC consensus refinement iterations
      random_state: RNG seed
      force_positive_z: if True, pre-flip all axes so z >= 0 (optional)

    Returns:
      dict with:
        n_hat       : (3,) unit consensus axis (sign arbitrary unless force_positive_z=True)
        p_hat       : (3,) consensus center (planar mean; along-axis = median of inliers)
        inliers     : np.ndarray of inlier indices
        support     : int, number of inliers
        angle_thresh_deg, dist_thresh_m
    """
    angle_thresh_deg=2.0   # inlier threshold on axis angle (degrees)
    dist_thresh_m=0.01     # inlier threshold on planar center distance (meters)
    max_trials=1000
    refine_iters=1         # extra consensus refinement passes
    random_state=42
    force_positive_z=False  # if True, pre-flip all axes so z >= 0 (optional prior)
    def _unit(v):
        v = np.asarray(v, float)
        n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-15
        return v / n

    def _angle_err_deg(n1, n2):
        # direction only → use |dot|
        c = np.clip(np.abs(np.dot(n1, n2)), -1.0, 1.0)
        return np.degrees(np.arccos(c))

    def _project_to_plane(p, n):
        # p_perp = p - (n·p) n
        return p - np.dot(n, p) * n

    n_arr = _unit(np.asarray(n_list, float))
    p_arr = np.asarray(p_list, float)
    N = len(n_arr)
    if N == 0:
        raise ValueError("Empty n_list / p_list.")

    if force_positive_z:
        # Optional prior if you know the axis should point roughly +Z
        for i in range(N):
            if n_arr[i, 2] < 0:
                n_arr[i] *= -1.0

    rng = np.random.default_rng(random_state)
    angle_thr = float(angle_thresh_deg)
    dist_thr = float(dist_thresh_m)

    best_score = -1
    best_idx = None
    best_n = None
    best_p_perp = None
    best_inliers = None

    trials = min(max_trials, N * 20)
    for _ in range(trials):
        i = int(rng.integers(0, N))
        n_cand = _unit(n_arr[i])
        p_perp_cand = _project_to_plane(p_arr[i], n_cand)

        # Score all samples
        ang_err = np.array([_angle_err_deg(n_cand, n_arr[j]) for j in range(N)])
        p_perp_all = np.array([_project_to_plane(p_arr[j], n_cand) for j in range(N)])
        dists = np.linalg.norm(p_perp_all - p_perp_cand, axis=1)

        inliers = np.where((ang_err <= angle_thr) & (dists <= dist_thr))[0]
        score = inliers.size

        if score > best_score:
            best_score = score
            best_idx = i
            best_n = n_cand
            best_p_perp = p_perp_cand
            best_inliers = inliers

    # Fallback if nothing passed thresholds: use all with robust means
    if best_score <= 0 or best_inliers.size == 0:
        # Align signs to the mean direction
        mean_dir = _unit(np.mean(n_arr, axis=0))
        aligned = np.array([v if np.dot(v, mean_dir) >= 0 else -v for v in n_arr])
        n_hat = _unit(np.mean(aligned, axis=0))
        p_perp = np.mean([_project_to_plane(p_arr[j], n_hat) for j in range(N)], axis=0)
        t_along = np.median([np.dot(n_hat, p_arr[j]) for j in range(N)])
        p_hat = p_perp + t_along * n_hat
        inliers = np.arange(N)
        return {
            "n_hat": n_hat,
            "p_hat": p_hat,
            "inliers": inliers,
            "support": int(inliers.size),
            "angle_thresh_deg": angle_thr,
            "dist_thresh_m": dist_thr
        }

    # Consensus refinement on inliers
    n_hat = best_n.copy()
    inliers = best_inliers.copy()
    for _ in range(max(0, int(refine_iters))):
        # Align inlier axes to current n_hat and average
        aligned = []
        for j in inliers:
            nj = n_arr[j]
            if np.dot(nj, n_hat) < 0:
                nj = -nj
            aligned.append(nj)
        n_hat = _unit(np.mean(aligned, axis=0))

        # Recompute inliers under refined axis/planar center
        p_perp_inliers = np.array([_project_to_plane(p_arr[j], n_hat) for j in inliers])
        p_perp_hat = np.mean(p_perp_inliers, axis=0)

        ang_err_all = np.array([_angle_err_deg(n_hat, n_arr[j]) for j in range(N)])
        p_perp_all = np.array([_project_to_plane(p_arr[j], n_hat) for j in range(N)])
        dists_all = np.linalg.norm(p_perp_all - p_perp_hat, axis=1)
        inliers = np.where((ang_err_all <= angle_thr) & (dists_all <= dist_thr))[0]

    # Final center: planar mean + along-axis median for gauge-fixing
    p_perp_final = np.mean([_project_to_plane(p_arr[j], n_hat) for j in inliers], axis=0)
    t_along = np.median([np.dot(n_hat, p_arr[j]) for j in inliers])
    p_hat = p_perp_final + t_along * n_hat

    # Optional: enforce positive z at end (if requested)
    if force_positive_z and n_hat[2] < 0:
        n_hat = -n_hat
        # Center is equivalent up to along-axis flip; keep p_hat as computed (either is fine)

    return {
        "n_hat": n_hat,
        "p_hat": p_hat,
        "inliers": inliers,
        "support": int(inliers.size),
        "angle_thresh_deg": angle_thr,
        "dist_thresh_m": dist_thr
    }
    
# def get_top_cameras():
#     camera_errors = {}
#     error_log_path = "./per_image_errors.txt"
    
#     if os.path.exists(error_log_path):
#         with open(error_log_path, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line or line.startswith('#'):
#                     continue
                
#                 # Parse line format: "   1 masked_scan-0001_light-0_camera-0_phi0.942_theta0.000.png obs=  4833 mean_err=1.0655px"
#                 parts = line.split()
#                 if len(parts) >= 4:
#                     filename = parts[1]
#                     # Extract camera ID from filename
#                     match = re.search(r'camera-(\d+)', filename)
#                     if match:
#                         camera_id = int(match.group(1))
#                         if camera_id <= 9:  # Only consider camera IDs 0-9
#                             # Extract mean error
#                             for part in parts:
#                                 if part.startswith('mean_err=') and part.endswith('px'):
#                                     error_val = float(part[9:-2])  # Remove 'mean_err=' and 'px'
#                                     if camera_id not in camera_errors:
#                                         camera_errors[camera_id] = []
#                                     camera_errors[camera_id].append(error_val)
#                                     break
    
#     # Compute average error per camera
#     camera_avg_errors = {}
#     for camera_id in range(10):
#         if camera_id in camera_errors:
#             camera_avg_errors[camera_id] = np.mean(camera_errors[camera_id])
#         else:
#             camera_avg_errors[camera_id] = float('inf')  # No data available
    
#     # Sort cameras by average error and select top 40%
#     sorted_cameras = sorted(camera_avg_errors.items(), key=lambda x: x[1])
#     num_top_cameras = max(1, int(0.4 * len([c for c in sorted_cameras if c[1] != float('inf')])))
#     top_cameras = [cam_id for cam_id, _ in sorted_cameras[:num_top_cameras] if _ != float('inf')]
    
#     print("Average reprojection errors per camera:")
#     for camera_id, avg_error in sorted_cameras:
#         if avg_error != float('inf'):
#             status = "✓" if camera_id in top_cameras else " "
#             print(f"  {status} Camera {camera_id}: {avg_error:.4f}px")
#         else:
#             print(f"    Camera {camera_id}: No data")
    
#     print(f"\nUsing top {len(top_cameras)} cameras (40%): {top_cameras}")
#     return top_cameras


        
def solve_axis_center_from_c2w(cam_c2w_dict: dict[str, np.ndarray], scan_log_path, R_c2g, t_c2g) -> dict:
    """
    Input: dict image_name -> 4x4 c2w (COLMAP). Assumes camera-0 underwent
    pure rotations about a fixed axis/center in COLMAP/world space.
    Output: {'axis': n (3,), 'center': p (3,), 'ref_name': str, 'num': N, 'condA': float}
    """
    n_list = []
    p_list = []
    condA_list = []
    
    # top_cameras = get_top_cameras()
    for camera_id in range(10):  # camera id 0 to 9
        items = _filter_camera_c2w_by_id(cam_c2w_dict, camera_id=camera_id)
        if len(items) >= 3:  # Need at least 3 poses to solve
            n, p, condA = solve_rotation(items)
            n_list.append(n)
            p_list.append(p)
            condA_list.append(condA)
    
    # Run RANSAC to get robust fused axis/center
    res = ransac_fuse_axis_center(n_list, p_list)

    # Extract final results
    n = res["n_hat"]       # robust averaged axis
    p = res["p_hat"]     # robust averaged center

    """ do shape matching """
    items = _filter_colmap_c2w_by_id_range(cam_c2w_dict, camera_id_range=(10, 48))
    cam_centres_base, cam_centres_world = load_pose_c2w_pairs(scan_log_path, items, R_c2g, t_c2g)
    # single Sim3 world->base
    s, R, t = sim3_umeyama(cam_centres_world, cam_centres_base)
    T_w2b = build_4x4(s * R, t)
    
    # Transform axis and center from COLMAP world space to robotic base space
    # Axis is a direction vector, so we only apply rotation (no translation or scaling)
    n_base = (T_w2b[:3, :3] @ n.reshape(-1, 1)).flatten()
    n = n_base / (np.linalg.norm(n_base) + 1e-15)  # Renormalize
    
    # Center is a point, so we apply full transformation
    p_homogeneous = np.array([p[0], p[1], p[2], 1.0])
    p = (T_w2b @ p_homogeneous)[:3]

    print("\nRotation axis solutions:")
    print(f"  Axis: {n}")
    print(f"  Center: {p}")
    
    W_h = np.hstack([cam_centres_world, np.ones((len(cam_centres_world), 1))])
    cam_centres_base_sim3  = (T_w2b @ W_h.T).T[:, :3]
    mean_err = np.mean(np.linalg.norm(cam_centres_base_sim3 - cam_centres_base, axis=1))
    print(f"[umeyama] mean error: {mean_err:.6f} m  (N={len(cam_centres_world)})")
    
    # cam_centres_base, cam_centres_world0 = load_pose_c2w0_pairs(scan_log_path, cam_c2w_dict, R_c2g, t_c2g, p, n)
    # W_h0 = np.hstack([cam_centres_world0, np.ones((len(cam_centres_world0), 1))])
    # cam_centres_base_sim3 = (T_w2b @ W_h0.T).T[:, :3]
    # mean_err = np.mean(np.linalg.norm(cam_centres_base_sim3 - cam_centres_base, axis=1))
    # print(f"[umeyama] mean error: {mean_err:.6f} m  (N={len(cam_centres_world0)})")
    
    
    return {
        "axis": n,
        "center": p,
    }

def _is_camera0(name: str) -> bool:
    m = re.search(r'camera-(\d+)', name)
    return bool(m) and int(m.group(1)) == 0

def _project_simple_radial(Xw, c2w, f, cx, cy, k1):
    """
    Project world point Xw (3,) into pixels using SIMPLE_RADIAL intrinsics.
    c2w: 4x4 camera-to-world. We compute Xc via w2c = inv(c2w).
    Returns (u,v,valid_flag)
    """
    # world -> camera
    R = c2w[:3,:3]
    t = c2w[:3, 3]
    Xc = R.T @ (Xw - t)
    Z = Xc[2]
    if Z <= 1e-9:
        return (np.nan, np.nan, False)

    x = Xc[0] / Z
    y = Xc[1] / Z

    r2 = x*x + y*y
    scale = 1.0 + k1 * r2
    xd = x * scale
    yd = y * scale

    u = f * xd + cx
    v = f * yd + cy
    return (float(u), float(v), True)

def _project_simple_radial_points(Pw, c2w, f, cx, cy, k1):
    """Vectorized projection for an array Pw (N,3)."""
    R = c2w[:3,:3]; t = c2w[:3,3]
    Xc = (Pw - t) @ R      # (N,3) since (Pw - t) * R == R.T@(Pw-t)^T but row-wise
    Z  = Xc[:,2]
    valid = Z > 1e-9
    uv = np.full((len(Pw), 2), np.nan, dtype=float)
    if np.any(valid):
        x = Xc[valid,0] / Z[valid]
        y = Xc[valid,1] / Z[valid]
        r2 = x*x + y*y
        s  = 1.0 + k1 * r2
        uv[valid,0] = f * (x * s) + cx
        uv[valid,1] = f * (y * s) + cy
    return uv, valid

# ---------- main drawing ----------
def draw_rotation_axis(
    colmap_c2w_dict,
    image_root,
    output_root,
    intrinsics,
    scan_log_path,
    R_c2g,
    t_c2g,
    color =(0, 255, 0), thickness=2, radius=6,
    rot_solution=None,
    span_meters=32.0,        # total axis length to sample (±span/2 around center)
    n_samples=20,          # dense sampling to ensure a smooth polyline
    draw_center_dot=False
):
    """
    Draws the projected turntable axis line for camera-0 images.
    - rot_solution: dict with keys 'center' (3,) and 'axis' (3,) in COLMAP/world.
    - If your input images are *already undistorted*, set k1=0.0 here.
    """
    os.makedirs(output_root, exist_ok=True)
    if rot_solution is None:
        rot_solution = solve_axis_center_from_c2w(colmap_c2w_dict, scan_log_path, R_c2g, t_c2g)

    p = np.asarray(rot_solution["center"], float)
    n = np.asarray(rot_solution["axis"], float)
    n = n / (np.linalg.norm(n) + 1e-15)

    # Extract intrinsics
    width = intrinsics.width
    height = intrinsics.height
    f = intrinsics.focal_length
    cx = intrinsics.cx
    cy = intrinsics.cy
    k1 = intrinsics.distortion

    # Build points along the 3D axis: p + t*n, t in [-L, L]
    L  = float(span_meters) * 0.5
    ts = np.linspace(-L, L, int(n_samples), dtype=float)
    Pw = p[None, :] + ts[:, None] * n[None, :]  # (N,3)

    num_drawn, num_skipped = 0, 0
    for name, c2w in colmap_c2w_dict.items():
        if not _is_camera0(name):
            continue

        # Load image
        in_path = os.path.join(image_root, name)
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            num_skipped += 1
            continue

        # Project all sampled axis points
        uv, validZ = _project_simple_radial_points(Pw, c2w, f, cx, cy, k1)
        if uv.size == 0:
            num_skipped += 1
            continue

        # Keep points with positive depth and inside the image
        inside = (
            validZ &
            (uv[:,0] >= 0) & (uv[:,0] < width) &
            (uv[:,1] >= 0) & (uv[:,1] < height)
        )
        pts = uv[inside]
        if len(pts) >= 2:
            # Draw a single line from first to last point
            pt1 = tuple(pts[0].astype(np.int32))
            pt2 = tuple(pts[-1].astype(np.int32))
            cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

        # Optional: mark the projected center (if visible)
        if draw_center_dot:
            u0, v0, ok0 = _project_simple_radial(p, c2w, f, cx, cy, k1)
            if ok0 and (0 <= u0 < width) and (0 <= v0 < height):
                cv2.circle(img, (int(round(u0)), int(round(v0))), radius, color, -1, lineType=cv2.LINE_AA)

        # Save
        out_path = os.path.join(output_root, name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ext = os.path.splitext(out_path)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
            out_path = out_path + ".png"
        cv2.imwrite(out_path, img)
        num_drawn += 1

    print(f"[draw_rotation_axis] center p = {p}, axis n = {n}")
    print(f"[draw_rotation_axis] wrote {num_drawn} images, skipped {num_skipped}")
    
def solve_and_draw_rotation_axis_images(
    images,                       # COLMAP images dict (read_model output)
    image_root,                   # folder containing the image files
    output_root,                  # where to write annotated images
    intrinsics,                   # intrinsics config
    scan_log_path,
    R_c2g,
    t_c2g,
    color=(0, 255, 0), radius=8, thickness=2
):
    """
    1) Solve center/axis in COLMAP space from all poses (your camera-0 subset is used when projecting).
    2) For each camera-0 image, project center and draw a marker.
    """
    os.makedirs(output_root, exist_ok=True)

    # Build name->c2w
    colmap_c2w_dict = parse_colmap_images_txt(images)

    # Solve center/axis in COLMAP/world space
    sol = solve_axis_center_from_c2w(colmap_c2w_dict, scan_log_path, R_c2g, t_c2g)

    draw_rotation_axis(
        colmap_c2w_dict,
        image_root,
        output_root,
        intrinsics=intrinsics,
        scan_log_path=scan_log_path,
        R_c2g=R_c2g,
        t_c2g=t_c2g,
        color=color, radius=radius, thickness=thickness, rot_solution=sol
    )

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
        c2w = np.linalg.inv(w2c)
        cam_c2w_dict[img.name] = c2w
    return cam_c2w_dict


@hydra.main(version_base=None, config_path="../../config/renderer", config_name="realcapture_area_emitter")
def main(cfg: DictConfig):
    # Hard-coded arguments from launch.json
    model_path = "/media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22/sparse"
    scan_log_path = "/media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22/scan_log.json"
    image_root = "/media/raid/cloth/No_cable_capture_Sep22/Axis_estimation_Sep22/masks/masked_images"
    output_root = image_root.rstrip('/').rsplit('/', 1)[0] + '/masked_images_axis'
    
    
    cameras, images = read_model(model_path, ext=".bin")
    
    # Extract camera2gripper transformation from config
    R_c2g = np.array(cfg.camera.R_c2g, dtype=float)
    t_c2g = np.array(cfg.camera.t_c2g, dtype=float)
    
    solve_and_draw_rotation_axis_images(
        images=images,
        image_root=image_root,
        output_root=output_root,
        intrinsics=cfg.camera.intrinsics,
        scan_log_path=scan_log_path,
        R_c2g=R_c2g,
        t_c2g=t_c2g
    )

if __name__ == "__main__":
    main()
