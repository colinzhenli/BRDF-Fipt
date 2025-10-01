#!/usr/bin/env python3
import os
import re
import json
import argparse
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

# COLMAP helpers (same ones you already use)
from read_write_model import read_model, qvec2rotmat

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
        if not img.name.startswith('mask'):
            continue
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

def find_matching_entry(fname, scan_log):
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

def print_unmatched_scan_ids(colmap_c2w_dict, scan_log):
    """
    Print scan_ids that are in scan_log but not found in any COLMAP filename.
    """
    # Extract all scan_ids from filenames
    matched_scan_ids = set()
    for fname in colmap_c2w_dict.keys():
        base = fname.replace(".png", "").replace(".jpg", "")
        m = re.search(r'scan-(\d+)', base)
        if m:
            matched_scan_ids.add(int(m.group(1)))
    
    # Find scan_ids in scan_log that weren't matched
    scan_log_ids = set(int(e["scan_id"]) for e in scan_log)
    unmatched_ids = scan_log_ids - matched_scan_ids
    
    if unmatched_ids:
        print(f"Scan IDs in scan_log but not in COLMAP filenames: {sorted(unmatched_ids)}")
    else:
        print("All scan_log IDs were matched in COLMAP filenames.")

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

def load_robot_poses_c2w0(scan_log_path, images, R_c2g, t_c2g, rotation_center, rotation_axis):
    """
    Load robot poses and match to COLMAP poses from images.txt via φ/θ in the filename.
    (Matching uses your scan-(light)-(camera) scheme; not θ.)
    """
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)

    colmap_c2w_dict = parse_colmap_images_txt(images)

    robot_poses, cam_c2w, scan_id = [], [], []
    for fname, c2w in colmap_c2w_dict.items():

        idx = find_matching_entry(fname, scan_log)
        if idx is None:
            continue

        # COLMAP camera pose
        cam_c2w.append(c2w)

        # Robot camera pose in 0-angle world (using current TURNTABLE_CENTER)
        entry = scan_log[idx]
        T = rotated_c2w(entry, R_c2g, t_c2g, rotation_center, rotation_axis)
        robot_poses.append(T)
        scan_id.append(idx)

    print_unmatched_scan_ids(colmap_c2w_dict, scan_log)
    
    print(f"Matched {len(robot_poses)}/{len(colmap_c2w_dict)} frames.")
    return robot_poses, cam_c2w, scan_id


# -------------------- world->base estimation (your pipeline kept) --------------------

def estimate_world2base(scan_log_path, images, R_c2g, t_c2g, rotation_center, rotation_axis, solve_center_xy=True):
    """
    Returns T_BW: world->base Sim3 mapping COLMAP world to robot base.
    If solve_center_xy=True, first refines ROTATION_CENTER[0:2] from data.
    """
    # Now build your matched robot & colmap poses using the (possibly) updated center
    robot_T, cam_c2w, scan_id = load_robot_poses_c2w0(scan_log_path, images, R_c2g, t_c2g, rotation_center, rotation_axis)

    # collect centres
    cam_centres_base, cam_centres_world = [], []
    for T_c2b, c2w in zip(robot_T, cam_c2w):
        cam_centres_base.append(T_c2b[:3, 3])
        cam_centres_world.append(c2w[:3, 3])
    cam_centres_base  = np.vstack(cam_centres_base)
    cam_centres_world = np.vstack(cam_centres_world)

    # single Sim3 world->base
    s, R, t = sim3_umeyama(cam_centres_world, cam_centres_base)
    T_BW = build_4x4(s * R, t)

    # debug
    print(f"[umeyama] scale: {s:.6f}")
    print(f"[umeyama] rotation matrix:\n{R}")
    print(f"[umeyama] translation:\n{t}")
    W_h = np.hstack([cam_centres_world, np.ones((len(cam_centres_world), 1))])
    W2B = (T_BW @ W_h.T).T[:, :3]
    mean_err = np.mean(np.linalg.norm(W2B - cam_centres_base, axis=1))
    print(f"[umeyama] mean error: {mean_err:.6f} m  (N={len(cam_centres_world)})")

    return T_BW, cam_c2w, scan_id

# -------------------- mesh transform & camera log save (kept) --------------------

def transform_mesh_to_base(mesh_path, T_BW, output_path=None):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh from {mesh_path}")
    mesh.transform(T_BW)
    if output_path is not None:
        ok = o3d.io.write_triangle_mesh(str(output_path), mesh)
        if not ok:
            raise RuntimeError(f"Failed to save mesh to {output_path}")
        print(f"Transformed mesh saved to: {output_path}")
    return mesh

def save_camera_log_from_colmap(camera_c2w, T_BW, scan_id, output_path):
    camera_log = []
    for i, C2W in enumerate(camera_c2w):
        C2B = T_BW @ np.asarray(C2W)
        position = C2B[:3, 3] * 1000.0  # m->mm to match scan_log
        rotation_matrix = C2B[:3, :3]
        camera_entry = {
            "overall_id": scan_id[i],
            "camera_id": scan_id[i],
            "position": position.tolist(),
            "rotation_matrix": rotation_matrix.tolist()
        }
        camera_log.append(camera_entry)
    with open(output_path, 'w') as f:
        json.dump(camera_log, f, indent=2)
    print(f"Camera log saved to: {output_path}")
    print(f"Saved {len(camera_log)} camera poses")
    return camera_log

# -------------------- main --------------------

def main_process(scan_log_path, images, mesh_path, camera_log_path, cfg):
    # Extract parameters from config
    R_c2g = np.array(cfg.camera.R_c2g)
    t_c2g = np.array(cfg.camera.t_c2g)
    rotation_center = np.array(cfg.emitter.turntable.center)
    rotation_axis = np.array(cfg.emitter.turntable.axis)
    
    T_BW, cam_c2w, scan_id = estimate_world2base(scan_log_path, images, R_c2g, t_c2g, rotation_center, rotation_axis)
    if mesh_path is not None:
        transform_mesh_to_base(mesh_path, T_BW, output_path=str(Path(mesh_path).with_name(Path(mesh_path).stem + "_transformed.ply")))
    save_camera_log_from_colmap(cam_c2w, T_BW, scan_id, camera_log_path)

@hydra.main(version_base=None, config_path="../../config/renderer", config_name="realcapture_area_emitter")
def main(cfg: DictConfig) -> None:
    # Hard-coded arguments from launch.json
    scan_log_path = "/media/raid/cloth/No_cable_capture_Sep22/BRDF_recon_Sep30/scan_log.json"
    mesh_path = None  # "None" from launch.json
    model_path = "/media/raid/cloth/No_cable_capture_Sep22/BRDF_recon_Sep30/Controlled_light/ldr/sparse"
    
    camera_log_path = str(Path(scan_log_path).parent / "rotated_camera.json")

    cameras, images = read_model(model_path, ext=".bin")

    main_process(scan_log_path, images, mesh_path, camera_log_path, cfg)

if __name__ == "__main__":
    main()
