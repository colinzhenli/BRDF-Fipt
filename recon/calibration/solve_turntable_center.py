#!/usr/bin/env python3
import os
import re
import json
import argparse
import numpy as np
from pathlib import Path

# COLMAP helpers (same ones you already use)
from read_write_model import read_model, qvec2rotmat

# -------------------- constants (your values kept) --------------------

TURNTABLE_CENTER = np.array([0.20, -0.1961, 6.0], dtype=float)  # meters; pz is unused by Rz anyway
R_CAMERA2GRIPPER = np.array(
    [[-0.00369406,  0.99992083,  0.01202885],
     [-0.00272167,  0.01201883, -0.99992407],
     [-0.99998947, -0.00372652,  0.00267706]], dtype=float
)
# meters:
t_CAMERA2GRIPPER = np.array([0.02634460753, -0.01919117879, 0.03014509088], dtype=float)

CLOCKWISE_ROTATION = True  # your log stores +θ as clockwise
ROTATION_FACTOR = 1.0405

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

def Rz_deg(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def T_about_point(R, p):
    """Homogeneous transform that rotates by R about world point p (3,)."""
    p = np.asarray(p, float).reshape(3)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = (np.eye(3) - R) @ p
    return T

# -------------------- your existing helpers (kept) --------------------

def find_matching_robot_pose(fname, scan_log):
    """Return index in scan_log that exactly matches camera_id and light_id."""
    base = fname.replace(".png", "").replace(".jpg", "")
    pattern = r'scan-(\d+)-(\d+)'
    match = re.search(pattern, base)
    if not match:
        raise ValueError(f"No 'scan-<id>-<id>' pattern found in filename: {fname}")
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

# -------------------- rotation undo and pose building --------------------

def rotated_c2w(json_entry, R_c2g, t_c2g, turntable_center):
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

    # undo world by +/- theta about center
    theta = float(json_entry.get("turn_angle", 0.0))
    theta = theta * ROTATION_FACTOR
    undo = +theta if CLOCKWISE_ROTATION else -theta
    R_undo = Rz_deg(undo)
    T_b2w0 = T_about_point(R_undo, turntable_center)

    # camera->0-angle world
    return T_b2w0 @ T_c2b

# -------------------- your matching function (unchanged) --------------------

def load_robot_poses_c2w0(scan_log_path, images):
    """
    Load robot poses and match to COLMAP poses from images.txt via φ/θ in the filename.
    (Matching uses your scan-(light)-(camera) scheme; not θ.)
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

        # Robot camera pose in 0-angle world (using current TURNTABLE_CENTER)
        entry = scan_log[idx]
        T = rotated_c2w(entry, R_CAMERA2GRIPPER, t_CAMERA2GRIPPER, TURNTABLE_CENTER)
        robot_poses.append(T)
        log_idx.append(idx)

    print(f"Matched {len(robot_poses)}/{len(colmap_c2w_dict)} frames.")
    return robot_poses, cam_c2w, log_idx

# -------------------- preprocessing: collect pairs (no pipeline changes) --------------------

def _collect_matched_centres(scan_log_path, images, R_c2g, t_c2g):
    """
    Build arrays:
      W: (N,3) COLMAP camera centres
      C: (N,3) camera centres from robot (camera->base) before any undo rotation
      theta: (N,) turntable angles (deg) read from scan_log entries (not filenames)
      entries: list of matched scan_log entries (for any extra debug)
    """
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)
    colmap_c2w_dict = parse_colmap_images_txt(images)

    W, C, theta, entries = [], [], [], []
    for fname, c2w in colmap_c2w_dict.items():
        if "theta" not in fname:
            continue
        idx = find_matching_robot_pose(fname, scan_log)
        if idx is None:
            continue
        entry = scan_log[idx]

        # COLMAP camera center (world)
        W.append(c2w[:3, 3])

        # camera->base from robot (no undo)
        R_g2b = np.asarray(entry["rotation_matrix"], float)
        t_g2b = np.asarray(entry["position"], float) / 1000.0
        T_g2b = build_4x4(R_g2b, t_g2b)
        T_c2g = build_4x4(R_c2g, t_c2g)
        T_c2b = T_g2b @ T_c2g
        C.append(T_c2b[:3, 3])

        # angle from log (deg)
        theta.append(float(entry.get("turn_angle", 0.0)))
        entries.append(entry)

    return np.asarray(W, float), np.asarray(C, float), np.asarray(theta, float), entries

# -------------------- center solver for (px,py), pz fixed --------------------

def solve_turntable_center_xy(scan_log_path, images,
                              R_c2g, t_c2g,
                              init_center,
                              angle_convention='CW',
                              max_iters=20, tol=1e-9,
                              solve_delta_angle=True, delta_search_deg=3.0, delta_step_deg=0.25):
    """
    Alternating solve for (px,py) that minimizes RMS after undoing rotations about p and
    fitting a single Sim3 (Umeyama) from COLMAP centres to robot centres.
    pz is kept fixed.
    """
    W, C, thetas, _ = _collect_matched_centres(scan_log_path, images, R_c2g, t_c2g)
    if len(W) == 0:
        raise ValueError("No matched frames with 'theta' in filenames after parsing images.")

    sign = +1.0 if angle_convention.upper() == 'CW' else -1.0
    I = np.eye(3)
    p = np.asarray(init_center, float).copy()
    pz = float(p[2])
    delta = 0.0
    last_err = None

    def Rs(δ):
        return [Rz_deg(sign * th + δ) for th in thetas]

    for it in range(max_iters):
        Rlist = Rs(delta)
        # undo-rotated robot centres Cp(p,δ): Cp_i = R_i C_i + (I - R_i) p
        Cp = np.empty_like(C)
        for i, Ri in enumerate(Rlist):
            Cp[i] = Ri @ C[i] + (I - Ri) @ p

        # Umeyama: W -> Cp
        s, R, t = sim3_umeyama(W, Cp, with_scale=True)
        Wmap = (s * (R @ W.T)).T + t
        err = np.sqrt(np.mean(np.sum((Wmap - Cp)**2, axis=1)))

        # optional small 1D search for global Δθ bias
        if solve_delta_angle:
            best = (err, delta)
            rng = np.arange(delta - delta_search_deg,
                            delta + delta_search_deg + 1e-9, delta_step_deg)
            for δ in rng:
                Rlist_δ = Rs(δ)
                Cp_δ = np.empty_like(C)
                for i, Ri in enumerate(Rlist_δ):
                    Cp_δ[i] = Ri @ C[i] + (I - Ri) @ p
                sδ, Rδ, tδ = sim3_umeyama(W, Cp_δ, with_scale=True)
                Wmap_δ = (sδ * (Rδ @ W.T)).T + tδ
                eδ = np.sqrt(np.mean(np.sum((Wmap_δ - Cp_δ)**2, axis=1)))
                if eδ < best[0]:
                    best = (eδ, δ)
            delta = best[1]
            # rebuild after Δθ update
            Rlist = Rs(delta)
            for i, Ri in enumerate(Rlist):
                Cp[i] = Ri @ C[i] + (I - Ri) @ p
            s, R, t = sim3_umeyama(W, Cp, with_scale=True)
            Wmap = (s * (R @ W.T)).T + t
            err = np.sqrt(np.mean(np.sum((Wmap - Cp)**2, axis=1)))

        # linear least-squares for p_x, p_y (keep pz fixed)
        A_rows, b_rows = [], []
        for i, Ri in enumerate(Rlist):
            A = (I - Ri)  # 3x3
            rhs = (s * (R @ W[i]) + t) - (Ri @ C[i])
            # xy rows only; subtract pz contribution (zero for pure Rz but keep generic)
            Axy = A[:2, :2]
            az  = A[:2, 2:3]
            bxy = rhs[:2] - az[:, 0] * pz
            A_rows.append(Axy); b_rows.append(bxy)
        A_big = np.vstack(A_rows)               # (2N,2)
        b_big = np.concatenate(b_rows, axis=0)  # (2N,)

        pxpy, *_ = np.linalg.lstsq(A_big, b_big, rcond=None)
        p_new = np.array([pxpy[0], pxpy[1], pz], float)

        if last_err is not None and np.linalg.norm(p_new - p) < tol and abs(err - last_err) < 1e-12:
            p = p_new
            break

        p = p_new
        last_err = err

    # final stats
    Rlist = Rs(delta)
    Cp = np.empty_like(C)
    for i, Ri in enumerate(Rlist):
        Cp[i] = Ri @ C[i] + (np.eye(3) - Ri) @ p
    s, R, t = sim3_umeyama(W, Cp, with_scale=True)
    Wmap = (s * (R @ W.T)).T + t
    rmse = np.sqrt(np.mean(np.sum((Wmap - Cp)**2, axis=1)))

    return {
        "center_est": p,                # (px,py,pz_fixed)
        "delta_angle_deg": float(delta),
        "rmse_after": float(rmse),
        "sim3": (s, R, t),
        "num_points": int(len(W)),
    }

# -------------------- world->base estimation (your pipeline kept) --------------------

def estimate_world2base(scan_log_path, images, solve_center_xy=True):
    """
    Returns T_BW: world->base Sim3 mapping COLMAP world to robot base.
    If solve_center_xy=True, first refines TURNTABLE_CENTER[0:2] from data.
    """
    global TURNTABLE_CENTER

    if solve_center_xy:
        init_center = TURNTABLE_CENTER.copy()
        res = solve_turntable_center_xy(
            scan_log_path, images,
            R_CAMERA2GRIPPER, t_CAMERA2GRIPPER,
            init_center=init_center,
            angle_convention=('CW' if CLOCKWISE_ROTATION else 'CCW'),
            max_iters=25, tol=1e-10,
            solve_delta_angle=True, delta_search_deg=3.0, delta_step_deg=0.25
        )
        print(f"[center-solve] estimated center (m): {res['center_est']}")
        print(f"[center-solve] Δθ bias (deg): {res['delta_angle_deg']:.4f}")
        print(f"[center-solve] RMS after: {res['rmse_after']:.6f} m")
        TURNTABLE_CENTER = res["center_est"]  # use solved px,py (pz unchanged)

    # Now build your matched robot & colmap poses using the (possibly) updated center
    robot_T, cam_c2w, log_idx = load_robot_poses_c2w0(scan_log_path, images)

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
    W_h = np.hstack([cam_centres_world, np.ones((len(cam_centres_world), 1))])
    W2B = (T_BW @ W_h.T).T[:, :3]
    mean_err = np.mean(np.linalg.norm(W2B - cam_centres_base, axis=1))
    print(f"[umeyama] mean error: {mean_err:.6f} m  (N={len(cam_centres_world)})")

    return T_BW, cam_c2w, log_idx

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

def save_camera_log_from_colmap(camera_c2w, T_BW, log_idx, output_path):
    camera_log = []
    for i, C2W in enumerate(camera_c2w):
        C2B = T_BW @ np.asarray(C2W)
        position = C2B[:3, 3] * 1000.0  # m->mm to match scan_log
        rotation_matrix = C2B[:3, :3]
        camera_entry = {
            "overall_id": log_idx[i],
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

def main(scan_log_path, images, mesh_path, camera_log_path, solve_center_xy=True):
    T_BW, cam_c2w, log_idx = estimate_world2base(scan_log_path, images, solve_center_xy=solve_center_xy)
    transform_mesh_to_base(mesh_path, T_BW, output_path=str(Path(mesh_path).with_name(Path(mesh_path).stem + "_transformed.ply")))
    save_camera_log_from_colmap(cam_c2w, T_BW, log_idx, camera_log_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform COLMAP world to robot base; optional center solve.")
    parser.add_argument("--scan_log_path", type=str, required=True, help="Path to robot scan log JSON")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to input mesh (.ply)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to COLMAP sparse model dir")
    parser.add_argument("--no_solve_center_xy", action="store_true",
                        help="Disable solving px,py; use TURNTABLE_CENTER as-is")
    args = parser.parse_args()

    scan_log_path = args.scan_log_path
    mesh_path     = args.mesh_path
    camera_log_path = str(Path(scan_log_path).parent / "rotated_camera.json")

    cameras, images = read_model(args.model_path, ext=".bin")

    main(scan_log_path, images, mesh_path, camera_log_path,
         solve_center_xy=(not args.no_solve_center_xy))
