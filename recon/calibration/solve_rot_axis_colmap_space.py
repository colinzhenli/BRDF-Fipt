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

# -------------------- External (from your repo) --------------------
from read_write_model import read_model, qvec2rotmat

# -------------------- Calibration constants (yours) --------------------
R_CAMERA2GRIPPER = np.array(
    [[ 6.28318646e-05,  9.99760635e-01,  2.18784947e-02],
     [-1.66959884e-04,  2.18785049e-02, -9.99760623e-01],
     [-9.99999984e-01,  5.91639931e-05,  1.68294587e-04]], dtype=float
)
t_CAMERA2GRIPPER = np.array([3.28324263e-2, 9.41618540e-3, 4.63816395e-02], dtype=float)  # meters

# ================================================================
# Math / Utility
# ================================================================

def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3]  = np.asarray(t, float).reshape(3)
    return T

def project_to_SO3(M: np.ndarray) -> np.ndarray:
    U,_,Vt = svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:,-1] *= -1
        R = U @ Vt
    return R

def _parse_cam_id(name: str) -> int | None:
    m = re.search(r'camera-(\d+)', name)
    return int(m.group(1)) if m else None

def extract_camera0_c2w(cam_c2w_dict: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    """
    Returns a list of (name, c2w) for camera-0 only, sorted by any scan-id if present, else by name.
    """
    items = [(k, T) for k, T in cam_c2w_dict.items() if _parse_cam_id(k) == 0]
    # Optional: stable sort by scan-<id> if present, else by name
    def _scan_id(name):
        m = re.search(r'scan-(\d+)', name)
        return int(m.group(1)) if m else 1_000_000_000
    items.sort(key=lambda kv: (_scan_id(kv[0]), kv[0]))
    return items

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

def solve_axis_center_from_c2w(cam_c2w_dict: dict[str, np.ndarray]) -> dict:
    """
    Input: dict image_name -> 4x4 c2w (COLMAP). Assumes camera-0 underwent
    pure rotations about a fixed axis/center in COLMAP/world space.
    Output: {'axis': n (3,), 'center': p (3,), 'ref_name': str, 'num': N, 'condA': float}
    """
    items = extract_camera0_c2w(cam_c2w_dict)
    if len(items) < 3:
        raise ValueError("Need at least 3 poses of camera-0 to solve.")

    names, Tlist = zip(*items)
    Rlist = [T[:3,:3] for T in Tlist]
    clist = [T[:3, 3] for T in Tlist]           # camera centers in world
    C = np.asarray(clist, float)
    N = len(Rlist)

    # Reference pose (index 0)
    R0 = Rlist[0]
    c0 = C[0]

    # Build stacked linear system for center p:
    # (I - R_i0) p = c_i - R_i0 c0, where R_i0 = R_i R0^T
    A_rows = []
    b_rows = []
    axes_from_rel = []
    for i in range(N):
        Ri0 = Rlist[i] @ R0.T
        Ai = np.eye(3) - Ri0
        bi = C[i] - (Ri0 @ c0)
        A_rows.append(Ai)
        b_rows.append(bi)
        if i != 0:
            axes_from_rel.append(_axis_from_R(Ri0))
    A = np.vstack(A_rows)     # (3N,3)
    b = np.hstack(b_rows)     # (3N,)

    # Solve with tiny Tikhonov to stabilize near-singular direction
    lam = 1e-10
    AtA = A.T @ A + lam * np.eye(3)
    Atb = A.T @ b
    p = np.linalg.solve(AtA, Atb)

    # Estimate axis in two ways and reconcile direction:
    # 1) average of axes from relative rotations
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

    # Optional: project p to plane orthogonal to n if you want the observable part only
    # (component along n is weakly/never observable for pure rotations)
    # p = p - (n @ p) * n

    # Conditioning info
    svals = svd(A, compute_uv=False)
    condA = (svals[0] / max(svals[-1], 1e-15))

    return {
        "axis": n.astype(float),
        "center": p.astype(float),
        "ref_name": names[0],
        "num": N,
        "condA": float(condA),
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
    width=3072, height=2048,
    f=6708.8357292721503, cx=1536.0, cy=1024.0, k1=-0.062586978580585914,
    color=(0, 255, 0), thickness=2, radius=6,
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
        rot_solution = solve_axis_center_from_c2w(colmap_c2w_dict)

    p = np.asarray(rot_solution["center"], float)
    n = np.asarray(rot_solution["axis"], float)
    n = n / (np.linalg.norm(n) + 1e-15)

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
    width=3072, height=2048,
    f=6708.8357292721503, cx=1536.0, cy=1024.0, k1=-0.062586978580585914,
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
    sol = solve_axis_center_from_c2w(colmap_c2w_dict)

    draw_rotation_axis(
        colmap_c2w_dict,
        image_root,
        output_root,
        width=width, height=height,
        f=f, cx=cx, cy=cy, k1=k1,
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


def main(model_path):
    cameras, images = read_model(model_path, ext=".bin")
    solve_and_draw_rotation_axis_images(
        images=images,
        image_root="/media/raid/cloth/New_turntable_Sep15/Axis_est_Sep14/masks/masked_images",           # folder that contains masked_scan-*.png
        output_root="/media/raid/cloth/New_turntable_Sep15/Axis_est_Sep14/rotation_axis",     # will be created
        width=3072, height=2048,
        f=6708.8357292721503, cx=1536.0, cy=1024.0, k1=-0.062586978580585914
    )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Axis/center estimation with CCW angles (no global delta).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to COLMAP sparse model dir")
    args = parser.parse_args()
    
    main(args.model_path)
