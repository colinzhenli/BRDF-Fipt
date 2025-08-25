#!/usr/bin/env python3
import os
import json
import re
import glob
import math
from typing import Tuple, List, Dict, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'calibration'))
from read_write_model import rotmat2qvec
import numpy as np
import cv2

# ---------------------------
# Config / inputs
# ---------------------------
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Build COLMAP camera model from robot scan data")
    parser.add_argument("--image_path", required=True, help="Path to directory containing scan images")
    parser.add_argument("--scan_log_json", required=True, help="Path to scan log JSON file")
    parser.add_argument("--out_dir", required=True, help="Output directory for COLMAP model files")
    return parser.parse_args()

args = parse_args()
IMAGE_PATH = args.image_path
SCAN_LOG_JSON = args.scan_log_json
OUT_DIR = args.out_dir

# Initial intrinsics guess
FX0, FY0 = 6666.0, 6666.0
# Distortion initial guess: OPENCV (k1, k2, p1, p2)
SIMPLE_RADIAL = True
DIST0 = [0.0, 0.0, 0.0, 0.0]

# Hand–eye (camera→gripper)
R_c2g = np.array([[-0.00369406,  0.99992083,  0.01202885],
                  [-0.00272167,  0.01201883, -0.99992407],
                  [-0.99998947, -0.00372652,  0.00267706]], dtype=float)
t_c2g = np.array([36.68630125, -24.61733549, 27.64501449], dtype=float)

# OpenGL → OpenCV rotation (rotate 180° about X)
R_gl_to_cv = np.array([[1.0,  0.0,  0.0],
                       [0.0, -1.0,  0.0],
                       [0.0,  0.0, -1.0]], dtype=float)

# Filename tolerance for matching robot record by (phi, theta)
MATCH_TOL = 1e-3


# ---------------------------
# Utilities
# ---------------------------
def to_h(T: np.ndarray) -> np.ndarray:
    """Ensure 4x4 homogeneous from 3x3/3x4."""
    if T.shape == (4, 4):
        return T
    if T.shape == (3, 3):
        H = np.eye(4)
        H[:3, :3] = T
        return H
    if T.shape == (3, 4):
        H = np.eye(4)
        H[:3, :4] = T
        return H
    raise ValueError("Unexpected shape for transform")

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    H = np.eye(4)
    H[:3, :3] = R
    H[:3,  3] = t
    return H

def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3,  3]
    H = np.eye(4)
    Rt = R.T
    H[:3, :3] = Rt
    H[:3,  3] = -Rt @ t
    return H

def mat_to_colmap_q(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Return quaternion (qw,qx,qy,qz) from rotation matrix (right-handed).
    COLMAP expects world->camera rotation.
    """
    # Use scipy if available; otherwise do a robust manual conversion
    K = np.array([
        [R[0,0]-R[1,1]-R[2,2], 0, 0, 0],
        [R[1,0]+R[0,1], R[1,1]-R[0,0]-R[2,2], 0, 0],
        [R[2,0]+R[0,2], R[2,1]+R[1,2], R[2,2]-R[0,0]-R[1,1], 0],
        [R[1,2]-R[2,1], R[2,0]-R[0,2], R[0,1]-R[1,0], R[0,0]+R[1,1]+R[2,2]]
    ])
    K /= 3.0
    w, V = np.linalg.eigh(K)
    q = V[:, np.argmax(w)]
    # reorder to (x,y,z,w) -> (w,x,y,z) if needed
    qx, qy, qz, qw = q
    # Normalize and ensure qw >= 0 (COLMAP convention allows either, but we standardize)
    qnorm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/qnorm, qx/qnorm, qy/qnorm, qz/qnorm
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    return float(qw), float(qx), float(qy), float(qz)

def extract_phi_theta_from_filename(filename: str) -> Tuple[float, float]:
    """Extract φ and θ from names like:
       'scan-0-phi0.079_theta3.142.png' OR '..._phi0.583_theta4.608.jpg'."""
    base = filename
    for ext in [".png", ".jpg", ".jpeg"]:
        if base.endswith(ext):
            base = base[: -len(ext)]
    base = base.replace("scan-", "")
    if "-phi" in base and "_theta" in base:
        phi_part = base.split("-phi")[1]
        phi_theta_parts = phi_part.split("_theta")
        phi = float(phi_theta_parts[0])
        theta = float(phi_theta_parts[1])
        return phi, theta
    parts = base.split("_")
    phi = float([p for p in parts if p.startswith("phi")][0][3:])
    theta = float([p for p in parts if p.startswith("theta")][0][5:])
    return phi, theta

def find_matching_robot_pose(trg_phi: float, trg_theta: float,
                             scan_log: List[Dict[str, Any]],
                             tol: float = MATCH_TOL) -> Tuple[int, float]:
    best_i, best_d = None, float("inf")
    for i, e in enumerate(scan_log):
        d = max(abs(trg_phi - float(e["phi"])), abs(trg_theta - float(e["theta"])))
        if d < best_d:
            best_d, best_i = d, i
    return (best_i, best_d) if best_d <= tol else (None, best_d)

def scan_id_key(path):
    name = os.path.basename(path)
    m = re.match(r"scan-(\d+)-phi", name)
    if not m:
        raise ValueError(f"Unexpected filename format: {name}")
    return int(m.group(1))


# ---------------------------
# Main
# ---------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Collect image files (sorted by scan ID)
img_files = sorted(
    [f for f in glob.glob(os.path.join(IMAGE_PATH, "*"))
     if f.lower().endswith((".png", ".jpg", ".jpeg"))],
    key=scan_id_key
)
if not img_files:
    raise RuntimeError(f"No images found in {IMAGE_PATH}")

# 2) Read image size from the first image
probe = cv2.imread(img_files[0], cv2.IMREAD_UNCHANGED)
if probe is None:
    raise RuntimeError(f"Failed to read image: {img_files[0]}")
H, W = probe.shape[:2]
CX0, CY0 = W / 2.0, H / 2.0

# 3) Load robot scan log JSON
with open(SCAN_LOG_JSON, "r") as f:
    scan_log = json.load(f)
if not isinstance(scan_log, list):
    raise RuntimeError("Expected scan_log JSON to be a list of pose entries.")

# 4) Build T_c2g, T_gl2cv homogeneous
T_c2g = make_T(R_c2g, t_c2g)
T_gl2cv = np.eye(4)
T_gl2cv[:3, :3] = R_gl_to_cv

# 5) Prepare outputs
cameras_txt = os.path.join(OUT_DIR, "cameras.txt")
images_txt  = os.path.join(OUT_DIR, "images.txt")
points3d_txt = os.path.join(OUT_DIR, "points3D.txt")

# 6) Write cameras.txt (single camera, OPENCV or SIMPLE_RADIAL)
with open(cameras_txt, "w") as fc:
    # Format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
    fx, fy, cx, cy = FX0, FY0, CX0, CY0
    k1, k2, p1, p2 = DIST0
    
    # Option to use SIMPLE_RADIAL (fx, cx, cy, k1) instead of OPENCV
    USE_SIMPLE_RADIAL = False  # Set to True for single radial distortion model
    
    if USE_SIMPLE_RADIAL:
        # SIMPLE_RADIAL params: fx cx cy k1 (assumes fx=fy)
        fc.write(f"1 SIMPLE_RADIAL {W} {H} {fx:.6f} {cx:.6f} {cy:.6f} {k1:.6f}\n")
    else:
        # OPENCV params: fx fy cx cy k1 k2 p1 p2
        fc.write(f"1 OPENCV {W} {H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} {k1:.6f} {k2:.6f} {p1:.6f} {p2:.6f}\n")

# 7) Compose images.txt
with open(images_txt, "w") as fi:
    # Header comment (optional)
    fi.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    fi.write("# Followed by a line with: POINTS2D[] as (x, y, point3D_id)\n")

    image_id = 1
    for path in img_files:
        name = os.path.basename(path)
        if "theta" not in name:
            # still allow, but skip if you only rely on phi/theta names
            continue

        # 7a) parse phi/theta from filename
        phi, theta = extract_phi_theta_from_filename(name)

        # 7b) find matching robot entry
        idx, d = find_matching_robot_pose(phi, theta, scan_log, MATCH_TOL)
        if idx is None:
            print(f"[WARN] No robot pose within tol for {name} (φ={phi}, θ={theta}). Best diff={d}")
            continue
        entry = scan_log[idx]

        # 7c) build Tg2b (gripper→base)
        R_g2b = np.array(entry["rotation_matrix"], dtype=float)
        t_g2b = np.array(entry["position"], dtype=float)
        T_g2b = make_T(R_g2b, t_g2b)   # maps gripper coords into base coords

        # 7d) camera absolute pose in base (OpenGL convention at this point)
        #     T_b_cGL = T_g2b * T_c2g   (camera->base)
        T_c2bGL = T_g2b @ T_c2g
        T_c2bCV = T_c2bGL @ T_gl2cv
        # T_c2bCV = T_gl2cv @ T_c2bGL
        T_b2cCV = invert_T(T_c2bCV)
        R_cw = T_b2cCV[:3, :3]
        t_cw = T_b2cCV[:3,  3] / 1000.0
        
        # 7g) Convert rotation to quaternion (qw,qx,qy,qz)
        # qw, qx, qy, qz = mat_to_colmap_q(R_cw)
        qw, qx, qy, qz = rotmat2qvec(R_cw)

        # 7h) Write image entry
        fi.write(f"{image_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                 f"{t_cw[0]:.10f} {t_cw[1]:.10f} {t_cw[2]:.10f} 1 {name}\n")
        fi.write("\n")  # empty 2D point line
        image_id += 1

# 8) Ensure points3D.txt exists and is empty
with open(points3d_txt, "w") as fp:
    pass

print(f"Written:\n  {cameras_txt}\n  {images_txt}\n  {points3d_txt}")
print("Done.")
