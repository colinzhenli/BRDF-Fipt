# -*- coding: utf-8 -*-
"""
Turn‑table axis estimation
==========================

A self‑contained script that can work with **either** a classic checkerboard
or an OpenCV **Charuco** board.  Set `BOARD_TYPE` below to "checkerboard" or
"charuco" before running.

Pipeline
--------
1.  *Intrinsic calibration*   (Zhang for checkerboard / Charuco for Charuco)
2.  *Per–frame pose*          (`solvePnP` for the chosen board)
3.  *Axis fit*                (common eigen‑axis + least‑squares centre)

Only NumPy & OpenCV (≥ 4.7 for AprilTag/Charuco) are required.
"""

from __future__ import annotations
import glob
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ──────────────────────── CONFIG ────────────────────────
BOARD_TYPE = "checkerboard"   #  "checkerboard"  or  "charuco"
# Checkerboard parameters (ignored when BOARD_TYPE == 'charuco')
CB_COLS, CB_ROWS = 10, 7       # internal corners across, down
CB_SQUARE = 0.025              # 25 mm
# Charuco parameters (ignored for checkerboard)
DICT_NAME = cv2.aruco.DICT_4X4_50
CU_COLS, CU_ROWS = 7, 5        # squares across, down (must include markers)
CU_SQUARE = 0.020              # 20 mm square length
CU_MARKER = 0.016              # 16 mm marker length

# Paths
CALIB_GLOB = "calib/*.png"     # images for intrinsic calibration
TURN_GLOB  = "turn/*.png"      # images with board on the turn‑table

# Hand–eye result (camera → robot‑base)
T_BASE_CAM = np.eye(4)         # ← replace with your calibrated 4×4

# ────────────────────── Helper functions ──────────────────────

def make_obj_points(cols: int, rows: int, square: float) -> np.ndarray:
    """Generate (rows × cols) object points in the Z = 0 plane."""
    jj, ii = np.meshgrid(np.arange(rows), np.arange(cols), indexing="xy")
    pts = np.stack([ii.ravel() * square, jj.ravel() * square, np.zeros(cols * rows)], axis=1)
    return pts.astype(np.float32)


def rodrigues_to_Rt(rvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t[:, 0]
    return R, t[:, 0], T


# ────────────────────── Axis estimation ──────────────────────

def estimate_axis_from_poses(R_list: List[np.ndarray], p_list: List[np.ndarray]):
    """Return axis direction d, point c, mean radius and RMS residual."""
    d_vecs = []
    R0 = R_list[0]
    for R in R_list[1:]:
        R_rel = R0.T @ R
        rvec, _ = cv2.Rodrigues(R_rel)
        ang = np.linalg.norm(rvec)
        if ang < 1e-6:
            continue
        axis = rvec[:, 0] / ang
        if d_vecs and np.dot(axis, d_vecs[0]) < 0:
            axis = -axis
        d_vecs.append(axis)
    d = np.mean(d_vecs, axis=0)
    d /= np.linalg.norm(d)
    # point on axis (least‑squares)
    P = np.eye(3) - np.outer(d, d)
    A = np.zeros((3, 3)); b = np.zeros(3)
    for p in p_list:
        A += P
        b += P @ p
    c = np.linalg.solve(A, b)
    radii = [np.linalg.norm(np.cross(p - c, d)) for p in p_list]
    residuals = [np.linalg.norm(P @ (p - c)) for p in p_list]
    return d, c, float(np.mean(radii)), float(np.sqrt(np.mean(np.square(residuals))))


# ──────────────────── Checkerboard branch ────────────────────

def reorder_obj_points(corners_img: np.ndarray, objp: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Re‑index objp so (0,0) stays the same physical corner across views."""
    v1 = corners_img[1] - corners_img[0]
    v2 = corners_img[cols] - corners_img[0]
    flip_x = v1[0] < 0
    flip_y = v2[1] < 0
    idx = np.arange(cols * rows).reshape(rows, cols)
    if flip_y:
        idx = idx[::-1, :]
    if flip_x:
        idx = idx[:, ::-1]
    return objp[idx.ravel()].astype(np.float32)


def calibrate_intrinsics_checkerboard(image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:
    objp = make_obj_points(CB_COLS, CB_ROWS, CB_SQUARE)
    objPoints, imgPoints = [], []
    imsize = None
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        imsize = img.shape[::-1]
        ok, corners = cv2.findChessboardCorners(img, (CB_COLS, CB_ROWS),
                                                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            continue
        corners = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1),
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        imgPoints.append(corners.reshape(-1, 1, 2))
        objPoints.append(objp.reshape(-1, 1, 3))
    if len(objPoints) < 8:
        raise RuntimeError("Need ≥ 8 views with detected corners for calibration.")
    ret, K, D, *_ = cv2.calibrateCamera(objPoints, imgPoints, imsize, None, None,
                                        flags=cv2.CALIB_RATIONAL_MODEL)
    return K, D, ret


def board_pose_pnp_checkerboard(img: np.ndarray, K: np.ndarray, D: np.ndarray):
    objp = make_obj_points(CB_COLS, CB_ROWS, CB_SQUARE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, (CB_COLS, CB_ROWS),
                                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    objp_use = reorder_obj_points(corners.reshape(-1, 2), objp, CB_COLS, CB_ROWS)
    ok, rvec, tvec = cv2.solvePnP(objp_use, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    return rodrigues_to_Rt(rvec, tvec)


# ────────────────────── Charuco branch ──────────────────────

DICT = cv2.aruco.getPredefinedDictionary(DICT_NAME)
CHARUCO_BOARD = cv2.aruco.CharucoBoard_create(squaresX=CU_COLS, squaresY=CU_ROWS,
                                             squareLength=CU_SQUARE, markerLength=CU_MARKER,
                                             dictionary=DICT)

def calibrate_intrinsics_charuco(image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:
    all_corners, all_ids = [], []
    imsize = None
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imsize = gray.shape[::-1]
        corners, ids, _ = cv2.aruco.detectMarkers(gray, DICT)
        if ids is None:
            continue
        cv2.aruco.refineDetectedMarkers(gray, CHARUCO_BOARD, corners, ids, rejectedCorners=None)
        ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, CHARUCO_BOARD)
        if ret < 4:
            continue
        all_corners.append(ch_corners)
        all_ids.append(ch_ids)
    if len(all_corners) < 8:
        raise RuntimeError("Need ≥ 8 views with detected markers for Charuco calibration.")
    ret, K, D, *_ = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, CHARUCO_BOARD, imsize, None, None)
    return K, D, ret


def board_pose_charuco(img: np.ndarray, K: np.ndarray, D: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, DICT)
    if ids is None:
        return None
    cv2.aruco.refineDetectedMarkers(gray, CHARUCO_BOARD, corners, ids, rejectedCorners=None)
    ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, CHARUCO_BOARD)
    if ret < 4:
        return None
    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(ch_corners, ch_ids, CHARUCO_BOARD, K, D)
    if not ok:
        return None
    return rodrigues_to_Rt(rvec, tvec)


# ────────────────────────── MAIN ──────────────────────────

def main() -> None:
    calib_imgs = sorted(glob.glob(CALIB_GLOB))
    turn_imgs = sorted(glob.glob(TURN_GLOB))
    if not calib_imgs:
        sys.exit("No calibration images found – check CALIB_GLOB pattern.")
    if not turn_imgs:
        sys.exit("No turn‑table images found – check TURN_GLOB pattern.")

    if BOARD_TYPE == "checkerboard":
        K, D, rms = calibrate_intrinsics_checkerboard(calib_imgs)
        pose_fn = board_pose_pnp_checkerboard
    elif BOARD_TYPE == "charuco":
        K, D, rms = calibrate_intrinsics_charuco(calib_imgs)
        pose_fn = board_pose_charuco
    else:
        raise ValueError("BOARD_TYPE must be 'checkerboard' or 'charuco'")

    print("\n=== Intrinsic calibration ===")
    print("K:\n", K)
    print("Distortion D:", D.ravel())
    print(f"Mean reprojection error: {rms:.3f} px\n")

    Rb_list: List[np.ndarray] = []
    pb_list: List[np.ndarray] = []
    for pth in turn_imgs:
        img = cv2.imread(pth)
        if img is None:
            continue
        pose = pose_fn(img, K, D)
        if pose is None:
            print(f"[warn] pose failed for {pth}")
            continue
        R_cam, t_cam, T_cam_board = pose
        T_base_board = T_BASE_CAM @ T_cam_board
        Rb_list.append(T_base_board[:3, :3])
        pb_list.append(T_base_board[:3, 3])

    if len(Rb_list) < 6:
        sys.exit("Need at least 6 valid board poses to fit an axis.")

    d, c, r_mean, rms_fit = estimate_axis_from_poses(Rb_list, pb_list)

    print("=== Axis result ===")
    print("Axis direction d (unit):", d)
    print("Axis point c (base):   ", c)
    print(f"Mean radius:           {r_mean:.4f} m")
    print(f"Fit RMS residual:      {rms_fit:.4f} m\n")


if __name__ == "__main__":
    main()
