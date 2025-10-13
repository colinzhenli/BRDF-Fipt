#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ColorChecker-based 3x3 color correction matrix for HDR (linear) images.

- Parses CGATS-like Lab(D50) text (post-2014 ColorChecker).
- Click 4 corners (TL->TR->BR->BL) to compute homography.
- Warps to a canonical 6x4 grid (matching your A..F rows × 1..4 cols).
- Averages each patch center region.
- Converts Lab(D50) -> XYZ(D50) -> XYZ(D65) [Bradford] -> linear sRGB(D65).
- Solves least-squares 3x3 (camera RGB -> linear sRGB).
- Reports ΔE00 per patch.
- Applies M to full HDR image and saves corrected EXR.

Usage:
  python calibrate_cc_hdr.py \
      --lab_file ColorChecker24_After_Nov2014.txt \
      --images img1.exr img2.exr img3.exr \
      --out_suffix _ccorr.exr
"""

import argparse
import os
import re
import sys
from typing import Tuple, List

import cv2
import numpy as np

# =========================
# Configuration / Grid
# =========================
# Your CGATS snippet lists A..F rows (6), 1..4 cols (4) => 24 patches.
GRID_ROWS = 6
GRID_COLS = 4
CELL_W = 120   # canonical cell width in pixels
CELL_H = 120   # canonical cell height in pixels
CENTER_MARGIN = 0.2  # take central 60% box per patch (avoid edges)

# PATCH_ORDER must match the CGATS names ordering that you want to align with
# the canonical warped grid indexing [row-major 0..ROWS-1][0..COLS-1].
PATCH_ORDER = [
    "A1","A2","A3","A4",
    "B1","B2","B3","B4",
    "C1","C2","C3","C4",
    "D1","D2","D3","D4",
    "E1","E2","E3","E4",
    "F1","F2","F3","F4",
]

# ==================================================
# CGATS parser: Lab(D50) with decimal commas support
# ==================================================
def parse_cgats_lab_d50(txt_path: str) -> np.ndarray:
    """
    Parse CGATS-like text listing SAMPLE_NAME and Lab_L Lab_a Lab_b for patches A1..F4.
    Returns (24,3) array in PATCH_ORDER.
    """
    num_re = re.compile(r'[-+]?\d+(?:[.,]\d+)?')
    lab_map = {}
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            # Match lines that start with patch name (A-F followed by digit)
            # The format is: A1\t37,54\t14,37\t14,92
            m = re.match(r'^\s*([A-F]\d)\s+(.+)', ln)
            if not m:
                continue
            name = m.group(1)
            # Split by tabs or multiple spaces to get the numeric fields
            fields = re.split(r'\t+|\s{2,}', m.group(2).strip())
            if len(fields) >= 3:
                # Replace commas with dots for decimal conversion
                L = float(fields[0].replace(',', '.'))
                a = float(fields[1].replace(',', '.'))
                b = float(fields[2].replace(',', '.'))
                lab_map[name] = (L, a, b)

    vals = []
    missing = []
    for p in PATCH_ORDER:
        if p in lab_map:
            vals.append(lab_map[p])
        else:
            missing.append(p)
    if missing:
        print("WARNING: missing patches in CGATS:", missing)
    return np.asarray(vals, dtype=np.float64)  # (24,3)

# =====================================
# Color space conversions (D50 & D65)
# =====================================
def lab_to_xyz_D50(Lab: np.ndarray) -> np.ndarray:
    """ CIE 1976 Lab (D50 white) -> XYZ (D50). """
    Xn, Yn, Zn = 0.9642, 1.0000, 0.8251
    L, a, b = Lab[..., 0], Lab[..., 1], Lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)
    delta = 6/29

    def inv_f(t):
        return np.where(t > delta, t**3, 3*delta**2*(t - 4/29))

    xr = inv_f(fx)
    yr = inv_f(fy)
    zr = inv_f(fz)
    X = xr * Xn
    Y = yr * Yn
    Z = zr * Zn
    return np.stack([X, Y, Z], axis=-1)

def bradford_D50_to_D65(XYZ_D50: np.ndarray) -> np.ndarray:
    """ Bradford chromatic adaptation: D50 -> D65. """
    M = np.array([[ 0.8951,  0.2664, -0.1614],
                  [-0.7502,  1.7135,  0.0367],
                  [ 0.0389, -0.0685,  1.0296]], dtype=np.float64)
    M_inv = np.linalg.inv(M)
    D50 = np.array([0.9642, 1.0000, 0.8251])
    D65 = np.array([0.95047, 1.00000, 1.08883])

    rho_D50 = M @ D50
    rho_D65 = M @ D65
    D = np.diag(rho_D65 / rho_D50)

    shp = XYZ_D50.shape
    XYZ = XYZ_D50.reshape(-1, 3).T
    XYZ_adapt = M_inv @ (D @ (M @ XYZ))
    return XYZ_adapt.T.reshape(shp)

def xyz_to_linear_srgb_D65(XYZ: np.ndarray) -> np.ndarray:
    """ XYZ(D65) -> linear sRGB(D65). """
    M = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660,  1.8760108,  0.0415560],
                  [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float64)
    return XYZ @ M.T

def linear_srgb_to_xyz_D65(rgb: np.ndarray) -> np.ndarray:
    """ linear sRGB(D65) -> XYZ(D65). """
    M_inv = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
    return rgb @ M_inv.T

def bradford_D65_to_D50(XYZ_D65: np.ndarray) -> np.ndarray:
    """ Bradford chromatic adaptation: D65 -> D50. """
    M = np.array([[ 0.8951,  0.2664, -0.1614],
                  [-0.7502,  1.7135,  0.0367],
                  [ 0.0389, -0.0685,  1.0296]], dtype=np.float64)
    M_inv = np.linalg.inv(M)
    D50 = np.array([0.9642, 1.0000, 0.8251])
    D65 = np.array([0.95047, 1.00000, 1.08883])

    rho_D50 = M @ D50
    rho_D65 = M @ D65
    D = np.diag(rho_D50 / rho_D65)

    shp = XYZ_D65.shape
    XYZ = XYZ_D65.reshape(-1, 3).T
    XYZ_adapt = M_inv @ (D @ (M @ XYZ))
    return XYZ_adapt.T.reshape(shp)

def xyz_to_lab_D50(XYZ_D50: np.ndarray) -> np.ndarray:
    """ XYZ(D50) -> Lab(D50). """
    Xn, Yn, Zn = 0.9642, 1.0, 0.8251
    xr = XYZ_D50[...,0] / Xn
    yr = XYZ_D50[...,1] / Yn
    zr = XYZ_D50[...,2] / Zn

    def f(t):
        delta = 6/29
        return np.where(t > delta**3, np.cbrt(t), (t/(3*delta**2)) + (4/29))

    fx = f(xr); fy = f(yr); fz = f(zr)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.stack([L, a, b], axis=-1)

# ================
# CIEDE2000 ΔE00
# ================
def _hp_f(a, b):
    h = np.degrees(np.arctan2(b, a))
    return np.where(h < 0, h + 360, h)

def deltaE2000(Lab1: np.ndarray, Lab2: np.ndarray) -> np.ndarray:
    L1,a1,b1 = Lab1[...,0], Lab1[...,1], Lab1[...,2]
    L2,a2,b2 = Lab2[...,0], Lab2[...,1], Lab2[...,2]
    kL=kC=kH=1.0

    C1 = np.sqrt(a1*a1 + b1*b1)
    C2 = np.sqrt(a2*a2 + b2*b2)
    Cm = (C1 + C2)/2.0
    G = 0.5*(1 - np.sqrt((Cm**7)/(Cm**7 + 25**7)))
    a1p = (1+G)*a1; a2p = (1+G)*a2
    C1p = np.sqrt(a1p*a1p + b1*b1)
    C2p = np.sqrt(a2p*a2p + b2*b2)
    h1p = _hp_f(a1p,b1)
    h2p = _hp_f(a2p,b2)

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dHp = 2*np.sqrt(C1p*C2p)*np.sin(np.radians(dhp/2))

    Lpm = (L1 + L2)/2
    Cpm = (C1p + C2p)/2
    hpm = (h1p + h2p)/2
    hpm = np.where(np.abs(h1p - h2p) > 180, hpm + 180, hpm)
    hpm = np.where(hpm >= 360, hpm - 360, hpm)

    T = 1 - 0.17*np.cos(np.radians(hpm - 30)) + \
            0.24*np.cos(np.radians(2*hpm)) + \
            0.32*np.cos(np.radians(3*hpm + 6)) - \
            0.20*np.cos(np.radians(4*hpm - 63))

    Sl = 1 + (0.015*(Lpm - 50)**2)/np.sqrt(20 + (Lpm - 50)**2)
    Sc = 1 + 0.045*Cpm
    Sh = 1 + 0.015*Cpm*T

    Rt = -2*np.sqrt((Cpm**7)/(Cpm**7 + 25**7))*np.sin(np.radians(60*np.exp(-((hpm-275)/25)**2)))
    dE = np.sqrt((dLp/(kL*Sl))**2 + (dCp/(kC*Sc))**2 + (dHp/(kH*Sh))**2 + Rt*(dCp/(kC*Sc))*(dHp/(kH*Sh)))
    return dE

# =======================
# Geometry / UI helpers
# =======================
def click_four_points(preview_bgr: np.ndarray, window_name="ClickCorners") -> np.ndarray:
    """
    Click 4 corners in order: TL -> TR -> BR -> BL. Returns (4,2) float32 array.
    """
    pts: List[Tuple[int,int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(preview_bgr, (x, y), 5, (0,255,0), -1)
            cv2.imshow(window_name, preview_bgr)

    cv2.imshow(window_name, preview_bgr)
    cv2.setMouseCallback(window_name, on_mouse)
    print("Click 4 corners in order: TL(brown) -> TR(green) -> BR(white) -> BL(black). Press 'q' to cancel.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if len(pts) == 4:
            break
        if key == ord('q'):
            pts.clear()
            break
    cv2.setMouseCallback(window_name, lambda *args: None)
    cv2.destroyWindow(window_name)
    return np.asarray(pts, dtype=np.float32)

def warp_and_sample_patches(img_linear: np.ndarray,
                            H_can_img: np.ndarray,
                            rows: int = GRID_ROWS,
                            cols: int = GRID_COLS,
                            cell_w: int = CELL_W,
                            cell_h: int = CELL_H,
                            margin: float = CENTER_MARGIN) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp image to canonical grid (cols x rows) and sample center means per patch.
    Returns (means_24x3, warped_image).
    """
    W = cols * cell_w
    H = rows * cell_h
    warped = cv2.warpPerspective(img_linear, H_can_img, (W, H), flags=cv2.INTER_LINEAR)

    means = []
    for r in range(rows):
        for c in range(cols):
            x0 = c * cell_w; y0 = r * cell_h
            mx = int(x0 + margin * cell_w)
            my = int(y0 + margin * cell_h)
            Mx = int(x0 + (1 - margin) * cell_w)
            My = int(y0 + (1 - margin) * cell_h)
            patch = warped[my:My, mx:Mx, :]
            if patch.size == 0:
                means.append([np.nan, np.nan, np.nan])
            else:
                means.append(np.nanmean(patch.reshape(-1, 3), axis=0))
    return np.asarray(means, dtype=np.float64), warped

# ======================
# Fitting & Application
# ======================
def fit_color_matrix(cam_rgb_24x3: np.ndarray, ref_rgb_24x3: np.ndarray) -> np.ndarray:
    """
    Solve cam * M ≈ ref  (least squares). Returns 3x3.
    """
    mask = np.isfinite(cam_rgb_24x3).all(axis=1) & np.isfinite(ref_rgb_24x3).all(axis=1)
    C = cam_rgb_24x3[mask]
    R = ref_rgb_24x3[mask]
    M, _, _, _ = np.linalg.lstsq(C, R, rcond=None)
    return M

def apply_matrix_image(img_linear: np.ndarray, M: np.ndarray) -> np.ndarray:
    h, w, c = img_linear.shape
    flat = img_linear.reshape(-1, 3)
    corr = flat @ M
    return corr.reshape(h, w, 3)

# =============
# Main driver
# =============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab_file", default="ColorChecker24_After_Nov2014.txt", help="CGATS Lab(D50) text for ColorChecker AFTER Nov 2014 (A1..F4)")
    ap.add_argument("--images_folder", required=True, help="Folder containing HDR linear images (PNG)")
    ap.add_argument("--out_suffix", default="_ccorr.png", help="Suffix for corrected output images")
    args = ap.parse_args()

    # 1) Load Lab(D50) reference (24x3)
    lab_ref = parse_cgats_lab_d50(args.lab_file)

    # 2) Convert to linear sRGB (D65): Lab(D50) -> XYZ(D50) -> XYZ(D65) -> lin sRGB(D65)
    XYZ_D50 = lab_to_xyz_D50(lab_ref)
    XYZ_D65 = bradford_D50_to_D65(XYZ_D50)
    ref_lin_srgb = xyz_to_linear_srgb_D65(XYZ_D65)  # (24,3)

    # Get all PNG images from folder
    image_paths = sorted([os.path.join(args.images_folder, f) for f in os.listdir(args.images_folder) if f.endswith('.png')])
    if not image_paths:
        print(f"ERROR: No PNG images found in {args.images_folder}")
        return

    Ms = []

    for img_path in image_paths:
        print(f"\n=== Processing: {img_path} ===")
        # Read HDR (linear)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = img[..., :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # important

        if img is None:
            print("ERROR: cannot read", img_path)
            continue
        img = img.astype(np.float64)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        # Build display preview for clicking
        prev = img / (np.percentile(img, 99.5) + 1e-12)
        prev = np.clip(prev, 0, 1)
        prev_bgr = (prev * 255).astype(np.uint8)
        try:
            cv2.namedWindow("ClickCorners", cv2.WINDOW_NORMAL)
        except cv2.error:
            # OpenCV built without GUI support, window will be created on first imshow
            pass
        cv2.resizeWindow("ClickCorners", 1200, 800)
        pts_img = click_four_points(prev_bgr, "ClickCorners")
        if pts_img.shape[0] != 4:
            print("Canceled or invalid clicks, skipping.")
            continue

        # Canonical rectangle (cols x rows)
        W = GRID_COLS * CELL_W
        H = GRID_ROWS * CELL_H
        pts_can = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)  # TL,TR,BR,BL

        # Find homography from canonical->image, invert for warp image->canonical
        H_img_can, _ = cv2.findHomography(pts_can, pts_img, method=0)
        H_can_img = np.linalg.inv(H_img_can)

        # 3) Warp & sample 24 patches
        cam_means, warped = warp_and_sample_patches(img, H_can_img,
                                                    rows=GRID_ROWS, cols=GRID_COLS,
                                                    cell_w=CELL_W, cell_h=CELL_H,
                                                    margin=CENTER_MARGIN)

        # 4) Fit 3x3
        M = fit_color_matrix(cam_means, ref_lin_srgb)
        Ms.append(M)
        print("Fitted 3x3 (camera -> linear sRGB D65):\n", M)

        # 5) Evaluate ΔE00 per patch
        corr_lin = cam_means @ M
        XYZ_D65_corr = linear_srgb_to_xyz_D65(corr_lin)
        XYZ_D50_corr = bradford_D65_to_D50(XYZ_D65_corr)
        Lab_corr = xyz_to_lab_D50(XYZ_D50_corr)
        dE = deltaE2000(Lab_corr, lab_ref)
        print("ΔE00 stats — mean: {:.3f}, median: {:.3f}, max: {:.3f}".format(np.mean(dE), np.median(dE), np.max(dE)))

        # 6) Apply to full image & save EXR
        img_corr = apply_matrix_image(img, M)
        out_path = os.path.splitext(img_path)[0] + args.out_suffix
        ok = cv2.imwrite(out_path, img_corr.astype(np.float32))
        print("Saved corrected HDR:", out_path if ok else "(failed to save)")

    if len(Ms) >= 2:
        M_avg = np.mean(np.stack(Ms, axis=0), axis=0)
        print("\nAverage 3x3 across processed images:\n", M_avg)

if __name__ == "__main__":
    main()
