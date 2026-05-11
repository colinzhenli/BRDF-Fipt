#!/usr/bin/env python3
"""Visualize ground-truth BRDF lobes from the Bonn dataset.

For selected materials and uniformly sampled surface points, produce 2D
slice plots of the measured BRDF values:

  1) Fix wi (shared light position), vary wo  → BRDF(wo)
  2) Fix wo (shared camera position), vary wi → BRDF(wi) × cos(θ_i)

Each produces elevation and azimuth slice sub-plots.

Usage:
    python scripts/visualize_brdf_lobes_bonn.py
"""

import argparse
import os
import sys
import re
import numpy as np
import scipy.io as spio
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Add project root to path so we can reuse bonn helpers ──────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.dataset.bonn import (
    _read_exr, _read_xyz_map, _parse_poly_channels, _read_gt_normal_map,
    _parse_pan_channels, _parse_lls_channels, _parse_poly2pan_weights,
    _pan_weights_for_image,
)

# ======================================================================
# Hard-coded parameters  (edit these)
# ======================================================================
ROOT_FOLDER  = "/media/raid/cloth/Bonn_train"
SVFRESNEL_DIR = "/media/raid/cloth/Bonn_train/Bonn_svfresnel"
OUTPUT_DIR   = "/media/raid/cloth/output/BRDF/visualizations_debug/lobes_2"
MATERIAL_IDS = [100]        # which material(s) to visualise
NUM_POINTS   = 20         # number of surface points (uniform grid from H×W)

# ======================================================================
# Helpers
# ======================================================================

def load_calibration(root_folder, mat_id):
    """Load calibration .mat file for one material (matches bonn.py)."""
    prefix = Path(root_folder) / f'mat{mat_id:04d}'
    path = f'{prefix}_calibration.mat'
    raw = spio.loadmat(path)
    calib = {}
    for rot_key in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
        rd = raw[rot_key][0, 0]
        rot_dict = {}
        for field in rd.dtype.names:
            val = rd[field]
            if field == 'llsCorners':
                rot_dict[field] = np.array(val, dtype=np.float32)
            else:
                rot_dict[field] = np.array(val, dtype=np.float32).flatten()
            m = re.match(r'(il|cv)(\d+)', field)
            if m and len(m.group(2)) < 3:
                padded = f'{m.group(1)}{int(m.group(2)):03d}'
                rot_dict[padded] = rot_dict[field]
        calib[rot_key] = rot_dict
    calib['llsAnglesDegrees'] = raw['llsAnglesDegrees'].flatten().astype(np.float64)
    return calib


def load_material(root_folder, mat_id, svfresnel_dir=None):
    """Load one material's xyz map, poly images, calibration, and (if
    available) the per-pixel GT surface normals from the svfresnel pack.

    Returns dict with keys:
        H, W, xyz_map (H,W,3), normals (H*W,3) or None,
        rgbs (K,V,3) float32, light_pos (K,3), cam_pos (K,3)
    """
    prefix = Path(root_folder) / f'mat{mat_id:04d}'
    calib = load_calibration(root_folder, mat_id)

    # xyz
    xyz_map, H, W = _read_xyz_map(f'{prefix}_xyz_rot000.exr')
    n_pixels = H * W

    # GT surface normals (per-pixel) — None if file missing
    normals_flat = None
    if svfresnel_dir is not None:
        normals_flat = _read_gt_normal_map(svfresnel_dir, mat_id, H, W)

    # poly EXR
    poly_data, poly_ch_names, pH, pW = _read_exr(f'{prefix}_poly.exr')
    assert (pH, pW) == (H, W)

    poly_images = _parse_poly_channels(poly_ch_names)
    n_poly = len(poly_images)
    assert n_poly > 0, f"No poly images found for mat{mat_id:04d}"

    # (H, W, n_poly*3) → (V, n_poly, 3) → (K, V, 3)
    poly_rgbs = poly_data.reshape(n_pixels, n_poly, 3).transpose(1, 0, 2)
    np.clip(poly_rgbs, 0, None, out=poly_rgbs)

    # light / camera positions per view
    light_pos = np.array([
        calib[im['rotation']][im['led']] for im in poly_images], dtype=np.float32)
    cam_pos = np.array([
        calib[im['rotation']][im['camera']] for im in poly_images], dtype=np.float32)

    # Per-view discrete indices extracted directly from EXR channel names.
    # cv_ids   : camera ring index 1–4  (cv01 → 1, cv02 → 2, …)
    # il_ids   : LED index              (il026 → 26, il027 → 27, …)
    # rot_azims: turntable azimuth (°)  (rot000 → 0, rot045 → 45, …)
    cv_ids    = np.array([int(re.search(r'\d+', im['camera']).group())
                          for im in poly_images], dtype=np.int32)
    il_ids    = np.array([int(re.search(r'\d+', im['led']).group())
                          for im in poly_images], dtype=np.int32)
    rot_azims = np.array([int(re.search(r'\d+', im['rotation']).group())
                          for im in poly_images], dtype=np.float32)

    return {
        'H': H, 'W': W,
        'xyz_map':   xyz_map,                            # (H, W, 3)
        'normals':   normals_flat,                       # (H*W, 3) or None
        'rgbs':      poly_rgbs.astype(np.float32),       # (K, V, 3)
        'light_pos': light_pos,                          # (K, 3)
        'cam_pos':   cam_pos,                            # (K, 3)
        'cv_ids':    cv_ids,                             # (K,) int  camera index 1..4
        'il_ids':    il_ids,                             # (K,) int  LED index e.g. 26–32
        'rot_azims': rot_azims,                          # (K,) float turntable azimuth °
    }


def load_material_all_sources(root_folder, mat_id, svfresnel_dir=None):
    """Load poly + pan + lls captures merged into pan-equivalent grayscale.

    For each source the per-view pan-equivalent value is:
      poly: rgb · per_il_weights        (3-vector dot product)
      pan : scalar (already pan-equivalent)
      lls : scalar (no empirical rescale — see bonn.py comment)

    Returns dict with concatenated arrays (K = K_poly + K_pan + K_lls):
      H, W, normals (HW,3)|None,
      gray_pan_equiv (K, V) float32,
      light_pos      (K, 3) float32,
      source_type    (K,)   object array of {'poly','pan','lls'}.
    """
    prefix = Path(root_folder) / f'mat{mat_id:04d}'
    raw_calib = spio.loadmat(f'{prefix}_calibration.mat')
    calib = load_calibration(root_folder, mat_id)
    w = _parse_poly2pan_weights(raw_calib)
    calib['poly2pan_per_il']     = w['per_il']
    calib['poly2pan_per_cv']     = w['per_cv']
    calib['poly2pan_global_avg'] = w['global_avg']

    xyz_map, H, W = _read_xyz_map(f'{prefix}_xyz_rot000.exr')
    n_pixels = H * W

    normals_flat = None
    if svfresnel_dir is not None:
        normals_flat = _read_gt_normal_map(svfresnel_dir, mat_id, H, W)

    gray_parts, light_parts, cam_parts, type_parts = [], [], [], []
    cv_parts, il_parts, rot_parts = [], [], []

    # ---- poly --------------------------------------------------------
    poly_data, poly_ch_names, pH, pW = _read_exr(f'{prefix}_poly.exr')
    assert (pH, pW) == (H, W)
    poly_images = _parse_poly_channels(poly_ch_names)
    if poly_images:
        n_poly = len(poly_images)
        poly_rgbs = poly_data.reshape(n_pixels, n_poly, 3).transpose(1, 0, 2)
        np.clip(poly_rgbs, 0, None, out=poly_rgbs)
        poly_w = np.stack([
            _pan_weights_for_image(calib, im['rotation'], im['camera'],
                                    im['led'], is_poly=True)
            for im in poly_images], axis=0).astype(np.float32)
        poly_gray = (poly_rgbs * poly_w[:, None, :]).sum(axis=-1)  # (K, V)
        poly_light = np.array([
            calib[im['rotation']][im['led']] for im in poly_images], dtype=np.float32)
        poly_cam = np.array([
            calib[im['rotation']][im['camera']] for im in poly_images], dtype=np.float32)
        poly_cv = np.array([int(re.search(r'\d+', im['camera']).group()) for im in poly_images], dtype=np.int32)
        poly_il = np.array([int(re.search(r'\d+', im['led']).group()) for im in poly_images], dtype=np.int32)
        poly_rot = np.array([int(re.search(r'\d+', im['rotation']).group()) for im in poly_images], dtype=np.float32)
        gray_parts.append(poly_gray)
        light_parts.append(poly_light)
        cam_parts.append(poly_cam)
        type_parts.append(np.array(['poly'] * n_poly, dtype=object))
        cv_parts.append(poly_cv); il_parts.append(poly_il); rot_parts.append(poly_rot)
    del poly_data

    # ---- pan ---------------------------------------------------------
    pan_path = f'{prefix}_pan.exr'
    if os.path.exists(pan_path):
        pan_data, pan_ch_names, pHp, pWp = _read_exr(pan_path)
        assert (pHp, pWp) == (H, W)
        name_to_idx = {n: i for i, n in enumerate(pan_ch_names)}
        pan_images = [im for im in _parse_pan_channels(pan_ch_names)
                       if int(im['led'][2:]) <= 24]
        if pan_images:
            n_pan = len(pan_images)
            pan_flat = pan_data.reshape(n_pixels, -1)
            ch_ci = np.array([name_to_idx[im['channel']] for im in pan_images])
            pan_gray = pan_flat[:, ch_ci].T.astype(np.float32)  # (K, V)
            np.clip(pan_gray, 0, None, out=pan_gray)
            pan_light = np.array([
                calib[im['rotation']][im['led']] for im in pan_images], dtype=np.float32)
            pan_cam = np.array([
                calib[im['rotation']][im['camera']] for im in pan_images], dtype=np.float32)
            pan_cv = np.array([int(re.search(r'\d+', im['camera']).group()) for im in pan_images], dtype=np.int32)
            pan_il = np.array([int(re.search(r'\d+', im['led']).group()) for im in pan_images], dtype=np.int32)
            pan_rot = np.array([int(re.search(r'\d+', im['rotation']).group()) for im in pan_images], dtype=np.float32)
            gray_parts.append(pan_gray)
            light_parts.append(pan_light)
            cam_parts.append(pan_cam)
            type_parts.append(np.array(['pan'] * n_pan, dtype=object))
            cv_parts.append(pan_cv); il_parts.append(pan_il); rot_parts.append(pan_rot)
        del pan_data

    # ---- lls ---------------------------------------------------------
    # LLS strips are extended sources; we treat the strip CENTER as the
    # effective wi position (matches what bonn.py uses for `light_pos`).
    # Raw LLS values are used directly — no empirical scale (the previous
    # 0.75 factor was a single-pixel fit; population median is ~1.0).
    lls_path = f'{prefix}_lls.exr'
    if os.path.exists(lls_path):
        lls_data, lls_ch_names, lHl, lWl = _read_exr(lls_path)
        assert (lHl, lWl) == (H, W)
        lls_angles = calib['llsAnglesDegrees']
        lls_angle_to_idx = {float(a): i for i, a in enumerate(lls_angles)}
        lls_name_to_idx = {n: i for i, n in enumerate(lls_ch_names)}
        lls_images = _parse_lls_channels(lls_ch_names)
        if lls_images:
            n_lls = len(lls_images)
            lls_flat = lls_data.reshape(n_pixels, -1)
            ch_ci = np.array([lls_name_to_idx[im['channel']] for im in lls_images])
            lls_gray = lls_flat[:, ch_ci].T.astype(np.float32)        # (K, V)
            np.clip(lls_gray, 0, None, out=lls_gray)
            # No empirical rescale — the previous _LLS_EMPIRICAL_SCALE = 0.75
            # was a single-pixel fit and the population median is ~1.0.

            center_list = []
            for im in lls_images:
                rd = calib[im['rotation']]
                ai = lls_angle_to_idx[im['angle']]
                c = rd['llsCorners'][:, :, ai].T                       # (4, 3)
                center_list.append(c.mean(axis=0))
            lls_light = np.array(center_list, dtype=np.float32)
            lls_cam = np.array([
                calib[im['rotation']][im['camera']] for im in lls_images],
                dtype=np.float32)
            lls_cv  = np.array([int(re.search(r'\d+', im['camera']).group())
                                for im in lls_images], dtype=np.int32)
            # Composite key per (strip, la_angle) so each unique LLS light
            # position gets its own ``il_id``. Stored in the same slot as
            # poly/pan il_ids; downstream code filters source-scoped so the
            # numeric ranges don't have to be disjoint. Encoded as
            # ``strip_idx * 1000 + la_angle_idx``.
            lls_il  = np.array([
                int(re.search(r'lls(\d+)', im['channel']).group(1)) * 1000
                + lls_angle_to_idx[im['angle']]
                for im in lls_images], dtype=np.int32)
            lls_rot = np.array([int(re.search(r'\d+', im['rotation']).group())
                                for im in lls_images], dtype=np.float32)
            gray_parts.append(lls_gray)
            light_parts.append(lls_light)
            cam_parts.append(lls_cam)
            type_parts.append(np.array(['lls'] * n_lls, dtype=object))
            cv_parts.append(lls_cv); il_parts.append(lls_il); rot_parts.append(lls_rot)
        del lls_data

    return {
        'H': H, 'W': W,
        'xyz_map': xyz_map,
        'normals': normals_flat,
        'gray_pan_equiv': np.concatenate(gray_parts, axis=0),    # (K, V)
        'light_pos':      np.concatenate(light_parts, axis=0),   # (K, 3)
        'cam_pos':        np.concatenate(cam_parts, axis=0),     # (K, 3)
        'source_type':    np.concatenate(type_parts),            # (K,)
        'cv_ids':         np.concatenate(cv_parts),              # (K,)
        'il_ids':         np.concatenate(il_parts),              # (K,)
        'rot_azims':      np.concatenate(rot_parts),             # (K,)
    }


def uniform_sample_point_indices(H, W, num_points):
    """Return pixel indices (into flattened H*W) on a uniform grid.

    Produces roughly sqrt(num_points) × sqrt(num_points) points,
    clamped to at most *num_points*.
    """
    n_side = max(1, int(np.ceil(np.sqrt(num_points))))
    rows = np.linspace(0, H - 1, n_side, dtype=int)
    cols = np.linspace(0, W - 1, n_side, dtype=int)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    flat = (rr * W + cc).ravel()
    return flat[:num_points]


def spherical_coords(dirs):
    """Convert (N,3) direction vectors to (elevation_deg, azimuth_deg).

    elevation = angle from XY-plane (positive toward +Z)
    azimuth   = angle in XY-plane from +X toward +Y
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    elev = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    azim = np.degrees(np.arctan2(y, x))
    return elev, azim


def compute_signed_theta(swept_dirs_cart, fixed_dir_cart):
    """Project swept directions onto the incidence plane of fixed_dir and
    return signed polar angle from the surface normal (+Z).

    Convention identical to UBO visualize_brdf_lobes_ubo_gt.compute_signed_theta:
      positive → specular side (opposite tangential hemisphere from fixed)
      negative → retro side    (same tangential hemisphere as fixed)
    """
    fi_t = np.array([fixed_dir_cart[0], fixed_dir_cart[1], 0.0])
    fi_t_norm = np.linalg.norm(fi_t)
    if fi_t_norm < 1e-8:
        forward = np.array([-1.0, 0.0, 0.0])
    else:
        forward = -fi_t / fi_t_norm
    in_plane = swept_dirs_cart[:, 0] * forward[0] + swept_dirs_cart[:, 1] * forward[1]
    return np.arctan2(in_plane, swept_dirs_cart[:, 2])


def compute_signed_theta_pairwise(swept_dirs, fixed_dirs):
    """Vectorized signed-theta where each swept direction has its OWN fixed
    direction (e.g. each Bonn view has its own wo, since the camera azimuth
    rotates with the turntable). ``swept_dirs`` and ``fixed_dirs`` are both
    (N, 3) and aligned row-by-row.

    Returns (N,) signed theta in radians, same convention as
    ``compute_signed_theta`` (positive = specular side).
    """
    fi_t = fixed_dirs.copy()
    fi_t[:, 2] = 0.0
    fi_t_norm = np.linalg.norm(fi_t, axis=1, keepdims=True)
    safe = fi_t_norm[:, 0] >= 1e-8
    forward = np.zeros_like(fi_t)
    # Default forward (when fixed_dir is along normal): -X
    forward[~safe, 0] = -1.0
    forward[safe] = -fi_t[safe] / fi_t_norm[safe]
    in_plane = swept_dirs[:, 0] * forward[:, 0] + swept_dirs[:, 1] * forward[:, 1]
    return np.arctan2(in_plane, swept_dirs[:, 2])


# ======================================================================
# Plotting
# ======================================================================

def plot_fix_wi_vary_wo(mat_id, point_idx, pixel_row, pixel_col,
                        xyz_point, rgbs_all, light_pos, cam_pos,
                        cv_ids, il_ids, rot_azims,
                        out_dir):
    """Fix wi (average over il elevations), plot BRDF vs wo angles.

    Index-based grouping (no heuristic tolerance):
      cv_ids    – camera ring index 1..4  (one elevation level each)
      il_ids    – LED index e.g. 26..32   (one elevation level each)
      rot_azims – turntable azimuth 0/45/90/135/180 (deg)

    Elevation slice  – 5 curves (one per rotation azimuth), 4 points each
                       (cv01–cv04).  y = mean BRDF over all il_ids at
                       that (rot, cv) cell.  x = ACTUAL mean wo_elev for
                       that specific (rot, cv) cell (not a global average).

    Azimuth slice    – 4 curves (one per cv_id), 5 points each
                       (one per rotation azimuth).  y = mean BRDF over all
                       il_ids at that (cv, rot) cell.  x = ACTUAL mean
                       wo_azim for that specific (cv, rot) cell.
    """
    # Compute direction vectors at this surface point
    wi_all = light_pos - xyz_point[None, :]
    wi_all /= np.maximum(np.linalg.norm(wi_all, axis=1, keepdims=True), 1e-8)
    wo_all = cam_pos - xyz_point[None, :]
    wo_all /= np.maximum(np.linalg.norm(wo_all, axis=1, keepdims=True), 1e-8)

    wo_elev, wo_azim = spherical_coords(wo_all)
    wi_elev, wi_azim = spherical_coords(wi_all)

    unique_cv  = sorted(np.unique(cv_ids).tolist())   # e.g. [1, 2, 3, 4]
    unique_il  = sorted(np.unique(il_ids).tolist())   # e.g. [26, 27, 28, 31, 32]
    unique_rot = sorted(np.unique(rot_azims).tolist()) # [0, 45, 90, 135, 180]

    # ── DEBUG: index → angle mapping ─────────────────────────────────
    # Each line shows the mean (and std) of the actual calibrated angle
    # for all views sharing the same index at this surface point.
    # Use this block to manually verify cv/il/rot ↔ angle correspondence.
    print(f"\n  [idx→angle  mat{mat_id:04d} pt({pixel_row},{pixel_col})]")
    for cv in unique_cv:
        m = cv_ids == cv
        print(f"    cv{cv:02d} → wo_elev = {np.mean(wo_elev[m]):6.1f}° "
              f"(std {np.std(wo_elev[m]):.2f}°, n={m.sum()})")
    for il in unique_il:
        m = il_ids == il
        print(f"    il{il:03d} → wi_elev = {np.mean(wi_elev[m]):6.1f}° "
              f"(std {np.std(wi_elev[m]):.2f}°, n={m.sum()})")
    for rot in unique_rot:
        m = rot_azims == rot
        print(f"    rot{rot:03.0f} → wo_azim = {np.mean(wo_azim[m]):6.1f}°, "
              f"wi_azim = {np.mean(wi_azim[m]):6.1f}°  (n={m.sum()})")
    # ── end DEBUG ─────────────────────────────────────────────────────

    # Global average wo_elev per cv_id – for legend labels (nominal camera elevation).
    avg_wo_elev_by_cv = {cv: float(np.mean(wo_elev[cv_ids == cv]))
                         for cv in unique_cv}
    # Mean wi azimuth per rotation – shown in elevation-slice curve labels so the
    # reader knows which azimuthal cut of wi each curve corresponds to.
    avg_wi_azim_by_rot = {rot: float(np.mean(wi_azim[rot_azims == rot]))
                          for rot in unique_rot}
    # Overall mean wi elevation (avg over all il_ids and rotations) – shown in
    # title so the reader knows the nominal elevation of the fixed light.
    mean_wi_elev = float(np.mean(wi_elev))

    cmap = plt.cm.tab10

    # ── Elevation slice ───────────────────────────────────────────────
    # 5 curves = 5 rotation azimuths; 4 points per curve = cv01–cv04
    # y = mean BRDF over all il_ids for the (rot, cv) cell
    # x = ACTUAL wo_elev for that specific (rot, cv) cell
    fig, ax = plt.subplots(figsize=(10, 6))
    for gi, rot in enumerate(unique_rot):
        x_vals, y_vals = [], []
        for cv in unique_cv:
            mask = (rot_azims == rot) & (cv_ids == cv)
            if not np.any(mask):
                continue
            y = float(np.mean(np.linalg.norm(rgbs_all[mask], axis=-1)))
            x_vals.append(float(np.mean(wo_elev[mask])))   # actual angle
            y_vals.append(y)
        x_vals, y_vals = zip(*sorted(zip(x_vals, y_vals))) if x_vals else ([], [])
        ax.plot(x_vals, y_vals, 'o-', markersize=5, color=cmap(gi),
                label=f"rot={rot:.0f}°  (wi_azim≈{avg_wi_azim_by_rot[rot]:.0f}°)")

    ax.set_xlabel("wo elevation (°)")
    ax.set_ylabel("BRDF magnitude (‖RGB‖)")
    ax.set_title(f"mat{mat_id:04d}  point({pixel_row},{pixel_col})  "
                 f"Fix wi (wi_elev≈{mean_wi_elev:.0f}°, avg over il) → BRDF vs wo elevation")
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir,
        f"mat{mat_id:04d}_pt{pixel_row:04d}x{pixel_col:04d}_fixWi_elev.png")
    plt.savefig(fname, dpi=150)
    plt.close()

    # ── Azimuth slice ─────────────────────────────────────────────────
    # 4 curves = cv01–cv04; 5 points per curve = 5 rotation azimuths
    # y = mean BRDF over all il_ids for the (cv, rot) cell
    # x = ACTUAL wo_azim for that specific (cv, rot) cell
    fig, ax = plt.subplots(figsize=(10, 6))
    for gi, cv in enumerate(unique_cv):
        x_vals, y_vals = [], []
        for rot in unique_rot:
            mask = (cv_ids == cv) & (rot_azims == rot)
            if not np.any(mask):
                continue
            y = float(np.mean(np.linalg.norm(rgbs_all[mask], axis=-1)))
            x_vals.append(float(np.mean(wo_azim[mask])))   # actual angle
            y_vals.append(y)
        x_vals, y_vals = zip(*sorted(zip(x_vals, y_vals))) if x_vals else ([], [])
        ax.plot(x_vals, y_vals, 'o-', markersize=5, color=cmap(gi),
                label=f"cv{cv:02d}  wo_elev≈{avg_wo_elev_by_cv[cv]:.0f}°")

    ax.set_xlabel("wo azimuth (°)")
    ax.set_ylabel("BRDF magnitude (‖RGB‖)")
    ax.set_title(f"mat{mat_id:04d}  point({pixel_row},{pixel_col})  "
                 f"Fix wi (wi_elev≈{mean_wi_elev:.0f}°, avg over il) → BRDF vs wo azimuth")
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir,
        f"mat{mat_id:04d}_pt{pixel_row:04d}x{pixel_col:04d}_fixWi_azim.png")
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_fix_wo_vary_wi(mat_id, point_idx, pixel_row, pixel_col,
                        xyz_point, rgbs_all, light_pos, cam_pos,
                        cv_ids, il_ids, rot_azims,
                        out_dir, apply_cos_wi=True):
    """Fix wo (average over cv elevations), plot BRDF×cos(θ_i) vs wi elevation.

    Index-based grouping (no heuristic tolerance):
      cv_ids    – camera ring index 1..4  (one elevation level each)
      il_ids    – LED index e.g. 26..32   (one elevation level each)
      rot_azims – turntable azimuth 0/45/90/135/180 (deg)

    Elevation slice  – 5 curves (one per rotation azimuth), 5 points each
                       (one per il_id).  y = mean BRDF×cos over all cv_ids
                       at that (rot, il) cell.  x = ACTUAL mean wi_elev for
                       that specific (rot, il) cell (not a global average).

    NOTE: The azimuth slice (curves by il_id, x = wi_azim) is intentionally
    omitted.  Because LEDs are not placed at a common azimuth, a given il_id
    does NOT correspond to a fixed wi elevation as the turntable rotates –
    both wi_elev and wi_azim change together.  Treating il_id as "fixed
    elevation" for an azimuth sweep is therefore geometrically invalid.
    """
    # Compute direction vectors at this surface point
    wi_all = light_pos - xyz_point[None, :]
    wi_all /= np.maximum(np.linalg.norm(wi_all, axis=1, keepdims=True), 1e-8)
    wo_all = cam_pos - xyz_point[None, :]
    wo_all /= np.maximum(np.linalg.norm(wo_all, axis=1, keepdims=True), 1e-8)

    # cos(θ_i) = dot(wi, surface_normal);  surface normal ≈ +Z
    normal = np.array([0.0, 0.0, 1.0])
    cos_theta_i = np.clip(wi_all @ normal, 0, None)   # (K,)

    wi_elev, wi_azim = spherical_coords(wi_all)
    wo_elev, wo_azim = spherical_coords(wo_all)

    unique_cv  = sorted(np.unique(cv_ids).tolist())
    unique_il  = sorted(np.unique(il_ids).tolist())
    unique_rot = sorted(np.unique(rot_azims).tolist())

    # ── DEBUG: index → angle mapping ─────────────────────────────────
    # (printed only from plot_fix_wi_vary_wo to avoid duplication; the
    #  same surface point is used so the mapping is identical.  Uncomment
    #  the block below if you want an independent printout here.)
    #
    # for il in unique_il:
    #     m = il_ids == il
    #     print(f"    il{il:03d} → wi_elev = {np.mean(wi_elev[m]):6.1f}° "
    #           f"(std {np.std(wi_elev[m]):.2f}°, n={m.sum()})")
    # for rot in unique_rot:
    #     m = rot_azims == rot
    #     print(f"    rot{rot:03.0f} → wi_azim = {np.mean(wi_azim[m]):6.1f}° "
    #           f"(n={m.sum()})")
    # ── end DEBUG ─────────────────────────────────────────────────────

    # Mean wo azimuth per rotation – shown in curve labels so the reader knows
    # which azimuthal cut of wo each curve represents.
    avg_wo_azim_by_rot = {rot: float(np.mean(wo_azim[rot_azims == rot]))
                          for rot in unique_rot}
    # Overall mean wo elevation (avg over all cv_ids and rotations) – shown in
    # title so the reader knows the nominal elevation of the fixed camera.
    mean_wo_elev = float(np.mean(wo_elev))

    cmap = plt.cm.tab10

    # ── Elevation slice ───────────────────────────────────────────────
    # 5 curves = 5 rotation azimuths; 5 points per curve = il_ids
    # y = mean BRDF×cos over all cv_ids for the (rot, il) cell
    # x = ACTUAL wi_elev for that specific (rot, il) cell
    fig, ax = plt.subplots(figsize=(10, 6))
    for gi, rot in enumerate(unique_rot):
        x_vals, y_vals = [], []
        for il in unique_il:
            mask = (rot_azims == rot) & (il_ids == il)
            if not np.any(mask):
                continue
            brdf_mag = np.linalg.norm(rgbs_all[mask], axis=-1)
            if apply_cos_wi:
                y = float(np.mean(brdf_mag * cos_theta_i[mask]))
            else:
                y = float(np.mean(brdf_mag))
            x_vals.append(float(np.mean(wi_elev[mask])))   # actual angle
            y_vals.append(y)
        x_vals, y_vals = zip(*sorted(zip(x_vals, y_vals))) if x_vals else ([], [])
        ax.plot(x_vals, y_vals, 'o-', markersize=5, color=cmap(gi),
                label=f"rot={rot:.0f}°  (wo_azim≈{avg_wo_azim_by_rot[rot]:.0f}°)")

    ax.set_xlabel("wi elevation (°)")
    if apply_cos_wi:
        ax.set_ylabel("BRDF × cos(θ_i)  (‖RGB‖ × cos)")
        title_metric = "BRDF×cos"
    else:
        ax.set_ylabel("BRDF magnitude (‖RGB‖)")
        title_metric = "BRDF"
    ax.set_title(f"mat{mat_id:04d}  point({pixel_row},{pixel_col})  "
                 f"Fix wo (wo_elev≈{mean_wo_elev:.0f}°, avg over cv) → {title_metric} vs wi elevation")
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir,
        f"mat{mat_id:04d}_pt{pixel_row:04d}x{pixel_col:04d}_fixWo_elev.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    # Azimuth slice omitted: see docstring for reason.


def plot_polar_fix_wi_vary_wo(mat_id, point_idx, pixel_row, pixel_col,
                              xyz_point, rgbs_all, light_pos, cam_pos,
                              il_ids,
                              out_dir):
    """UBO-style polar plot: one curve per il_id (fixed wi elevation).

    For each il_id group, the mean wi defines an incidence plane; every wo
    in the group is projected into that plane and plotted at its signed
    polar angle. Mirrors ``visualize_brdf_lobes_ubo_gt.plot_vary_wo_gt``.
    """
    wi_all = light_pos - xyz_point[None, :]
    wi_all /= np.maximum(np.linalg.norm(wi_all, axis=1, keepdims=True), 1e-8)
    wo_all = cam_pos - xyz_point[None, :]
    wo_all /= np.maximum(np.linalg.norm(wo_all, axis=1, keepdims=True), 1e-8)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    unique_il = sorted(np.unique(il_ids).tolist())
    cmap = plt.cm.tab10

    for ci, il in enumerate(unique_il):
        mask = il_ids == il
        if not np.any(mask):
            continue
        # Each view in the group has its own (rotated) wi; compute signed_theta
        # for each wo against the matching wi to keep the incidence plane valid.
        wi_group = wi_all[mask]
        wo_group = wo_all[mask]
        wi_theta_deg = float(np.degrees(
            np.arccos(np.clip(wi_group[:, 2], -1, 1))).mean())
        signed_theta = compute_signed_theta_pairwise(wo_group, wi_group)
        brdf_mag = np.linalg.norm(rgbs_all[mask], axis=-1)
        order = np.argsort(signed_theta)
        ax.plot(signed_theta[order], brdf_mag[order], 'o-', markersize=4,
                color=cmap(ci),
                label=f'il{il:03d}  θ_i≈{wi_theta_deg:.0f}°')
        ax.axvline(x=np.deg2rad(wi_theta_deg), color=cmap(ci),
                   linestyle='--', alpha=0.5)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax.set_title(f'mat{mat_id:04d}  pt({pixel_row},{pixel_col})\n'
                 f'GT BRDF Polar Plot (Fixed wi, vary wo)')
    plt.tight_layout()
    fname = os.path.join(out_dir,
        f"mat{mat_id:04d}_pt{pixel_row:04d}x{pixel_col:04d}_polar_vary_wo.png")
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_polar_fix_wo_vary_wi(mat_id, point_idx, pixel_row, pixel_col,
                              xyz_point, rgbs_all, light_pos, cam_pos,
                              cv_ids,
                              out_dir, apply_cos_wi=True):
    """UBO-style polar plot: one curve per cv_id (fixed wo elevation).

    For each cv_id group, the mean wo defines an incidence plane; every wi
    in the group is projected into that plane. y = BRDF (× cos(θ_i) when
    ``apply_cos_wi``). Mirrors ``visualize_brdf_lobes_ubo_gt.plot_vary_wi_gt``.
    """
    wi_all = light_pos - xyz_point[None, :]
    wi_all /= np.maximum(np.linalg.norm(wi_all, axis=1, keepdims=True), 1e-8)
    wo_all = cam_pos - xyz_point[None, :]
    wo_all /= np.maximum(np.linalg.norm(wo_all, axis=1, keepdims=True), 1e-8)
    cos_theta_i = np.clip(wi_all[:, 2], 0, None)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    unique_cv = sorted(np.unique(cv_ids).tolist())
    cmap = plt.cm.tab10

    for ci, cv in enumerate(unique_cv):
        mask = cv_ids == cv
        if not np.any(mask):
            continue
        wi_group = wi_all[mask]
        wo_group = wo_all[mask]
        wo_theta_deg = float(np.degrees(
            np.arccos(np.clip(wo_group[:, 2], -1, 1))).mean())
        signed_theta = compute_signed_theta_pairwise(wi_group, wo_group)
        brdf_mag = np.linalg.norm(rgbs_all[mask], axis=-1)
        y = brdf_mag * cos_theta_i[mask] if apply_cos_wi else brdf_mag
        order = np.argsort(signed_theta)
        ax.plot(signed_theta[order], y[order], 'o-', markersize=4,
                color=cmap(ci),
                label=f'cv{cv:02d}  θ_o≈{wo_theta_deg:.0f}°')
        ax.axvline(x=np.deg2rad(wo_theta_deg), color=cmap(ci),
                   linestyle='--', alpha=0.5)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    title_lhs = 'GT BRDF × cos(θ_i)' if apply_cos_wi else 'GT BRDF'
    ax.set_title(f'mat{mat_id:04d}  pt({pixel_row},{pixel_col})\n'
                 f'{title_lhs} Polar Plot (Fixed wo, vary wi)')
    plt.tight_layout()
    fname = os.path.join(out_dir,
        f"mat{mat_id:04d}_pt{pixel_row:04d}x{pixel_col:04d}_polar_vary_wi.png")
    plt.savefig(fname, dpi=150)
    plt.close()


def compute_brdf_vs_costhetai_data(xyz_point, rgbs_all, light_pos, il_ids,
                                   apply_cos_wi=True, normal=None):
    """Return (cos_theta_i_per_view, y_per_view) for one surface point.

    cos(θ_i) is computed against the per-point surface normal when supplied
    (Bonn provides a GT normal map via svfresnel) — otherwise it falls back
    to wi·(+Z), which is only correct for a perfectly flat sample.

    y is BRDF × cos(θ_i) when apply_cos_wi, else raw BRDF magnitude.
    """
    wi_all = light_pos - xyz_point[None, :]
    wi_all /= np.maximum(np.linalg.norm(wi_all, axis=1, keepdims=True), 1e-8)
    if normal is None:
        cos_theta_i = np.clip(wi_all[:, 2], 0.0, None)
    else:
        n = np.asarray(normal, dtype=np.float32).reshape(3)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            cos_theta_i = np.clip(wi_all[:, 2], 0.0, None)
        else:
            n = n / n_norm
            cos_theta_i = np.clip(wi_all @ n, 0.0, None)
    brdf_mag = np.linalg.norm(rgbs_all, axis=-1)
    y_all = brdf_mag * cos_theta_i if apply_cos_wi else brdf_mag
    return cos_theta_i, y_all


def plot_brdf_vs_costhetai(mat_id, point_idx, pixel_row, pixel_col,
                           xyz_point, rgbs_all, light_pos, cam_pos,
                           il_ids,
                           out_dir, apply_cos_wi=True, normal=None):
    """Per-point: mean BRDF vs cos(θ_i), averaged over all wi azimuths & wo.

    Shows individual views as scatter (one dot per view) plus a line
    connecting the per-il_id mean. No errorbars.
    """
    cos_theta_i, y_all = compute_brdf_vs_costhetai_data(
        xyz_point, rgbs_all, light_pos, il_ids, apply_cos_wi, normal=normal)

    unique_il = sorted(np.unique(il_ids).tolist())
    rows = []
    for il in unique_il:
        mask = il_ids == il
        if not np.any(mask):
            continue
        rows.append((float(cos_theta_i[mask].mean()),
                     float(y_all[mask].mean()),
                     int(mask.sum()), il))
    rows.sort(key=lambda r: r[0])
    cos_means = np.array([r[0] for r in rows])
    y_means   = np.array([r[1] for r in rows])
    counts    = [r[2] for r in rows]
    il_labels = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cos_theta_i, y_all, alpha=0.35, s=18, color='C0',
               label='individual views')
    ax.plot(cos_means, y_means, 'o-', markersize=8, linewidth=2.0,
            color='C1', label='per-il mean')
    for x, y, il, n in zip(cos_means, y_means, il_labels, counts):
        ax.annotate(f'il{il:03d} (n={n})', (x, y),
                    textcoords='offset points', xytext=(6, 6), fontsize=8)

    metric = 'BRDF × cos(θ_i)' if apply_cos_wi else 'BRDF magnitude (‖RGB‖)'
    ax.set_xlabel('cos(θ_i)')
    ax.set_ylabel(metric)
    ax.set_title(f'mat{mat_id:04d}  pt({pixel_row},{pixel_col})\n'
                 f'{metric} averaged over all wi azimuths and all wo')
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    fname = os.path.join(out_dir,
        f"mat{mat_id:04d}_pt{pixel_row:04d}x{pixel_col:04d}_brdf_vs_costhetai.png")
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_phi_binned_slices_per_pixel(mat_id, mat_all, pix_indices, out_dir,
                                     theta_o_centers=(15.0, 45.0, 75.0),
                                     phi_i_tol=45.0,
                                     theta_o_tol=10.0,
                                     n_theta_i_bins=18):
    """Per-pixel θ_i lobe slices, controlled for anisotropy (φ_i) and view (θ_o).

    1×3 grid per pixel: three θ_o panels at the *dominant* φ_i for this
    pixel (the φ_i window with the most views across all three sources).
    Within each panel: views are filtered to that (φ_i, θ_o) cell, signed
    θ_i is computed, and binned along θ_i so each colored line is a clean
    aggregated lobe slice — no raw-scatter dots so the figure stays
    readable.

    φ_i / φ_o use the per-pixel surface-local frame: GT normal + projected
    world +X as tangent.
    """
    H = mat_all['H']; W = mat_all['W']
    light_pos = mat_all['light_pos']
    cam_pos   = mat_all['cam_pos']
    src_type  = mat_all['source_type']
    gray      = mat_all['gray_pan_equiv']
    normals   = mat_all['normals']

    palette = {'poly': 'C0', 'pan': 'C1', 'lls': 'C2'}
    N_theta = len(theta_o_centers)

    theta_edges   = np.linspace(-np.pi / 2, np.pi / 2, n_theta_i_bins + 1)
    theta_centers_rad = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    world_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    world_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Probe these φ_i centers, then auto-select the one with most data.
    phi_i_probe = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)

    for pix_idx in pix_indices:
        r, c = int(pix_idx) // W, int(pix_idx) % W
        xyz_pt = mat_all['xyz_map'][r, c]
        n_pt = (normals[pix_idx] if normals is not None else
                np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if np.linalg.norm(n_pt) < 1e-8:
            n_pt = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        n = n_pt / np.linalg.norm(n_pt)

        t = world_x - (world_x @ n) * n
        if np.linalg.norm(t) < 1e-6:
            t = world_y - (world_y @ n) * n
        t = t / np.linalg.norm(t)
        b = np.cross(n, t)

        wi = light_pos - xyz_pt[None, :]
        wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
        wo = cam_pos - xyz_pt[None, :]
        wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)

        wi_phi = np.degrees(np.arctan2(wi @ b, wi @ t)) % 360.0
        wo_theta = np.degrees(np.arccos(np.clip(wo @ n, -1, 1)))
        y_all = gray[:, int(pix_idx)]

        # Auto-pick the φ_i window with most total views, summed across
        # θ_o panels & sources. Keeps the plot informative pixel-by-pixel
        # without burying the user in a 4×3 grid most cells of which are
        # sparse.
        def _coverage(phi_c):
            d_phi = np.abs(((wi_phi - phi_c + 180.0) % 360.0) - 180.0)
            n = 0
            for tc in theta_o_centers:
                n += int(((d_phi <= phi_i_tol) &
                          (np.abs(wo_theta - tc) <= theta_o_tol)).sum())
            return n
        phi_c = max(phi_i_probe, key=_coverage)

        fig, axes = plt.subplots(
            1, N_theta, figsize=(7.0 * N_theta, 7.0),
            subplot_kw={'projection': 'polar'},
            squeeze=False)
        axes = axes[0]

        for i_theta, theta_c in enumerate(theta_o_centers):
            ax = axes[i_theta]
            d_phi = np.abs(((wi_phi - phi_c + 180.0) % 360.0) - 180.0)
            cell_mask = (d_phi <= phi_i_tol) & \
                        (np.abs(wo_theta - theta_c) <= theta_o_tol)

            for src in ['poly', 'pan', 'lls']:
                bin_mask = (src_type == src) & cell_mask
                if not np.any(bin_mask):
                    continue
                wi_g = wi[bin_mask]
                wo_g = wo[bin_mask]
                y_g  = y_all[bin_mask]
                signed_theta = compute_signed_theta_pairwise(wi_g, wo_g)
                idx = np.clip(np.digitize(signed_theta, theta_edges) - 1,
                              0, n_theta_i_bins - 1)
                means = np.full(n_theta_i_bins, np.nan)
                for bb in range(n_theta_i_bins):
                    sel = idx == bb
                    if sel.sum() > 0:
                        means[bb] = y_g[sel].mean()
                valid = ~np.isnan(means)
                if valid.any():
                    ax.plot(theta_centers_rad[valid], means[valid], 'o-',
                            linewidth=2.0, markersize=6,
                            color=palette[src], alpha=0.95,
                            label=f'{src} (n={int(bin_mask.sum())})')

            ax.set_theta_zero_location('N'); ax.set_theta_direction(1)
            ax.set_thetamin(-90); ax.set_thetamax(90)
            ax.set_title(
                f'θ_o = {theta_c:.0f}° ± {theta_o_tol:.0f}°',
                fontsize=14, pad=10)
            ax.legend(loc='upper right', bbox_to_anchor=(1.30, 1.05),
                      fontsize=11)
            ax.tick_params(labelsize=10)

        fig.suptitle(
            f'mat{mat_id:04d}  pixel ({r}, {c})   '
            f'φ_i = {phi_c:.0f}° ± {phi_i_tol:.0f}°  '
            f'→ sweep signed θ_i (scalar, pan-equiv)',
            fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
            f'mat{mat_id:04d}_pt{r:04d}x{c:04d}_phi_binned.png'),
            dpi=180, bbox_inches='tight')
        plt.close()


def plot_strict_slices_per_pixel(mat_id, mat_all, pix_indices, out_dir,
                                 ref_cv=1, ref_rot=0.0):
    """True per-pixel slices: every dot is a single view, no aggregation.

    Two figures per pixel, both polar:
      A) light-sweep — fix cv=ref_cv and rot=ref_rot; vary the light index
         (LED id for poly/pan, strip id for LLS). One curve per source.
         Each dot is one observation; line connects dots sorted by signed
         θ_i (wi projected into the fixed-wo plane).
      B) view-sweep  — for each source, pick the light index whose median
         wi-elevation at this pixel is closest to 45°; fix that light and
         rot=ref_rot; vary cv (1–4). 4 dots per source-curve. Signed θ_o
         (wo projected into the fixed-wi plane).

    These are sparse but every dot is a single view at the same pixel —
    no projection collision from mixing different φ. If a curve is spiky
    here, the GT really is spiky.
    """
    H = mat_all['H']; W = mat_all['W']
    light_pos = mat_all['light_pos']
    cam_pos   = mat_all['cam_pos']
    src_type  = mat_all['source_type']
    cv_ids    = mat_all['cv_ids']
    il_ids    = mat_all['il_ids']
    rot_azims = mat_all['rot_azims']
    gray      = mat_all['gray_pan_equiv']
    normals   = mat_all['normals']

    palette = {'poly': 'C0', 'pan': 'C1', 'lls': 'C2'}

    for pix_idx in pix_indices:
        r, c = int(pix_idx) // W, int(pix_idx) % W
        xyz_pt = mat_all['xyz_map'][r, c]
        n_pt = (normals[pix_idx] if normals is not None
                else np.array([0.0, 0.0, 1.0], dtype=np.float32))
        n = n_pt / max(np.linalg.norm(n_pt), 1e-8)

        wi = light_pos - xyz_pt[None, :]
        wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
        wo = cam_pos - xyz_pt[None, :]
        wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)
        wi_elev = np.degrees(np.arccos(np.clip(wi @ n, -1, 1)))
        y_all = gray[:, int(pix_idx)]

        # ---- Figure A: light sweep (fix cv+rot, vary light index) ----
        fig, ax = plt.subplots(figsize=(7.5, 7),
                               subplot_kw={'projection': 'polar'})
        for src in ['poly', 'pan', 'lls']:
            mask = ((src_type == src) & (cv_ids == ref_cv) &
                    (rot_azims == ref_rot))
            if not np.any(mask):
                continue
            wi_g = wi[mask]
            wo_g = wo[mask]
            y_g  = y_all[mask]
            signed_theta = compute_signed_theta_pairwise(wi_g, wo_g)
            order = np.argsort(signed_theta)
            ax.plot(signed_theta[order], y_g[order], 'o-',
                    markersize=6, linewidth=1.5,
                    color=palette[src], alpha=0.95,
                    label=f'{src} (n={int(mask.sum())})')
        ax.set_theta_zero_location('N'); ax.set_theta_direction(1)
        ax.set_thetamin(-90); ax.set_thetamax(90)
        ax.set_title(
            f'mat{mat_id:04d} pixel ({r},{c}) — light sweep\n'
            f'fix cv={ref_cv:02d}, rot={ref_rot:.0f}°; vary LED / LLS strip; '
            f'x = signed θ_i, y = scalar (pan-equiv)',
            fontsize=10)
        ax.legend(loc='upper right', bbox_to_anchor=(1.40, 1.05), fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
            f'mat{mat_id:04d}_pt{r:04d}x{c:04d}_strict_lightsweep.png'), dpi=140)
        plt.close()

        # ---- Figure B: view sweep (fix il+rot, vary cv) ----
        fig, ax = plt.subplots(figsize=(7.5, 7),
                               subplot_kw={'projection': 'polar'})
        for src in ['poly', 'pan', 'lls']:
            src_rot_mask = (src_type == src) & (rot_azims == ref_rot)
            if not np.any(src_rot_mask):
                continue
            # Choose the per-source light index whose median wi-elevation
            # at this pixel is closest to 45° (mid-hemisphere).
            il_pool = []
            for ll in np.unique(il_ids[src_rot_mask]):
                mm = src_rot_mask & (il_ids == ll)
                if mm.any():
                    il_pool.append((int(ll), float(np.median(wi_elev[mm]))))
            if not il_pool:
                continue
            il_pool.sort(key=lambda t: abs(t[1] - 45.0))
            chosen_il, chosen_il_elev = il_pool[0]
            mask = src_rot_mask & (il_ids == chosen_il)
            if not np.any(mask):
                continue
            wi_g = wi[mask]
            wo_g = wo[mask]
            y_g  = y_all[mask]
            signed_theta = compute_signed_theta_pairwise(wo_g, wi_g)
            order = np.argsort(signed_theta)
            ax.plot(signed_theta[order], y_g[order], 'o-',
                    markersize=7, linewidth=1.7,
                    color=palette[src], alpha=0.95,
                    label=f'{src} (light_id={chosen_il}, θ_i≈{chosen_il_elev:.0f}°, n={int(mask.sum())})')
        ax.set_theta_zero_location('N'); ax.set_theta_direction(1)
        ax.set_thetamin(-90); ax.set_thetamax(90)
        ax.set_title(
            f'mat{mat_id:04d} pixel ({r},{c}) — view sweep\n'
            f'fix rot={ref_rot:.0f}°, fix per-source mid-elevation LED; vary cv (1–4); '
            f'x = signed θ_o, y = scalar (pan-equiv)',
            fontsize=10)
        ax.legend(loc='upper right', bbox_to_anchor=(1.50, 1.05), fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
            f'mat{mat_id:04d}_pt{r:04d}x{c:04d}_strict_viewsweep.png'), dpi=140)
        plt.close()


def plot_lobe_alignment_per_pixel(mat_id, mat_all, pix_indices, out_dir,
                                  elev_targets=(15.0, 30.0, 45.0, 60.0, 75.0),
                                  elev_tol=7.5):
    """Per-pixel polar lobe slices overlaid for poly/pan/lls.

    Mirrors the trainer's ``visualize_brdf_lobe`` convention — fix wi at a
    sequence of θ_i targets (default {15,30,45,60,75}°), and within each
    target's elevation band plot scalar response vs *signed* θ_o (wo
    projected into the per-view wi-incidence plane via
    ``compute_signed_theta_pairwise``). Then the dual: fix wo at the same
    θ_o targets and sweep wi. Each panel overlays the three source curves
    in colour:
        poly=C0, pan=C1, lls=C2 (pan-equivalent grayscale).

    No binning of the scalar — each marker is a raw view, ordered along
    signed-θ. If the three colours trace the same curve in each slice, the
    poly→pan weights put the sources on a single lobe. Vertical offsets
    between colours are the residual calibration gap.

    Two PNGs per pixel:
      mat{ID}_pt{R}x{C}_fixWi_slices.png  (fix wi, vary wo;  1×N polar grid)
      mat{ID}_pt{R}x{C}_fixWo_slices.png  (fix wo, vary wi;  1×N polar grid)
    """
    H = mat_all['H']; W = mat_all['W']
    light_pos = mat_all['light_pos']
    cam_pos   = mat_all['cam_pos']
    src_type  = mat_all['source_type']
    gray      = mat_all['gray_pan_equiv']
    normals   = mat_all['normals']

    palette = {'poly': 'C0', 'pan': 'C1', 'lls': 'C2'}
    N = len(elev_targets)

    # Number of bins along the sweep axis (signed-θ). The trainer's
    # ``visualize_brdf_lobe`` queries the decoder at 128 evenly spaced θ_o
    # values to draw a smooth curve; for discrete GT we do the analogous
    # thing — aggregate within each Δθ slice WITHIN the fixed-elevation
    # band so the curve traces the lobe shape instead of zig-zagging
    # between adjacent same-angle observations.
    n_theta_bins = 12
    theta_edges = np.linspace(-np.pi / 2, np.pi / 2, n_theta_bins + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    def _draw_grid(fixed_elev, sweep_elev, fixed_dirs, sweep_dirs,
                   y, sources_present, axes, fixed_label):
        """Populate a 1×N polar grid with one panel per elev_target."""
        for axi, target in enumerate(elev_targets):
            ax = axes[axi] if N > 1 else axes
            for src in ['poly', 'pan', 'lls']:
                src_mask = src_type == src
                bin_mask = src_mask & (np.abs(fixed_elev - target) <= elev_tol)
                if not np.any(bin_mask):
                    continue
                signed_theta = compute_signed_theta_pairwise(
                    sweep_dirs[bin_mask], fixed_dirs[bin_mask])
                y_g = y[bin_mask]
                # Raw observations (faint scatter for transparency).
                ax.scatter(signed_theta, y_g, s=6, alpha=0.18,
                           color=palette[src])
                # Aggregate along signed-θ to recover a smooth lobe curve.
                idx = np.clip(np.digitize(signed_theta, theta_edges) - 1,
                              0, n_theta_bins - 1)
                means = np.full(n_theta_bins, np.nan)
                for b in range(n_theta_bins):
                    sel = idx == b
                    if sel.sum() > 0:
                        means[b] = y_g[sel].mean()
                valid = ~np.isnan(means)
                if valid.any():
                    ax.plot(theta_centers[valid], means[valid], 'o-',
                            markersize=4, linewidth=1.6,
                            color=palette[src], alpha=0.95,
                            label=f'{src} n={int(bin_mask.sum())}')
            ax.axvline(x=np.deg2rad(target), color='k', linestyle='--', alpha=0.35)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(1)
            ax.set_thetamin(-90); ax.set_thetamax(90)
            ax.set_title(f'{fixed_label}={target:.0f}° ± {elev_tol:.0f}°',
                         fontsize=10)
            ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.05),
                      fontsize=7)

    for pix_idx in pix_indices:
        r, c = int(pix_idx) // W, int(pix_idx) % W
        xyz_pt = mat_all['xyz_map'][r, c]
        n_pt = (normals[pix_idx] if normals is not None
                else np.array([0.0, 0.0, 1.0], dtype=np.float32))
        n = n_pt / max(np.linalg.norm(n_pt), 1e-8)

        wi = light_pos - xyz_pt[None, :]
        wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
        wo = cam_pos - xyz_pt[None, :]
        wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)
        wi_elev = np.degrees(np.arccos(np.clip(wi @ n, -1, 1)))
        wo_elev = np.degrees(np.arccos(np.clip(wo @ n, -1, 1)))
        y_all = gray[:, int(pix_idx)]

        sources = sorted(set(src_type.tolist()))

        # ---- Fix wi, vary wo ----
        fig, axes = plt.subplots(1, N, figsize=(4.2 * N, 4.7),
                                 subplot_kw={'projection': 'polar'})
        _draw_grid(wi_elev, wo_elev, wi, wo, y_all, sources, axes, 'θ_i')
        fig.suptitle(
            f'mat{mat_id:04d}  pixel ({r}, {c})   Fix wi → vary wo   '
            f'(signed θ_o; scalar in pan-equiv units; sources={sources})',
            fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
            f'mat{mat_id:04d}_pt{r:04d}x{c:04d}_fixWi_slices.png'), dpi=140)
        plt.close()

        # ---- Fix wo, vary wi ----
        fig, axes = plt.subplots(1, N, figsize=(4.2 * N, 4.7),
                                 subplot_kw={'projection': 'polar'})
        _draw_grid(wo_elev, wi_elev, wo, wi, y_all, sources, axes, 'θ_o')
        fig.suptitle(
            f'mat{mat_id:04d}  pixel ({r}, {c})   Fix wo → vary wi   '
            f'(signed θ_i; scalar in pan-equiv units; sources={sources})',
            fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
            f'mat{mat_id:04d}_pt{r:04d}x{c:04d}_fixWo_slices.png'), dpi=140)
        plt.close()


def plot_aggregate_with_without_cos(mat_id, mat_all, pix_indices, out_dir,
                                    cos_bins=20):
    """Side-by-side cosine-check across poly+pan+lls sources.

    Diagnoses whether raw LLS GT already includes the cos(θ_i) factor by
    comparing each source's lobe shape against the same cos(θ_i) axis.

    LEFT  panel — raw GT vs cos(θ_i)         (no cos applied)
    RIGHT panel — GT × cos(θ_i) vs cos(θ_i)  (cos applied a posteriori)

    Reading the comparison:
      * LLS GT *includes* cos already
            LEFT:  lls drops to 0 at cos→0 (just like poly/pan after × cos)
            RIGHT: lls under-shoots poly/pan (cos is now applied twice)
      * LLS GT *does NOT* include cos
            LEFT:  lls stays high at cos→0 like raw poly/pan
            RIGHT: lls aligns with poly/pan

    The aggregate uses the per-pixel GT normal (when available) and the
    LLS strip CENTER as the effective wi position — same effective-direction
    convention used by ``light_pos`` in the dataset loader.
    """
    H = mat_all['H']; W = mat_all['W']
    light_pos = mat_all['light_pos']
    src_type  = mat_all['source_type']
    gray      = mat_all['gray_pan_equiv']
    normals   = mat_all['normals']

    cos_all, y_all, src_all = [], [], []
    for pix_idx in pix_indices:
        r, c = int(pix_idx) // W, int(pix_idx) % W
        xyz = mat_all['xyz_map'][r, c]
        n_pt = (normals[pix_idx] if normals is not None
                else np.array([0.0, 0.0, 1.0], dtype=np.float32))
        wi = light_pos - xyz[None, :]
        wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
        n = n_pt / max(np.linalg.norm(n_pt), 1e-8)
        cos_t_i = np.clip(wi @ n, 0.0, None)
        cos_all.append(cos_t_i)
        y_all.append(gray[:, int(pix_idx)])
        src_all.append(src_type)
    cos_all = np.concatenate(cos_all)
    y_all   = np.concatenate(y_all)
    src_all = np.concatenate(src_all)

    palette = {'poly': 'C0', 'pan': 'C1', 'lls': 'C2'}
    bin_edges = np.linspace(0.0, 1.0, cos_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    for ax, apply_cos, label_y, panel in zip(
            axes, [False, True],
            ['raw GT (pan-equiv)', 'GT × cos(θ_i)'],
            ['LEFT — raw GT (no cos applied)', 'RIGHT — GT × cos(θ_i)']):
        # Track the largest binned-mean across sources so we can autoscale y
        # to the lobe trend (specular outliers would otherwise flatten the
        # binned-mean lines, which are the actual diagnostic).
        max_binned = 0.0
        for src in ['poly', 'pan', 'lls']:
            mask = src_all == src
            if not np.any(mask):
                continue
            x = cos_all[mask]
            y = (y_all[mask] * x) if apply_cos else y_all[mask]
            ax.scatter(x, y, s=8, alpha=0.15, color=palette[src],
                       label=f'{src}  (n={int(mask.sum())})')
            bin_idx = np.clip(np.digitize(x, bin_edges) - 1, 0, cos_bins - 1)
            means = np.full(cos_bins, np.nan)
            for b in range(cos_bins):
                sel = bin_idx == b
                if sel.sum() > 0:
                    means[b] = y[sel].mean()
            valid = ~np.isnan(means)
            ax.plot(bin_centers[valid], means[valid], '-o',
                    color=palette[src], linewidth=2.0, markersize=5,
                    label=f'{src} binned mean')
            if valid.any():
                max_binned = max(max_binned, float(np.nanmax(means[valid])))
        ax.set_xlabel('cos(θ_i)')
        ax.set_ylabel(label_y)
        ax.set_title(panel)
        ax.set_xlim(0.0, 1.0)
        # Headroom = 4× the largest binned-mean: keeps trend lines readable
        # while still showing the spread of individual scatter points.
        if max_binned > 0:
            ax.set_ylim(0.0, 4.0 * max_binned)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    sources = sorted(set(src_type.tolist()))
    fig.suptitle(
        f'mat{mat_id:04d}  ({len(pix_indices)} pts × {sources})  —  LLS cosine check\n'
        f'lls drops to 0 in LEFT  ⇒  cos already baked into LLS GT  '
        f'(use the existing _lls_monte_carlo, do NOT apply cos again)\n'
        f'lls stays high in LEFT, aligns with poly/pan in RIGHT  ⇒  cos NOT baked in  '
        f'(apply cos to BRDF inside _lls_monte_carlo)',
        fontsize=10)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'mat{mat_id:04d}_lls_cos_check.png')
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_aggregate_brdf_vs_costhetai_multisource(
        mat_id, mat_all, pix_indices, out_dir, apply_cos_wi=True,
        cos_bins=20):
    """Aggregate poly + pan + lls onto one figure.

    For each sampled pixel × view we compute cos(θ_i) (using the GT normal
    if available) and the pan-equivalent grayscale BRDF. Points are plotted
    as scatter coloured by source. A solid binned-mean line is drawn per
    source so we can see whether the three line up.
    """
    H = mat_all['H']
    W = mat_all['W']
    light_pos = mat_all['light_pos']
    src_type  = mat_all['source_type']
    gray      = mat_all['gray_pan_equiv']
    normals   = mat_all['normals']

    cos_all, y_all, src_all = [], [], []
    for pix_idx in pix_indices:
        r, c = pix_idx // W, pix_idx % W
        xyz = mat_all['xyz_map'][r, c]
        n_pt = (normals[pix_idx] if normals is not None
                else np.array([0.0, 0.0, 1.0], dtype=np.float32))
        wi = light_pos - xyz[None, :]
        wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
        n = n_pt / max(np.linalg.norm(n_pt), 1e-8)
        cos_t_i = np.clip(wi @ n, 0.0, None)
        y = gray[:, pix_idx]
        if apply_cos_wi:
            y = y * cos_t_i
        cos_all.append(cos_t_i)
        y_all.append(y)
        src_all.append(src_type)
    cos_all = np.concatenate(cos_all)
    y_all   = np.concatenate(y_all)
    src_all = np.concatenate(src_all)

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {'poly': 'C0', 'pan': 'C1', 'lls': 'C2'}
    bin_edges = np.linspace(0.0, 1.0, cos_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for src in ['poly', 'pan', 'lls']:
        mask = src_all == src
        if not np.any(mask):
            continue
        x = cos_all[mask]
        y = y_all[mask]
        ax.scatter(x, y, s=8, alpha=0.15, color=palette[src],
                   label=f'{src}  (n={mask.sum()})')
        # Binned mean line for this source.
        bin_idx = np.clip(np.digitize(x, bin_edges) - 1, 0, cos_bins - 1)
        means = np.full(cos_bins, np.nan)
        for b in range(cos_bins):
            sel = bin_idx == b
            if sel.sum() > 0:
                means[b] = y[sel].mean()
        valid = ~np.isnan(means)
        ax.plot(bin_centers[valid], means[valid], '-o',
                color=palette[src], linewidth=2.0, markersize=5,
                label=f'{src} binned mean')

    metric = 'BRDF × cos(θ_i)' if apply_cos_wi else 'BRDF magnitude (pan-equiv)'
    ax.set_xlabel('cos(θ_i)')
    ax.set_ylabel(metric)
    ax.set_title(f'mat{mat_id:04d}  ({len(pix_indices)} pts × poly+pan+lls)\n'
                 f'{metric} averaged in cos(θ_i) bins (per source)')
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    fname = os.path.join(out_dir,
        f"mat{mat_id:04d}_brdf_vs_costhetai_all_sources.png")
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_aggregate_brdf_vs_costhetai(mat_id, per_point_records, out_dir,
                                     apply_cos_wi=True):
    """Aggregate across ALL sampled points for one material.

    per_point_records: list of dicts each with keys
        pixel_row, pixel_col, cos_theta_i (K,), y_all (K,), il_ids (K,).

    Output: one PNG. Shows
      - per-point per-il_id mean as a scatter dot (alpha-blended)
      - bold line through the median across points per il_id
    No vertical errorbars.
    """
    if not per_point_records:
        return

    all_il_ids = sorted({int(il) for r in per_point_records
                                  for il in np.unique(r['il_ids']).tolist()})

    per_il = {il: {'cos': [], 'y': []} for il in all_il_ids}
    for r in per_point_records:
        for il in all_il_ids:
            mask = r['il_ids'] == il
            if not np.any(mask):
                continue
            per_il[il]['cos'].append(float(r['cos_theta_i'][mask].mean()))
            per_il[il]['y'].append(float(r['y_all'][mask].mean()))

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.tab10
    for ci, il in enumerate(all_il_ids):
        cos_arr = np.array(per_il[il]['cos'])
        y_arr = np.array(per_il[il]['y'])
        if cos_arr.size == 0:
            continue
        ax.scatter(cos_arr, y_arr, alpha=0.35, s=22,
                   color=cmap(ci), label=f'il{il:03d}  (n_pts={cos_arr.size})')

    metric = 'BRDF × cos(θ_i)' if apply_cos_wi else 'BRDF magnitude (‖RGB‖)'
    ax.set_xlabel('cos(θ_i)')
    ax.set_ylabel(metric)
    ax.set_title(f'mat{mat_id:04d}  ({len(per_point_records)} surface points)\n'
                 f'{metric} averaged over all wi azimuths and all wo')
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"mat{mat_id:04d}_brdf_vs_costhetai_all_points.png")
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_fix_wo_vary_wi_combined(mat_id, point_idx, pixel_row, pixel_col,
                                 xyz_point, gray_all, light_pos,
                                 cv_ids, il_ids, rot_azims, source_type,
                                 out_dir, apply_cos_wi=True, normal=None):
    """Same axes as plot_fix_wo_vary_wi but combines poly + pan LEDs.

    y is pan-equivalent grayscale BRDF (× cos(θ_i) when apply_cos_wi).
    cos(θ_i) uses the per-pixel GT normal when supplied.
    """
    wi_all = light_pos - xyz_point[None, :]
    wi_all /= np.maximum(np.linalg.norm(wi_all, axis=1, keepdims=True), 1e-8)
    if normal is not None:
        n = np.asarray(normal, dtype=np.float32).reshape(3)
        n_norm = np.linalg.norm(n)
        n = n / max(n_norm, 1e-8) if n_norm >= 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cos_theta_i = np.clip(wi_all @ n, 0.0, None)
    wi_elev = np.degrees(np.arctan2(wi_all[:, 2],
                                     np.sqrt(wi_all[:, 0]**2 + wi_all[:, 1]**2)))

    unique_rot = sorted(np.unique(rot_azims).tolist())
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(10, 6))
    for gi, rot in enumerate(unique_rot):
        rot_mask = rot_azims == rot
        # One point per il_id seen in this rotation, averaged over cv_ids.
        unique_il_in_rot = sorted(np.unique(il_ids[rot_mask]).tolist())
        x_vals, y_vals, src_used = [], [], []
        for il in unique_il_in_rot:
            cell = rot_mask & (il_ids == il)
            if not np.any(cell):
                continue
            y_pixel_views = gray_all[cell]              # (n_views,)
            if apply_cos_wi:
                y_pixel_views = y_pixel_views * cos_theta_i[cell]
            x_vals.append(float(np.mean(wi_elev[cell])))
            y_vals.append(float(np.mean(y_pixel_views)))
            src_used.append(set(source_type[cell].tolist()))
        if not x_vals:
            continue
        x_arr, y_arr = zip(*sorted(zip(x_vals, y_vals)))
        ax.plot(x_arr, y_arr, 'o-', markersize=5, color=cmap(gi),
                label=f"rot={rot:.0f}° (n_il={len(x_arr)})")

    metric = 'BRDF × cos(θ_i)' if apply_cos_wi else 'BRDF magnitude (pan-equiv)'
    ax.set_xlabel("wi elevation (°)")
    ax.set_ylabel(metric)
    n_poly = int((source_type == 'poly').sum())
    n_pan  = int((source_type == 'pan').sum())
    n_lls  = int((source_type == 'lls').sum())
    ax.set_title(f"mat{mat_id:04d}  point({pixel_row},{pixel_col})  "
                 f"Fix wo, vary wi  [poly={n_poly}, pan={n_pan}, lls={n_lls}]")
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir,
        f"mat{mat_id:04d}_pt{pixel_row:04d}x{pixel_col:04d}_fixWo_elev_polypan.png")
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_3d_lobe_fix_wi(mat_id, point_idx, pixel_row, pixel_col,
                        xyz_point, rgbs_all, light_pos, cam_pos,
                        rot_azims,
                        out_dir):
    """3D lobe: scatter all views coloured by rotation azimuth.

    x = wo elevation, y = wo azimuth, z = BRDF magnitude.
    Each rotation (turntable azimuth) gets its own colour so the lobe
    shape at different azimuthal cuts is visible simultaneously.
    """
    wo_all = cam_pos - xyz_point[None, :]
    wo_all /= np.maximum(np.linalg.norm(wo_all, axis=1, keepdims=True), 1e-8)

    wo_elev, wo_azim = spherical_coords(wo_all)

    unique_rot = sorted(np.unique(rot_azims).tolist())
    cmap = plt.cm.tab10
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for gi, rot in enumerate(unique_rot):
        mask = rot_azims == rot
        brdf_mag = np.linalg.norm(rgbs_all[mask], axis=-1)
        ax.scatter(wo_elev[mask], wo_azim[mask], brdf_mag,
                   color=cmap(gi), s=30, label=f"rot={rot:.0f}°")

    ax.set_xlabel("wo elevation (°)")
    ax.set_ylabel("wo azimuth (°)")
    ax.set_zlabel("BRDF magnitude (‖RGB‖)")
    ax.set_title(f"mat{mat_id:04d}  point({pixel_row},{pixel_col})  "
                 f"All views → 3D BRDF lobe (coloured by rotation)")
    ax.legend(fontsize=7, loc='best')
    plt.tight_layout()
    fname = os.path.join(out_dir,
        f"mat{mat_id:04d}_pt{pixel_row:04d}x{pixel_col:04d}_fixWi_3d.png")
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_3d_lobe_fix_wo(mat_id, point_idx, pixel_row, pixel_col,
                        xyz_point, rgbs_all, light_pos, cam_pos,
                        rot_azims,
                        out_dir, apply_cos_wi=True):
    """3D lobe: scatter all views coloured by rotation azimuth.

    x = wi elevation, y = wi azimuth, z = BRDF magnitude × cos(θ_i)
    when ``apply_cos_wi`` is True; raw BRDF magnitude otherwise.
    Each rotation (turntable azimuth) gets its own colour.
    """
    wi_all = light_pos - xyz_point[None, :]
    wi_all /= np.maximum(np.linalg.norm(wi_all, axis=1, keepdims=True), 1e-8)

    normal = np.array([0.0, 0.0, 1.0])
    cos_theta_i = np.clip(wi_all @ normal, 0, None)

    wi_elev, wi_azim = spherical_coords(wi_all)

    unique_rot = sorted(np.unique(rot_azims).tolist())
    cmap = plt.cm.tab10
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for gi, rot in enumerate(unique_rot):
        mask = rot_azims == rot
        brdf_mag = np.linalg.norm(rgbs_all[mask], axis=-1)
        z_vals = brdf_mag * cos_theta_i[mask] if apply_cos_wi else brdf_mag
        ax.scatter(wi_elev[mask], wi_azim[mask], z_vals,
                   color=cmap(gi), s=30, label=f"rot={rot:.0f}°")

    ax.set_xlabel("wi elevation (°)")
    ax.set_ylabel("wi azimuth (°)")
    if apply_cos_wi:
        ax.set_zlabel("BRDF × cos(θ_i)  (‖RGB‖ × cos)")
        title_metric = "BRDF×cos"
    else:
        ax.set_zlabel("BRDF magnitude (‖RGB‖)")
        title_metric = "BRDF"
    ax.set_title(f"mat{mat_id:04d}  point({pixel_row},{pixel_col})  "
                 f"All views → 3D {title_metric} lobe (coloured by rotation)")
    ax.legend(fontsize=7, loc='best')
    plt.tight_layout()
    fname = os.path.join(out_dir,
        f"mat{mat_id:04d}_pt{pixel_row:04d}x{pixel_col:04d}_fixWo_3d.png")
    plt.savefig(fname, dpi=150)
    plt.close()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize Bonn GT BRDF lobes")
    parser.add_argument('--root_folder', default=ROOT_FOLDER,
                        help='Bonn dataset root (with matXXXX_*.exr files)')
    parser.add_argument('--svfresnel_dir', default=SVFRESNEL_DIR,
                        help='svfresnel root with per-pixel GT normal maps. '
                             'Pass empty string to fall back to assuming +Z normal.')
    parser.add_argument('--output_dir', default=OUTPUT_DIR,
                        help='Output directory for lobe PNGs')
    parser.add_argument('--material_ids', type=int, nargs='+', default=MATERIAL_IDS,
                        help='Material IDs to visualize')
    parser.add_argument('--num_points', type=int, default=NUM_POINTS,
                        help='Number of surface points (uniform grid)')
    parser.add_argument('--apply_cos_wi', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Multiply vary-wi GT lobe by cos(θ_i). '
                             'Pass --no-apply_cos_wi to plot raw BRDF.')
    parser.add_argument('--aggregate_only', action='store_true',
                        help='Skip per-point Cartesian/3D/polar plots; '
                             'emit only the aggregate brdf_vs_costhetai figure.')
    parser.add_argument('--all_sources', action='store_true',
                        help='Use poly + pan + lls (pan-equivalent grayscale) '
                             'instead of poly RGB only. Implies --aggregate_only.')
    args = parser.parse_args()
    if args.all_sources:
        args.aggregate_only = True

    os.makedirs(args.output_dir, exist_ok=True)

    for mat_id in args.material_ids:
        print(f"\n{'='*60}")
        print(f"Loading material {mat_id:04d} from {args.root_folder}")
        print(f"{'='*60}")

        svfresnel = args.svfresnel_dir if args.svfresnel_dir else None
        if args.all_sources:
            mat_all = load_material_all_sources(
                args.root_folder, mat_id, svfresnel_dir=svfresnel)
            H, W = mat_all['H'], mat_all['W']
            K = mat_all['gray_pan_equiv'].shape[0]
            sources = sorted(set(mat_all['source_type'].tolist()))
            print(f"  H={H}, W={W}, K={K} views, sources={sources}, "
                  f"normals={'GT' if mat_all['normals'] is not None else '+Z fallback'}")
            polypan_dir = os.path.join(args.output_dir, "fixWo_polypan",
                                        f"mat{mat_id:04d}")
            cos_check_dir = os.path.join(args.output_dir, "lls_cos_check")
            align_dir = os.path.join(args.output_dir, "lobe_alignment",
                                     f"mat{mat_id:04d}")
            strict_dir = os.path.join(args.output_dir, "lobe_strict",
                                      f"mat{mat_id:04d}")
            phibin_dir = os.path.join(args.output_dir, "lobe_phi_binned",
                                      f"mat{mat_id:04d}")
            os.makedirs(polypan_dir, exist_ok=True)
            os.makedirs(cos_check_dir, exist_ok=True)
            os.makedirs(align_dir, exist_ok=True)
            os.makedirs(strict_dir, exist_ok=True)
            os.makedirs(phibin_dir, exist_ok=True)
            pix_indices = uniform_sample_point_indices(H, W, args.num_points)
            print(f"  Sampling {len(pix_indices)} points (uniform grid)")
            for pi, pix_idx in enumerate(pix_indices):
                pixel_row = int(pix_idx) // W
                pixel_col = int(pix_idx) % W
                xyz_pt = mat_all['xyz_map'][pixel_row, pixel_col]
                normal_pt = (mat_all['normals'][pix_idx]
                             if mat_all['normals'] is not None else None)
                gray_views = mat_all['gray_pan_equiv'][:, int(pix_idx)]
                plot_fix_wo_vary_wi_combined(
                    mat_id, pi, pixel_row, pixel_col,
                    xyz_pt, gray_views,
                    mat_all['light_pos'],
                    mat_all['cv_ids'], mat_all['il_ids'], mat_all['rot_azims'],
                    mat_all['source_type'],
                    polypan_dir, apply_cos_wi=args.apply_cos_wi,
                    normal=normal_pt)
            print(f"  Saved fixWo poly+pan(+lls) plots to {polypan_dir}")

            # LLS cosine-check (one figure per material, all sources pooled).
            plot_aggregate_with_without_cos(
                mat_id, mat_all, pix_indices, cos_check_dir)
            print(f"  Saved LLS cosine check to {cos_check_dir}")

            # Per-pixel poly/pan/lls overlay — diagnoses radiometric alignment.
            plot_lobe_alignment_per_pixel(
                mat_id, mat_all, pix_indices, align_dir)
            print(f"  Saved per-pixel lobe alignment to {align_dir}")

            # Strict slices — each dot is one view, no aggregation.
            plot_strict_slices_per_pixel(
                mat_id, mat_all, pix_indices, strict_dir)
            print(f"  Saved strict per-view slices to {strict_dir}")

            # (φ_i, θ_o)-binned θ_i sweep — controls for anisotropy AND view.
            plot_phi_binned_slices_per_pixel(
                mat_id, mat_all, pix_indices, phibin_dir)
            print(f"  Saved φ-binned θ_i sweeps to {phibin_dir}")
            continue

        mat = load_material(args.root_folder, mat_id, svfresnel_dir=svfresnel)
        H, W = mat['H'], mat['W']
        K = mat['rgbs'].shape[0]
        print(f"  H={H}, W={W}, K={K} views, "
              f"normals={'GT' if mat['normals'] is not None else '+Z fallback'}")

        # Uniform grid sampling of surface points
        pix_indices = uniform_sample_point_indices(H, W, args.num_points)
        print(f"  Sampling {len(pix_indices)} points (uniform grid)")

        mat_out_dir = os.path.join(args.output_dir, f"mat{mat_id:04d}")
        os.makedirs(mat_out_dir, exist_ok=True)
        cos_out_dir = os.path.join(args.output_dir, "brdf_vs_costhetai", f"mat{mat_id:04d}")
        os.makedirs(cos_out_dir, exist_ok=True)

        per_point_records = []

        for pi, pix_idx in enumerate(pix_indices):
            pixel_row = pix_idx // W
            pixel_col = pix_idx % W
            xyz_point = mat['xyz_map'][pixel_row, pixel_col]  # (3,)
            normal_point = (mat['normals'][pix_idx]
                            if mat['normals'] is not None else None)

            # BRDF values (RGB) for this pixel across all K views
            rgbs_point = mat['rgbs'][:, pix_idx, :]  # (K, 3)

            if not args.aggregate_only:
                print(f"  Point {pi+1}/{len(pix_indices)}: "
                      f"pixel=({pixel_row},{pixel_col})  xyz={xyz_point}")

                # 1) Fix wi (avg over il), vary wo – 2D elevation + azimuth slices
                plot_fix_wi_vary_wo(
                    mat_id, pi, pixel_row, pixel_col,
                    xyz_point, rgbs_point,
                    mat['light_pos'], mat['cam_pos'],
                    mat['cv_ids'], mat['il_ids'], mat['rot_azims'],
                    mat_out_dir)

                # 2) Fix wo (avg over cv), vary wi – 2D elevation + azimuth slices
                plot_fix_wo_vary_wi(
                    mat_id, pi, pixel_row, pixel_col,
                    xyz_point, rgbs_point,
                    mat['light_pos'], mat['cam_pos'],
                    mat['cv_ids'], mat['il_ids'], mat['rot_azims'],
                    mat_out_dir, apply_cos_wi=args.apply_cos_wi)

                # 3) All views – 3D BRDF lobe coloured by rotation azimuth
                plot_3d_lobe_fix_wi(
                    mat_id, pi, pixel_row, pixel_col,
                    xyz_point, rgbs_point,
                    mat['light_pos'], mat['cam_pos'],
                    mat['rot_azims'],
                    mat_out_dir)

                # 4) All views – 3D BRDF×cos lobe coloured by rotation azimuth
                plot_3d_lobe_fix_wo(
                    mat_id, pi, pixel_row, pixel_col,
                    xyz_point, rgbs_point,
                    mat['light_pos'], mat['cam_pos'],
                    mat['rot_azims'],
                    mat_out_dir, apply_cos_wi=args.apply_cos_wi)

                # 5) UBO-style polar plot: fix wi, vary wo (one curve per il_id)
                plot_polar_fix_wi_vary_wo(
                    mat_id, pi, pixel_row, pixel_col,
                    xyz_point, rgbs_point,
                    mat['light_pos'], mat['cam_pos'],
                    mat['il_ids'],
                    mat_out_dir)

                # 6) UBO-style polar plot: fix wo, vary wi (one curve per cv_id)
                plot_polar_fix_wo_vary_wi(
                    mat_id, pi, pixel_row, pixel_col,
                    xyz_point, rgbs_point,
                    mat['light_pos'], mat['cam_pos'],
                    mat['cv_ids'],
                    mat_out_dir, apply_cos_wi=args.apply_cos_wi)

                # 7) BRDF vs cos(θ_i): averaged over all wi azimuths and all wo
                #    (saved to a separate sub-folder for easy comparison)
                plot_brdf_vs_costhetai(
                    mat_id, pi, pixel_row, pixel_col,
                    xyz_point, rgbs_point,
                    mat['light_pos'], mat['cam_pos'],
                    mat['il_ids'],
                    cos_out_dir, apply_cos_wi=args.apply_cos_wi,
                    normal=normal_point)

            # Collect per-point data for the across-points aggregate plot.
            cos_view, y_view = compute_brdf_vs_costhetai_data(
                xyz_point, rgbs_point, mat['light_pos'], mat['il_ids'],
                apply_cos_wi=args.apply_cos_wi, normal=normal_point)
            per_point_records.append({
                'pixel_row': pixel_row,
                'pixel_col': pixel_col,
                'cos_theta_i': cos_view,
                'y_all': y_view,
                'il_ids': mat['il_ids'],
            })

        # Across-points aggregate (one figure per material)
        plot_aggregate_brdf_vs_costhetai(
            mat_id, per_point_records, cos_out_dir,
            apply_cos_wi=args.apply_cos_wi)

        print(f"  Saved plots to {mat_out_dir}")

    print(f"\nDone. All outputs in {args.output_dir}")


if __name__ == '__main__':
    main()
