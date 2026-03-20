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
from utils.dataset.bonn import _read_exr, _read_xyz_map, _parse_poly_channels

# ======================================================================
# Hard-coded parameters  (edit these)
# ======================================================================
ROOT_FOLDER  = "/media/raid/cloth/Bonn_train"
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


def load_material(root_folder, mat_id):
    """Load one material's xyz map, poly images, and calibration.

    Returns dict with keys:
        H, W, xyz_map (H,W,3), rgbs (K,V,3) float32,
        light_pos (K,3), cam_pos (K,3)
    """
    prefix = Path(root_folder) / f'mat{mat_id:04d}'
    calib = load_calibration(root_folder, mat_id)

    # xyz
    xyz_map, H, W = _read_xyz_map(f'{prefix}_xyz_rot000.exr')
    n_pixels = H * W

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
        'rgbs':      poly_rgbs.astype(np.float32),       # (K, V, 3)
        'light_pos': light_pos,                          # (K, 3)
        'cam_pos':   cam_pos,                            # (K, 3)
        'cv_ids':    cv_ids,                             # (K,) int  camera index 1..4
        'il_ids':    il_ids,                             # (K,) int  LED index e.g. 26–32
        'rot_azims': rot_azims,                          # (K,) float turntable azimuth °
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
                        out_dir):
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
            y = float(np.mean(brdf_mag * cos_theta_i[mask]))
            x_vals.append(float(np.mean(wi_elev[mask])))   # actual angle
            y_vals.append(y)
        x_vals, y_vals = zip(*sorted(zip(x_vals, y_vals))) if x_vals else ([], [])
        ax.plot(x_vals, y_vals, 'o-', markersize=5, color=cmap(gi),
                label=f"rot={rot:.0f}°  (wo_azim≈{avg_wo_azim_by_rot[rot]:.0f}°)")

    ax.set_xlabel("wi elevation (°)")
    ax.set_ylabel("BRDF × cos(θ_i)  (‖RGB‖ × cos)")
    ax.set_title(f"mat{mat_id:04d}  point({pixel_row},{pixel_col})  "
                 f"Fix wo (wo_elev≈{mean_wo_elev:.0f}°, avg over cv) → BRDF×cos vs wi elevation")
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir,
        f"mat{mat_id:04d}_pt{pixel_row:04d}x{pixel_col:04d}_fixWo_elev.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    # Azimuth slice omitted: see docstring for reason.


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
                        out_dir):
    """3D lobe: scatter all views coloured by rotation azimuth.

    x = wi elevation, y = wi azimuth, z = BRDF magnitude × cos(θ_i).
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
        brdf_cos = np.linalg.norm(rgbs_all[mask], axis=-1) * cos_theta_i[mask]
        ax.scatter(wi_elev[mask], wi_azim[mask], brdf_cos,
                   color=cmap(gi), s=30, label=f"rot={rot:.0f}°")

    ax.set_xlabel("wi elevation (°)")
    ax.set_ylabel("wi azimuth (°)")
    ax.set_zlabel("BRDF × cos(θ_i)  (‖RGB‖ × cos)")
    ax.set_title(f"mat{mat_id:04d}  point({pixel_row},{pixel_col})  "
                 f"All views → 3D BRDF×cos lobe (coloured by rotation)")
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for mat_id in MATERIAL_IDS:
        print(f"\n{'='*60}")
        print(f"Loading material {mat_id:04d} from {ROOT_FOLDER}")
        print(f"{'='*60}")

        mat = load_material(ROOT_FOLDER, mat_id)
        H, W = mat['H'], mat['W']
        K = mat['rgbs'].shape[0]
        print(f"  H={H}, W={W}, K={K} views")

        # Uniform grid sampling of surface points
        pix_indices = uniform_sample_point_indices(H, W, NUM_POINTS)
        print(f"  Sampling {len(pix_indices)} points (uniform grid)")

        mat_out_dir = os.path.join(OUTPUT_DIR, f"mat{mat_id:04d}")
        os.makedirs(mat_out_dir, exist_ok=True)

        for pi, pix_idx in enumerate(pix_indices):
            pixel_row = pix_idx // W
            pixel_col = pix_idx % W
            xyz_point = mat['xyz_map'][pixel_row, pixel_col]  # (3,)

            # BRDF values (RGB) for this pixel across all K views
            rgbs_point = mat['rgbs'][:, pix_idx, :]  # (K, 3)

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
                mat_out_dir)

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
                mat_out_dir)

        print(f"  Saved plots to {mat_out_dir}")

    print(f"\nDone. All outputs in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
