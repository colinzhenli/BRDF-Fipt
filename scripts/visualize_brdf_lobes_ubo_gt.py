#!/usr/bin/env python3
"""Visualize ground-truth BRDF lobes from UBO2014 BTF data.

Produces polar plots that are directly comparable to the prediction lobes
saved by ``Stage2Trainer_UBO.visualize_brdf_lobe``.

Two plot types per point:
  1) Fix wi (in phi=0 plane), vary wo  → BRDF(wo) polar plot
  2) Fix wo (in phi=0 plane), vary wi  → BRDF(wi)×cos(θ_i) polar plot

Direction convention (matching the prediction code):
  - wi = (sin θ_i, 0, cos θ_i)           on the +x side
  - θ_o > 0 → wo = (-sin θ_o, 0, cos θ_o)  on the -x side (specular side)
  - θ_o < 0 → wo = ( sin|θ_o|, 0, cos|θ_o|) on the +x side (retro side)
  - θ_o = 0 → wo along normal (+z)

UBO dome → signed-theta mapping (phi=0 plane slice):
  - view at (θ_v, φ_v=0°)   → wo on +x side → signed θ_o = -θ_v  (radians)
  - view at (θ_v, φ_v=180°) → wo on -x side → signed θ_o = +θ_v  (radians)

Usage:
    python scripts/visualize_brdf_lobes_ubo_gt.py [--btf_path PATH] [--pred_lobe_dir DIR] [--output_dir DIR]
"""

import argparse
import os
import re
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ======================================================================
# Defaults
# ======================================================================
DEFAULT_BTF_PATH = "/mnt/data/colin/colin/Bonn_BTF/carpet07_W400xH400_L151xV151.btf"
DEFAULT_PRED_LOBE_DIR = (
    "/media/raid/cloth/output/BRDF/Bonn-Theia2/"
    "Stage-2_UBO_carpet07_from-Bonn_Logrel-Learnable-factor_run_2/images/brdf_lobes"
)
DEFAULT_OUTPUT_DIR = (
    "/media/raid/cloth/output/BRDF/Bonn-Theia2/"
    "Stage-2_UBO_carpet07_from-Bonn_Logrel-Learnable-factor_run_2/images/brdf_lobes_gt"
)


# ======================================================================
# Helpers
# ======================================================================

def parse_point_ids_from_folder(lobe_dir):
    """Extract sorted unique point IDs from prediction lobe filenames.

    Filenames look like: polar_brdf_vary_wo_pt_2409.png
    """
    ids = set()
    for fn in os.listdir(lobe_dir):
        m = re.search(r'pt_(\d+)\.png$', fn)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def sph2cart(theta_deg, phi_deg):
    """(theta, phi) in degrees → unit Cartesian (x, y, z).

    Same convention as UBO dome / utils.dataset.ubo._sph2cart:
      theta = polar angle from +Z, phi = azimuth from +X toward +Y.
    """
    theta = np.deg2rad(np.float64(theta_deg))
    phi = np.deg2rad(np.float64(phi_deg))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z], dtype=np.float64)


def get_pixel_brdf(btf, theta_l, phi_l, theta_v, phi_v, row, col):
    """Get RGB BRDF value for one pixel at one angle combo.

    Returns (3,) float32, RGB order.
    """
    img = btf.angles_to_image(theta_l, phi_l, theta_v, phi_v)  # BGR
    rgb = img[row, col, ::-1].copy().astype(np.float32)
    np.clip(rgb, 0.0, None, out=rgb)
    return rgb


def find_inplane_directions(all_dirs, phi_tol=0.5):
    """Find dome directions near the phi=0 plane.

    Returns dict mapping signed_theta_rad → (theta_deg, phi_deg).
      phi ≈ 0°   → positive x side → signed theta = -theta_rad
      phi ≈ 180° → negative x side → signed theta = +theta_rad
      theta = 0  → signed theta = 0 (only appears at phi=0)

    Parameters
    ----------
    all_dirs : list of (theta_deg, phi_deg)
    phi_tol : float
        Tolerance in degrees for phi=0 and phi=180 selection.
    """
    result = {}
    for theta_d, phi_d in all_dirs:
        theta_rad = np.deg2rad(theta_d)
        if theta_d == 0.0:
            # Zenith: signed theta = 0 regardless of phi
            result[0.0] = (theta_d, phi_d)
        elif abs(phi_d) <= phi_tol or abs(phi_d - 360.0) <= phi_tol:
            # phi ≈ 0 → +x side → signed theta = -theta_rad
            result[-theta_rad] = (theta_d, phi_d)
        elif abs(phi_d - 180.0) <= phi_tol:
            # phi ≈ 180 → -x side → signed theta = +theta_rad
            result[theta_rad] = (theta_d, phi_d)
    return result


def find_nearest_dome_dir(target_theta_deg, target_phi_deg, all_dirs):
    """Find the dome direction closest to (target_theta, target_phi) in
    Cartesian distance.  Returns (theta_deg, phi_deg) of the best match.
    """
    target = sph2cart(target_theta_deg, target_phi_deg)
    best, best_dist = None, np.inf
    for theta_d, phi_d in all_dirs:
        v = sph2cart(theta_d, phi_d)
        d = np.linalg.norm(v - target)
        if d < best_dist:
            best_dist = d
            best = (theta_d, phi_d)
    return best


def compute_signed_theta(swept_dirs_cart, fixed_dir_cart):
    """Compute the signed in-plane angle for each swept direction.

    Projects every swept direction onto the incidence plane (defined by
    the surface normal n=(0,0,1) and the fixed direction), then returns
    the signed polar angle from n.

      positive → specular side (opposite tangential hemisphere from fixed)
      negative → retro side   (same tangential hemisphere as fixed)
      zero     → along normal

    Parameters
    ----------
    swept_dirs_cart : (N, 3) array of Cartesian unit vectors
    fixed_dir_cart  : (3,)   Cartesian unit vector of the fixed direction

    Returns
    -------
    signed_theta : (N,) array in radians, range [-pi/2, pi/2]
    """
    # Forward direction in tangent plane: toward specular side
    fi_t = np.array([fixed_dir_cart[0], fixed_dir_cart[1], 0.0])
    fi_t_norm = np.linalg.norm(fi_t)
    if fi_t_norm < 1e-8:
        # Fixed dir along normal → fall back to -x as forward
        forward = np.array([-1.0, 0.0, 0.0])
    else:
        forward = -fi_t / fi_t_norm

    # In-plane component of each swept direction
    swept_t = swept_dirs_cart[:, :2]  # (N, 2) — xy part
    in_plane = swept_t[:, 0] * forward[0] + swept_t[:, 1] * forward[1]  # dot
    signed_theta = np.arctan2(in_plane, swept_dirs_cart[:, 2])
    return signed_theta


# Prediction colors: matplotlib default prop_cycle (C0, C1, C2, ...)
PRED_COLORS = [f'C{i}' for i in range(10)]


# ======================================================================
# Plotting
# ======================================================================

def _collect_inplane_views(all_view_dirs, phi_fix):
    """Collect view directions in the incidence plane of a fixed direction at phi_fix.

    Returns list of (signed_theta_rad, theta_deg, phi_deg) sorted by angle.
    In-plane means phi_v ≈ phi_fix (retro side, signed = -theta)
    or phi_v ≈ phi_fix+180 (specular side, signed = +theta).
    """
    results = []
    phi_opp = (phi_fix + 180.0) % 360.0
    for theta_d, phi_d in all_view_dirs:
        theta_rad = np.deg2rad(theta_d)
        if theta_d == 0.0:
            results.append((0.0, theta_d, phi_d))
            continue
        dphi_same = min(abs(phi_d - phi_fix), 360.0 - abs(phi_d - phi_fix))
        if dphi_same < 1.0:
            results.append((-theta_rad, theta_d, phi_d))
            continue
        dphi_opp = min(abs(phi_d - phi_opp), 360.0 - abs(phi_d - phi_opp))
        if dphi_opp < 1.0:
            results.append((theta_rad, theta_d, phi_d))
    results.sort(key=lambda t: t[0])
    return results


def plot_vary_wo_gt(btf, point_id, H, W, all_light_dirs, all_view_dirs, output_dir):
    """Fix wi, vary wo in the incidence plane only — GT polar plot.

    Matches the prediction exactly: both wi and wo stay in the same
    azimuthal plane (y=0 in the prediction). Only wo elevation varies.

    Uses the same θ_i values as the prediction: [15°, 30°, 45°, 60°, 75°].
    """
    row, col = point_id // W, point_id % W
    theta_i_values = [15.0, 30.0, 45.0, 60.0, 75.0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    for ci, theta_i_deg in enumerate(theta_i_values):
        tl, pl = find_nearest_dome_dir(theta_i_deg, 0.0, all_light_dirs)
        # Only view dirs in the same azimuthal plane as the fixed light
        inplane = _collect_inplane_views(all_view_dirs, pl)

        x_vals, y_vals = [], []
        for signed_theta, tv, pv in inplane:
            rgb = get_pixel_brdf(btf, tl, pl, tv, pv, row, col)
            x_vals.append(signed_theta)
            y_vals.append(rgb.mean())

        ax.plot(x_vals, y_vals, 'o-', markersize=4, color=PRED_COLORS[ci],
                label=f'θ_i={theta_i_deg}°')
        ax.axvline(x=np.deg2rad(theta_i_deg), color=PRED_COLORS[ci],
                   linestyle='--', alpha=0.5)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title(f'GT BRDF Polar Plot (vary wo) - Point {point_id}\n'
                 f'(Fixed wi, vary wo; 0°=normal, dashed=specular direction)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             f'polar_brdf_vary_wo_pt_{point_id}.png'), dpi=150)
    plt.close()


def plot_vary_wi_gt(btf, point_id, H, W, all_light_dirs, all_view_dirs, output_dir):
    """Fix wo, vary wi in the incidence plane only — GT polar plot.

    Uses the same θ_o values as the prediction: [15°, 30°, 45°, 60°].
    Plots BRDF × cos(θ_i).
    """
    row, col = point_id // W, point_id % W
    theta_o_values = [15.0, 30.0, 45.0, 60.0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    for ci, theta_o_deg in enumerate(theta_o_values):
        tv, pv = find_nearest_dome_dir(theta_o_deg, 0.0, all_view_dirs)
        # Only light dirs in the same azimuthal plane as the fixed view
        inplane = _collect_inplane_views(all_light_dirs, pv)

        x_vals, y_vals = [], []
        for signed_theta, tl, pl in inplane:
            rgb = get_pixel_brdf(btf, tl, pl, tv, pv, row, col)
            cos_theta_i = max(0.0, np.cos(abs(signed_theta)))
            x_vals.append(signed_theta)
            y_vals.append(rgb.mean() * cos_theta_i)

        ax.plot(x_vals, y_vals, 'o-', markersize=4, color=PRED_COLORS[ci],
                label=f'θ_o={theta_o_deg}°')
        ax.axvline(x=np.deg2rad(theta_o_deg), color=PRED_COLORS[ci],
                   linestyle='--', alpha=0.5)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title(f'GT BRDF × cos(θ_i) Polar Plot - Point {point_id}\n'
                 f'(Fixed wo, vary wi; 0°=normal, dashed=specular direction)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             f'polar_brdf_vary_wi_pt_{point_id}.png'), dpi=150)
    plt.close()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize GT BRDF lobes from UBO BTF data")
    parser.add_argument('--btf_path', default=DEFAULT_BTF_PATH,
                        help='Path to .btf file')
    parser.add_argument('--pred_lobe_dir', default=DEFAULT_PRED_LOBE_DIR,
                        help='Directory containing prediction lobe PNGs (to parse point IDs)')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for GT lobe plots')
    parser.add_argument('--point_ids', type=int, nargs='*', default=None,
                        help='Explicit point IDs (overrides --pred_lobe_dir parsing)')
    args = parser.parse_args()

    # ---- Resolve point IDs ----
    if args.point_ids:
        point_ids = sorted(args.point_ids)
    else:
        print(f"Parsing point IDs from: {args.pred_lobe_dir}")
        point_ids = parse_point_ids_from_folder(args.pred_lobe_dir)
    print(f"Point IDs ({len(point_ids)}): {point_ids}")

    # ---- Load BTF ----
    from btf_extractor import Ubo2014
    print(f"\nLoading BTF: {args.btf_path}")
    btf = Ubo2014(args.btf_path)
    H, W, _ = btf.img_shape
    n_pixels = H * W
    print(f"  Image shape: {H}×{W}  ({n_pixels:,} pixels)")

    # Validate point IDs
    for pid in point_ids:
        assert 0 <= pid < n_pixels, f"point_id {pid} out of range [0, {n_pixels})"

    # ---- Enumerate dome directions ----
    all_angles = sorted(btf.angles_set)
    all_light_dirs = sorted(set((a[0], a[1]) for a in all_angles))
    all_view_dirs = sorted(set((a[2], a[3]) for a in all_angles))
    print(f"  {len(all_light_dirs)} light dirs, {len(all_view_dirs)} view dirs, "
          f"{len(all_angles)} total combos")

    # Show in-plane directions for reference
    light_inplane = find_inplane_directions(all_light_dirs)
    view_inplane = find_inplane_directions(all_view_dirs)
    print(f"  In-plane light dirs (signed θ rad): {sorted(light_inplane.keys())}")
    print(f"  In-plane view  dirs (signed θ rad): {sorted(view_inplane.keys())}")

    # ---- Generate plots ----
    os.makedirs(args.output_dir, exist_ok=True)

    for i, pid in enumerate(point_ids):
        row, col = pid // W, pid % W
        print(f"\n[{i+1}/{len(point_ids)}] Point {pid}  (row={row}, col={col})")

        plot_vary_wo_gt(btf, pid, H, W, all_light_dirs, all_view_dirs, args.output_dir)
        plot_vary_wi_gt(btf, pid, H, W, all_light_dirs, all_view_dirs, args.output_dir)

    print(f"\nDone. GT lobe plots saved to {args.output_dir}")
    print(f"Compare with prediction lobes in: {args.pred_lobe_dir}")


if __name__ == '__main__':
    main()
