#!/usr/bin/env python3
"""Debug GT BRDF lobe visualization from UBO2014 BTF data — RAW (no cos applied).

Used to check whether the cosine falloff in incoming direction is already
baked into the BTF measurements. If it is, the trainer's
``BRDF × cos(θ_i)`` curve effectively double-applies cos(θ_i), and the raw
lobe plotted here is what we should be comparing against.

For each (wi_phi, wo_phi) configuration in the incidence plane:
  - Fix wo at multiple θ_o values (one curve each)
  - Sweep wi over the dome directions lying in the wi_phi plane
  - Plot signed θ_i  vs.  raw BRDF (no cos multiplication)

Filename pattern:
    polar_brdf_raw_vary_wi_pt_<pid>_wiphi<NNN>_wophi<NNN>.png

Refer to ``visualize_brdf_lobes_bonn_gt.py`` for the analogous multi-slice
debug viz on Bonn data.
"""

import argparse
import os
import re
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_BTF_PATH = "/media/raid/cloth/BTF/carpet07_W400xH400_L151xV151.btf"
DEFAULT_PRED_LOBE_DIR = (
    "/media/raid/cloth/output/BRDF/Bonn-Theia2/"
    "Stage-2_UBO_carpet07_from-Bonn_Logrel-Learnable-factor_run_2/images/brdf_lobes"
)
DEFAULT_OUTPUT_DIR = (
    "/media/raid/cloth/output/BRDF/Bonn-Theia2/"
    "Stage-2_UBO_carpet07_from-Bonn_Logrel-Learnable-factor_run_2/images/"
    "visualization_debug"
)

PALETTE = [f'C{i}' for i in range(10)]


def parse_point_ids_from_folder(lobe_dir):
    ids = set()
    for fn in os.listdir(lobe_dir):
        m = re.search(r'pt_(\d+)', fn)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def sph2cart(theta_deg, phi_deg):
    theta = np.deg2rad(np.float64(theta_deg))
    phi = np.deg2rad(np.float64(phi_deg))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z], dtype=np.float64)


def get_pixel_brdf(btf, theta_l, phi_l, theta_v, phi_v, row, col):
    img = btf.angles_to_image(theta_l, phi_l, theta_v, phi_v)  # BGR
    rgb = img[row, col, ::-1].copy().astype(np.float32)
    np.clip(rgb, 0.0, None, out=rgb)
    return rgb


def find_nearest_dome_dir(target_theta_deg, target_phi_deg, all_dirs):
    """Closest dome direction to (target_theta, target_phi) by Cartesian dist."""
    target = sph2cart(target_theta_deg, target_phi_deg)
    best, best_dist = None, np.inf
    for theta_d, phi_d in all_dirs:
        v = sph2cart(theta_d, phi_d)
        d = np.linalg.norm(v - target)
        if d < best_dist:
            best_dist = d
            best = (theta_d, phi_d)
    return best


def _collect_inplane_dirs(all_dirs, phi_fix, phi_tol=1.0):
    """Dome directions lying in the wi/wo incidence plane at azimuth phi_fix.

    Returns list of (signed_theta_rad, theta_deg, phi_deg) sorted by angle.
      phi ≈ phi_fix      → retro side    → signed = -theta_rad
      phi ≈ phi_fix+180  → specular side → signed = +theta_rad
      theta = 0          → zenith        → signed =  0
    """
    results = []
    phi_opp = (phi_fix + 180.0) % 360.0
    for theta_d, phi_d in all_dirs:
        theta_rad = np.deg2rad(theta_d)
        if theta_d == 0.0:
            results.append((0.0, theta_d, phi_d))
            continue
        dphi_same = min(abs(phi_d - phi_fix), 360.0 - abs(phi_d - phi_fix))
        if dphi_same < phi_tol:
            results.append((-theta_rad, theta_d, phi_d))
            continue
        dphi_opp = min(abs(phi_d - phi_opp), 360.0 - abs(phi_d - phi_opp))
        if dphi_opp < phi_tol:
            results.append((theta_rad, theta_d, phi_d))
    results.sort(key=lambda t: t[0])
    return results


def plot_vary_wi_raw(btf, point_id, H, W, all_light_dirs, all_view_dirs,
                     wi_phi_deg, wo_phi_deg, output_dir,
                     theta_o_values=(15.0, 30.0, 45.0, 60.0, 75.0),
                     phi_tol=1.0):
    """Sweep wi over the wi_phi plane at multiple fixed θ_o (in the wo_phi plane).

    Y axis is RAW BRDF — no cos(θ_i) multiplication. The signed θ_i convention
    matches the prediction plots:
      + side → wi on the opposite tangential half from wo_phi (specular side)
      - side → wi on the same tangential half as wo_phi
    """
    row, col = point_id // W, point_id % W

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    # Sweep wi within the wi_phi plane.
    inplane = _collect_inplane_dirs(all_light_dirs, wi_phi_deg, phi_tol=phi_tol)
    if not inplane:
        print(f"    (wi_φ={wi_phi_deg:.0f}°: no in-plane light dirs, skipping)")
        plt.close()
        return

    any_curve = False
    for ci, theta_o_deg in enumerate(theta_o_values):
        # Fix wo at the closest dome view to (theta_o, wo_phi).
        tv, pv = find_nearest_dome_dir(theta_o_deg, wo_phi_deg, all_view_dirs)

        x_vals, y_vals = [], []
        for signed_theta, tl, pl in inplane:
            rgb = get_pixel_brdf(btf, tl, pl, tv, pv, row, col)
            x_vals.append(signed_theta)
            y_vals.append(rgb.mean())

        ax.plot(x_vals, y_vals, 'o-', markersize=4,
                color=PALETTE[ci % len(PALETTE)],
                label=f'θ_o={theta_o_deg:.0f}° (dome: θ={tv:.1f}°, φ={pv:.1f}°)')
        ax.axvline(x=np.deg2rad(theta_o_deg),
                   color=PALETTE[ci % len(PALETTE)],
                   linestyle='--', alpha=0.4)
        any_curve = True

    if not any_curve:
        plt.close()
        return

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.0), fontsize=8)
    ax.set_title(
        f'GT BRDF (RAW, no cos) — pt {point_id} '
        f'(row={row}, col={col})\n'
        f'wi_φ={wi_phi_deg:.0f}°  wo_φ={wo_phi_deg:.0f}°,  '
        f'vary wi θ, fix wo θ\n'
        f'(0°=normal, dashed = specular θ_o)'
    )
    plt.tight_layout()
    fname = (f'polar_brdf_raw_vary_wi_pt_{point_id}'
             f'_wiphi{int(round(wi_phi_deg)):03d}'
             f'_wophi{int(round(wo_phi_deg)):03d}.png')
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()


def parse_phi_configs(spec):
    """Parse 'a,b;c,d;...' into [(a, b), (c, d), ...] of floats."""
    out = []
    for tok in spec.split(';'):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(',')
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid phi config '{tok}', expected 'wi_phi,wo_phi'")
        out.append((float(parts[0]), float(parts[1])))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Visualize raw GT BRDF lobes from UBO BTF data (no cos applied) "
                    "to check whether cosine falloff is baked into the BTF.")
    parser.add_argument('--btf_path', default=DEFAULT_BTF_PATH,
                        help='Path to .btf file')
    parser.add_argument('--pred_lobe_dir', default=DEFAULT_PRED_LOBE_DIR,
                        help='Directory containing prediction lobe PNGs (to parse point IDs)')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for raw GT lobe plots')
    parser.add_argument('--point_ids', type=int, nargs='*', default=None,
                        help='Explicit point IDs (overrides --pred_lobe_dir parsing)')
    parser.add_argument('--phi_configs', type=str,
                        default='0,0;0,90;0,180;90,90;90,180',
                        help="Semicolon-separated 'wi_phi,wo_phi' pairs in degrees. "
                             "Each pair produces one polar plot.")
    parser.add_argument('--theta_o_values', type=float, nargs='+',
                        default=[15.0, 30.0, 45.0, 60.0, 75.0],
                        help='Fixed θ_o values (one line each) per slice.')
    parser.add_argument('--phi_tol', type=float, default=1.0,
                        help='Tolerance (deg) for "in-plane" dome direction matching.')
    args = parser.parse_args()

    if args.point_ids:
        point_ids = sorted(args.point_ids)
    else:
        print(f"Parsing point IDs from: {args.pred_lobe_dir}")
        point_ids = parse_point_ids_from_folder(args.pred_lobe_dir)
    print(f"Point IDs ({len(point_ids)}): {point_ids}")

    phi_configs = parse_phi_configs(args.phi_configs)
    print(f"Phi configs ({len(phi_configs)}): {phi_configs}")
    print(f"θ_o values: {args.theta_o_values}")

    from btf_extractor import Ubo2014
    print(f"\nLoading BTF: {args.btf_path}")
    btf = Ubo2014(args.btf_path)
    H, W, _ = btf.img_shape
    n_pixels = H * W
    print(f"  Image shape: {H}×{W}  ({n_pixels:,} pixels)")

    for pid in point_ids:
        assert 0 <= pid < n_pixels, f"point_id {pid} out of range [0, {n_pixels})"

    all_angles = sorted(btf.angles_set)
    all_light_dirs = sorted(set((a[0], a[1]) for a in all_angles))
    all_view_dirs = sorted(set((a[2], a[3]) for a in all_angles))
    print(f"  {len(all_light_dirs)} light dirs, {len(all_view_dirs)} view dirs, "
          f"{len(all_angles)} total combos")

    os.makedirs(args.output_dir, exist_ok=True)

    for i, pid in enumerate(point_ids):
        row, col = pid // W, pid % W
        print(f"\n[{i+1}/{len(point_ids)}] Point {pid}  (row={row}, col={col})")
        for wi_phi, wo_phi in phi_configs:
            print(f"  wi_φ={wi_phi:.0f}°  wo_φ={wo_phi:.0f}°")
            plot_vary_wi_raw(
                btf, pid, H, W, all_light_dirs, all_view_dirs,
                wi_phi_deg=wi_phi, wo_phi_deg=wo_phi,
                output_dir=args.output_dir,
                theta_o_values=tuple(args.theta_o_values),
                phi_tol=args.phi_tol,
            )

    print(f"\nDone. Raw GT lobe plots saved to {args.output_dir}")


if __name__ == '__main__':
    main()
