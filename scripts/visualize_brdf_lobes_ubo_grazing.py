#!/usr/bin/env python3
"""Multi-material grazing-angle BRDF lobe visualization on UBO2014.

Convention (confirmed):
  - One polar plot per (wi_phi, wo_phi) slice.
  - x-axis = signed θ_i (radians), filled by sweeping wi over dome lights
    lying in the wi_phi plane.
  - y-axis = raw BRDF value (mean RGB) — NO cos(θ_i) applied so we can see
    directly whether the lobe falls to zero at grazing.
  - One curve per fixed θ_o value (wo = nearest dome view to (θ_o, wo_phi)).

Signed-θ_i convention (matches the existing UBO viz):
  - phi ≈ wi_phi      → retro side    → signed = -theta_rad
  - phi ≈ wi_phi+180  → specular side → signed = +theta_rad
  - theta=0 (zenith)  → signed = 0

UBO2014 caveats inspected first:
  - Max θ in the dome is 75° (no literal 90° samples).
  - Theta layers alternate between two phi grids offset by ~7.5–15°, so a
    strict in-plane match would skip ~half the layers. We use a wider
    phi_tol (default 10°) to capture both grids in each slice.
"""

import argparse
import os
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from btf_extractor import Ubo2014


DEFAULT_BTF_DIR = "/media/raid/cloth/BTF"
DEFAULT_OUT_DIR = "/media/raid/cloth/output/BRDF/visualization_debug/ubo_grazing_lobes"

PALETTE = [f'C{i}' for i in range(10)]


def sph2cart(theta_deg, phi_deg):
    t = np.deg2rad(np.float64(theta_deg))
    p = np.deg2rad(np.float64(phi_deg))
    return np.array([np.sin(t) * np.cos(p),
                     np.sin(t) * np.sin(p),
                     np.cos(t)], dtype=np.float64)


def find_nearest_dome_dir(target_theta_deg, target_phi_deg, all_dirs):
    target = sph2cart(target_theta_deg, target_phi_deg)
    best, best_dist = None, np.inf
    for theta_d, phi_d in all_dirs:
        v = sph2cart(theta_d, phi_d)
        d = np.linalg.norm(v - target)
        if d < best_dist:
            best_dist, best = d, (theta_d, phi_d)
    return best


def _collect_inplane_dirs(all_dirs, phi_fix, phi_tol_deg):
    """Dome directions whose azimuth lies within phi_tol of phi_fix (retro side,
    signed = -theta) or phi_fix+180 (specular side, signed = +theta).
    """
    results = []
    phi_opp = (phi_fix + 180.0) % 360.0
    for theta_d, phi_d in all_dirs:
        theta_rad = np.deg2rad(theta_d)
        if theta_d == 0.0:
            results.append((0.0, theta_d, phi_d))
            continue
        dphi_same = min(abs(phi_d - phi_fix), 360.0 - abs(phi_d - phi_fix))
        if dphi_same <= phi_tol_deg:
            results.append((-theta_rad, theta_d, phi_d))
            continue
        dphi_opp = min(abs(phi_d - phi_opp), 360.0 - abs(phi_d - phi_opp))
        if dphi_opp <= phi_tol_deg:
            results.append((theta_rad, theta_d, phi_d))
    results.sort(key=lambda t: t[0])
    return results


def plot_one_slice(btf, mat_name, pid, row, col, H, W,
                   all_light_dirs, all_view_dirs,
                   wi_phi_deg, wo_phi_deg, output_dir,
                   theta_o_values, phi_tol_deg, apply_cos_wi):
    inplane = _collect_inplane_dirs(all_light_dirs, wi_phi_deg, phi_tol_deg)
    if not inplane:
        print(f"    wi_φ={wi_phi_deg:.0f}°: no in-plane lights, skipping")
        return

    fig, ax = plt.subplots(figsize=(8.5, 8), subplot_kw={'projection': 'polar'})

    any_curve = False
    for ci, theta_o_deg in enumerate(theta_o_values):
        tv, pv = find_nearest_dome_dir(theta_o_deg, wo_phi_deg, all_view_dirs)
        x_vals, y_vals = [], []
        for signed_theta, tl, pl in inplane:
            img = btf.angles_to_image(tl, pl, tv, pv)  # BGR
            rgb = img[row, col, ::-1].astype(np.float32)
            np.clip(rgb, 0.0, None, out=rgb)
            val = float(rgb.mean())
            if apply_cos_wi:
                val *= max(0.0, np.cos(abs(signed_theta)))
            x_vals.append(signed_theta)
            y_vals.append(val)
        ax.plot(x_vals, y_vals, 'o-', markersize=4,
                color=PALETTE[ci % len(PALETTE)],
                label=f'θ_o={theta_o_deg:.0f}°  (dome: θ={tv:.1f}°, φ={pv:.1f}°)')
        ax.axvline(x=np.deg2rad(theta_o_deg),
                   color=PALETTE[ci % len(PALETTE)], linestyle='--', alpha=0.35)
        any_curve = True

    if not any_curve:
        plt.close()
        return

    # Visual hints for the dome's grazing reach.
    ax.axvline(x=np.deg2rad(75), color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=-np.deg2rad(75), color='gray', linestyle=':', alpha=0.5)
    ax.text(np.deg2rad(75), ax.get_rmax() * 0.98, '75°', fontsize=8,
            color='gray', ha='left', va='top')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.legend(loc='upper right', bbox_to_anchor=(1.55, 1.0), fontsize=8)
    cos_note = ' × cos(θ_i)' if apply_cos_wi else ' (RAW, no cos)'
    ax.set_title(
        f'{mat_name}  pt {pid} (row={row}, col={col})\n'
        f'GT BRDF{cos_note}  —  wi_φ={wi_phi_deg:.0f}°, wo_φ={wo_phi_deg:.0f}°,  '
        f'phi_tol=±{phi_tol_deg:.0f}°\n'
        f'(vary wi θ, one curve per θ_o;  0°=normal, dashed=specular θ_o, dotted=dome 75° edge)'
    )
    plt.tight_layout()
    fname = (f'{mat_name}_pt{pid}_wiphi{int(round(wi_phi_deg)):03d}_'
             f'wophi{int(round(wo_phi_deg)):03d}'
             f'{"_cos" if apply_cos_wi else "_raw"}.png')
    plt.savefig(Path(output_dir) / fname, dpi=140)
    plt.close()


def run_one_material(mat_name, btf_dir, out_root, pixel, phi_configs,
                     theta_o_values, phi_tol_deg, apply_cos_wi):
    btf_path = Path(btf_dir) / f"{mat_name}_W400xH400_L151xV151.btf"
    if not btf_path.exists():
        print(f"  (skip) BTF not found: {btf_path}")
        return
    btf = Ubo2014(str(btf_path))
    H, W, _ = btf.img_shape
    row, col = pixel
    pid = row * W + col

    angles_set = btf.angles_set
    all_light = sorted(set((a[0], a[1]) for a in angles_set))
    all_view  = sorted(set((a[2], a[3]) for a in angles_set))

    mat_out = Path(out_root) / mat_name
    mat_out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {mat_name}  ({H}×{W}, pixel ({row},{col}), pid={pid}) ===")
    for wi_phi, wo_phi in phi_configs:
        print(f"  wi_φ={wi_phi:.0f}°  wo_φ={wo_phi:.0f}°")
        plot_one_slice(btf, mat_name, pid, row, col, H, W,
                       all_light, all_view, wi_phi, wo_phi,
                       mat_out, theta_o_values, phi_tol_deg, apply_cos_wi)


def parse_phi_configs(spec):
    out = []
    for tok in spec.split(';'):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split(',')
        out.append((float(a), float(b)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--materials', nargs='+',
                    default=['carpet07', 'felt01', 'felt09', 'fabric01', 'fabric08'],
                    help='UBO2014 material names (no _W400... suffix)')
    ap.add_argument('--btf_dir', default=DEFAULT_BTF_DIR)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR,
                    help='Output root. Per-material subfolders are created inside.')
    ap.add_argument('--pixel', type=int, nargs=2, default=[200, 200],
                    help='(row, col) of pixel to evaluate')
    ap.add_argument('--phi_configs', type=str,
                    default='0,0;0,90;0,180;90,0;90,90;90,180',
                    help="Semicolon-separated 'wi_phi,wo_phi' pairs (degrees).")
    ap.add_argument('--theta_o_values', type=float, nargs='+',
                    default=[15, 30, 45, 60, 75],
                    help='Fixed θ_o values for the per-curve sweep.')
    ap.add_argument('--phi_tol', type=float, default=10.0,
                    help='Tolerance (deg) around wi_phi / wi_phi+180 for in-plane '
                         'wi sweep. UBO2014 dome alternates phi grids offset by '
                         '7.5–15° between theta layers — set 10° to catch both.')
    ap.add_argument('--apply_cos_wi', action='store_true',
                    help='Multiply curves by cos(θ_i) (the trainer convention).')
    args = ap.parse_args()

    phi_configs = parse_phi_configs(args.phi_configs)

    print(f"Materials      : {args.materials}")
    print(f"Pixel          : ({args.pixel[0]}, {args.pixel[1]})")
    print(f"Phi configs    : {phi_configs}")
    print(f"θ_o values     : {args.theta_o_values}")
    print(f"phi_tol        : {args.phi_tol}°")
    print(f"apply_cos_wi   : {args.apply_cos_wi}")
    print(f"Output root    : {args.out_dir}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for mat in args.materials:
        run_one_material(
            mat_name=mat, btf_dir=args.btf_dir, out_root=args.out_dir,
            pixel=tuple(args.pixel), phi_configs=phi_configs,
            theta_o_values=args.theta_o_values, phi_tol_deg=args.phi_tol,
            apply_cos_wi=args.apply_cos_wi,
        )
    print(f"\nDone. Outputs in {args.out_dir}/<material>/")


if __name__ == '__main__':
    main()
