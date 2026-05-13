#!/usr/bin/env python3
"""Verify the LLS quad corner ordering assumed by ``sample_quad_uniform``.

``sample_quad_uniform`` (in trainers/stage1_trainer_bonn.py and the dataset
loaders) interpolates::

    sample = (1-u)(1-v) c0 + u(1-v) c1 + u v c2 + (1-u) v c3

which is correct only when the four corners trace a NON-CROSSING loop:

    c0 ---- c1
    |       |
    c3 ---- c2

If the actual order in the Bonn calibration is different (e.g.
c0 → c2 → c1 → c3 is a bow-tie), bilinear sampling parameterises a twisted
quad and the wi_k directions become wrong.

This script checks each (la_angle, rotation) corner set in a sample of
materials against rectangle properties:

  * opposite edges have equal length (within ε)
  * adjacent edges meet at ~90°
  * diagonals are equal and longer than either edge
  * the polygon is convex (no self-crossing) once projected to its best-fit plane

Plus it renders a 3D PNG with the four corners connected in the assumed
cyclic order, so a bow-tie would jump out visually.

Run::

    conda run -n fipt_copy python scripts/tests/check_lls_corner_ordering.py

Outputs:
  * stdout summary stats per la_angle (median + worst case)
  * /tmp/lls_corner_check/mat<ID>_rot<NNN>_la<AA>.png — one figure per quad
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import scipy.io as spio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_ROOT = '/media/raid/cloth/Bonn_train'
OUT_DIR   = Path('/tmp/lls_corner_check')


def load_calibration(mat_id):
    raw = spio.loadmat(f'{DATA_ROOT}/mat{mat_id:04d}_calibration.mat')
    rotations = ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']
    out = {}
    for r in rotations:
        rd = raw[r][0, 0]
        # llsCorners is shape (3, 4, n_angles) according to docs.
        # In the Bonn loaders the convention is corners.T → (n_angles=any → 4, 3)
        out[r] = {'llsCorners': np.array(rd['llsCorners'], dtype=np.float32)}
    out['llsAnglesDegrees'] = np.array(raw['llsAnglesDegrees'],
                                        dtype=np.float64).flatten()
    return out


def quad_geometry(c):
    """c shape (4, 3). Return dict of edge lengths, diagonals, angles."""
    e01 = float(np.linalg.norm(c[1] - c[0]))
    e12 = float(np.linalg.norm(c[2] - c[1]))
    e23 = float(np.linalg.norm(c[3] - c[2]))
    e30 = float(np.linalg.norm(c[0] - c[3]))
    d02 = float(np.linalg.norm(c[2] - c[0]))
    d13 = float(np.linalg.norm(c[3] - c[1]))
    # Interior angles at each corner (cyclic neighbours).
    def angle(a, b, c_):
        v1 = a - b; v2 = c_ - b
        cosa = (v1 @ v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-12)
        return float(np.degrees(np.arccos(np.clip(cosa, -1, 1))))
    angles = [angle(c[3], c[0], c[1]),
              angle(c[0], c[1], c[2]),
              angle(c[1], c[2], c[3]),
              angle(c[2], c[3], c[0])]
    return {
        'edges':     (e01, e12, e23, e30),
        'diagonals': (d02, d13),
        'angles':    angles,
    }


def is_bowtie_2d(c):
    """Project to best-fit plane, then check whether sides c0-c1 and c2-c3
    intersect, or c1-c2 and c3-c0 intersect. Either means the cyclic order
    crosses itself."""
    centroid = c.mean(axis=0)
    cc = c - centroid
    # SVD for best-fit plane normal
    _, _, vh = np.linalg.svd(cc, full_matrices=False)
    # 2 dominant directions span the plane
    basis = vh[:2]   # (2, 3)
    p = cc @ basis.T  # (4, 2)
    def seg_cross(a1, a2, b1, b2):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) - (B[1]-A[1]) * (C[0]-A[0])
        return ((ccw(a1, b1, b2) * ccw(a2, b1, b2) < 0) and
                (ccw(b1, a1, a2) * ccw(b2, a1, a2) < 0))
    cross1 = seg_cross(p[0], p[1], p[2], p[3])
    cross2 = seg_cross(p[1], p[2], p[3], p[0])
    return cross1 or cross2


def visualize_quad(c, title, out_path):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    # cyclic loop c0→c1→c2→c3→c0
    loop = np.concatenate([c, c[:1]], axis=0)
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], '-o', color='C0', linewidth=2,
            markersize=8)
    for i in range(4):
        ax.text(c[i, 0], c[i, 1], c[i, 2] + 1, f'c{i}', fontsize=11)
    # diagonals dashed
    ax.plot(c[[0, 2], 0], c[[0, 2], 1], c[[0, 2], 2], '--', color='C1', alpha=0.5,
            label='diag c0-c2')
    ax.plot(c[[1, 3], 0], c[[1, 3], 1], c[[1, 3], 2], '--', color='C2', alpha=0.5,
            label='diag c1-c3')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    ax.set_title(title, fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--material_ids', type=int, nargs='+',
                    default=[1, 100, 200, 300])
    ap.add_argument('--n_visualize',  type=int, default=14,
                    help='How many quads to plot per material (one per la_angle '
                         'at rot000 by default).')
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}      # la_angle → list of (e_opp_ratio, ang_var, has_bowtie)
    print(f'{"mat":>5s} {"rot":>5s} {"la":>6s}   '
          f'{"e01":>6s} {"e12":>6s} {"e23":>6s} {"e30":>6s}   '
          f'{"d02":>6s} {"d13":>6s}   '
          f'{"opp01_23":>8s} {"opp12_30":>8s} {"d02/d13":>8s}   '
          f'{"min_ang":>7s} {"max_ang":>7s}   {"bowtie":>7s}')
    bowtie_count = 0
    total = 0
    for mat_id in args.material_ids:
        calib = load_calibration(mat_id)
        las = calib['llsAnglesDegrees']
        for rot in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
            corners_all = calib[rot]['llsCorners']     # (3, 4, n_angles)
            for ai, la in enumerate(las):
                c = corners_all[:, :, ai].T            # (4, 3) per loader convention
                g = quad_geometry(c)
                e01, e12, e23, e30 = g['edges']
                d02, d13 = g['diagonals']
                # rectangle: e01 ≈ e23 (opposite sides match)
                #           e12 ≈ e30 (opposite sides match)
                #           d02 ≈ d13 (diagonals equal)
                opp01_23 = abs(e01 - e23) / max(0.5 * (e01 + e23), 1e-8)
                opp12_30 = abs(e12 - e30) / max(0.5 * (e12 + e30), 1e-8)
                d_ratio  = abs(d02 - d13) / max(0.5 * (d02 + d13), 1e-8)
                min_ang = min(g['angles'])
                max_ang = max(g['angles'])
                bow = is_bowtie_2d(c)
                bowtie_count += int(bow)
                total += 1
                summary.setdefault(float(la), []).append(
                    (opp01_23, opp12_30, d_ratio,
                     min_ang, max_ang, bow))
                # log a sample row for the first material at rot000
                if mat_id == args.material_ids[0] and rot == 'rot000':
                    print(f'{mat_id:>5d} {rot:>5s} {float(la):>6.2f}   '
                          f'{e01:>6.1f} {e12:>6.1f} {e23:>6.1f} {e30:>6.1f}   '
                          f'{d02:>6.1f} {d13:>6.1f}   '
                          f'{opp01_23:>8.3%} {opp12_30:>8.3%} {d_ratio:>8.3%}   '
                          f'{min_ang:>7.2f} {max_ang:>7.2f}   '
                          f'{"YES" if bow else "no":>7s}')
                    if ai < args.n_visualize:
                        title = (f'mat{mat_id:04d} {rot} la={float(la):.0f}°\n'
                                 f'edges {e01:.1f} {e12:.1f} {e23:.1f} {e30:.1f}  '
                                 f'diags {d02:.1f} {d13:.1f}\n'
                                 f'min/max angle {min_ang:.1f}/{max_ang:.1f}°  '
                                 f'bowtie={bow}')
                        out = OUT_DIR / f'mat{mat_id:04d}_{rot}_la{int(la):+04d}.png'
                        visualize_quad(c, title, out)

    print(f'\n=== Population summary over {total} quads '
          f'({len(args.material_ids)} mats × 5 rots × 14 la_angles) ===')
    print(f'  Bowtie cases (cyclic loop self-crosses in 2D projection): '
          f'{bowtie_count} / {total}')
    print(f'\n  Per-la_angle stats (median across mat × rot):')
    print(f'  {"la":>6s}  {"med_opp01_23":>12s}  {"med_opp12_30":>12s}  '
          f'{"med_d02/d13":>12s}  {"med_min_ang":>11s}  {"med_max_ang":>11s}  '
          f'{"bowtie":>7s}')
    for la in sorted(summary):
        rows = summary[la]
        opp1 = np.median([r[0] for r in rows])
        opp2 = np.median([r[1] for r in rows])
        d    = np.median([r[2] for r in rows])
        amin = np.median([r[3] for r in rows])
        amax = np.median([r[4] for r in rows])
        bows = sum(r[5] for r in rows)
        print(f'  {la:>6.2f}  {opp1:>12.2%}  {opp2:>12.2%}  '
              f'{d:>12.2%}  {amin:>11.2f}  {amax:>11.2f}  '
              f'{bows}/{len(rows):>3d}')

    print(f'\nVisualizations saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
