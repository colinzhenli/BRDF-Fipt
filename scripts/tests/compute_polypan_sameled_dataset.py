#!/usr/bin/env python3
"""Direct same-(cv,il,rot) poly→pan comparison across the population.

For each material, the calibration `.mat` provides ``poly2panIndices`` —
a 100-entry mapping from poly captures (il026/27/28/31/32 × 4 cv × 5 rot)
to the *corresponding* pan captures (same cv/il/rot, color-filtered LED).
The Bonn docs say:

    Panchromatic images can be obtained from polychromatic ones via
    weighted averaging of the three RGB channels.

So `poly2panWeights[cv, il]` should make:

    poly_pan_equiv = sum(rgb * poly2panWeights[cv, il])
    pan_direct     = pan_exr[matched_index]

equal at the same pixel. This script tests that directly (no KNN — these
are exact pairings) and reports the ratio per (cv, il) cell across a
population sample.

Run:
    conda run -n fipt_copy python scripts/tests/compute_polypan_sameled_dataset.py
"""
import os
import sys
import re
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as spio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.dataset.bonn import (
    _read_exr, _parse_poly_channels, _parse_pan_channels, _parse_poly2pan_weights,
)

DATA_ROOT = '/media/raid/cloth/Bonn_train'


def uniform_pixel_indices(H, W, n):
    n_side = max(1, int(np.ceil(np.sqrt(n))))
    rows = np.linspace(int(0.05 * H), int(0.95 * H) - 1, n_side, dtype=int)
    cols = np.linspace(int(0.05 * W), int(0.95 * W) - 1, n_side, dtype=int)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    return (rr * W + cc).ravel()[:n]


def stats(values):
    a = np.asarray(values, dtype=np.float32)
    if a.size == 0:
        return None
    return {
        'n': int(a.size),
        'median': float(np.median(a)),
        'mean':   float(a.mean()),
        'p10':    float(np.percentile(a, 10)),
        'p90':    float(np.percentile(a, 90)),
        'iqr':    float(np.percentile(a, 75) - np.percentile(a, 25)),
    }


def process_material(mat_id, n_pixels):
    """Returns list of dicts {cv, il, rot, ratio} for same-(cv,il,rot) pairs."""
    prefix = Path(DATA_ROOT) / f'mat{mat_id:04d}'
    raw = spio.loadmat(f'{prefix}_calibration.mat')
    w = _parse_poly2pan_weights(raw)
    per_il = w['per_il']  # {(cv, il) -> (3,)}
    global_avg = w['global_avg']

    poly2pan = raw['poly2panIndices'].flatten().astype(np.int64)  # 100 entries, 1-based

    # Read poly EXR
    poly_data, poly_ch, _, _ = _read_exr(f'{prefix}_poly.exr')
    poly_imgs = _parse_poly_channels(poly_ch)           # 100 imgs
    H, W = poly_data.shape[:2]
    n_pixels_total = H * W
    poly_rgbs = poly_data.reshape(n_pixels_total, len(poly_imgs), 3).astype(np.float32)
    np.clip(poly_rgbs, 0, None, out=poly_rgbs)

    # Read pan EXR (all channels including il026-32 pan counterparts)
    pan_data, pan_ch, _, _ = _read_exr(f'{prefix}_pan.exr')
    pan_imgs = _parse_pan_channels(pan_ch)              # ~388 imgs (all LEDs)
    pan_flat = pan_data.reshape(n_pixels_total, len(pan_imgs)).astype(np.float32)
    np.clip(pan_flat, 0, None, out=pan_flat)

    pix_indices = uniform_pixel_indices(H, W, n_pixels)

    out = []
    for kp, im in enumerate(poly_imgs):
        cv = im['camera']; il = im['led']; rot = im['rotation']
        w_cvil = per_il.get((cv, il), global_avg).astype(np.float32)
        # The matched pan index (1-based per Bonn convention).
        idx_pan_1based = poly2pan[kp]
        if idx_pan_1based <= 0 or idx_pan_1based > len(pan_imgs):
            continue
        idx_pan = int(idx_pan_1based - 1)
        # Sanity: the matched pan image should have same (cv, il, rot).
        pim = pan_imgs[idx_pan]
        if pim['camera'] != cv or pim['rotation'] != rot:
            # Some pan channel ordering differs; skip if cv/rot mismatch.
            continue

        poly_pan_equiv = (poly_rgbs[pix_indices, kp, :] * w_cvil).sum(axis=-1)
        pan_direct     = pan_flat[pix_indices, idx_pan]
        keep = (poly_pan_equiv > 1e-6) & (pan_direct > 1e-6)
        if not keep.any():
            continue
        ratios = (poly_pan_equiv[keep] / pan_direct[keep]).tolist()
        for rv in ratios:
            out.append({'cv': cv, 'il': il, 'rot': rot, 'ratio': float(rv)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_materials', type=int, default=25)
    ap.add_argument('--n_pixels',    type=int, default=200)
    ap.add_argument('--out_json',    default='/tmp/polypan_sameled_dataset.json')
    args = ap.parse_args()

    all_paths = sorted(Path(DATA_ROOT).glob('mat????_lls.exr'))
    all_ids = [int(p.name[3:7]) for p in all_paths]
    step = max(1, len(all_ids) // args.n_materials)
    chosen = all_ids[::step][:args.n_materials]
    print(f'Sampling {len(chosen)} materials × {args.n_pixels} pixels each')

    records = []
    for mat_id in chosen:
        try:
            rec = process_material(mat_id, args.n_pixels)
        except Exception as e:
            print(f'mat{mat_id:04d}: failed ({e})'); continue
        for r in rec:
            r['mat_id'] = int(mat_id)
        records.extend(rec)
        print(f'  mat{mat_id:04d}: {len(rec)} same-LED pairs')

    if not records:
        print('No matches collected.'); return

    ratios = np.array([r['ratio'] for r in records])
    cvs    = np.array([int(re.search(r'\d+', r['cv']).group()) for r in records])
    ils    = np.array([int(re.search(r'\d+', r['il']).group()) for r in records])

    g = stats(ratios)
    print(f'\n=== Global SAME-LED poly→pan ratio ===')
    print(f'  n={g["n"]}  median={g["median"]:.4f}  mean={g["mean"]:.4f}  '
          f'p10={g["p10"]:.4f}  p90={g["p90"]:.4f}  iqr={g["iqr"]:.4f}')

    print(f'\n=== Per cv ===')
    print(f'  {"cv":>3s}  {"n":>7s}  {"median":>7s}  {"mean":>7s}  {"p10":>7s}  {"p90":>7s}  {"iqr":>7s}')
    per_cv = {}
    for cv in sorted(np.unique(cvs)):
        sel = cvs == cv
        s = stats(ratios[sel])
        per_cv[int(cv)] = s
        print(f'  {int(cv):>3d}  {s["n"]:>7d}  {s["median"]:>7.4f}  '
              f'{s["mean"]:>7.4f}  {s["p10"]:>7.4f}  {s["p90"]:>7.4f}  '
              f'{s["iqr"]:>7.4f}')

    print(f'\n=== Per il LED ===')
    print(f'  {"il":>4s}  {"n":>7s}  {"median":>7s}  {"mean":>7s}  {"p10":>7s}  {"p90":>7s}  {"iqr":>7s}')
    per_il = {}
    for il in sorted(np.unique(ils)):
        sel = ils == il
        s = stats(ratios[sel])
        per_il[int(il)] = s
        print(f'  {int(il):>4d}  {s["n"]:>7d}  {s["median"]:>7.4f}  '
              f'{s["mean"]:>7.4f}  {s["p10"]:>7.4f}  {s["p90"]:>7.4f}  '
              f'{s["iqr"]:>7.4f}')

    print(f'\n=== Per (cv, il) — table form ===')
    print(f'  {"cv":>3s} {"il":>4s}  {"n":>6s}  {"median":>7s}  {"iqr":>7s}')
    per_cv_il = {}
    for cv in sorted(np.unique(cvs)):
        for il in sorted(np.unique(ils)):
            sel = (cvs == cv) & (ils == il)
            if sel.sum() < 5:
                continue
            s = stats(ratios[sel])
            per_cv_il[f'cv{int(cv):02d}_il{int(il):03d}'] = s
            print(f'  {int(cv):>3d} {int(il):>4d}  {s["n"]:>6d}  '
                  f'{s["median"]:>7.4f}  {s["iqr"]:>7.4f}')

    out = {
        'global':    g,
        'per_cv':    per_cv,
        'per_il':    per_il,
        'per_cv_il': per_cv_il,
        'n_materials': len(set(r['mat_id'] for r in records)),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {args.out_json}')


if __name__ == '__main__':
    main()
