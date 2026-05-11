#!/usr/bin/env python3
"""Population poly↔pan consistency stats.

For each (material, pixel) sample, project poly RGB to pan-equivalent grayscale
using the per-(cv, il) ``poly2panWeights`` already loaded into ``mat['pan_weights']``,
then KNN-match each poly view to the nearest pan view in (L, V) cosine
similarity space. Report

  ratio = poly_pan_equiv / pan_raw

distribution per:
  - global  (sanity check: should be ≈ 1.0 if the per-(cv, il) weights are
            calibrated correctly)
  - poly cv (4 cameras)
  - poly il (5 filter LEDs: il026/27/28/31/32)
  - (cv, il) (20 cells)

Also reports the same broken out for *same-LED matches only* (poly cv01_il026
↔ pan cv01_il026), which isolates the calibration error from BRDF
disagreement at different LEDs.

Run:
    conda run -n fipt_copy python scripts/tests/compute_polypan_factors_dataset.py
"""
import os
import sys
import re
import argparse
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.dataset.bonn import (
    _load_single_material_full,
    _read_exr, _parse_poly_channels, _parse_pan_channels,
    DTYPE_POLY, DTYPE_PAN,
)

DATA_ROOT = '/media/raid/cloth/Bonn_train'


def normalize(v, axis=-1):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-8)


def parse_poly_per_view_meta(prefix):
    _, ch_names, _, _ = _read_exr(f'{prefix}_poly.exr')
    imgs = _parse_poly_channels(ch_names)
    cv_ids = np.array([int(re.search(r'\d+', im['camera']).group()) for im in imgs],
                      dtype=np.int32)
    il_ids = np.array([int(re.search(r'\d+', im['led']).group())    for im in imgs],
                      dtype=np.int32)
    return cv_ids, il_ids


def parse_pan_per_view_meta(prefix):
    _, ch_names, _, _ = _read_exr(f'{prefix}_pan.exr')
    imgs = [im for im in _parse_pan_channels(ch_names)
            if int(im['led'][2:]) <= 24 or int(im['led'][2:]) in {26, 27, 28, 31, 32}]
    # Keep all 388 pan images for the matching pool — the bonn loader filters
    # to il001-24 only for the gray_parts. Use the loader's filter:
    pan_imgs = [im for im in _parse_pan_channels(ch_names)
                if int(im['led'][2:]) <= 24]
    cv_ids = np.array([int(re.search(r'\d+', im['camera']).group()) for im in pan_imgs],
                      dtype=np.int32)
    il_ids = np.array([int(re.search(r'\d+', im['led']).group())    for im in pan_imgs],
                      dtype=np.int32)
    return cv_ids, il_ids


def uniform_pixel_indices(H, W, n):
    n_side = max(1, int(np.ceil(np.sqrt(n))))
    rows = np.linspace(int(0.05 * H), int(0.95 * H) - 1, n_side, dtype=int)
    cols = np.linspace(int(0.05 * W), int(0.95 * W) - 1, n_side, dtype=int)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    return (rr * W + cc).ravel()[:n]


def per_pixel_polypan(mat, p, knn_thr, poly_meta, pan_meta):
    """KNN-match poly→pan at one pixel.

    Returns list of {poly_cv, poly_il, pan_cv, pan_il, ratio, cosdist}.
    """
    K_poly = mat['rgbs'].shape[0] if mat['rgbs'] is not None else 0
    if K_poly == 0 or mat['gray_vals'] is None:
        return []
    poly_cv, poly_il = poly_meta
    if len(poly_cv) != K_poly:
        return []  # metadata mismatch — skip
    pan_cv, pan_il = pan_meta

    # K_gray = K_pan + K_lls; we only need K_pan.
    K_pan = int((mat['data_type'] == DTYPE_PAN).sum())
    if K_pan == 0:
        return []
    if len(pan_cv) != K_pan:
        return []

    xyz_p = mat['xyz'][p]

    # Build poly arrays.
    Lp = normalize(mat['light_pos'][:K_poly] - xyz_p, axis=1)
    Vp = normalize(mat['cam_pos'][:K_poly]   - xyz_p, axis=1)
    rgb = mat['rgbs'][:, p, :].astype(np.float32)        # (K_poly, 3)
    w   = mat['pan_weights'][:K_poly]                    # (K_poly, 3)
    poly_val = (rgb * w).sum(axis=-1)                    # (K_poly,)

    # Build pan arrays. Pan is the first K_pan rows of gray_vals; its
    # light/cam positions live at offsets K_poly..K_poly+K_pan in mat['light_pos'].
    pan_slice = slice(K_poly, K_poly + K_pan)
    Lq = normalize(mat['light_pos'][pan_slice] - xyz_p, axis=1)
    Vq = normalize(mat['cam_pos'][pan_slice]   - xyz_p, axis=1)
    pan_val = mat['gray_vals'][:K_pan, p].astype(np.float32)

    if poly_val.size == 0 or pan_val.size == 0:
        return []

    # KNN: for each poly view, find best pan.
    sim = 0.5 * (Lp @ Lq.T + Vp @ Vq.T)
    dist = 1.0 - sim
    best_idx = np.argmin(dist, axis=1)
    best_d   = dist[np.arange(dist.shape[0]), best_idx]
    keep = (best_d < knn_thr) & (poly_val > 0) & (pan_val[best_idx] > 0)

    records = []
    for i in np.where(keep)[0]:
        records.append({
            'poly_cv': int(poly_cv[i]),
            'poly_il': int(poly_il[i]),
            'pan_cv':  int(pan_cv[best_idx[i]]),
            'pan_il':  int(pan_il[best_idx[i]]),
            'ratio':   float(poly_val[i]) / float(pan_val[best_idx[i]]),
            'cosdist': float(best_d[i]),
        })
    return records


def stats(values):
    a = np.asarray(values, dtype=np.float32)
    if a.size == 0:
        return {'n': 0, 'median': float('nan'), 'mean': float('nan'),
                'p10': float('nan'), 'p90': float('nan'),
                'iqr': float('nan')}
    return {
        'n': int(a.size),
        'median': float(np.median(a)),
        'mean':   float(a.mean()),
        'p10':    float(np.percentile(a, 10)),
        'p90':    float(np.percentile(a, 90)),
        'iqr':    float(np.percentile(a, 75) - np.percentile(a, 25)),
    }


def report_group(label, values_by_key, decimal_key=False):
    print(f'\n=== {label} ===')
    print(f'  {"key":>14s}  {"n":>7s}  {"median":>7s}  {"mean":>7s}  '
          f'{"p10":>7s}  {"p90":>7s}  {"iqr":>7s}')
    out = {}
    for k in sorted(values_by_key):
        s = stats(values_by_key[k])
        out[str(k)] = s
        if decimal_key:
            key_str = f'{k:>14.2f}'
        else:
            key_str = f'{str(k):>14s}'
        print(f'  {key_str}  {s["n"]:>7d}  {s["median"]:>7.3f}  '
              f'{s["mean"]:>7.3f}  {s["p10"]:>7.3f}  {s["p90"]:>7.3f}  '
              f'{s["iqr"]:>7.3f}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_materials', type=int, default=25)
    ap.add_argument('--n_pixels',    type=int, default=200)
    ap.add_argument('--knn_thr',     type=float, default=0.02,
                    help='Tighter than the LLS run (0.05) since poly and pan '
                         'have many more views and we want stricter geometry match.')
    ap.add_argument('--out_json',    default='/tmp/polypan_factor_dataset.json')
    args = ap.parse_args()

    all_paths = sorted(Path(DATA_ROOT).glob('mat????_lls.exr'))
    all_ids = [int(p.name[3:7]) for p in all_paths]
    step = max(1, len(all_ids) // args.n_materials)
    chosen = all_ids[::step][:args.n_materials]
    print(f'Sampling {len(chosen)} materials × {args.n_pixels} pixels each '
          f'(KNN cosdist < {args.knn_thr})')

    records = []
    for mat_id in chosen:
        prefix = Path(DATA_ROOT) / f'mat{mat_id:04d}'
        try:
            mat = _load_single_material_full(DATA_ROOT, mat_id,
                                             use_pan=True, use_lls=False)
        except Exception as e:
            print(f'mat{mat_id:04d}: load failed ({e})'); continue
        if mat is None:
            print(f'mat{mat_id:04d}: empty'); continue
        try:
            poly_meta = parse_poly_per_view_meta(prefix)
            pan_meta  = parse_pan_per_view_meta(prefix)
        except Exception as e:
            print(f'mat{mat_id:04d}: meta parse failed: {e}'); continue

        H, W = mat['H'], mat['W']
        pix_indices = uniform_pixel_indices(H, W, args.n_pixels)
        mat_n = 0
        for p in pix_indices:
            try:
                recs = per_pixel_polypan(mat, int(p), args.knn_thr,
                                         poly_meta, pan_meta)
            except Exception:
                continue
            for r in recs:
                r['mat_id'] = int(mat_id)
                records.append(r)
            mat_n += len(recs)
        print(f'  mat{mat_id:04d}: H={H} W={W} → {mat_n} matches')

    if not records:
        print('No matches collected.')
        return

    ratios = np.array([r['ratio'] for r in records])
    poly_cv = np.array([r['poly_cv'] for r in records])
    poly_il = np.array([r['poly_il'] for r in records])
    pan_il  = np.array([r['pan_il']  for r in records])

    same_led = poly_il == pan_il
    print(f'\nTotal matches: {len(records)} '
          f'(across {len(set(r["mat_id"] for r in records))} mats); '
          f'same-LED matches: {int(same_led.sum())} ({100*same_led.mean():.1f}%)')

    print(f'\n=== Global poly→pan ratio (ALL matches) ===')
    g_all = stats(ratios)
    print(f'  n={g_all["n"]}  median={g_all["median"]:.3f}  '
          f'mean={g_all["mean"]:.3f}  p10={g_all["p10"]:.3f}  '
          f'p90={g_all["p90"]:.3f}  iqr={g_all["iqr"]:.3f}')

    print(f'\n=== Global poly→pan ratio (SAME-LED matches only) ===')
    g_same = stats(ratios[same_led])
    print(f'  n={g_same["n"]}  median={g_same["median"]:.3f}  '
          f'mean={g_same["mean"]:.3f}  p10={g_same["p10"]:.3f}  '
          f'p90={g_same["p90"]:.3f}  iqr={g_same["iqr"]:.3f}')
    print(f'  (Tests the per-(cv, il) poly→pan weights in isolation: a '
          f'well-calibrated weight gives median ≈ 1.0)')

    per_cv_all = {int(k): ratios[poly_cv == k].tolist()
                  for k in np.unique(poly_cv)}
    per_il_all = {int(k): ratios[poly_il == k].tolist()
                  for k in np.unique(poly_il)}
    per_cv_same = {int(k): ratios[(poly_cv == k) & same_led].tolist()
                   for k in np.unique(poly_cv)}
    per_il_same = {int(k): ratios[(poly_il == k) & same_led].tolist()
                   for k in np.unique(poly_il)}

    per_cv_all_stats   = report_group('Per cv camera (ALL matches)', per_cv_all)
    per_il_all_stats   = report_group('Per poly il LED (ALL matches)', per_il_all)
    per_cv_same_stats  = report_group('Per cv camera (SAME-LED matches)', per_cv_same)
    per_il_same_stats  = report_group('Per poly il LED (SAME-LED matches)', per_il_same)

    # Per (cv, il) for same-LED only — this is the calibration table test.
    print(f'\n=== Per (cv, il) — SAME-LED only ===')
    print(f'  {"cv":>3s} {"il":>4s}  {"n":>6s}  {"median":>7s}  {"p10":>6s}  {"p90":>6s}')
    per_cv_il = {}
    for cv_v in np.unique(poly_cv):
        for il_v in np.unique(poly_il):
            sel = (poly_cv == cv_v) & (poly_il == il_v) & same_led
            if sel.sum() < 5:
                continue
            s = stats(ratios[sel])
            per_cv_il[f'cv{int(cv_v):02d}_il{int(il_v):03d}'] = s
            print(f'  {int(cv_v):>3d} {int(il_v):>4d}  {s["n"]:>6d}  '
                  f'{s["median"]:>7.3f}  {s["p10"]:>6.3f}  {s["p90"]:>6.3f}')

    out = {
        'global_all':   g_all,
        'global_same':  g_same,
        'per_cv_all':   per_cv_all_stats,
        'per_il_all':   per_il_all_stats,
        'per_cv_same':  per_cv_same_stats,
        'per_il_same':  per_il_same_stats,
        'per_cv_il_same': per_cv_il,
        'knn_thr': args.knn_thr,
        'n_materials': len(set(r['mat_id'] for r in records)),
        'n_pixels_per_material_target': args.n_pixels,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {args.out_json}')


if __name__ == '__main__':
    main()
