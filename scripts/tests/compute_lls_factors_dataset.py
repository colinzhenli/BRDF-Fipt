#!/usr/bin/env python3
"""Estimate per-LLS-emitter scale factors from a subset of (material, pixel)
samples.

The previous single-pixel single-material constant `_LLS_EMPIRICAL_SCALE
= 0.7511` (fit from mat0001 pixel (256, 256)) has been removed — the
population median across 25 materials × 200 pixels is ~0.98 (≈ 1.0), and
the per-la_angle factor ranges from ~0.75 to ~1.05, so a single global
constant was the wrong abstraction. This script reproduces that
diagnostic on demand: KNN-match LLS↔(pan+poly), group by LLS la_angle,
and report the distribution of `c = ref_val / lls_val` per group.

Also reports the overall median and per-cv breakdown for sanity.

Run:
    conda run -n fipt_copy python scripts/tests/compute_lls_factors_dataset.py
"""
import os
import sys
import re
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.dataset.bonn import (
    _load_single_material_full,
    _read_exr, _parse_lls_channels,
    DTYPE_POLY, DTYPE_PAN, DTYPE_LLS,
)

DATA_ROOT = '/media/raid/cloth/Bonn_train'


def normalize(v, axis=-1):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-8)


def parse_lls_per_view_meta(prefix):
    """Re-parse LLS channels for (strip_id, la_angle, cv_id) per LLS view,
    in the same order the bonn loader puts them in ``gray_vals``."""
    _, ch_names, _, _ = _read_exr(f'{prefix}_lls.exr')
    imgs = _parse_lls_channels(ch_names)
    if not imgs:
        return None
    strip_ids = np.array([int(re.search(r'lls(\d+)', im['channel']).group(1))
                          for im in imgs], dtype=np.int32)
    la_angles = np.array([im['angle'] for im in imgs], dtype=np.float32)
    cv_ids = np.array([int(re.search(r'\d+', im['camera']).group())
                       for im in imgs], dtype=np.int32)
    return strip_ids, la_angles, cv_ids


def uniform_pixel_indices(H, W, n):
    """Roughly sqrt(n) × sqrt(n) interior pixel grid."""
    n_side = max(1, int(np.ceil(np.sqrt(n))))
    # Pad away from the borders by 5% so we sample valid pixels.
    rows = np.linspace(int(0.05 * H), int(0.95 * H) - 1, n_side, dtype=int)
    cols = np.linspace(int(0.05 * W), int(0.95 * W) - 1, n_side, dtype=int)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    return (rr * W + cc).ravel()[:n]


def per_pixel_matches(mat, p, normal_p, lls_meta, knn_thr):
    """For one (material, pixel) location, KNN-match LLS↔(pan+poly).

    Returns list of dicts: {la_angle, lls_cv, ref_val, lls_val} per match.
    """
    K_poly = mat['rgbs'].shape[0] if mat['rgbs'] is not None else 0
    if mat['gray_vals'] is None or K_poly == 0:
        return []
    xyz_p = mat['xyz'][p]

    # Build per-view (L, V, val) for each modality at this pixel.
    L_all = np.zeros((K_poly + mat['gray_vals'].shape[0], 3), dtype=np.float32)
    V_all = np.zeros_like(L_all)
    val   = np.zeros(L_all.shape[0], dtype=np.float32)
    dtype = mat['data_type']

    # Poly
    for k in range(K_poly):
        Lw = normalize(mat['light_pos'][k] - xyz_p)
        Vw = normalize(mat['cam_pos'][k]   - xyz_p)
        rgb = mat['rgbs'][k, p].astype(np.float32)
        w   = mat['pan_weights'][k]
        L_all[k] = Lw; V_all[k] = Vw; val[k] = float((rgb * w).sum())

    # Pan + LLS (gray_vals)
    gray = mat['gray_vals']
    for k_total in range(K_poly, L_all.shape[0]):
        li = k_total - K_poly
        if dtype[k_total] == DTYPE_LLS:
            corners = mat['lls_corners'][k_total]
            Lw = normalize(corners.mean(axis=0) - xyz_p)
        else:
            Lw = normalize(mat['light_pos'][k_total] - xyz_p)
        Vw = normalize(mat['cam_pos'][k_total] - xyz_p)
        L_all[k_total] = Lw; V_all[k_total] = Vw
        val[k_total] = float(gray[li, p])

    is_poly = dtype == DTYPE_POLY
    is_pan  = dtype == DTYPE_PAN
    is_lls  = dtype == DTYPE_LLS
    is_ref  = is_poly | is_pan
    if not is_lls.any() or not is_ref.any():
        return []

    # KNN: for each LLS view, find best ref view by (L·L' + V·V')/2.
    Llls, Vlls, Vlls_val = L_all[is_lls], V_all[is_lls], val[is_lls]
    Lref, Vref, Vref_val = L_all[is_ref], V_all[is_ref], val[is_ref]
    sim = 0.5 * (Llls @ Lref.T + Vlls @ Vref.T)
    dist = 1.0 - sim
    best_idx = np.argmin(dist, axis=1)
    best_d   = dist[np.arange(dist.shape[0]), best_idx]
    keep = best_d < knn_thr

    # Map LLS rows back to (la_angle, cv) using lls_meta.
    # lls_meta indices align with the LLS-only subset, in the same order
    # they appear in gray_vals (after pan). The LLS slice in L_all/V_all is
    # exactly the lls rows of gray_vals.
    lls_global_indices = np.where(is_lls)[0]
    lls_local_indices  = lls_global_indices - K_poly - int(is_pan.sum())

    strip_ids, la_angles, cv_ids = lls_meta
    records = []
    for i_lls, k_ok in enumerate(keep):
        if not k_ok or Vlls_val[i_lls] <= 0:
            continue
        ref_v = Vref_val[best_idx[i_lls]]
        if ref_v <= 0:
            continue
        li = lls_local_indices[i_lls]
        records.append({
            'la': float(la_angles[li]),
            'cv': int(cv_ids[li]),
            'ref': float(ref_v),
            'lls': float(Vlls_val[i_lls]),
            'cosdist': float(best_d[i_lls]),
        })
    return records


def stats(values):
    a = np.asarray(values, dtype=np.float32)
    return {
        'n': int(a.size),
        'median': float(np.median(a)),
        'mean':   float(a.mean()),
        'p10':    float(np.percentile(a, 10)),
        'p90':    float(np.percentile(a, 90)),
        'iqr':    float(np.percentile(a, 75) - np.percentile(a, 25)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_materials', type=int, default=25,
                    help='How many materials to sample (stride over the LLS-available set).')
    ap.add_argument('--n_pixels',    type=int, default=200,
                    help='Pixels per material (uniform grid).')
    ap.add_argument('--knn_thr',     type=float, default=0.05,
                    help='Max (1 - cosine similarity) for KNN match.')
    ap.add_argument('--out_json',    default='/tmp/lls_factor_dataset.json')
    args = ap.parse_args()

    all_paths = sorted(Path(DATA_ROOT).glob('mat????_lls.exr'))
    all_ids = [int(p.name[3:7]) for p in all_paths]
    step = max(1, len(all_ids) // args.n_materials)
    chosen = all_ids[::step][:args.n_materials]
    print(f'Sampling {len(chosen)} materials × {args.n_pixels} pixels each')
    print(f'Material IDs: {chosen}')
    print(f'KNN cosdist threshold: {args.knn_thr}')

    records = []
    for mat_id in chosen:
        prefix = Path(DATA_ROOT) / f'mat{mat_id:04d}'
        try:
            mat = _load_single_material_full(DATA_ROOT, mat_id,
                                             use_pan=True, use_lls=True)
        except Exception as e:
            print(f'mat{mat_id:04d}: load failed ({e})'); continue
        if mat is None:
            print(f'mat{mat_id:04d}: empty'); continue

        lls_meta = parse_lls_per_view_meta(prefix)
        if lls_meta is None:
            print(f'mat{mat_id:04d}: no LLS meta'); continue

        H, W = mat['H'], mat['W']
        pix_indices = uniform_pixel_indices(H, W, args.n_pixels)
        mat_records = 0
        for p in pix_indices:
            normal_p = (mat['gt_normals'][p]
                        if mat['gt_normals'] is not None else
                        np.array([0., 0., 1.], dtype=np.float32))
            try:
                recs = per_pixel_matches(mat, int(p), normal_p, lls_meta,
                                         args.knn_thr)
            except Exception as e:
                continue
            for r in recs:
                r['mat_id'] = int(mat_id)
                records.append(r)
            mat_records += len(recs)
        print(f'  mat{mat_id:04d}: H={H} W={W} → {mat_records} matches')

    if not records:
        print('No matches collected. Increase --knn_thr or sample.')
        return

    ratios = np.array([r['ref'] / r['lls'] for r in records])
    la = np.array([r['la']  for r in records])
    cv = np.array([r['cv']  for r in records])
    mat_ids = np.array([r['mat_id'] for r in records])

    print(f'\nTotal matches: {len(records)} (across {len(set(mat_ids.tolist()))} mats)')
    print(f'\n=== Global factor (all matches pooled) ===')
    g = stats(ratios)
    print(f'  n={g["n"]}  median={g["median"]:.3f}  mean={g["mean"]:.3f}  '
          f'p10={g["p10"]:.3f}  p90={g["p90"]:.3f}  iqr={g["iqr"]:.3f}')
    print(f'  → previous _LLS_EMPIRICAL_SCALE was 0.7511, now removed')

    print(f'\n=== Per la_angle (14 LLS emitter positions) ===')
    print(f'  {"la_deg":>7s}  {"n":>6s}  {"median":>7s}  {"mean":>7s}  '
          f'{"p10":>7s}  {"p90":>7s}  {"iqr":>7s}')
    per_la = {}
    for la_v in sorted(np.unique(la)):
        sel = la == la_v
        s = stats(ratios[sel])
        per_la[float(la_v)] = s
        print(f'  {float(la_v):>7.2f}  {s["n"]:>6d}  {s["median"]:>7.3f}  '
              f'{s["mean"]:>7.3f}  {s["p10"]:>7.3f}  {s["p90"]:>7.3f}  '
              f'{s["iqr"]:>7.3f}')

    print(f'\n=== Per cv camera (4 cameras) ===')
    print(f'  {"cv":>4s}  {"n":>6s}  {"median":>7s}  {"mean":>7s}  '
          f'{"p10":>7s}  {"p90":>7s}')
    per_cv = {}
    for cv_v in sorted(np.unique(cv)):
        sel = cv == cv_v
        s = stats(ratios[sel])
        per_cv[int(cv_v)] = s
        print(f'  {int(cv_v):>4d}  {s["n"]:>6d}  {s["median"]:>7.3f}  '
              f'{s["mean"]:>7.3f}  {s["p10"]:>7.3f}  {s["p90"]:>7.3f}')

    print(f'\n=== Per (la_angle, cv) — top variance cells ===')
    per_la_cv = {}
    for la_v in sorted(np.unique(la)):
        for cv_v in sorted(np.unique(cv)):
            sel = (la == la_v) & (cv == cv_v)
            if sel.sum() < 5:
                continue
            s = stats(ratios[sel])
            per_la_cv[f'{float(la_v):+06.2f}_cv{int(cv_v):02d}'] = s

    out = {
        'global':   g,
        'per_la':   per_la,
        'per_cv':   per_cv,
        'per_la_cv': per_la_cv,
        'n_materials': int(len(set(mat_ids.tolist()))),
        'n_pixels_per_material_target': args.n_pixels,
        'knn_thr': args.knn_thr,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {args.out_json}')


if __name__ == '__main__':
    main()
