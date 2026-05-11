#!/usr/bin/env python3
"""Test whether MC quad-sampling resolves the per-la_angle factor variation.

`compute_lls_factors_dataset.py` treated each LLS view as a point light at
the strip CENTROID, then KNN-matched LLS↔(pan+poly) by single-direction
similarity. Per la_angle the median ratio varied 0.75–1.05 — most likely
because the strip subtends a wide solid angle at grazing la_angles and
the centroid direction stops representing the strip integral well.

This script repeats the same population sweep but with MC sampling: for
each LLS view, sample SPP points on the quad, KNN-match each sample to
its nearest pan/poly view, and form an MC-weighted reference:

    ref_mc = Σ_k (ref_val_k · w_k) / Σ_k w_k   with w_k = cos_k / dist_k²

If MC integration is the right model, the per-la_angle factor variation
should flatten near 1.0 — confirming that the trainer's `_lls_monte_carlo`
already handles the angular mismatch.

Run:
    conda run -n fipt_copy python scripts/tests/compute_lls_factors_mc_dataset.py
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
    _read_exr, _parse_lls_channels,
    DTYPE_POLY, DTYPE_PAN, DTYPE_LLS,
)

DATA_ROOT = '/media/raid/cloth/Bonn_train'


def normalize(v, axis=-1):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-8)


def parse_lls_per_view_meta(prefix):
    _, ch_names, _, _ = _read_exr(f'{prefix}_lls.exr')
    imgs = _parse_lls_channels(ch_names)
    if not imgs:
        return None
    la_angles = np.array([im['angle'] for im in imgs], dtype=np.float32)
    cv_ids    = np.array([int(re.search(r'\d+', im['camera']).group())
                          for im in imgs], dtype=np.int32)
    return la_angles, cv_ids


def uniform_pixel_indices(H, W, n):
    n_side = max(1, int(np.ceil(np.sqrt(n))))
    rows = np.linspace(int(0.05 * H), int(0.95 * H) - 1, n_side, dtype=int)
    cols = np.linspace(int(0.05 * W), int(0.95 * W) - 1, n_side, dtype=int)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    return (rr * W + cc).ravel()[:n]


def sample_quad_uniform(corners, spp, rng):
    """Bilinear interpolation sampling of a (4,3) quad. Returns (spp, 3)."""
    u = rng.random((spp, 1)).astype(np.float32)
    v = rng.random((spp, 1)).astype(np.float32)
    c0, c1, c2, c3 = corners[0], corners[1], corners[2], corners[3]
    return ((1 - u) * (1 - v) * c0 + u * (1 - v) * c1
            + u * v * c2 + (1 - u) * v * c3)


def per_pixel_mc(mat, p, knn_thr, lls_meta, spp, rng):
    """For one pixel, compute centroid-based and MC-based ratios per LLS view.

    Returns list of {la_angle, cv, ratio_centroid, ratio_mc}.
    """
    K_poly = mat['rgbs'].shape[0] if mat['rgbs'] is not None else 0
    if K_poly == 0 or mat['gray_vals'] is None:
        return []
    la_angles, lls_cv_ids = lls_meta
    xyz_p = mat['xyz'][p]

    is_poly = mat['data_type'] == DTYPE_POLY
    is_pan  = mat['data_type'] == DTYPE_PAN
    is_lls  = mat['data_type'] == DTYPE_LLS

    # Build pan+poly reference pool at this pixel.
    ref_mask = is_poly | is_pan
    n_ref = int(ref_mask.sum())
    if n_ref == 0:
        return []
    L_ref = np.zeros((n_ref, 3), dtype=np.float32)
    V_ref = np.zeros((n_ref, 3), dtype=np.float32)
    v_ref = np.zeros(n_ref, dtype=np.float32)
    j = 0
    for k in range(L_ref.shape[0] + int(is_lls.sum())):
        if not ref_mask[k]:
            continue
        L_ref[j] = normalize(mat['light_pos'][k] - xyz_p)
        V_ref[j] = normalize(mat['cam_pos'][k]   - xyz_p)
        if is_poly[k]:
            rgb = mat['rgbs'][k, p].astype(np.float32)
            w   = mat['pan_weights'][k]
            v_ref[j] = float((rgb * w).sum())
        else:
            li = k - K_poly
            v_ref[j] = float(mat['gray_vals'][li, p])
        j += 1

    # LLS views
    lls_global_idx = np.where(is_lls)[0]
    out = []
    for li_local, k_global in enumerate(lls_global_idx):
        corners = mat['lls_corners'][k_global]
        cam = mat['cam_pos'][k_global]
        Vw  = normalize(cam - xyz_p)
        lls_val = float(mat['gray_vals'][k_global - K_poly, p])
        if lls_val <= 0:
            continue

        # ---- Centroid-based reference (old method) ----
        cent = corners.mean(axis=0)
        Lw_c = normalize(cent - xyz_p)
        sim_c = 0.5 * (L_ref @ Lw_c + V_ref @ Vw)
        d_c = 1.0 - sim_c
        i_c = int(np.argmin(d_c))
        if d_c[i_c] >= knn_thr or v_ref[i_c] <= 0:
            ref_c = None
        else:
            ref_c = float(v_ref[i_c])

        # ---- MC reference ----
        samples = sample_quad_uniform(corners, spp, rng)        # (spp, 3)
        diff = samples - xyz_p[None, :]
        dist_sq = (diff * diff).sum(-1, keepdims=True).clip(1e-8)
        wi_k = diff / np.sqrt(dist_sq)                          # (spp, 3)
        # Local surface normal at this pixel — use the GT if available.
        n = (mat['gt_normals'][p] if mat['gt_normals'] is not None
             else np.array([0., 0., 1.], dtype=np.float32))
        n = n / max(np.linalg.norm(n), 1e-8)
        cos_k = np.clip(wi_k @ n, 0., None).reshape(-1, 1)      # (spp, 1)
        w_k = cos_k / dist_sq                                    # (spp, 1)

        # KNN per MC sample
        sim_mc = 0.5 * (wi_k @ L_ref.T + (Vw @ V_ref.T)[None, :])  # (spp, n_ref)
        d_mc = 1.0 - sim_mc
        i_mc = np.argmin(d_mc, axis=1)
        d_best = d_mc[np.arange(spp), i_mc]
        keep = (d_best < knn_thr) & (v_ref[i_mc] > 0)
        if not keep.any() or w_k[keep].sum() < 1e-8:
            ref_mc_val = None
        else:
            ref_vals = v_ref[i_mc[keep]]
            w_keep = w_k[keep].squeeze(-1)
            ref_mc_val = float((ref_vals * w_keep).sum() / w_keep.sum())

        record = {
            'la': float(la_angles[li_local]),
            'cv': int(lls_cv_ids[li_local]),
            'lls': lls_val,
            'ref_c':  ref_c,
            'ref_mc': ref_mc_val,
        }
        out.append(record)
    return out


def stats(values):
    a = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=np.float32)
    if a.size == 0:
        return None
    return {
        'n': int(a.size),
        'median': float(np.median(a)),
        'mean':   float(a.mean()),
        'p10':    float(np.percentile(a, 10)),
        'p90':    float(np.percentile(a, 90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_materials', type=int, default=15)
    ap.add_argument('--n_pixels',    type=int, default=80)
    ap.add_argument('--knn_thr',     type=float, default=0.05)
    ap.add_argument('--spp',         type=int, default=16)
    ap.add_argument('--seed',        type=int, default=0)
    ap.add_argument('--out_json',    default='/tmp/lls_factor_mc_dataset.json')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    all_paths = sorted(Path(DATA_ROOT).glob('mat????_lls.exr'))
    all_ids = [int(p.name[3:7]) for p in all_paths]
    step = max(1, len(all_ids) // args.n_materials)
    chosen = all_ids[::step][:args.n_materials]
    print(f'Sampling {len(chosen)} materials × {args.n_pixels} pixels each '
          f'(spp={args.spp}, KNN cosdist < {args.knn_thr})')

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
            continue
        H, W = mat['H'], mat['W']
        pix = uniform_pixel_indices(H, W, args.n_pixels)
        n_mat = 0
        for p in pix:
            try:
                recs = per_pixel_mc(mat, int(p), args.knn_thr, lls_meta,
                                    args.spp, rng)
            except Exception:
                continue
            for r in recs:
                r['mat_id'] = int(mat_id)
                records.append(r)
            n_mat += len(recs)
        print(f'  mat{mat_id:04d}: {n_mat} LLS views processed')

    if not records:
        print('No records.'); return

    # Compute ratios
    la = np.array([r['la'] for r in records])
    lls = np.array([r['lls'] for r in records])
    ref_c  = np.array([r['ref_c']  if r['ref_c']  is not None else np.nan
                       for r in records])
    ref_mc = np.array([r['ref_mc'] if r['ref_mc'] is not None else np.nan
                       for r in records])
    ratio_c  = ref_c  / lls
    ratio_mc = ref_mc / lls

    print(f'\nTotal LLS-view records: {len(records)}')
    print(f'  centroid match available: {int(np.isfinite(ratio_c).sum())}')
    print(f'  MC match available:       {int(np.isfinite(ratio_mc).sum())}')

    print(f'\n=== Global ratio comparison ===')
    g_c  = stats(ratio_c.tolist())
    g_mc = stats(ratio_mc.tolist())
    print(f'  centroid: n={g_c["n"]:>7d}  median={g_c["median"]:.3f}  '
          f'p10/p90=[{g_c["p10"]:.3f}, {g_c["p90"]:.3f}]')
    print(f'  MC      : n={g_mc["n"]:>7d}  median={g_mc["median"]:.3f}  '
          f'p10/p90=[{g_mc["p10"]:.3f}, {g_mc["p90"]:.3f}]')

    print(f'\n=== Per la_angle — centroid vs MC ===')
    print(f'  {"la":>6s}  {"n_c":>6s} {"med_c":>6s} {"iqr_c":>6s}   '
          f'{"n_mc":>6s} {"med_mc":>6s} {"iqr_mc":>6s}   {"Δmed":>7s}')
    per_la = {}
    for la_v in sorted(np.unique(la)):
        sel = la == la_v
        rc = ratio_c[sel]
        rm = ratio_mc[sel]
        sc = stats(rc.tolist())
        sm = stats(rm.tolist())
        if sc is None or sm is None: continue
        per_la[float(la_v)] = {'centroid': sc, 'mc': sm}
        c_iqr = float(np.percentile(rc[np.isfinite(rc)], 75) -
                      np.percentile(rc[np.isfinite(rc)], 25))
        m_iqr = float(np.percentile(rm[np.isfinite(rm)], 75) -
                      np.percentile(rm[np.isfinite(rm)], 25))
        d = sm['median'] - sc['median']
        print(f'  {float(la_v):>6.2f}  {sc["n"]:>6d} {sc["median"]:>6.3f} '
              f'{c_iqr:>6.3f}   {sm["n"]:>6d} {sm["median"]:>6.3f} '
              f'{m_iqr:>6.3f}   {d:>+7.3f}')

    out = {
        'global_centroid': g_c,
        'global_mc':       g_mc,
        'per_la':          per_la,
        'spp':             args.spp,
        'knn_thr':         args.knn_thr,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {args.out_json}')


if __name__ == '__main__':
    main()
