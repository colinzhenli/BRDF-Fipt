#!/usr/bin/env python3
"""Check whether the per-pixel scalar response is consistent across the three
Bonn modalities (poly, pan, LLS) at the *same* surface point and (close)
geometry.

For a chosen pixel:
  - poly: scalar = RGB(observed) · pan_weights[cv, il]  (per-image calibrated)
  - pan : scalar = directly recorded
  - lls : scalar = directly recorded; treat the LLS source either as a point
                   at the quad CENTER, or via Monte-Carlo over the quad
                   (cosθ/r²-weighted average of corner samples).

Each measurement carries (theta_i, phi_i, theta_o, phi_o) defined w.r.t.
the local surface normal at that pixel.  We then:

  1. Save 3D hemisphere scatter plots (one per modality) — light position
     colored by scalar response.
  2. Save a polar (theta_i vs scalar) projection, colored by modality.
  3. For each pair (poly,pan), (poly,lls), (pan,lls), find K-nearest
     neighbour matches in (light direction, view direction) space and
     report the median / 90th-percentile of relative differences.

Outputs land in:
    /media/raid/cloth/output/BRDF/diagnostics/brdf_consistency/mat0001_pix{...}/
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import utils.dataset.bonn as bonn_mod
from utils.dataset.bonn import (
    _load_single_material_full,
    DTYPE_POLY, DTYPE_PAN, DTYPE_LLS,
)

DATA_ROOT = '/media/raid/cloth/Bonn_train'
OUT_BASE  = Path('/media/raid/cloth/output/BRDF/diagnostics/brdf_consistency')

MAT_ID    = 1                # which material
PIXEL_RC  = (256, 256)       # which pixel (row, col); center of 512x512 grid
LLS_SPP   = 16               # MC samples on the LLS quad


# ----------------------------------------------------------------------
# Geometry helpers (operate per-image at one pixel)
# ----------------------------------------------------------------------
def _normalize(v, axis=-1, eps=1e-8):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def _project_to_local(vec_world, normal):
    """Return (theta, phi) of vec_world in a local frame whose +z is `normal`.

    A simple frame with arbitrary tangent is sufficient for visualization.
    """
    n = _normalize(normal)
    # arbitrary tangent: pick world +x then orthogonalize; if degenerate use +y
    t = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(t, n)) > 0.9:
        t = np.array([0.0, 1.0, 0.0])
    t = _normalize(t - np.dot(t, n) * n)
    b = np.cross(n, t)
    x = np.dot(vec_world, t)
    y = np.dot(vec_world, b)
    z = np.dot(vec_world, n)
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi   = np.arctan2(y, x)
    return theta, phi, np.array([x, y, z])


def _quad_area(corners):
    """Area of a planar quad given by 4 corners (3D)."""
    a = corners[1] - corners[0]
    b = corners[3] - corners[0]
    c = corners[1] - corners[2]
    d = corners[3] - corners[2]
    return 0.5 * (np.linalg.norm(np.cross(a, b)) + np.linalg.norm(np.cross(c, d)))


def _lls_geom_reweight(corners, xyz_p, normal, n_samples=64, seed=0):
    """Return per-image multiplicative factor for converting a strip
    response to a "centroid-equivalent" response.

    The strip is treated as an area emitter with the same per-unit-emittance
    intensity as the point LED.  The centroid-vs-strip irradiance ratio at
    the surface point is then independent of the strip area:

        factor = (cos_c / r_c²) / mean_quad(cos / r²)

    Multiplying the recorded LLS scalar by this factor "moves" the apparent
    irradiance to what a point light at the strip centroid would produce.
    """
    p_c = corners.mean(axis=0)
    L_c = _normalize(p_c - xyz_p)
    r2_c = max(float(((p_c - xyz_p) ** 2).sum()), 1e-12)
    cos_c = max(float(np.dot(L_c, normal)), 0.0)
    centroid_term = cos_c / r2_c

    rng = np.random.RandomState(seed)
    uv = rng.uniform(size=(n_samples, 2))
    top = corners[0] * (1 - uv[:, :1]) + corners[1] * uv[:, :1]
    bot = corners[3] * (1 - uv[:, :1]) + corners[2] * uv[:, :1]
    pts = top * (1 - uv[:, 1:]) + bot * uv[:, 1:]
    dirs = pts - xyz_p[None, :]
    r2 = np.maximum((dirs ** 2).sum(axis=1), 1e-12)
    dirs_n = dirs / np.sqrt(r2)[:, None]
    cos_t = np.maximum((dirs_n * normal[None, :]).sum(axis=1), 0.0)
    mean_w = float((cos_t / r2).mean())
    A = _quad_area(corners)
    if mean_w <= 1e-12 or centroid_term <= 1e-12:
        return 1.0, centroid_term, mean_w, A
    return centroid_term / mean_w, centroid_term, mean_w, A


# ----------------------------------------------------------------------
def main():
    mat = _load_single_material_full(DATA_ROOT, MAT_ID, use_pan=True, use_lls=True)
    if mat is None:
        raise RuntimeError(f'failed to load mat{MAT_ID:04d}')

    H, W = mat['H'], mat['W']
    # Adjust pixel if outside this material's grid
    r = min(PIXEL_RC[0], H - 1)
    c = min(PIXEL_RC[1], W - 1)
    p = r * W + c
    xyz_p = mat['xyz'][p]

    K_poly = mat['rgbs'].shape[0]
    K_gray = mat['gray_vals'].shape[0] if mat['gray_vals'] is not None else 0
    K_total = K_poly + K_gray

    out_dir = OUT_BASE / f'mat{MAT_ID:04d}_pix_r{r}_c{c}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'pixel (r={r}, c={c})  ->  xyz={xyz_p}')
    print(f'images: poly={K_poly}, gray={K_gray}, total={K_total}')
    print(f'output: {out_dir}')

    # Approx surface normal: assume the sample is roughly flat with normal +z
    # (Bonn fabrics are flat patches).  Check via gt_normals if available.
    if mat['gt_normals'] is not None:
        normal = mat['gt_normals'][p]
        normal = _normalize(normal)
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    print(f'surface normal at pixel: {normal}')

    # Containers per modality
    records = {DTYPE_POLY: [], DTYPE_PAN: [], DTYPE_LLS: []}

    for k in range(K_total):
        dt = int(mat['data_type'][k])

        # Light direction = light_pos - xyz, normalized
        if dt == DTYPE_LLS:
            corners = mat['lls_corners'][k]            # (4, 3)
            light_pos = corners.mean(axis=0)           # quad center
        else:
            light_pos = mat['light_pos'][k]
        cam_pos = mat['cam_pos'][k]

        L_world = _normalize(light_pos - xyz_p)
        V_world = _normalize(cam_pos   - xyz_p)
        theta_i, phi_i, _ = _project_to_local(L_world, normal)
        theta_o, phi_o, _ = _project_to_local(V_world, normal)

        # Scalar value at the pixel
        if dt == DTYPE_POLY:
            rgb = mat['rgbs'][k, p].astype(np.float32)  # (3,)
            w   = mat['pan_weights'][k]
            val = float((rgb * w).sum())
            val_rw = val
            geom_factor = 1.0
            cterm = iterm = A_strip = float('nan')
        else:
            li = k - K_poly
            val = float(mat['gray_vals'][li, p])
            if dt == DTYPE_LLS:
                geom_factor, cterm, iterm, A_strip = _lls_geom_reweight(
                    mat['lls_corners'][k], xyz_p, normal,
                    n_samples=LLS_SPP * 4, seed=k,
                )
                val_rw = val * geom_factor
            else:
                val_rw = val
                geom_factor = 1.0
                cterm = iterm = A_strip = float('nan')

        records[dt].append({
            'L': L_world, 'V': V_world,
            'theta_i': theta_i, 'phi_i': phi_i,
            'theta_o': theta_o, 'phi_o': phi_o,
            'val': val,
            'val_rw': val_rw,
            'geom_factor': geom_factor,
            'centroid_term': cterm,
            'integrated_term': iterm,
            'A_strip': A_strip,
        })

    # Convert to arrays
    arrs = {}
    for dt, recs in records.items():
        if not recs:
            arrs[dt] = None
            continue
        arrs[dt] = {
            'L':         np.stack([r['L'] for r in recs]),
            'V':         np.stack([r['V'] for r in recs]),
            'theta_i':   np.array([r['theta_i'] for r in recs]),
            'phi_i':     np.array([r['phi_i'] for r in recs]),
            'theta_o':   np.array([r['theta_o'] for r in recs]),
            'phi_o':     np.array([r['phi_o'] for r in recs]),
            'val':       np.array([r['val'] for r in recs]),
            'val_rw':    np.array([r['val_rw'] for r in recs]),
            'geom_factor': np.array([r['geom_factor'] for r in recs]),
        }
        print(f'  modality={dt}: n={len(recs)}, '
              f'val[min/median/max]='
              f'{arrs[dt]["val"].min():.4f} / '
              f'{np.median(arrs[dt]["val"]):.4f} / '
              f'{arrs[dt]["val"].max():.4f}'
              f'   val_rw[min/median/max]='
              f'{arrs[dt]["val_rw"].min():.4f} / '
              f'{np.median(arrs[dt]["val_rw"]):.4f} / '
              f'{arrs[dt]["val_rw"].max():.4f}')

    # Print LLS reweight diagnostics
    if arrs[DTYPE_LLS] is not None:
        gf = arrs[DTYPE_LLS]['geom_factor']
        print(f'\n  LLS geom-factor (centroid/integrated): '
              f'min={gf.min():.4g}  median={np.median(gf):.4g}  '
              f'max={gf.max():.4g}  mean={gf.mean():.4g}')

    # ----------------------------------------------------------------
    # Plot 1 — 3D hemisphere scatter, one panel per modality
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(18, 6))
    titles = {DTYPE_POLY: 'POLY (RGB·pan_w)', DTYPE_PAN: 'PAN', DTYPE_LLS: 'LLS (quad center)'}
    # shared colour scale
    vmin = min(a['val'].min() for a in arrs.values() if a is not None)
    vmax = max(np.percentile(a['val'], 99) for a in arrs.values() if a is not None)
    for i, dt in enumerate([DTYPE_POLY, DTYPE_PAN, DTYPE_LLS]):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        a = arrs[dt]
        if a is None:
            ax.set_title(f'{titles[dt]} (no data)')
            continue
        sc = ax.scatter(a['L'][:, 0], a['L'][:, 1], a['L'][:, 2],
                        c=a['val'], cmap='viridis', vmin=vmin, vmax=vmax, s=40)
        # draw normal arrow
        ax.quiver(0, 0, 0, normal[0], normal[1], normal[2], color='red', linewidth=2)
        ax.set_xlabel('Lx'); ax.set_ylabel('Ly'); ax.set_zlabel('Lz')
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-0.1, 1.1)
        ax.set_title(f'{titles[dt]} (n={len(a["val"])})')
        plt.colorbar(sc, ax=ax, shrink=0.6, label='scalar value')
    fig.suptitle(f'mat{MAT_ID:04d}, pixel (r={r}, c={c})  '
                 f'— light directions colored by scalar response')
    plt.tight_layout()
    fig.savefig(out_dir / 'hemisphere_3d.png', dpi=140)
    plt.close(fig)

    # ----------------------------------------------------------------
    # Plot 2 — polar (theta_i, val) overlay across modalities
    # Two panels: raw values vs LLS-reweighted values
    # ----------------------------------------------------------------
    color = {DTYPE_POLY: 'tab:blue', DTYPE_PAN: 'tab:orange', DTYPE_LLS: 'tab:green'}
    label = {DTYPE_POLY: 'poly (RGB·w)', DTYPE_PAN: 'pan', DTYPE_LLS: 'lls (centroid)'}
    fig, axes = plt.subplots(1, 2, figsize=(18, 9),
                             subplot_kw={'projection': 'polar'})
    for ax, vkey, sub in zip(axes, ('val', 'val_rw'),
                             ('LLS raw', 'LLS reweighted (centroid-equiv)')):
        for dt in [DTYPE_POLY, DTYPE_PAN, DTYPE_LLS]:
            a = arrs[dt]
            if a is None:
                continue
            signed_theta = np.where(np.cos(a['phi_i']) >= 0,
                                    a['theta_i'], -a['theta_i'])
            ax.scatter(signed_theta, a[vkey], s=18, alpha=0.6,
                       c=color[dt], label=label[dt])
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)
        ax.set_thetamin(-90); ax.set_thetamax(90)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title(sub)
    fig.suptitle(f'Scalar response vs theta_i (signed)  '
                 f'mat{MAT_ID:04d}, pixel (r={r}, c={c})')
    plt.tight_layout()
    fig.savefig(out_dir / 'polar_theta_i_vs_value.png', dpi=140)
    plt.close(fig)

    # ----------------------------------------------------------------
    # Plot 3 — scalar vs theta_h (half-vector elevation)
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, vkey, sub in zip(axes, ('val', 'val_rw'),
                             ('LLS raw', 'LLS reweighted (centroid-equiv)')):
        for dt in [DTYPE_POLY, DTYPE_PAN, DTYPE_LLS]:
            a = arrs[dt]
            if a is None:
                continue
            H_dir = _normalize(a['L'] + a['V'], axis=1)
            cos_h = (H_dir * normal[None, :]).sum(axis=1).clip(-1, 1)
            theta_h = np.degrees(np.arccos(cos_h))
            ax.scatter(theta_h, a[vkey], s=18, alpha=0.6,
                       c=color[dt], label=label[dt])
        ax.set_xlabel('theta_h (deg)')
        ax.set_ylabel('scalar value')
        ax.set_yscale('log')
        ax.legend()
        ax.set_title(sub)
    fig.suptitle(f'Scalar vs half-vector elevation  '
                 f'mat{MAT_ID:04d}, pixel (r={r}, c={c})')
    fig.savefig(out_dir / 'scatter_theta_h_vs_value.png', dpi=140)
    plt.close(fig)

    # ----------------------------------------------------------------
    # Quantitative consistency: nearest-neighbour matching
    # Match each measurement of source A to its nearest measurement of
    # source B in (L, V) joint cosine distance, then report relative diff.
    # ----------------------------------------------------------------
    def _knn_match(a, b, k=1):
        """For each row in a return idx in b minimizing 1 - 0.5*(L·L' + V·V')."""
        L_dot = a['L'] @ b['L'].T          # (Na, Nb)
        V_dot = a['V'] @ b['V'].T
        sim   = 0.5 * (L_dot + V_dot)
        dist  = 1.0 - sim                  # in [0, 2]
        idx   = np.argmin(dist, axis=1)
        d_best = np.take_along_axis(dist, idx[:, None], axis=1).squeeze(1)
        return idx, d_best

    def _stats(name, a, b, vkey='val', max_dist=0.05):
        """Median / 90p relative diff for matches within angular threshold."""
        idx, d = _knn_match(a, b)
        keep = d < max_dist
        if not keep.any():
            print(f'  {name}: no matches within angular threshold {max_dist}')
            return
        va = a[vkey][keep]
        vb = b[vkey][idx[keep]]
        denom = np.maximum(0.5 * (np.abs(va) + np.abs(vb)), 1e-6)
        rel = np.abs(va - vb) / denom
        ratio = (va + 1e-9) / (vb + 1e-9)
        print(f'  {name} (n_matches={keep.sum()}):')
        print(f'    rel-diff: median={np.median(rel)*100:.1f}%  '
              f'90p={np.percentile(rel, 90)*100:.1f}%  '
              f'mean={rel.mean()*100:.1f}%')
        print(f'    ratio  A/B: median={np.median(ratio):.3f}  '
              f'10–90p=[{np.percentile(ratio, 10):.3f}, '
              f'{np.percentile(ratio, 90):.3f}]')

    def _all_stats(vkey, header):
        print(f'\n=== {header} ===')
        if arrs[DTYPE_POLY] is not None and arrs[DTYPE_PAN] is not None:
            _stats('poly→pan', arrs[DTYPE_POLY], arrs[DTYPE_PAN], vkey)
            _stats('pan→poly', arrs[DTYPE_PAN], arrs[DTYPE_POLY], vkey)
        if arrs[DTYPE_POLY] is not None and arrs[DTYPE_LLS] is not None:
            _stats('poly→lls', arrs[DTYPE_POLY], arrs[DTYPE_LLS], vkey)
            _stats('lls→poly', arrs[DTYPE_LLS], arrs[DTYPE_POLY], vkey)
        if arrs[DTYPE_PAN] is not None and arrs[DTYPE_LLS] is not None:
            _stats('pan→lls',  arrs[DTYPE_PAN],  arrs[DTYPE_LLS], vkey)
            _stats('lls→pan',  arrs[DTYPE_LLS],  arrs[DTYPE_PAN], vkey)

    _all_stats('val',    'Quantitative consistency — RAW LLS values')
    _all_stats('val_rw', 'Quantitative consistency — LLS PER-IMAGE GEOM '
                         'reweight (strip→centroid-equivalent irradiance)')

    # ----------------------------------------------------------------
    # Reusable full-angular-consistency analyzer
    # ----------------------------------------------------------------
    def _within_modality_baseline(dt, vkey='val'):
        a = arrs[dt]
        if a is None or a[vkey].size < 4:
            return None
        n = a[vkey].size
        perm = np.random.RandomState(0).permutation(n)
        half = n // 2
        slc1 = {k: v[perm[:half]] for k, v in a.items() if isinstance(v, np.ndarray)}
        slc2 = {k: v[perm[half:]] for k, v in a.items() if isinstance(v, np.ndarray)}
        i2, d2 = _knn_match(slc1, slc2)
        keep = d2 < 0.05
        if not keep.any():
            return None
        va = slc1[vkey][keep]; vb = slc2[vkey][i2[keep]]
        rel = np.abs(va - vb) / np.maximum(0.5*(np.abs(va)+np.abs(vb)), 1e-6)
        return rel

    def _angular_analysis(label, A, B, vkey_a, vkey_b, plot_tag,
                          baselines=()):
        """Run threshold sweep + theta_h binning + CDF/log-log plots for
        the matched pair (A, B).  `baselines` is a list of (name, rel_array)
        for within-modality CDF overlays."""
        print(f'\n=== FULL angular consistency: {label} ===')
        idx_all, d_all = _knn_match(A, B)
        v_a = A[vkey_a]
        v_b = B[vkey_b][idx_all]
        denom = np.maximum(0.5 * (np.abs(v_a) + np.abs(v_b)), 1e-6)
        rel_all = np.abs(v_a - v_b) / denom
        ratio_all = (v_a + 1e-9) / (v_b + 1e-9)

        print('  threshold sweep:    n   med_rel%   90p%   ratio_med')
        for thr in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
            keep_t = d_all < thr
            if keep_t.sum() < 3:
                continue
            print(f'    cosdist<{thr:5.3f}: '
                  f'n={keep_t.sum():4d}   '
                  f'med={np.median(rel_all[keep_t])*100:5.1f}   '
                  f'90p={np.percentile(rel_all[keep_t], 90)*100:6.1f}   '
                  f'ratio_med={np.median(ratio_all[keep_t]):.3f}')

        keep_main = d_all < 0.05
        if keep_main.any():
            H = _normalize(A['L'] + A['V'], axis=1)
            cos_h = (H * normal[None, :]).sum(axis=1).clip(-1, 1)
            theta_h_deg = np.degrees(np.arccos(cos_h))
            print('  rel-diff binned by source-A theta_h (deg):')
            edges = [0, 5, 10, 20, 30, 45, 60, 90]
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = keep_main & (theta_h_deg >= lo) & (theta_h_deg < hi)
                if m.sum() >= 3:
                    print(f'    {lo:3d}–{hi:3d}°  n={m.sum():3d}  '
                          f'med_rel={np.median(rel_all[m])*100:5.1f}%  '
                          f'90p={np.percentile(rel_all[m], 90)*100:6.1f}%  '
                          f'ratio_med={np.median(ratio_all[m]):.3f}')

            # CDF plot
            fig, ax = plt.subplots(figsize=(8, 6))
            x = np.sort(rel_all[keep_main]); y = np.linspace(0, 1, len(x))
            ax.plot(x * 100, y, lw=2, color='tab:purple',
                    label=f'{label}  (n={keep_main.sum()})')
            for bname, brel, bcol in baselines:
                if brel is not None:
                    bx = np.sort(brel); by = np.linspace(0, 1, len(bx))
                    ax.plot(bx * 100, by, lw=1.5, ls='--',
                            color=bcol, label=f'{bname}  (n={len(brel)})')
            ax.set_xlabel('relative difference (%)')
            ax.set_ylabel('CDF')
            ax.set_xlim(0, 200); ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.legend(loc='lower right')
            ax.set_title(f'CDF of rel-diff at matched (L,V), cosdist<0.05\n'
                         f'mat{MAT_ID:04d}, pixel (r={r}, c={c})')
            fig.savefig(out_dir / f'rel_diff_cdf_{plot_tag}.png',
                        dpi=140, bbox_inches='tight')
            plt.close(fig)

            # log-log scatter
            fig, ax = plt.subplots(figsize=(7, 7))
            sc = ax.scatter(v_b[keep_main], v_a[keep_main],
                            c=theta_h_deg[keep_main], s=24, alpha=0.7,
                            cmap='viridis')
            plt.colorbar(sc, ax=ax, label='theta_h (deg)')
            lo = max(min(v_b[keep_main].min(), v_a[keep_main].min()), 1e-4)
            hi = max(v_b[keep_main].max(), v_a[keep_main].max())
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='y = x')
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('source-B value')
            ax.set_ylabel('source-A value')
            ax.legend()
            ax.set_title(f'{label}  (n={keep_main.sum()})\n'
                         f'mat{MAT_ID:04d}, pixel (r={r}, c={c})')
            fig.savefig(out_dir / f'scatter_loglog_{plot_tag}.png',
                        dpi=140, bbox_inches='tight')
            plt.close(fig)

    # =======================================================
    # PASS 1: pan ↔ poly (sanity — should be near noise floor)
    # =======================================================
    if arrs[DTYPE_PAN] is not None and arrs[DTYPE_POLY] is not None:
        rel_pan_self  = _within_modality_baseline(DTYPE_PAN,  'val')
        rel_poly_self = _within_modality_baseline(DTYPE_POLY, 'val')
        baselines = [('pan-self',  rel_pan_self,  'tab:orange'),
                     ('poly-self', rel_poly_self, 'tab:blue')]
        _angular_analysis('pan ↔ poly  (no scaling)',
                          arrs[DTYPE_PAN], arrs[DTYPE_POLY],
                          'val', 'val', 'pan_vs_poly',
                          baselines=baselines)
        _angular_analysis('poly ↔ pan  (no scaling)',
                          arrs[DTYPE_POLY], arrs[DTYPE_PAN],
                          'val', 'val', 'poly_vs_pan',
                          baselines=baselines)

    # Empirical constant scale: pick c minimizing median |lls·c - ref_match|/avg
    # using the lls→(pan+poly) KNN matches.  We use the COMBINED reference
    # (pan + poly, since poly is converted to scalar via pan_weights too) to
    # avoid being biased by one source.
    ref = None
    if arrs[DTYPE_PAN] is not None and arrs[DTYPE_POLY] is not None:
        ref = {
            'L':   np.concatenate([arrs[DTYPE_PAN]['L'],   arrs[DTYPE_POLY]['L']]),
            'V':   np.concatenate([arrs[DTYPE_PAN]['V'],   arrs[DTYPE_POLY]['V']]),
            'val': np.concatenate([arrs[DTYPE_PAN]['val'], arrs[DTYPE_POLY]['val']]),
        }
    elif arrs[DTYPE_PAN] is not None:
        ref = arrs[DTYPE_PAN]

    if arrs[DTYPE_LLS] is not None and ref is not None:
        idx, d = _knn_match(arrs[DTYPE_LLS], ref)
        keep = d < 0.05
        v_lls = arrs[DTYPE_LLS]['val'][keep]
        v_ref = ref['val'][idx[keep]]
        c_med = float(np.median(v_ref / np.maximum(v_lls, 1e-9)))
        arrs[DTYPE_LLS]['val_const'] = arrs[DTYPE_LLS]['val'] * c_med
        for dt in (DTYPE_POLY, DTYPE_PAN):
            if arrs[dt] is not None:
                arrs[dt]['val_const'] = arrs[dt]['val']
        print(f'\n[empirical constant LLS scale chosen: c = {c_med:.4f} '
              f'(fitted on {keep.sum()} lls↔pan+poly matches)]')
        _all_stats('val_const',
                   f'LLS scaled by CONSTANT c={c_med:.3f}')

        # ============================================================
        # PASS 2: LLS·c ↔ pan+poly (after radiometric scaling)
        # ============================================================
        ref_arr = {**ref}
        ref_arr['val_const'] = ref['val']  # ensure key exists
        rel_pan_self  = _within_modality_baseline(DTYPE_PAN,  'val')
        rel_poly_self = _within_modality_baseline(DTYPE_POLY, 'val')
        rel_lls_self  = _within_modality_baseline(DTYPE_LLS,  'val')
        baselines = [('pan-self',  rel_pan_self,  'tab:orange'),
                     ('poly-self', rel_poly_self, 'tab:blue'),
                     ('lls-self',  rel_lls_self,  'tab:green')]
        _angular_analysis(f'LLS·{c_med:.3f} ↔ pan+poly',
                          arrs[DTYPE_LLS], ref_arr,
                          'val_const', 'val_const', 'lls_vs_panpoly',
                          baselines=baselines)

    # Same-modality baseline (sanity floor): split each modality randomly
    # and KNN-match its halves; rel-diff here is the within-source noise.
    print('\n=== Within-modality baseline (random 50/50 split) ===')
    rng = np.random.RandomState(0)
    for dt, name in [(DTYPE_POLY, 'poly'), (DTYPE_PAN, 'pan'), (DTYPE_LLS, 'lls')]:
        a = arrs[dt]
        if a is None or a['val'].size < 4:
            continue
        n = a['val'].size
        perm = rng.permutation(n)
        half = n // 2
        slc1 = {k: v[perm[:half]] for k, v in a.items() if isinstance(v, np.ndarray)}
        slc2 = {k: v[perm[half:]] for k, v in a.items() if isinstance(v, np.ndarray)}
        _stats(f'{name}-self (raw)',         slc1, slc2, 'val')
        if dt == DTYPE_LLS:
            _stats(f'{name}-self (rw)',  slc1, slc2, 'val_rw')

    print(f'\n[done] saved visualizations to {out_dir}')


if __name__ == '__main__':
    main()
