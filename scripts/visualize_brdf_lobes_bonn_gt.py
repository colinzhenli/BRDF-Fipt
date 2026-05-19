#!/usr/bin/env python3
"""Ground-truth BRDF lobe visualizations for the same random latent points
the trainer's ``visualize_brdf_lobe`` sweeps, with three-source overlays.

Reproduces the exact set of polar plots written by
``Stage1Trainer_Bonn.visualize_brdf_lobe`` (10 random points × {fix wi /
vary wo} + {fix wo / vary wi at 5 (wi_phi, wo_phi) configs}), but driven
by the GT poly/pan/lls measurements instead of the trained decoder. Each
plot overlays one curve per source so the calibration and lobe shape can
be compared directly against the decoder's output.

Pipeline:
  * Read ``bonn_point_metadata.json``, apply the same point_subsample_ratio
    that the model uses, and reconstruct the global → (mat_id, local_pid)
    offset table.
  * Seed-match the trainer (``torch.manual_seed(42); randint(0, total)``).
  * For each global pid: resolve to (mat_id, row, col), group by mat_id so
    each material is loaded once via ``_load_single_material_full``.
  * For each (mat, pixel) compute per-view (θ_i, φ_i, θ_o, φ_o) in the
    per-pixel local frame (GT normal + projected world +X tangent), and
    per-view scalar value in pan-equivalent grayscale:
        poly: (rgb · pan_weights).sum()
        pan : raw gray value
        lls : raw gray value  (no _LLS_EMPIRICAL_SCALE divide anymore)
  * Bin views per (θ target, azimuth band) and aggregate the polar
    line along signed-θ using the same convention as
    ``plot_phi_binned_slices_per_pixel`` so the GT curves are smooth.

Outputs are written to ``<out_dir>/gt_brdf_lobes/`` with file names that
mirror the trainer's:
    gt_polar_brdf_vary_wo_pt_<pid>.png            (1×5 grid: θ_i targets)
    gt_polar_brdf_vary_wi_pt_<pid>_wiphi<NNN>_wophi<NNN>.png   (1×4 grid: θ_o targets)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch  # only used to match the trainer's seeded sample

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.dataset.bonn import (
    _load_single_material_full,
    DTYPE_POLY, DTYPE_PAN, DTYPE_LLS,
)


PALETTE = {'poly': 'C0', 'pan': 'C1', 'lls': 'C2'}
WORLD_X = np.array([1.0, 0.0, 0.0], dtype=np.float32)
WORLD_Y = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def load_metadata(data_folder, point_subsample_ratio=1.0,
                  debug=False, debug_num=1):
    """Mirror BonnLatentBRDF._load_point_metadata to build the same offset table."""
    meta_path = Path(data_folder) / 'bonn_point_metadata.json'
    with open(meta_path) as f:
        raw = json.load(f)
    keys = sorted(raw.keys(), key=int)
    if debug:
        keys = keys[:debug_num]
    materials, total = [], 0
    for k in keys:
        n = raw[k]['num_points']
        if point_subsample_ratio < 1.0:
            n = max(1, int(n * point_subsample_ratio))
        materials.append({
            'mat_id':   int(k),
            'H':        int(raw[k]['H']),
            'W':        int(raw[k]['W']),
            'num_points': n,
            'offset':   total,
        })
        total += n
    return materials, total


def resolve_global_pid(materials, pid):
    for m in materials:
        if m['offset'] <= pid < m['offset'] + m['num_points']:
            local_pid = pid - m['offset']
            return m['mat_id'], local_pid, m['H'], m['W']
    return None


def local_frame(normal):
    """Build a tangent-space basis matching the trainer's MLP frame.

    The trainer's ``eval_brdf`` calls ``world_to_local(wi, predicted_normal,
    predicted_tangent)`` where ``predicted_tangent = latent[-3:]`` is
    initialised to ``(0, 1, 0)``. So during training the MLP sees
    ``wi_local = (wi·tangent, wi·bitangent, wi·normal)`` with
    ``tangent ≈ +Y`` (projected onto the tangent plane for tilted normals).

    The trainer's ``visualize_brdf_lobe`` bypasses ``world_to_local`` and
    feeds ``wi_canon`` straight into the MLP, so the viz's "canonical
    frame" IS that latent-tangent frame. To put GT views in the same
    frame, we project world ``+Y`` (not ``+X``) onto the tangent plane.
    Earlier (``+X`` projection) caused a 90° azimuth rotation between
    GT and decoder lobes — visible as left/right peak swaps.
    """
    n = normal / max(np.linalg.norm(normal), 1e-8)
    t = WORLD_Y - (WORLD_Y @ n) * n   # tangent ≈ +Y (latent's initial tangent)
    if np.linalg.norm(t) < 1e-6:
        t = WORLD_X - (WORLD_X @ n) * n
    t = t / np.linalg.norm(t)
    b = np.cross(n, t)
    return n, t, b


def normalize(v, axis=-1):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-8)


def per_view_geometry(mat, pixel_idx, use_gt_normal=False):
    """Returns dict of arrays for all views at this pixel:
    source ('poly'|'pan'|'lls'), L, V, val (pan-equiv),
    theta_i, phi_i, theta_o, phi_o.

    Frame:
      * use_gt_normal=False (default): directions are in WORLD frame, i.e.
        surface normal is assumed to be (0, 0, 1). Matches the trainer's
        ``visualize_brdf_lobe`` which evaluates the decoder with
        ``local_normal = (0, 0, 1)`` regardless of the per-point GT normal.
      * use_gt_normal=True: rotate (L, V) into the per-pixel local frame
        built from the GT surface normal (from svfresnel/<mat>_Normal.exr,
        loaded via ``_read_gt_normal_map``) + projected world +X tangent.
        Then θ_i/θ_o are angles from the GT normal, φ_i/φ_o are in the
        tangent plane. This is the "correct" geometry for the BRDF lobe at
        the actual surface point but the rig coverage rotates with the
        normal, so the latent's intrinsic-frame and the trainer's plot may
        appear shifted in azimuth.
    """
    K_poly = mat['rgbs'].shape[0] if mat['rgbs'] is not None else 0
    gray = mat['gray_vals']
    K_gray = gray.shape[0] if gray is not None else 0
    K_total = K_poly + K_gray

    xyz = mat['xyz'][pixel_idx]

    if use_gt_normal:
        if mat['gt_normals'] is None:
            raise ValueError('use_gt_normal requested but mat["gt_normals"] is '
                             'None. Make sure Bonn_svfresnel/<matID>_svfresnel/'
                             '<matID>_svfresnel_Normal.exr exists for this '
                             'material.')
        n_pt = mat['gt_normals'][pixel_idx]
        if np.linalg.norm(n_pt) < 1e-8:
            n_pt = np.array([0., 0., 1.], dtype=np.float32)
        n, t, b = local_frame(n_pt)
        # 3×3 world→local rotation: columns are t, b, n so that
        # (Lw @ frame)[i] = Lw · {t,b,n}_i.
        frame = np.stack([t, b, n], axis=1).astype(np.float32)
    else:
        frame = None

    dtype = mat['data_type']
    source = np.empty(K_total, dtype=object)
    L = np.zeros((K_total, 3), dtype=np.float32)
    V = np.zeros((K_total, 3), dtype=np.float32)
    val = np.zeros(K_total, dtype=np.float32)

    for k in range(K_total):
        if dtype[k] == DTYPE_LLS:
            light_pos = mat['lls_corners'][k].mean(axis=0)
            source[k] = 'lls'
        else:
            light_pos = mat['light_pos'][k]
            source[k] = 'poly' if dtype[k] == DTYPE_POLY else 'pan'
        cam_pos = mat['cam_pos'][k]
        Lw = normalize(light_pos - xyz)
        Vw = normalize(cam_pos   - xyz)
        if frame is not None:
            L[k] = Lw @ frame
            V[k] = Vw @ frame
        else:
            L[k] = Lw
            V[k] = Vw

        if dtype[k] == DTYPE_POLY:
            rgb = mat['rgbs'][k, pixel_idx].astype(np.float32)
            w   = mat['pan_weights'][k]
            val[k] = float((rgb * w).sum())
        else:
            li = k - K_poly
            val[k] = float(gray[li, pixel_idx])

    np.clip(val, 0, None, out=val)
    L = L.astype(np.float32)
    V = V.astype(np.float32)
    theta_i = np.arccos(np.clip(L[:, 2], -1, 1))
    phi_i = np.arctan2(L[:, 1], L[:, 0])
    theta_o = np.arccos(np.clip(V[:, 2], -1, 1))
    phi_o = np.arctan2(V[:, 1], V[:, 0])
    return {
        'source': source, 'L': L, 'V': V, 'val': val,
        'theta_i': theta_i, 'phi_i': phi_i,
        'theta_o': theta_o, 'phi_o': phi_o,
    }


def cyclic_dist_deg(a, b):
    """Smallest distance between two angles in degrees (a, b can be arrays)."""
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def signed_theta_in_axis(swept_local, axis):
    """Signed θ of swept_local relative to a fixed azimuth direction `axis`
    (unit 3-vec in the local frame with axis_z = 0). Mirrors the trainer's
    sweep parameterisation: positive signed θ → swept on the opposite-azimuth
    side from `axis` (e.g. axis = wo_axis → positive signed θ_i means wi is
    on the specular side of wo).
    """
    forward = -axis  # (3,)
    in_plane = swept_local[:, 0] * forward[0] + swept_local[:, 1] * forward[1]
    return np.arctan2(in_plane, swept_local[:, 2])


def bin_and_aggregate(signed_theta, y, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(signed_theta, edges) - 1, 0, len(edges) - 2)
    means = np.full(len(centers), np.nan)
    for k in range(len(centers)):
        sel = idx == k
        if sel.any():
            means[k] = y[sel].mean()
    valid = ~np.isnan(means)
    return centers[valid], means[valid]


def plot_vary_wo(pid, mat_id, row, col, vd, out_path,
                 theta_i_targets=(15., 30., 45., 60., 75.),
                 theta_tol_deg=10., phi_tol_deg=25., n_theta_bins=18,
                 return_counts=True):
    """Vis 1: 1×N polar grid. Each panel = fix θ_i target (wi in x-z plane).
    Every view is plotted as one source-coloured marker; a single grey line
    runs through the pooled bin-mean across all three sources so the
    consensus lobe shape is visible alongside per-source disagreement."""
    edges = np.linspace(-np.pi / 2, np.pi / 2, n_theta_bins + 1)
    N = len(theta_i_targets)
    fig, axes = plt.subplots(1, N, figsize=(4.5 * N, 4.7),
                             subplot_kw={'projection': 'polar'},
                             squeeze=False)
    axes = axes[0]
    phi_deg = np.degrees(vd['phi_i'])
    theta_deg_i = np.degrees(vd['theta_i'])
    # wi_phi = 0 means wi in x-z plane: phi_i ≈ 0 or ±180.
    phi_band = np.minimum(cyclic_dist_deg(phi_deg, 0.0),
                          cyclic_dist_deg(phi_deg, 180.0))
    n_empty = 0
    for ai, theta_i_target in enumerate(theta_i_targets):
        ax = axes[ai]
        cell_mask = (np.abs(theta_deg_i - theta_i_target) <= theta_tol_deg) \
                    & (phi_band <= phi_tol_deg)
        pooled_signed, pooled_y = [], []
        any_data = False
        for src in ['poly', 'pan', 'lls']:
            sel = cell_mask & (vd['source'] == src)
            if not sel.any():
                continue
            wi_g = vd['L'][sel]
            wo_g = vd['V'][sel]
            y_g  = vd['val'][sel]
            signed_o = np.arctan2(-wo_g[:, 0] * wi_g[:, 0]
                                  - wo_g[:, 1] * wi_g[:, 1],
                                  wo_g[:, 2] * np.sqrt(wi_g[:, 0]**2
                                                       + wi_g[:, 1]**2 + 1e-12))
            ax.scatter(signed_o, y_g, s=18, alpha=0.85,
                       color=PALETTE[src], edgecolors='black', linewidths=0.3,
                       label=f'{src} (n={int(sel.sum())})')
            pooled_signed.append(signed_o)
            pooled_y.append(y_g)
            any_data = True
        if pooled_signed:
            sig_all = np.concatenate(pooled_signed)
            y_all   = np.concatenate(pooled_y)
            # Direct connection (CRIMSON solid): every observation sorted by
            # signed θ. Shows the raw, unaggregated lobe.
            order = np.argsort(sig_all)
            ax.plot(sig_all[order], y_all[order], '-', linewidth=1.0,
                    color='crimson', alpha=0.7,
                    label='raw (sort by θ)' if ai == 0 else None)
            # Fitted binned-mean line (BLACK dotted): smoothed lobe shape.
            xs, ys = bin_and_aggregate(sig_all, y_all, edges)
            if xs.size > 1:
                ax.plot(xs, ys, ':', linewidth=2.0, color='black',
                        alpha=0.9,
                        label='binned mean' if ai == 0 else None)
        ax.axvline(x=np.deg2rad(theta_i_target), color='k',
                   linestyle=':', alpha=0.4)
        ax.set_theta_zero_location('N'); ax.set_theta_direction(1)
        ax.set_thetamin(-90); ax.set_thetamax(90)
        ax.tick_params(axis='y', labelsize=6)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_rlabel_position(75)
        ax.set_title(f'θ_i = {theta_i_target:.0f}° ± {theta_tol_deg:.0f}°',
                     fontsize=11)
        if any_data:
            ax.legend(loc='upper right', bbox_to_anchor=(1.40, 1.05), fontsize=7)
        else:
            n_empty += 1
            ax.text(0, 0.0, '(no data)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='red')
    fig.suptitle(
        f'mat{mat_id:04d}  pixel ({row}, {col})  pid={pid}\n'
        f'GT BRDF (vary wo, wi in x-z plane).  Compare to '
        f'polar_brdf_vary_wo_pt_{pid}.png', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    if return_counts:
        return n_empty, len(theta_i_targets)


def plot_vary_wo_trainer_style(pid, mat_id, row, col, vd, out_path,
                               theta_i_targets=(15., 30., 45., 60., 75.),
                               theta_tol_deg=10., phi_tol_deg=25.,
                               n_theta_bins=18):
    """Single polar plot with 5 θ_i curves overlaid — mirrors the trainer's
    ``polar_brdf_vary_wo_pt_<pid>.png`` exact layout. All 3 sources are
    pooled into one binned curve per θ_i target so the shape is directly
    comparable to the decoder's analytic curve."""
    edges = np.linspace(-np.pi / 2, np.pi / 2, n_theta_bins + 1)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    cmap = plt.cm.tab10
    phi_deg = np.degrees(vd['phi_i'])
    theta_deg_i = np.degrees(vd['theta_i'])
    phi_band = np.minimum(cyclic_dist_deg(phi_deg, 0.0),
                          cyclic_dist_deg(phi_deg, 180.0))
    for ai, theta_i_target in enumerate(theta_i_targets):
        cell_mask = (np.abs(theta_deg_i - theta_i_target) <= theta_tol_deg) \
                    & (phi_band <= phi_tol_deg)
        if not cell_mask.any():
            continue
        wi_g = vd['L'][cell_mask]
        wo_g = vd['V'][cell_mask]
        y_g  = vd['val'][cell_mask]
        signed_o = np.arctan2(-wo_g[:, 0] * wi_g[:, 0]
                              - wo_g[:, 1] * wi_g[:, 1],
                              wo_g[:, 2] * np.sqrt(wi_g[:, 0]**2
                                                   + wi_g[:, 1]**2 + 1e-12))
        col_ = cmap(ai)
        # Direct connection (SOLID, thin) — every observation sorted by θ.
        order = np.argsort(signed_o)
        ax.plot(signed_o[order], y_g[order], '-', linewidth=1.0,
                color=col_, alpha=0.6)
        # Fitted binned-mean (DOTTED, thick, with small markers).
        xs, ys = bin_and_aggregate(signed_o, y_g, edges)
        if xs.size > 1:
            ax.plot(xs, ys, 'o:', linewidth=2.0, markersize=3,
                    color=col_, alpha=0.95,
                    label=f'θ_i={theta_i_target:.0f}° (n={int(cell_mask.sum())})')
        # Specular reference — dashed faint vertical so it's not confused
        # with the solid raw line or the dotted fit.
        ax.axvline(x=np.deg2rad(theta_i_target),
                   color=col_, linestyle='--', alpha=0.4)
    ax.set_theta_zero_location('N'); ax.set_theta_direction(1)
    ax.set_thetamin(-90); ax.set_thetamax(90)
    ax.tick_params(axis='y', labelsize=6)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_rlabel_position(75)
    ax.legend(loc='upper right', bbox_to_anchor=(1.30, 1.0), fontsize=8)
    ax.set_title(
        f'mat{mat_id:04d}  pixel ({row}, {col})  pid={pid}\n'
        f'GT BRDF (vary wo). Pooled poly+pan+lls. '
        f'Compare to polar_brdf_vary_wo_pt_{pid}.png',
        fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_vary_wi(pid, mat_id, row, col, vd, out_path,
                 wi_phi_deg, wo_phi_deg,
                 theta_o_targets=(15., 30., 45., 60.),
                 theta_tol_deg=10., phi_tol_deg=25., n_theta_bins=18,
                 return_counts=True):
    """Vis 2: 1×N polar grid for one (wi_phi, wo_phi) config. Each panel =
    fix θ_o target, sweep signed θ_i; y = scalar × cos(θ_i)."""
    edges = np.linspace(-np.pi / 2, np.pi / 2, n_theta_bins + 1)
    N = len(theta_o_targets)
    fig, axes = plt.subplots(1, N, figsize=(4.5 * N, 4.7),
                             subplot_kw={'projection': 'polar'},
                             squeeze=False)
    axes = axes[0]

    wi_phi_rad = np.deg2rad(wi_phi_deg)
    wo_phi_rad = np.deg2rad(wo_phi_deg)
    wi_axis = np.array([np.cos(wi_phi_rad), np.sin(wi_phi_rad), 0.0],
                       dtype=np.float32)
    wo_axis = np.array([np.cos(wo_phi_rad), np.sin(wo_phi_rad), 0.0],
                       dtype=np.float32)
    phi_o_deg = np.degrees(vd['phi_o'])
    phi_i_deg = np.degrees(vd['phi_i'])
    theta_o_deg = np.degrees(vd['theta_o'])

    # wo aligned with wo_phi.
    phi_o_band = cyclic_dist_deg(phi_o_deg, wo_phi_deg)
    # wi aligned with ±wi_phi (sweep covers both sides of the zenith).
    phi_i_band = np.minimum(cyclic_dist_deg(phi_i_deg, wi_phi_deg),
                            cyclic_dist_deg(phi_i_deg, wi_phi_deg + 180.0))

    n_empty = 0
    for ai, theta_o_target in enumerate(theta_o_targets):
        ax = axes[ai]
        cell_mask = (np.abs(theta_o_deg - theta_o_target) <= theta_tol_deg) \
                    & (phi_o_band <= phi_tol_deg) \
                    & (phi_i_band <= phi_tol_deg)
        pooled_signed, pooled_y = [], []
        any_data = False
        for src in ['poly', 'pan', 'lls']:
            sel = cell_mask & (vd['source'] == src)
            if not sel.any():
                continue
            wi_g = vd['L'][sel]
            y_g  = vd['val'][sel]
            signed_i = signed_theta_in_axis(wi_g, wi_axis)
            cos_i = wi_g[:, 2].clip(0, None)
            y_g_cos = y_g * cos_i  # match trainer's y = BRDF × cos(θ_i)
            ax.scatter(signed_i, y_g_cos, s=18, alpha=0.85,
                       color=PALETTE[src], edgecolors='black', linewidths=0.3,
                       label=f'{src} (n={int(sel.sum())})')
            pooled_signed.append(signed_i)
            pooled_y.append(y_g_cos)
            any_data = True
        if pooled_signed:
            sig_all = np.concatenate(pooled_signed)
            y_all   = np.concatenate(pooled_y)
            # Direct connection (CRIMSON solid) vs fitted binned-mean
            # (BLACK dotted).
            order = np.argsort(sig_all)
            ax.plot(sig_all[order], y_all[order], '-', linewidth=1.0,
                    color='crimson', alpha=0.7,
                    label='raw (sort by θ)' if ai == 0 else None)
            xs, ys = bin_and_aggregate(sig_all, y_all, edges)
            if xs.size > 1:
                ax.plot(xs, ys, ':', linewidth=2.0, color='black',
                        alpha=0.9,
                        label='binned mean' if ai == 0 else None)
        ax.axvline(x=np.deg2rad(theta_o_target), color='k',
                   linestyle=':', alpha=0.4)
        ax.set_theta_zero_location('N'); ax.set_theta_direction(1)
        ax.set_thetamin(-90); ax.set_thetamax(90)
        ax.tick_params(axis='y', labelsize=6)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_rlabel_position(75)
        ax.set_title(f'θ_o = {theta_o_target:.0f}° ± {theta_tol_deg:.0f}°',
                     fontsize=11)
        if any_data:
            ax.legend(loc='upper right', bbox_to_anchor=(1.40, 1.05), fontsize=7)
        else:
            n_empty += 1
            ax.text(0, 0.0, '(no data)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='red')
    fig.suptitle(
        f'mat{mat_id:04d}  pixel ({row}, {col})  pid={pid}   '
        f'wi_φ={wi_phi_deg:.0f}° wo_φ={wo_phi_deg:.0f}°\n'
        f'GT BRDF × cos(θ_i) (vary wi).  Compare to '
        f'polar_brdf_vary_wi_pt_{pid}_wiphi{int(wi_phi_deg):03d}_'
        f'wophi{int(wo_phi_deg):03d}.png', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    if return_counts:
        return n_empty, len(theta_o_targets)


def plot_vary_wi_overlay(pid, mat_id, row, col, vd, out_path,
                         wi_phi_deg, wo_phi_deg,
                         theta_o_targets=(15., 30., 45., 60.),
                         theta_tol_deg=10., phi_tol_deg=25.):
    """Trainer-style single polar overlay: all θ_o curves on one plot.

    Mirrors the decoder's ``polar_brdf_vary_wi_pt_<pid>_wiphi<NNN>_wophi<NNN>.png``
    layout — single polar axis, one direct-connection curve per θ_o target,
    consistent colour per θ_o so curves can be compared 1:1 against the
    decoder lobe of the same colour. Skips the binned/fitted line entirely.
    """
    wi_phi_rad = np.deg2rad(wi_phi_deg)
    wo_phi_rad = np.deg2rad(wo_phi_deg)
    wi_axis = np.array([np.cos(wi_phi_rad), np.sin(wi_phi_rad), 0.0],
                       dtype=np.float32)
    phi_o_deg = np.degrees(vd['phi_o'])
    phi_i_deg = np.degrees(vd['phi_i'])
    theta_o_deg = np.degrees(vd['theta_o'])
    phi_o_band = cyclic_dist_deg(phi_o_deg, wo_phi_deg)
    phi_i_band = np.minimum(cyclic_dist_deg(phi_i_deg, wi_phi_deg),
                            cyclic_dist_deg(phi_i_deg, wi_phi_deg + 180.0))

    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    cmap = plt.cm.tab10
    any_data = False
    src_counts = {'poly': 0, 'pan': 0, 'lls': 0}
    theta_o_handles = []  # (handle, label) — one per θ_o curve
    for ai, to_t in enumerate(theta_o_targets):
        cell = ((np.abs(theta_o_deg - to_t) <= theta_tol_deg) &
                (phi_o_band <= phi_tol_deg) &
                (phi_i_band <= phi_tol_deg))
        if not cell.any():
            continue
        col_ = cmap(ai)
        # Aggregate all sources for the θ_o curve (line is per θ_o, colour
        # matched to the decoder's same-θ_o curve).
        wi_g_all   = vd['L'][cell]
        y_g_all    = vd['val'][cell]
        signed_all = signed_theta_in_axis(wi_g_all, wi_axis)
        cos_all    = wi_g_all[:, 2].clip(0, None)
        y_cos_all  = y_g_all * cos_all
        order      = np.argsort(signed_all)
        line, = ax.plot(signed_all[order], y_cos_all[order], '-',
                        linewidth=2.0, color=col_, alpha=0.9)
        theta_o_handles.append((line, f'θ_o={to_t:.0f}° (n={int(cell.sum())})'))
        # Per-source scatter inside this θ_o cell — dot colour is the
        # SOURCE colour (PALETTE), so you can read both θ_o (line colour)
        # and acquisition path (dot colour) at once.
        for src in ['poly', 'pan', 'lls']:
            sel = cell & (vd['source'] == src)
            if not sel.any():
                continue
            wi_g = vd['L'][sel]
            y_g  = vd['val'][sel]
            sig  = signed_theta_in_axis(wi_g, wi_axis)
            ycos = y_g * wi_g[:, 2].clip(0, None)
            ax.scatter(sig, ycos, s=18, color=PALETTE[src],
                       edgecolors='black', linewidths=0.3, alpha=0.9,
                       zorder=5)
            src_counts[src] += int(sel.sum())
        # Specular reference (dashed thin) in the line colour.
        ax.axvline(x=np.deg2rad(to_t), color=col_, linestyle='--', alpha=0.45)
        any_data = True

    ax.set_theta_zero_location('N'); ax.set_theta_direction(1)
    ax.set_thetamin(-90); ax.set_thetamax(90)
    ax.tick_params(axis='y', labelsize=7)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_rlabel_position(75)
    if any_data:
        # Two legends: (1) θ_o curves (line colour),
        #              (2) source dots (marker colour). Avoids ambiguity.
        leg1 = ax.legend([h for h, _ in theta_o_handles],
                         [l for _, l in theta_o_handles],
                         loc='upper right', bbox_to_anchor=(1.30, 1.00),
                         title='θ_o (line)', fontsize=9, title_fontsize=9)
        ax.add_artist(leg1)
        src_handles = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=PALETTE[s],
                              markeredgecolor='black', markersize=7,
                              label=f'{s} (n={src_counts[s]})')
                       for s in ['poly', 'pan', 'lls'] if src_counts[s] > 0]
        if src_handles:
            ax.legend(handles=src_handles,
                      loc='upper right', bbox_to_anchor=(1.30, 0.60),
                      title='source (dots)', fontsize=9, title_fontsize=9)
    else:
        ax.text(0, 0.0, '(no data)', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='red')
    ax.set_title(
        f'mat{mat_id:04d}  pixel ({row}, {col})  pid={pid}\n'
        f'wi_φ={wi_phi_deg:.0f}°, wo_φ={wo_phi_deg:.0f}°   '
        f'GT BRDF × cos(θ_i) (vary wi, overlay).  Compare to '
        f'polar_brdf_vary_wi_pt_{pid}_wiphi{int(wi_phi_deg):03d}_'
        f'wophi{int(wo_phi_deg):03d}.png',
        fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_folder', default='/media/raid/cloth/Bonn_train')
    ap.add_argument('--out_dir',
                    default='/media/raid/cloth/output/BRDF/visualizations_debug/gt_brdf_lobes')
    ap.add_argument('--num_latents', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--point_subsample_ratio', type=float, default=1.0)
    ap.add_argument('--debug',    action='store_true',
                    help='Restrict to the first --debug_num materials (matches '
                         'BonnLatentBRDF debug mode).')
    ap.add_argument('--debug_num', type=int, default=1)
    ap.add_argument('--total_points', type=int, default=None,
                    help='Override total_points used for the seeded sample. '
                         'Pass the size of the trainer\'s '
                         'point_latent_bank.num_embeddings to reproduce the '
                         'exact pids written to brdf_lobes/ during training. '
                         'Defaults to the metadata-derived total.')
    ap.add_argument('--pid_list', type=int, nargs='+', default=None,
                    help='Explicit list of global pids to visualize. Overrides '
                         '--seed / --num_latents / --total_points. Use this '
                         'when you have the exact pids from the trainer log '
                         'or filename listing.')
    ap.add_argument('--theta_tol_deg', type=float, default=10.0)
    ap.add_argument('--phi_tol_deg',   type=float, default=25.0,
                    help='Tolerance around each (wi_phi, wo_phi) target. The '
                         'Bonn LED grid sits at 8 azimuths spaced 45° apart '
                         'offset by ~24° from the cardinals, so a 24-25° '
                         'tolerance is the minimum that catches the nearest '
                         'LED row for trainer-style targets at {0°, 90°, 180°}.')
    ap.add_argument('--n_theta_bins',  type=int,   default=18)
    ap.add_argument('--use_gt_normal', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='Rotate (L, V) into the per-pixel GT surface-normal '
                         'frame before computing (θ, φ) [default ON]. Pass '
                         '--no-use_gt_normal to fall back to normal=(0,0,1) '
                         '(the trainer\'s visualize_brdf_lobe convention).')
    ap.add_argument('--overlay_out_dir', default=None,
                    help='If set, also write trainer-style overlay plots '
                         '(single polar / all θ_o overlaid, direct-connection '
                         'only) into this folder. Defaults to None (disabled). '
                         'Pass a sibling path to keep overlay plots separate '
                         'from the per-panel grids.')
    args = ap.parse_args()

    out = Path(args.out_dir) / 'brdf_lobes'
    out.mkdir(parents=True, exist_ok=True)
    overlay_out = None
    if args.overlay_out_dir is not None:
        overlay_out = Path(args.overlay_out_dir) / 'brdf_lobes_overlay'
        overlay_out.mkdir(parents=True, exist_ok=True)

    materials, total_points = load_metadata(
        args.data_folder, args.point_subsample_ratio,
        args.debug, args.debug_num)
    print(f'Loaded metadata: {len(materials)} materials, '
          f'{total_points:,} total points')

    if args.pid_list is not None:
        pids = list(args.pid_list)
        print(f'Using explicit pids: {pids}')
    else:
        bank_total = args.total_points if args.total_points is not None else total_points
        torch.manual_seed(args.seed)
        pids = torch.randint(0, bank_total, (args.num_latents,)).tolist()
        print(f'Sampled global pids (seed={args.seed}, bank_total={bank_total}): {pids}')

    # Group pids by material so each material is loaded once.
    by_mat = {}
    for pid in pids:
        res = resolve_global_pid(materials, pid)
        if res is None:
            print(f'  pid={pid} not in any material range — skipping'); continue
        mat_id, local_pid, H, W = res
        row, col = divmod(local_pid, W)
        by_mat.setdefault(mat_id, []).append((pid, row, col, H, W))

    # Match the trainer's updated phi_configs (also rig-aligned to wo_phi ∈
    # {90°, 180°}). Same 5 (wi_phi, wo_phi) tuples → file names line up
    # 1:1 with the decoder lobes the trainer writes during validation.
    phi_configs = [
        (0.,   90.),
        (90.,  90.),
        (180., 90.),
        (0.,  180.),
        (90., 180.),
    ]

    per_pid_counts = []   # list of (pid, mat_id, empty, total)
    for mat_id, points in by_mat.items():
        print(f'\n--- mat{mat_id:04d}: {len(points)} pixels '
              f'(pids {[p[0] for p in points]}) ---')
        try:
            mat = _load_single_material_full(
                args.data_folder, mat_id, use_pan=True, use_lls=True)
        except Exception as e:
            print(f'  load failed: {e}'); continue
        if mat is None:
            print(f'  empty material'); continue
        # ``_load_single_material_full`` always loads the FULL H*W pixel set
        # — it does not subsample. So mat['xyz'] is indexed by raw flat
        # pixel id. When the trainer used point_subsample_ratio < 1.0, the
        # trainer's local_pid is a dense index into the subsampled subset
        # (BonnDataset / BonnValDataset build sub_idx via
        # `rng = np.random.default_rng(mat_id); rng.choice(n_pixels, N_sub,
        # replace=False); np.sort(...)`). To recover the raw flat pixel idx
        # we replay the same RNG here.
        n_pixels_full = mat['xyz'].shape[0]
        if args.point_subsample_ratio < 1.0:
            ratio_rng = np.random.default_rng(mat_id)
            N_sub = max(1, int(n_pixels_full * args.point_subsample_ratio))
            sub_idx = np.sort(ratio_rng.choice(n_pixels_full, size=N_sub,
                                               replace=False))
        else:
            sub_idx = None  # identity mapping

        for (pid, row, col, H, W) in points:
            local_pid = row * W + col   # from divmod(local_pid, W)
            if sub_idx is not None:
                if local_pid >= sub_idx.shape[0]:
                    print(f'  pid={pid}: local_pid {local_pid} >= '
                          f'subsample size {sub_idx.shape[0]} — skipping')
                    continue
                pix_idx = int(sub_idx[local_pid])
                raw_row, raw_col = divmod(pix_idx, W)
            else:
                pix_idx = local_pid
                raw_row, raw_col = row, col
            if pix_idx >= n_pixels_full:
                print(f'  pid={pid}: pix_idx {pix_idx} out of range '
                      f'(mat has {n_pixels_full} points) — skipping')
                continue
            print(f'  pid={pid} → local={local_pid} → raw pix '
                  f'({raw_row}, {raw_col})')
            row, col = raw_row, raw_col  # use raw coords in titles
            vd = per_view_geometry(mat, pix_idx,
                                   use_gt_normal=args.use_gt_normal)

            empty_pid = 0
            total_pid = 0
            # Vis 1: 1×5 grid (per-source overlay panels).
            n_e, n_t = plot_vary_wo(
                pid, mat_id, row, col, vd,
                out / f'gt_polar_brdf_vary_wo_pt_{pid}.png',
                theta_tol_deg=args.theta_tol_deg,
                phi_tol_deg=args.phi_tol_deg,
                n_theta_bins=args.n_theta_bins)
            empty_pid += n_e; total_pid += n_t
            # Vis 1b: trainer-style single polar with all 5 θ_i overlaid.
            plot_vary_wo_trainer_style(
                pid, mat_id, row, col, vd,
                out / f'gt_polar_brdf_vary_wo_pt_{pid}_trainer_style.png',
                theta_tol_deg=args.theta_tol_deg,
                phi_tol_deg=args.phi_tol_deg,
                n_theta_bins=args.n_theta_bins)
            # Vis 2 × 5 phi configs
            for wi_phi, wo_phi in phi_configs:
                fname = (f'gt_polar_brdf_vary_wi_pt_{pid}'
                         f'_wiphi{int(wi_phi):03d}_wophi{int(wo_phi):03d}.png')
                n_e, n_t = plot_vary_wi(
                    pid, mat_id, row, col, vd, out / fname,
                    wi_phi_deg=wi_phi, wo_phi_deg=wo_phi,
                    theta_tol_deg=args.theta_tol_deg,
                    phi_tol_deg=args.phi_tol_deg,
                    n_theta_bins=args.n_theta_bins)
                empty_pid += n_e; total_pid += n_t
                # Vis 2b: single-axis overlay (one figure per (wi_phi, wo_phi),
                # all 4 θ_o curves overlaid in tab10 colours, direct-connection
                # only). Saved into a separate folder when --overlay_out_dir set.
                if overlay_out is not None:
                    overlay_fname = (
                        f'gt_polar_brdf_vary_wi_overlay_pt_{pid}'
                        f'_wiphi{int(wi_phi):03d}_wophi{int(wo_phi):03d}.png')
                    plot_vary_wi_overlay(
                        pid, mat_id, row, col, vd,
                        overlay_out / overlay_fname,
                        wi_phi_deg=wi_phi, wo_phi_deg=wo_phi,
                        theta_tol_deg=args.theta_tol_deg,
                        phi_tol_deg=args.phi_tol_deg)
            per_pid_counts.append((pid, mat_id, empty_pid, total_pid))

    if per_pid_counts:
        print(f'\n=== Empty-panel summary (theta_tol={args.theta_tol_deg}°, '
              f'phi_tol={args.phi_tol_deg}°) ===')
        print(f'  {"pid":>10s}  {"mat":>5s}  {"empty/total":>12s}  '
              f'{"non-empty %":>12s}')
        for pid, mat_id, e, t in per_pid_counts:
            pct = 100.0 * (1.0 - e / t)
            print(f'  {pid:>10d}  {mat_id:>5d}  {e:>5d}/{t:<6d}  '
                  f'{pct:>11.1f}%')
        total_e = sum(e for _, _, e, _ in per_pid_counts)
        total_t = sum(t for _, _, _, t in per_pid_counts)
        print(f'  {"TOTAL":>10s}  {"":>5s}  '
              f'{total_e:>5d}/{total_t:<6d}  '
              f'{100.0*(1.0-total_e/total_t):>11.1f}%')
    print(f'\nWrote outputs to {out}')


if __name__ == '__main__':
    main()
