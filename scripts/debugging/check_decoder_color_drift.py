#!/usr/bin/env python3
"""Per-channel RGB color-drift analysis for Bonn stage-1 decoders.

Hypothesis under test:
  A decoder trained with pan loss (Σ w·RGB = scalar) leaves a 2-DOF color
  null-space unsupervised. On held-out poly rays, we expect a *systematic
  per-channel bias* (not per-sample noise) relative to GT RGB.

Inputs:
  --ckpt  : full stage-1 PL checkpoint (tens of GB — loaded ONCE).
  --materials : list of Bonn mat_ids (e.g. 1 2 3 4 5).
  --data-root : Bonn_train root.
  --cache-dir : where to persist the distilled state + predictions.

Pipeline:
  Phase 1 (slow, cached once):
    (a) Build material module from ckpt's saved hyper-parameters.
    (b) Load full state_dict → extract ONLY what we need:
          decoder.* + material_offset_tensor + latent-bank rows
          for the requested material IDs.
        Write that to cache_dir/extracted_state.pt  (~ MB).
    (c) For each mat_id, sample N poly rays, run material.eval_brdf,
        save (pred, gt, wi, wo, confidence) to cache_dir/mat{id}.npz.

  Phase 2 (fast, idempotent):
    Load all .npz → per-channel stats + plots:
      - signed error histograms per channel
      - scatter GT vs pred per channel (log-log)
      - pan-projection consistency (should be tight for a pan-trained decoder)
      - chromaticity (R/(R+G+B), G/(R+G+B), B/(R+G+B)) scatter

Typical use:
    # first run — slow (loads 44 GB ckpt, extracts, evaluates, plots)
    python scripts/debugging/check_decoder_color_drift.py \
        --ckpt /media/raid/.../last.ckpt --materials 1 2 3 5 7

    # tweak plots — fast (uses cache)
    python scripts/debugging/check_decoder_color_drift.py \
        --materials 1 2 3 5 7 --phase plot
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset.bonn import (  # noqa: E402
    _parse_poly_channels,
    _parse_poly2pan_weights,
    _pan_weights_for_image,
    _read_exr,
    _read_xyz_map,
)

DEFAULT_CKPT = ('/media/raid/cloth/output/BRDF/Bonn-Theia2/'
                'Stage-1_Logrel_Softplus_Fir_decoder-lr-1e-4_All-data_'
                'Latent-24_Color_All-RGB-Pan-0.5_run_1/training/training/'
                'model_0.20_0.20/last.ckpt')
DEFAULT_DATA_ROOT = '/media/raid/cloth/Bonn_train'
DEFAULT_CACHE_DIR = Path(os.environ.get(
    'COLOR_DRIFT_CACHE',
    '/media/raid/cloth/output/BRDF/diagnostics/color_drift'))

# ---------------------------------------------------------------------------
# Phase 1.a — build the material module from ckpt's saved cfg
# ---------------------------------------------------------------------------

def _cfg_from_ckpt(ckpt: dict):
    if 'hyper_parameters' not in ckpt:
        raise RuntimeError('ckpt has no hyper_parameters; cannot rebuild cfg')
    cfg = OmegaConf.create(ckpt['hyper_parameters'])
    return cfg


def _build_material(cfg):
    mod = importlib.import_module(cfg.material.module)
    cls = getattr(mod, cfg.material.type)
    return cls(cfg.material)


# ---------------------------------------------------------------------------
# Phase 1.b — slim the huge ckpt down to what we need
# ---------------------------------------------------------------------------

LATENT_KEYS = (
    'material.point_latent_bank.weight',
    'material.material_offset_tensor',
)


def _distill_state(ckpt: dict, mat_ids, out_path: Path):
    """Persist a tiny subset of the ckpt (decoder + relevant latents)."""
    full_sd = ckpt['state_dict']
    slim = {}

    # Decoder.
    for k, v in full_sd.items():
        if 'material.decoder.' in k:
            slim[k] = v.cpu()

    # Offsets & latents.
    offsets_key = 'material.material_offset_tensor'
    bank_key = 'material.point_latent_bank.weight'
    if offsets_key not in full_sd or bank_key not in full_sd:
        raise RuntimeError(f'ckpt missing {offsets_key} or {bank_key}')

    offsets = full_sd[offsets_key].cpu()          # (max_mat_id+1,)
    bank = full_sd[bank_key].cpu()                # (N_total, D)

    # Keep full offsets (small); for the bank keep only the rows of the
    # requested materials. We also need to know where each material ended,
    # so we look up the NEXT larger offset as the end index.
    sorted_offs = torch.sort(offsets)[0].tolist()
    sorted_offs_set = sorted(set(sorted_offs) | {int(bank.shape[0])})

    def next_offset_after(start):
        for s in sorted_offs_set:
            if s > start:
                return s
        return int(bank.shape[0])

    slim_offsets = offsets.clone()
    slim_bank_rows = {}
    for mat_id in mat_ids:
        if mat_id >= offsets.shape[0]:
            raise RuntimeError(
                f'mat_id {mat_id} out of range; offsets length {offsets.shape[0]}')
        start = int(offsets[mat_id].item())
        end = next_offset_after(start)
        slim_bank_rows[int(mat_id)] = (start, end, bank[start:end].clone())

    slim['material.material_offset_tensor'] = slim_offsets
    slim['__bank_rows_per_material'] = slim_bank_rows
    slim['__bank_total_rows'] = int(bank.shape[0])
    slim['__bank_dim'] = int(bank.shape[1])
    slim['__hyper_parameters'] = ckpt['hyper_parameters']

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out_path)
    mb = out_path.stat().st_size / 1e6
    print(f'[distill] wrote {out_path}  ({mb:.1f} MB, '
          f'{len(slim_bank_rows)} mats, {sum(v[2].shape[0] for v in slim_bank_rows.values())} latent rows)')


def _load_distilled(distilled_path: Path):
    return torch.load(distilled_path, map_location='cpu', weights_only=False)


def _apply_distilled_to_material(material, distilled):
    """Populate material's decoder + latent bank (partial) + offsets."""
    sd = material.state_dict()
    # Decoder.
    for k, v in distilled.items():
        if k.startswith('material.decoder.'):
            stripped = k.replace('material.', '', 1)
            if stripped in sd:
                sd[stripped] = v
            else:
                raise RuntimeError(f'unexpected decoder key {k}')
    # Offsets.
    off_key = 'material_offset_tensor'
    if off_key in sd:
        sd[off_key] = distilled['material.material_offset_tensor']

    # Latent bank.  The embedding is sized at construction time via cfg;
    # it should already have the right full-bank shape.  We overwrite only
    # the rows of the requested materials.
    bank_key = 'point_latent_bank.weight'
    if bank_key not in sd:
        raise RuntimeError(f'material has no {bank_key}')
    full_bank = sd[bank_key].clone()
    for mat_id, (start, end, rows) in distilled['__bank_rows_per_material'].items():
        if end > full_bank.shape[0]:
            full_bank = torch.cat([
                full_bank,
                torch.zeros(end - full_bank.shape[0], full_bank.shape[1],
                            dtype=full_bank.dtype, device=full_bank.device)], dim=0)
        full_bank[start:end] = rows
    sd[bank_key] = full_bank

    material.load_state_dict(sd, strict=False)
    material.eval()


# ---------------------------------------------------------------------------
# Phase 1.c — load a material's poly data and run predictions
# ---------------------------------------------------------------------------

def _load_calibration(root, mat_id):
    prefix = Path(root) / f'mat{mat_id:04d}'
    raw = spio.loadmat(f'{prefix}_calibration.mat')
    calib = {}
    for rot_key in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
        rd = raw[rot_key][0, 0]
        rot_dict = {}
        for field in rd.dtype.names:
            val = rd[field]
            if field == 'llsCorners':
                rot_dict[field] = np.array(val, dtype=np.float32)
            else:
                rot_dict[field] = np.array(val, dtype=np.float32).flatten()
            m = re.match(r'(il|cv)(\d+)', field)
            if m and len(m.group(2)) < 3:
                padded = f'{m.group(1)}{int(m.group(2)):03d}'
                rot_dict[padded] = rot_dict[field]
        calib[rot_key] = rot_dict
    calib['llsAnglesDegrees'] = raw['llsAnglesDegrees'].flatten().astype(np.float64)
    w = _parse_poly2pan_weights(raw)
    calib['poly2pan_per_il'] = w['per_il']
    calib['poly2pan_per_cv'] = w['per_cv']
    calib['poly2pan_global_avg'] = w['global_avg']
    return calib


def _load_material_poly(root, mat_id):
    """Load poly images + geometry for one material. No pan/lls."""
    prefix = Path(root) / f'mat{mat_id:04d}'
    calib = _load_calibration(root, mat_id)

    xyz_map, H, W = _read_xyz_map(f'{prefix}_xyz_rot000.exr')
    V = H * W
    xyz_pts = xyz_map.reshape(V, 3).astype(np.float32)

    poly_data, ch_names, pH, pW = _read_exr(f'{prefix}_poly.exr')
    assert (pH, pW) == (H, W)
    poly_images = _parse_poly_channels(ch_names)
    K = len(poly_images)
    poly_rgbs = poly_data.reshape(V, K, 3).transpose(1, 0, 2).astype(np.float32)
    np.clip(poly_rgbs, 0, None, out=poly_rgbs)

    light_pos = np.array(
        [calib[im['rotation']][im['led']] for im in poly_images], dtype=np.float32)
    cam_pos = np.array(
        [calib[im['rotation']][im['camera']] for im in poly_images], dtype=np.float32)
    pan_w = np.stack([
        _pan_weights_for_image(calib, im['rotation'], im['camera'],
                               im['led'], is_poly=True)
        for im in poly_images], axis=0).astype(np.float32)

    return {
        'H': H, 'W': W, 'V': V, 'K': K,
        'xyz': xyz_pts,                 # (V, 3)
        'rgbs': poly_rgbs,              # (K, V, 3) RGB
        'light_pos': light_pos,         # (K, 3)
        'cam_pos': cam_pos,             # (K, 3)
        'pan_weights': pan_w,           # (K, 3)
        'poly_images': poly_images,
    }


def _sample_rays(mat_data, n_rays, seed=0):
    rng = np.random.default_rng(seed)
    V, K = mat_data['V'], mat_data['K']
    img_i = rng.integers(0, K, n_rays)
    pix_i = rng.integers(0, V, n_rays)

    xyz = mat_data['xyz'][pix_i]
    rgbs_gt = mat_data['rgbs'][img_i, pix_i]       # (N, 3)
    light = mat_data['light_pos'][img_i]
    cam = mat_data['cam_pos'][img_i]
    pan_w = mat_data['pan_weights'][img_i]

    wi = light - xyz
    wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
    wo = cam - xyz
    wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)

    confidence = (rgbs_gt.sum(axis=-1) > 0).astype(np.float32)
    return dict(xyz=xyz, rgbs_gt=rgbs_gt, wi=wi, wo=wo,
                pan_w=pan_w, pix_i=pix_i, img_i=img_i,
                confidence=confidence)


@torch.no_grad()
def _run_brdf(material, rays, mat_id, device, chunk=8192):
    """Run material.eval_brdf on the sampled rays. Returns pred (N, 3)."""
    xyz = torch.from_numpy(rays['xyz']).to(device)
    wi = torch.from_numpy(rays['wi']).to(device)
    wo = torch.from_numpy(rays['wo']).to(device)
    pids = torch.from_numpy(rays['pix_i'].astype(np.int64)).to(device)
    mids = torch.full_like(pids, int(mat_id))
    # Dummy normals (predict_frame=True picks its own).
    normals = torch.zeros_like(wi)
    normals[..., 2] = 1.0

    N = xyz.shape[0]
    out = torch.zeros(N, 3, device=device)
    for i in range(0, N, chunk):
        s = slice(i, min(N, i + chunk))
        brdf, _, _, _ = material.eval_brdf(
            xyz[s], wi[s], wo[s], normals[s],
            point_ids=pids[s], material_ids=mids[s])
        out[s] = brdf
    return out.cpu().numpy()


# ---------------------------------------------------------------------------
# Phase 2 — analysis + plots
# ---------------------------------------------------------------------------

CHANNEL_NAMES = ['R', 'G', 'B']
CHANNEL_COLORS = ['tab:red', 'tab:green', 'tab:blue']


def _load_all_caches(cache_dir: Path, mat_ids):
    out = {}
    for mat_id in mat_ids:
        f = cache_dir / f'mat{mat_id:04d}.npz'
        if not f.exists():
            print(f'[warn] cache missing: {f}')
            continue
        out[mat_id] = dict(np.load(f))
    return out


def _stats_table(caches):
    print('\nper-material per-channel stats (valid rays only)')
    print('-' * 84)
    hdr = f'{"mat":>5}  {"N":>8}  {"ch":>2}  {"mean(gt)":>10}  ' \
          f'{"mean(pred)":>10}  {"bias":>10}  {"bias%":>8}  {"corr":>6}'
    print(hdr)
    print('-' * 84)
    for mat_id, d in caches.items():
        gt = d['rgbs_gt']
        pr = d['pred']
        mask = d['confidence'].astype(bool)
        gt = gt[mask]
        pr = pr[mask]
        N = gt.shape[0]
        for c in range(3):
            gm = float(gt[:, c].mean())
            pm = float(pr[:, c].mean())
            bias = pm - gm
            bias_pct = 100.0 * bias / (gm + 1e-8)
            corr = float(np.corrcoef(gt[:, c], pr[:, c])[0, 1]) if N > 1 else 0.0
            print(f'{mat_id:>5d}  {N:>8d}  {CHANNEL_NAMES[c]:>2}  '
                  f'{gm:>10.4f}  {pm:>10.4f}  {bias:>+10.4f}  '
                  f'{bias_pct:>+7.1f}%  {corr:>6.3f}')


def _plot_per_channel_scatter(caches, out_path: Path):
    fig, axes = plt.subplots(len(caches), 3, figsize=(14, 3.8 * len(caches)),
                             squeeze=False)
    for row, (mat_id, d) in enumerate(caches.items()):
        mask = d['confidence'].astype(bool)
        gt = d['rgbs_gt'][mask]
        pr = d['pred'][mask]
        for c in range(3):
            ax = axes[row, c]
            g = gt[:, c]
            p = pr[:, c]
            ax.scatter(g, p, s=1.2, alpha=0.3, color=CHANNEL_COLORS[c])
            lo = max(1e-3, min(g.min(), p.min()))
            hi = max(g.max(), p.max(), 1e-3) * 1.1
            ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel(f'GT {CHANNEL_NAMES[c]}'); ax.set_ylabel(f'pred {CHANNEL_NAMES[c]}')
            if c == 0:
                ax.set_title(f'mat{mat_id:04d}  {CHANNEL_NAMES[c]}')
            else:
                ax.set_title(CHANNEL_NAMES[c])
            ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'[plot] {out_path}')


def _plot_signed_error_hist(caches, out_path: Path):
    fig, axes = plt.subplots(len(caches), 1, figsize=(10, 3.2 * len(caches)),
                             squeeze=False)
    for row, (mat_id, d) in enumerate(caches.items()):
        mask = d['confidence'].astype(bool)
        diff = d['pred'][mask] - d['rgbs_gt'][mask]
        ax = axes[row, 0]
        for c in range(3):
            ax.hist(diff[:, c], bins=121, range=(-0.4, 0.4),
                    histtype='step', lw=1.5, color=CHANNEL_COLORS[c],
                    label=f'{CHANNEL_NAMES[c]} (μ={diff[:, c].mean():+.4f})')
        ax.axvline(0, color='k', lw=0.8)
        ax.set_title(f'mat{mat_id:04d}  — signed error = pred − GT')
        ax.set_xlabel('pred − GT')
        ax.set_ylabel('count')
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'[plot] {out_path}')


def _plot_pan_consistency(caches, out_path: Path):
    fig, axes = plt.subplots(1, len(caches), figsize=(4.2 * len(caches), 4.2),
                             squeeze=False)
    for col, (mat_id, d) in enumerate(caches.items()):
        mask = d['confidence'].astype(bool)
        gt = d['rgbs_gt'][mask]
        pr = d['pred'][mask]
        w = d['pan_w'][mask]
        gt_pan = (gt * w).sum(axis=-1)
        pr_pan = (pr * w).sum(axis=-1)
        ax = axes[0, col]
        ax.scatter(gt_pan, pr_pan, s=1.2, alpha=0.3, color='tab:orange')
        lo = max(1e-3, min(gt_pan.min(), pr_pan.min()))
        hi = max(gt_pan.max(), pr_pan.max(), 1e-3) * 1.1
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel('GT pan = Σ w·RGB')
        ax.set_ylabel('pred pan')
        ax.set_title(f'mat{mat_id:04d}  pan consistency')
        ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'[plot] {out_path}')


def _plot_chromaticity(caches, out_path: Path):
    """Plot normalized-RGB chromaticity to isolate color drift from brightness."""
    fig, axes = plt.subplots(len(caches), 3, figsize=(13, 3.8 * len(caches)),
                             squeeze=False)
    eps = 1e-6
    for row, (mat_id, d) in enumerate(caches.items()):
        mask = d['confidence'].astype(bool)
        gt = d['rgbs_gt'][mask]
        pr = d['pred'][mask]
        sgt = gt.sum(axis=-1, keepdims=True) + eps
        spr = pr.sum(axis=-1, keepdims=True) + eps
        chrom_gt = gt / sgt
        chrom_pr = pr / spr
        for c in range(3):
            ax = axes[row, c]
            g = chrom_gt[:, c]; p = chrom_pr[:, c]
            ax.scatter(g, p, s=1.2, alpha=0.3, color=CHANNEL_COLORS[c])
            ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel(f'GT chrom {CHANNEL_NAMES[c]}')
            ax.set_ylabel(f'pred chrom {CHANNEL_NAMES[c]}')
            ax.set_title(f'mat{mat_id:04d}  chrom {CHANNEL_NAMES[c]}'
                         if c == 0 else f'chrom {CHANNEL_NAMES[c]}')
            ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'[plot] {out_path}')


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _phase_extract(args, distilled_path: Path):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if distilled_path.exists() and not args.force_distill:
        print(f'[extract] reusing {distilled_path}')
        distilled = _load_distilled(distilled_path)
    else:
        print(f'[extract] loading full ckpt {args.ckpt}  (will take a while)')
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        _distill_state(ckpt, args.materials, distilled_path)
        del ckpt
        distilled = _load_distilled(distilled_path)

    cfg = OmegaConf.create(distilled['__hyper_parameters'])
    # Saved cfg bakes the training-time data path (CC: /home/zla247/scratch/...).
    # Override to wherever the user actually has the data on this host.
    if hasattr(cfg, 'material'):
        cfg.material.data_folder = args.data_root
    if hasattr(cfg, 'dataset_folder'):
        cfg.dataset_folder = args.data_root
    print(f'[extract] using data_folder={args.data_root}')
    material = _build_material(cfg)
    _apply_distilled_to_material(material, distilled)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    material.to(device)

    for mat_id in args.materials:
        npz = cache_dir / f'mat{mat_id:04d}.npz'
        if npz.exists() and not args.force_eval:
            print(f'[extract] {npz} exists; skip (use --force-eval to redo)')
            continue
        print(f'[extract] mat{mat_id:04d}: loading poly data…')
        mat_data = _load_material_poly(args.data_root, mat_id)
        print(f'[extract] mat{mat_id:04d}: {mat_data["V"]} px × {mat_data["K"]} imgs, '
              f'sampling {args.num_rays} rays')
        rays = _sample_rays(mat_data, args.num_rays, seed=args.seed + mat_id)
        pred = _run_brdf(material, rays, mat_id, device, chunk=args.chunk)
        np.savez(npz,
                 pred=pred.astype(np.float32),
                 rgbs_gt=rays['rgbs_gt'].astype(np.float32),
                 wi=rays['wi'].astype(np.float32),
                 wo=rays['wo'].astype(np.float32),
                 pan_w=rays['pan_w'].astype(np.float32),
                 confidence=rays['confidence'].astype(np.float32),
                 pix_i=rays['pix_i'].astype(np.int64),
                 img_i=rays['img_i'].astype(np.int64),
                 mat_id=mat_id)
        print(f'[extract] wrote {npz}  (N={pred.shape[0]})')


def _phase_plot(args):
    cache_dir = Path(args.cache_dir)
    plots_dir = cache_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    caches = _load_all_caches(cache_dir, args.materials)
    if not caches:
        raise RuntimeError('no per-material caches found; run --phase extract first')
    _stats_table(caches)
    _plot_per_channel_scatter(caches, plots_dir / 'per_channel_scatter.png')
    _plot_signed_error_hist(caches, plots_dir / 'per_channel_signed_error_hist.png')
    _plot_pan_consistency(caches, plots_dir / 'pan_consistency.png')
    _plot_chromaticity(caches, plots_dir / 'chromaticity.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=DEFAULT_CKPT,
                    help='full stage-1 PL ckpt (loaded once; distilled to cache)')
    ap.add_argument('--materials', nargs='+', type=int, required=True,
                    help='Bonn material IDs to evaluate (e.g. 1 2 3 5 7)')
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--cache-dir', default=str(DEFAULT_CACHE_DIR))
    ap.add_argument('--num-rays', type=int, default=20000,
                    help='number of random poly rays per material')
    ap.add_argument('--chunk', type=int, default=8192)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--phase', choices=['extract', 'plot', 'all'],
                    default='all')
    ap.add_argument('--force-distill', action='store_true',
                    help='redo the ckpt → distilled_state extraction')
    ap.add_argument('--force-eval', action='store_true',
                    help='redo BRDF eval even if cache exists')
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    distilled_path = cache_dir / 'distilled_state.pt'

    if args.phase in ('extract', 'all'):
        _phase_extract(args, distilled_path)
    if args.phase in ('plot', 'all'):
        _phase_plot(args)


if __name__ == '__main__':
    main()
