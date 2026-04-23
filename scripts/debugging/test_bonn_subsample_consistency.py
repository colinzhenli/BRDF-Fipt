"""Unit-test the per-material point subsampling on the Bonn pipeline.

What this verifies (matching the existing MultiMaterialDenseDataset behaviour):
  1. The subsample seed is deterministic per material ID — re-running
     ``np.random.default_rng(mat_id).choice(...)`` selects the SAME indices.
  2. BonnDataset and BonnValDataset (when loading the same mat_id) pick
     the IDENTICAL pixel subset, so xyz / gt_normals at local index k
     refer to the same physical pixel in train and val.
  3. BonnLatentBRDF sizes its latent bank to ``sum(int(num_points * ratio))``
     across materials, with per-material offsets that match the dataloader's
     emitted ``(material_id, point_id)`` tuples.
  4. A sampled training batch always satisfies
     ``0 <= point_ids[k] < num_points[material_ids[k]]`` after subsampling.
  5. ``get_global_point_id`` indexes valid rows of the bank for every
     emitted (material, local point) pair.

Run::

    conda run -n fipt_copy python scripts/debugging/test_bonn_subsample_consistency.py \\
        --dataset_folder /media/raid/cloth/Bonn_train --debug_num 10 --ratio 0.1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running from the repo root or scripts/debugging.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from utils.dataset.bonn import BonnDataset, BonnValDataset
from model.neural_brdf_refactored import BonnLatentBRDF


# ---------------------------------------------------------------------------
# Cfg construction helpers
# ---------------------------------------------------------------------------
def _build_cfg(dataset_folder: str, debug_num: int, ratio: float,
               use_pan: bool, use_lls: bool, rays_num: int) -> OmegaConf:
    """Mimic the Hydra-composed cfg used by main.py, scoped to what BonnDataset,
    BonnValDataset and BonnLatentBRDF actually read.
    """
    cfg = OmegaConf.create({
        'dataset_folder': dataset_folder,
        'data': {
            'debug': True,
            'debug_num': debug_num,
            'debug_rotate': False,
            'debug_swap_channels': False,
            'use_pan': use_pan,
            'use_lls': use_lls,
            'rays_num': rays_num,
            'num_load_workers': 0,
            'training_list_path': '',  # fall through to glob discovery
            'point_subsample_ratio': ratio,
            'val_materials': max(1, debug_num),
            'val_points': 500,
            'valid_num': 2,
            'overfit_mat_id': None,
            'val_view_ratio': 0.2,
            'val_seed': 42,
        },
        'material': {
            'data_folder': dataset_folder,
            'debug': True,
            'debug_num': debug_num,
            'debug_rotate': False,
            'debug_swap_channels': False,
            'latent_dim': 16,
            'predict_frame': True,
            'init_normal_from_gt': False,
            'use_pos_enc': True,
            'different_decoder': False,
            'learnable_factor': False,
            'init_std': 0.1,
            'point_subsample_ratio': ratio,
            'single_material_id': None,
            'optimizer': {'name': 'Adam'},  # dense embedding for the test
            'decoder': {
                'use_film': False,
                'use_color_decomp': False,
                'color_latent_dim': 6,
                'smooth_reg': False,
                'smooth_reg_eps': 0.02,
                'use_skip_connection': True,
                'skip_layer': 2,
                'activation': 'softplus',
                'hidden_layers': [256, 256, 256, 256],
                'intermediate_activation': 'leakyrelu',
                'output_channels': 3,
                'degree': 3,
            },
        },
    })
    return cfg


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
def _check(label: str, condition: bool, detail: str = "") -> None:
    marker = "PASS" if condition else "FAIL"
    print(f"  [{marker}] {label}" + (f"  — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"{label}: {detail}")


def _expected_subset(n_full: int, mat_id: int, ratio: float) -> np.ndarray:
    n_sub = max(1, int(n_full * ratio))
    rng = np.random.default_rng(mat_id)
    return np.sort(rng.choice(n_full, size=n_sub, replace=False))


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_folder', default='/media/raid/cloth/Bonn_train')
    p.add_argument('--debug_num', type=int, default=10)
    p.add_argument('--ratio', type=float, default=0.1)
    p.add_argument('--use_pan', action='store_true', default=True)
    p.add_argument('--use_lls', action='store_true', default=True)
    p.add_argument('--rays_num', type=int, default=8192)
    p.add_argument('--n_batches', type=int, default=4,
                   help='How many sampled batches to spot-check.')
    args = p.parse_args()

    print(f"\n{'='*70}")
    print(f"Bonn point-subsample consistency test")
    print(f"  dataset_folder        = {args.dataset_folder}")
    print(f"  debug_num             = {args.debug_num}")
    print(f"  point_subsample_ratio = {args.ratio}")
    print(f"  use_pan / use_lls     = {args.use_pan} / {args.use_lls}")
    print(f"{'='*70}\n")

    cfg = _build_cfg(
        dataset_folder=args.dataset_folder,
        debug_num=args.debug_num,
        ratio=args.ratio,
        use_pan=args.use_pan,
        use_lls=args.use_lls,
        rays_num=args.rays_num,
    )

    # ------------------------------------------------------------------
    # 0. Determinism of the subsample RNG itself.
    # ------------------------------------------------------------------
    print("[0] Subsample RNG determinism")
    for n_full, mid in [(262144, 1), (1440348, 2), (1514100, 5)]:
        a = _expected_subset(n_full, mid, args.ratio)
        b = _expected_subset(n_full, mid, args.ratio)
        _check(f"mat{mid:04d} (n_full={n_full})",
               np.array_equal(a, b),
               f"len={len(a)}, head={a[:5].tolist()}")

    # ------------------------------------------------------------------
    # 1. BonnLatentBRDF metadata + bank size.
    # ------------------------------------------------------------------
    print("\n[1] BonnLatentBRDF latent bank")
    brdf = BonnLatentBRDF(cfg.material)
    metadata = brdf.metadata

    expected_total = 0
    for mat_info in metadata['materials']:
        mid = mat_info['material_id']
        n_pts = mat_info['num_points']
        H, W = mat_info['H'], mat_info['W']
        expected_n = max(1, int(H * W * args.ratio))
        _check(f"mat{mid:04d} num_points",
               n_pts == expected_n,
               f"got {n_pts:,}, expected {expected_n:,} (H*W={H*W:,})")
        expected_total += n_pts

    bank_rows = brdf.point_latent_bank.weight.shape[0]
    _check("bank num_embeddings == sum(num_points)",
           bank_rows == expected_total,
           f"bank={bank_rows:,}, sum={expected_total:,}")
    _check("bank embedding_dim == total_latent_dim",
           brdf.point_latent_bank.weight.shape[1] == brdf.total_latent_dim)

    # Per-material offset tensor monotonic and aligned.
    print("  Material offsets:")
    for mat_info in metadata['materials']:
        mid = mat_info['material_id']
        offset_buf = int(brdf.material_offset_tensor[mid])
        offset_meta = mat_info['point_range'][0]
        _check(f"  offset[mat{mid:04d}]",
               offset_buf == offset_meta,
               f"buffer={offset_buf:,}, meta={offset_meta:,}")

    # ------------------------------------------------------------------
    # 2. BonnDataset (training) — per-material counts match the BRDF bank.
    # ------------------------------------------------------------------
    print("\n[2] BonnDataset (split=train)")
    train_ds = BonnDataset(cfg, args.dataset_folder, split='train')
    train_mats = train_ds._all_data.materials
    train_count_by_id = {m['mat_id']: m['xyz'].shape[0] for m in train_mats}
    print("  Per-material V (after subsample):")
    for mat_info in metadata['materials']:
        mid = mat_info['material_id']
        if mid not in train_count_by_id:
            print(f"    (mat{mid:04d} not in train split — debug_num filter)")
            continue
        v_train = train_count_by_id[mid]
        v_brdf = mat_info['num_points']
        _check(f"  mat{mid:04d} V matches",
               v_train == v_brdf,
               f"train={v_train:,}, brdf={v_brdf:,}")

    # Verify BonnDataset's actual subsample matches the canonical RNG selection.
    print("  Canonical-subset verification (xyz comparison):")
    for m in train_mats[: min(3, len(train_mats))]:
        mid = m['mat_id']
        # Re-load full-resolution xyz from EXR to compare
        from utils.dataset.bonn import _read_xyz_map
        prefix = Path(args.dataset_folder) / f'mat{mid:04d}'
        full_xyz, H, W = _read_xyz_map(f'{prefix}_xyz_rot000.exr')
        full_flat = full_xyz.reshape(-1, 3).astype(np.float32)
        sel = _expected_subset(H * W, mid, args.ratio)
        expected_xyz = full_flat[sel]

        got_xyz = m['xyz']
        _check(f"  mat{mid:04d} xyz subset",
               got_xyz.shape == expected_xyz.shape and
               np.allclose(got_xyz, expected_xyz, atol=1e-6),
               f"shape got={got_xyz.shape}, want={expected_xyz.shape}")

        got_pids = m['point_ids']
        _check(f"  mat{mid:04d} point_ids dense 0..N-1",
               np.array_equal(got_pids, np.arange(got_xyz.shape[0], dtype=np.int64)),
               f"got head={got_pids[:5].tolist()}, tail={got_pids[-5:].tolist()}")

    # ------------------------------------------------------------------
    # 3. Sampled training batches stay in range AND map into the bank.
    # ------------------------------------------------------------------
    print("\n[3] Sampled-batch index validity")
    bank_size = brdf.point_latent_bank.weight.shape[0]
    it = iter(train_ds)
    for b in range(args.n_batches):
        batch = next(it)
        pids = batch['point_ids'].numpy()
        mids = batch['material_ids'].numpy()

        # Per-material range check
        bad = 0
        for mid in np.unique(mids):
            v_train = train_count_by_id.get(int(mid))
            mask = mids == mid
            if v_train is None:
                bad += int(mask.sum())
                continue
            sub = pids[mask]
            if (sub.min() < 0) or (sub.max() >= v_train):
                bad += int(((sub < 0) | (sub >= v_train)).sum())
        _check(f"  batch {b}: per-material point_ids in [0, num_points)",
               bad == 0,
               f"out-of-range count = {bad}")

        # Global ID range check
        global_ids = brdf.get_global_point_id(
            torch.from_numpy(mids).long(),
            torch.from_numpy(pids).long(),
        ).numpy()
        _check(f"  batch {b}: global ids in [0, bank_size)",
               (global_ids.min() >= 0) and (global_ids.max() < bank_size),
               f"min={global_ids.min()}, max={global_ids.max()}, bank={bank_size}")

    # ------------------------------------------------------------------
    # 4. BonnValDataset uses the SAME pixel subset as training for mat_id=1.
    # ------------------------------------------------------------------
    print("\n[4] BonnValDataset alignment")
    val_ds = BonnValDataset(cfg, args.dataset_folder)
    val_mid = val_ds.mat_id
    val_xyz = val_ds._items[0]['xyz'].numpy()
    val_pids = val_ds._items[0]['point_ids'].numpy()

    # Look up the same material in BonnDataset
    if val_mid in train_count_by_id:
        train_mat = next(m for m in train_mats if m['mat_id'] == val_mid)
        train_xyz = train_mat['xyz']
        _check(f"  mat{val_mid:04d} val_xyz == train_xyz",
               val_xyz.shape == train_xyz.shape and
               np.allclose(val_xyz, train_xyz, atol=1e-6),
               f"val={val_xyz.shape}, train={train_xyz.shape}")
        _check(f"  mat{val_mid:04d} val pids dense 0..N-1",
               np.array_equal(val_pids, np.arange(val_xyz.shape[0], dtype=np.int64)),
               f"val head={val_pids[:5].tolist()}")
    else:
        print(f"  (mat{val_mid:04d} not in train debug split — skipping)")

    # ------------------------------------------------------------------
    # 5. Cleanup + summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("All consistency checks PASSED")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
