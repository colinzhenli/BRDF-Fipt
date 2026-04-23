"""Fast (mock-based) unit-test for the Bonn point-subsample feature.

The end-to-end ``test_bonn_subsample_consistency.py`` test loads real ~135 MB
poly EXR files per material — over slow NFS that can take hours and stall on
disk contention. This script instead monkey-patches every file-reader in
``utils.dataset.bonn`` to return tiny synthetic arrays (e.g. 32×32 instead of
512×512), so the full BonnDataset / BonnValDataset / BonnLatentBRDF pipeline
runs in seconds while still exercising the real subsample / indexing code.

What it verifies (matching MultiMaterialDenseDataset semantics):
  1. RNG determinism — same (mat_id, ratio) yields the same indices.
  2. BonnDataset._load_single_material compacts xyz / rgbs / gt_normals onto
     the canonical RNG-selected subset and resets point_ids to 0..N_sub-1.
  3. BonnValDataset selects the IDENTICAL subset for the same mat_id, so
     val xyz at index k refers to the same physical pixel as train xyz[k].
  4. Sampled training batches stay in [0, num_points) per material.
  5. BonnLatentBRDF latent bank size = sum(int(num_points * ratio)) and
     get_global_point_id maps every emitted (mat, local) into a valid row.

Run:
    conda run -n fipt_copy python scripts/debugging/test_bonn_subsample_fast.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Synthetic fixture: 10 materials with varying H, W and stable per-mat content.
# ---------------------------------------------------------------------------
MAT_SHAPES = {
    1: (32, 32),
    2: (64, 48),
    5: (40, 56),
    6: (24, 24),
    8: (48, 48),
    9: (32, 16),
    10: (40, 40),
    11: (16, 64),
    12: (28, 28),
    13: (52, 36),
}


def _shape_for(mat_id: int) -> tuple[int, int]:
    return MAT_SHAPES.get(mat_id, (32, 32))


def _fake_xyz(mat_id: int) -> np.ndarray:
    H, W = _shape_for(mat_id)
    rng = np.random.default_rng(1000 + mat_id)
    return rng.standard_normal((H, W, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# Monkey-patch helpers — all called BEFORE BonnDataset / Val / BRDF imports
# pick up the real readers.
# ---------------------------------------------------------------------------
def _make_patches(n_poly_imgs: int = 4):
    """Build patches that replace the slow EXR readers with tiny in-memory data.

    n_poly_imgs controls the number of synthetic poly images per material —
    each EXR has 3*n_poly_imgs channels (BGR triplets) like real poly EXRs.
    """

    def fake_read_xyz_map(filepath):
        # extract mat_id from path "<root>/matNNNN_xyz_rot000.exr"
        name = Path(filepath).name
        mat_id = int(name[3:7])
        xyz = _fake_xyz(mat_id)
        H, W = xyz.shape[:2]
        return xyz, H, W

    def fake_read_exr(filepath):
        name = Path(filepath).name
        mat_id = int(name[3:7])
        H, W = _shape_for(mat_id)
        kind = 'pan' if '_pan.' in name else ('lls' if '_lls.' in name else 'poly')

        if kind == 'poly':
            # Build n_poly_imgs synthetic images, channel order matches the
            # real EXR: alphabetically sorted (poly_cv01_il001_rot000_{B,G,R}, ...)
            ch_names = []
            for i in range(n_poly_imgs):
                cam = f'cv0{(i % 4) + 1}'
                led = f'il{(i % 5) + 26:03d}'  # only 026/027/028/029/030 to keep parser happy
                rot = 'rot000'
                for c in ('B', 'G', 'R'):
                    ch_names.append(f'poly_{cam}_{led}_{rot}_{c}')
            data = np.random.default_rng(2000 + mat_id).standard_normal(
                (H, W, len(ch_names)), dtype=np.float32).astype(np.float16)
            # ensure non-zero so confidence mask is mostly True
            data = np.abs(data) + 0.01
            return data, ch_names, H, W

        if kind == 'pan':
            ch_names = [f'pan_cv01_il{i:03d}_rot000' for i in range(1, 5)]
            data = np.random.default_rng(3000 + mat_id).standard_normal(
                (H, W, len(ch_names)), dtype=np.float32).astype(np.float16)
            return np.abs(data) + 0.01, ch_names, H, W

        # lls
        ch_names = [
            f'lls_cv01_lls01_la{ang:.2f}_rot000'
            for ang in (-30.0, 0.0, 30.0)
        ]
        data = np.random.default_rng(4000 + mat_id).standard_normal(
            (H, W, len(ch_names)), dtype=np.float32).astype(np.float16)
        return np.abs(data) + 0.01, ch_names, H, W

    def fake_read_gt_normal_map(svfresnel_dir, mat_id, H, W):
        rng = np.random.default_rng(5000 + mat_id)
        n = rng.standard_normal((H * W, 3), dtype=np.float32)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
        return n

    def fake_load_calibration(self, mat_id):
        # Minimal calibration with a single rotation key 'rot000' and the
        # 4 cameras × 5 LEDs that the synthetic poly channels reference.
        rot_dict = {}
        for cv_i in range(1, 5):
            rot_dict[f'cv{cv_i:02d}'] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            rot_dict[f'cv{cv_i:03d}'] = rot_dict[f'cv{cv_i:02d}']
        for il in range(1, 35):
            rot_dict[f'il{il:02d}'] = np.array([0.1, 0.0, 0.5], dtype=np.float32)
            rot_dict[f'il{il:03d}'] = rot_dict[f'il{il:02d}']
        # llsCorners shape: (3, 4, n_lls_angles); we map 3 angles
        rot_dict['llsCorners'] = np.zeros((3, 4, 3), dtype=np.float32)
        return {
            'rot000': rot_dict, 'rot045': rot_dict, 'rot090': rot_dict,
            'rot135': rot_dict, 'rot180': rot_dict,
            'llsAnglesDegrees': np.array([-30.0, 0.0, 30.0], dtype=np.float64),
            'poly2pan_per_il': {},
            'poly2pan_per_cv': {f'cv{i:02d}': np.array([0.34, 0.36, 0.28], dtype=np.float32)
                                 for i in range(1, 5)},
            'poly2pan_global_avg': np.array([0.34, 0.36, 0.28], dtype=np.float32),
        }

    return [
        patch('utils.dataset.bonn._read_xyz_map', side_effect=fake_read_xyz_map),
        patch('utils.dataset.bonn._read_exr', side_effect=fake_read_exr),
        patch('utils.dataset.bonn._read_gt_normal_map', side_effect=fake_read_gt_normal_map),
        # Calibration is a method so patch on the class
        patch('utils.dataset.bonn.BonnDataset._load_calibration',
              side_effect=fake_load_calibration, autospec=True),
        patch('utils.dataset.bonn.BonnValDataset._load_calibration',
              side_effect=fake_load_calibration, autospec=True),
    ]


# ---------------------------------------------------------------------------
# Cfg builder
# ---------------------------------------------------------------------------
def _build_cfg(dataset_folder: str, debug_num: int, ratio: float,
               rays_num: int = 4096) -> OmegaConf:
    return OmegaConf.create({
        'dataset_folder': dataset_folder,
        'data': {
            'debug': True,
            'debug_num': debug_num,
            'debug_rotate': False,
            'debug_swap_channels': False,
            'use_pan': True,
            'use_lls': True,
            'rays_num': rays_num,
            'num_load_workers': 0,
            'training_list_path': '',
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
            'optimizer': {'name': 'Adam'},
            'decoder': {
                'use_film': False, 'use_color_decomp': False, 'color_latent_dim': 6,
                'smooth_reg': False, 'smooth_reg_eps': 0.02,
                'use_skip_connection': True, 'skip_layer': 2,
                'activation': 'softplus',
                'hidden_layers': [128, 128, 128, 128],
                'intermediate_activation': 'leakyrelu',
                'output_channels': 3, 'degree': 3,
            },
        },
    })


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------
def _check(label: str, condition: bool, detail: str = "") -> None:
    marker = "PASS" if condition else "FAIL"
    print(f"  [{marker}] {label}" + (f"  — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"{label}: {detail}")


def main():
    import json
    import tempfile

    DEBUG_NUM = 10
    RATIO = 0.1
    print(f"\n{'='*70}")
    print(f"Bonn point-subsample FAST consistency test")
    print(f"  debug_num             = {DEBUG_NUM}")
    print(f"  point_subsample_ratio = {RATIO}")
    print(f"  fixture               = 10 synthetic mats, {dict(MAT_SHAPES)}")
    print(f"{'='*70}\n")

    # Synthetic dataset folder + bonn_point_metadata.json
    with tempfile.TemporaryDirectory() as tmpdir:
        meta = {str(mid): {'H': h, 'W': w, 'num_points': h * w}
                for mid, (h, w) in MAT_SHAPES.items()}
        with open(Path(tmpdir) / 'bonn_point_metadata.json', 'w') as f:
            json.dump(meta, f)

        # Touch synthetic mat files so glob discovery picks them up
        for mid in MAT_SHAPES:
            (Path(tmpdir) / f'mat{mid:04d}_poly.exr').touch()

        cfg = _build_cfg(tmpdir, debug_num=DEBUG_NUM, ratio=RATIO)

        patches = _make_patches(n_poly_imgs=4)
        for p in patches:
            p.start()
        try:
            from utils.dataset.bonn import BonnDataset, BonnValDataset
            from model.neural_brdf_refactored import BonnLatentBRDF

            # ------------------------------------------------------------------
            # 0. RNG determinism
            # ------------------------------------------------------------------
            print("[0] RNG determinism")
            for mid, (h, w) in list(MAT_SHAPES.items())[:3]:
                n_full = h * w
                rng_a = np.random.default_rng(mid).choice(n_full,
                    size=max(1, int(n_full * RATIO)), replace=False)
                rng_b = np.random.default_rng(mid).choice(n_full,
                    size=max(1, int(n_full * RATIO)), replace=False)
                _check(f"mat{mid:04d} (n_full={n_full})",
                       np.array_equal(np.sort(rng_a), np.sort(rng_b)))

            # ------------------------------------------------------------------
            # 1. BonnLatentBRDF latent bank size & offsets
            # ------------------------------------------------------------------
            print("\n[1] BonnLatentBRDF bank")
            brdf = BonnLatentBRDF(cfg.material)
            metadata = brdf.metadata
            expected_total = 0
            for mat_info in metadata['materials']:
                mid = mat_info['material_id']
                H, W = mat_info['H'], mat_info['W']
                expected_n = max(1, int(H * W * RATIO))
                _check(f"mat{mid:04d} num_points",
                       mat_info['num_points'] == expected_n,
                       f"got {mat_info['num_points']}, expected {expected_n}")
                expected_total += mat_info['num_points']
            bank_rows = brdf.point_latent_bank.weight.shape[0]
            _check("bank rows == sum(num_points)",
                   bank_rows == expected_total,
                   f"bank={bank_rows}, sum={expected_total}")

            # ------------------------------------------------------------------
            # 2. BonnDataset compacted xyz matches canonical RNG selection
            # ------------------------------------------------------------------
            print("\n[2] BonnDataset (split=train) — xyz subset")
            train_ds = BonnDataset(cfg, tmpdir, split='train')
            train_mats = train_ds._all_data.materials
            train_count_by_id = {m['mat_id']: m['xyz'].shape[0] for m in train_mats}

            for m in train_mats:
                mid = m['mat_id']
                H, W = MAT_SHAPES[mid]
                n_full = H * W
                expected_sub = np.sort(
                    np.random.default_rng(mid).choice(
                        n_full, size=max(1, int(n_full * RATIO)),
                        replace=False))
                full_xyz = _fake_xyz(mid).reshape(-1, 3)
                expected_xyz = full_xyz[expected_sub]
                _check(f"mat{mid:04d} xyz subset",
                       m['xyz'].shape == expected_xyz.shape and
                       np.allclose(m['xyz'], expected_xyz, atol=1e-6),
                       f"shape={m['xyz'].shape}")
                _check(f"mat{mid:04d} point_ids dense 0..N-1",
                       np.array_equal(m['point_ids'],
                                      np.arange(m['xyz'].shape[0], dtype=np.int64)))
                # rgbs and gray_vals second axis must equal the new V
                v_sub = m['xyz'].shape[0]
                _check(f"mat{mid:04d} rgbs V == V_sub",
                       m['rgbs'].shape[1] == v_sub,
                       f"rgbs V={m['rgbs'].shape[1]}, want={v_sub}")
                if m['gray_vals'] is not None:
                    _check(f"mat{mid:04d} gray V == V_sub",
                           m['gray_vals'].shape[1] == v_sub,
                           f"gray V={m['gray_vals'].shape[1]}, want={v_sub}")
                if m['gt_normals'] is not None:
                    _check(f"mat{mid:04d} gt_normals V == V_sub",
                           m['gt_normals'].shape[0] == v_sub)
                # The bank num_points slot for this mat must equal v_sub
                bank_slot = next(mi for mi in metadata['materials']
                                 if mi['material_id'] == mid)
                _check(f"mat{mid:04d} bank slot matches dataloader V",
                       bank_slot['num_points'] == v_sub,
                       f"bank={bank_slot['num_points']}, V={v_sub}")

            # ------------------------------------------------------------------
            # 3. Sampled batch index validity
            # ------------------------------------------------------------------
            print("\n[3] Sampled-batch index validity")
            it = iter(train_ds)
            for b in range(4):
                batch = next(it)
                pids = batch['point_ids'].numpy()
                mids = batch['material_ids'].numpy()
                bad = 0
                for mid in np.unique(mids):
                    v = train_count_by_id.get(int(mid))
                    sub = pids[mids == mid]
                    if v is None or sub.size == 0:
                        continue
                    bad += int(((sub < 0) | (sub >= v)).sum())
                _check(f"  batch {b}: per-material point_ids in [0, V_sub)",
                       bad == 0, f"out-of-range={bad}")

                global_ids = brdf.get_global_point_id(
                    torch.from_numpy(mids).long(),
                    torch.from_numpy(pids).long(),
                ).numpy()
                _check(f"  batch {b}: global ids in [0, bank_size)",
                       (global_ids.min() >= 0) and (global_ids.max() < bank_rows),
                       f"min={global_ids.min()}, max={global_ids.max()}, bank={bank_rows}")

            # ------------------------------------------------------------------
            # 4. BonnValDataset uses identical subset for the same mat_id
            # ------------------------------------------------------------------
            print("\n[4] BonnValDataset alignment")
            val_ds = BonnValDataset(cfg, tmpdir)
            val_mid = val_ds.mat_id
            val_xyz = val_ds._items[0]['xyz'].numpy()
            val_pids = val_ds._items[0]['point_ids'].numpy()

            train_mat = next((m for m in train_mats if m['mat_id'] == val_mid), None)
            _check(f"  val mat_id={val_mid} exists in train split",
                   train_mat is not None)
            _check(f"  val xyz == train xyz (identical pixel subset)",
                   val_xyz.shape == train_mat['xyz'].shape and
                   np.allclose(val_xyz, train_mat['xyz'], atol=1e-6),
                   f"val={val_xyz.shape}, train={train_mat['xyz'].shape}")
            _check(f"  val pids dense 0..N-1",
                   np.array_equal(val_pids,
                                  np.arange(val_xyz.shape[0], dtype=np.int64)))

            # ------------------------------------------------------------------
            # 4b. sub_indices field — must be a 1-D LongTensor whose length
            # equals the supervised V_sub and whose values index back into the
            # original H*W flat grid. The trainer scatters per-pixel arrays
            # onto a zero canvas at these positions for visualisation.
            # ------------------------------------------------------------------
            print("\n[4b] sub_indices field (visualisation scatter map)")
            val_sub_idx = val_ds._items[0]['sub_indices'].numpy()
            H, W = MAT_SHAPES[val_mid]
            n_full = H * W
            expected_sub = np.sort(np.random.default_rng(val_mid).choice(
                n_full, size=max(1, int(n_full * RATIO)), replace=False))
            _check("sub_indices length == V_sub",
                   val_sub_idx.shape[0] == val_xyz.shape[0],
                   f"got {val_sub_idx.shape[0]}, want {val_xyz.shape[0]}")
            _check("sub_indices values in [0, H*W)",
                   (val_sub_idx.min() >= 0) and (val_sub_idx.max() < n_full),
                   f"min={val_sub_idx.min()}, max={val_sub_idx.max()}, H*W={n_full}")
            _check("sub_indices == canonical RNG subset",
                   np.array_equal(val_sub_idx, expected_sub),
                   f"head got={val_sub_idx[:5].tolist()}, want={expected_sub[:5].tolist()}")

            # Spot-check the scatter math the trainer will use:
            # canvas[sub_indices] = val_rgbs   →   reshape to (H, W, 3)
            # → at each (row, col) the pixel value should be either zero
            # (non-supervised) or the matching val_rgbs entry.
            val_rgbs = val_ds._items[0]['rgbs'].numpy()
            canvas = np.zeros((n_full, 3), dtype=val_rgbs.dtype)
            canvas[val_sub_idx] = val_rgbs
            canvas_img = canvas.reshape(H, W, 3)
            n_nonzero = int((canvas.sum(-1) != 0).sum())
            n_supervised = int((val_rgbs.sum(-1) != 0).sum())
            _check("scatter places supervised values at correct positions",
                   n_nonzero == n_supervised,
                   f"canvas non-zero={n_nonzero}, val non-zero={n_supervised}")
            _check("non-supervised pixels are zero (black)",
                   (canvas[np.setdiff1d(np.arange(n_full), val_sub_idx)] == 0).all())
            _check("scatter image shape == (H, W, 3)",
                   canvas_img.shape == (H, W, 3),
                   f"got {canvas_img.shape}")

            # ------------------------------------------------------------------
            # 5. Ratio=1.0 is a no-op — full V everywhere
            # ------------------------------------------------------------------
            print("\n[5] Ratio=1.0 (no subsample) is a no-op")
            cfg_full = _build_cfg(tmpdir, debug_num=DEBUG_NUM, ratio=1.0)
            brdf_full = BonnLatentBRDF(cfg_full.material)
            for mat_info in brdf_full.metadata['materials']:
                mid = mat_info['material_id']
                H, W = MAT_SHAPES[mid]
                _check(f"mat{mid:04d} num_points == H*W",
                       mat_info['num_points'] == H * W,
                       f"got {mat_info['num_points']}, want {H*W}")
            ds_full = BonnDataset(cfg_full, tmpdir, split='train')
            for m in ds_full._all_data.materials:
                H, W = MAT_SHAPES[m['mat_id']]
                _check(f"mat{m['mat_id']:04d} V == H*W",
                       m['xyz'].shape[0] == H * W,
                       f"V={m['xyz'].shape[0]}, want {H*W}")
        finally:
            for p in patches:
                p.stop()

    print(f"\n{'='*70}")
    print("All FAST consistency checks PASSED")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
