#!/usr/bin/env python3
"""Unit tests for per-image pan_weights loading in utils/dataset/bonn.py.

Verifies:
  1. `_parse_poly2pan_weights` extracts all 20 per-(cv, il) entries from a
     real calibration .mat, each a 3-vector with reasonable magnitudes.
     Also returns `global_avg` over all 20 entries.
  2. Per-cv averaged weights approximately match the global fallback
     [0.34, 0.36, 0.28] but are camera-specific (not identical across cv).
  3. Per-image `pan_weights` attached to loaded materials align with the
     calibration for each image's (cv, il):
         poly (il026/027/028/031/032) → per-(cv, il) entry,
         poly (other LEDs)            → global_avg,
         pan                          → global_avg,
         lls                          → global_avg × _LLS_EMPIRICAL_SCALE.
  4. A `_sample_batch` draws include pan_weights matching the chosen image.
  5. BonnSingleMaterialDataset and its val counterpart also expose
     pan_weights of the right shape.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from pathlib import Path

import numpy as np
import scipy.io as spio
import torch
from omegaconf import OmegaConf

import utils.dataset.bonn as bonn_mod
from utils.dataset.bonn import (
    _parse_poly2pan_weights,
    _pan_weights_for_image,
    _DEFAULT_PAN_WEIGHTS,
    _LLS_EMPIRICAL_SCALE,
    BonnDataset,
    BonnValDataset,
    BonnSingleMaterialDataset,
    BonnSingleMaterialValDataset,
    DTYPE_POLY, DTYPE_PAN, DTYPE_LLS,
)

DATA_ROOTS = [
    '/media/raid/cloth/Bonn_train',
    '/home/zla247/scratch/data/Bonn/train',
]
DATA_ROOT = next((p for p in DATA_ROOTS if Path(p).exists()), None)
assert DATA_ROOT, f'no Bonn data root found: {DATA_ROOTS}'

TRAIN_LIST = Path(DATA_ROOT) / 'UBOFAB19_train_meas.txt'
# Build a tiny list file with just a couple materials we know exist.
TINY_LIST = Path('/tmp/_bonn_test_train_list.txt')


def _write_tiny_list(mat_ids):
    TINY_LIST.write_text('\n'.join(str(i) for i in mat_ids) + '\n')


def _build_cfg(mat_ids, use_pan=True, use_lls=True, overfit_id=None):
    cfg = OmegaConf.create({
        'data': {
            'rays_num': 128,
            'debug': False,
            'points_per_material': 100,
            'random_observations': True,
            'subsample_ratio': 1.0,
            'val_materials': 1,
            'val_points': 50,
            'use_pan': use_pan,
            'use_lls': use_lls,
            'training_list_path': str(TINY_LIST),
            'val_view_ratio': 0.2,
            'val_seed': 42,
            'valid_num': 1,
        }
    })
    _write_tiny_list(mat_ids)
    if overfit_id is not None:
        cfg.data.overfit_mat_id = overfit_id
    return cfg


# ----------------------------------------------------------------------
# Test 1 — raw parser
# ----------------------------------------------------------------------
def test_parser_structure():
    path = f'{DATA_ROOT}/mat0001_calibration.mat'
    raw = spio.loadmat(path)
    w = _parse_poly2pan_weights(raw)
    per_il = w['per_il']
    per_cv = w['per_cv']
    global_avg = w['global_avg']

    # 4 cameras × 5 poly LEDs = 20 entries
    assert len(per_il) == 20, f'expected 20 (cv, il) entries, got {len(per_il)}'
    for key, vec in per_il.items():
        cv, il = key
        assert re.fullmatch(r'cv\d{2}', cv), cv
        assert re.fullmatch(r'il\d{3}', il), il
        assert vec.shape == (3,), vec.shape
        assert vec.dtype == np.float32
        assert vec.min() > 0.0 and vec.max() < 1.0, f'odd range for {key}: {vec}'
        # each entry sums to ≈1 (they're RGB blending weights)
        assert abs(vec.sum() - 1.0) < 0.05, f'{key} sums to {vec.sum()}'

    # per-cv averages: 4 cameras, each close-to-global-default but not identical
    assert set(per_cv.keys()) == {'cv01', 'cv02', 'cv03', 'cv04'}
    defaults = _DEFAULT_PAN_WEIGHTS
    for cv, vec in per_cv.items():
        assert vec.shape == (3,) and vec.dtype == np.float32
        assert np.allclose(vec, defaults, atol=0.02), (cv, vec, defaults)
    # sanity: cv01 != cv04 (camera-specific)
    assert not np.allclose(per_cv['cv01'], per_cv['cv04'], atol=1e-4), \
        'expected per-cv differences, but all cameras produced identical weights'

    # global_avg: 3-vector near default, equals mean over all 20 per_il entries
    assert global_avg.shape == (3,) and global_avg.dtype == np.float32
    assert np.allclose(global_avg, defaults, atol=0.02), (global_avg, defaults)
    expected_global = np.stack(list(per_il.values()), axis=0).mean(axis=0)
    assert np.allclose(global_avg, expected_global, atol=1e-6)
    # global_avg ≈ average of the 4 per_cv means as well
    per_cv_mean = np.stack([per_cv[f'cv{i:02d}'] for i in range(1, 5)], axis=0).mean(axis=0)
    assert np.allclose(global_avg, per_cv_mean, atol=1e-6)
    print('[PASS] test_parser_structure')


# ----------------------------------------------------------------------
# Test 2 — cross-material identity (weights are instrument-only)
# ----------------------------------------------------------------------
def test_cross_material_identity():
    path1 = f'{DATA_ROOT}/mat0001_calibration.mat'
    path2 = f'{DATA_ROOT}/mat0002_calibration.mat'
    w1 = _parse_poly2pan_weights(spio.loadmat(path1))
    w2 = _parse_poly2pan_weights(spio.loadmat(path2))
    for key in w1['per_il']:
        assert np.allclose(w1['per_il'][key], w2['per_il'][key], atol=1e-6), \
            f'mat-to-mat weight mismatch at {key}'
    for cv in w1['per_cv']:
        assert np.allclose(w1['per_cv'][cv], w2['per_cv'][cv], atol=1e-6)
    assert np.allclose(w1['global_avg'], w2['global_avg'], atol=1e-6)
    print('[PASS] test_cross_material_identity')


# ----------------------------------------------------------------------
# Test 3 — BonnDataset attaches per-image pan_weights
# ----------------------------------------------------------------------
def test_bonn_dataset_loads_pan_weights():
    cfg = _build_cfg([1], use_pan=True, use_lls=True)
    ds = BonnDataset(cfg, DATA_ROOT, split='train')
    assert len(ds._all_data.materials) == 1
    mat = ds._all_data.materials[0]

    K_poly = mat['rgbs'].shape[0]
    K_gray = mat['gray_vals'].shape[0] if mat['gray_vals'] is not None else 0
    K_total = K_poly + K_gray
    assert mat['pan_weights'].shape == (K_total, 3), mat['pan_weights'].shape
    assert mat['pan_weights'].dtype == np.float32

    # Reload raw calibration and confirm a sample poly image's weight
    # matches per-(cv, il) entry in the .mat.
    mat_id = mat['mat_id']
    raw = spio.loadmat(f'{DATA_ROOT}/mat{mat_id:04d}_calibration.mat')
    w = _parse_poly2pan_weights(raw)
    global_avg = w['global_avg']
    lls_expected = (global_avg * _LLS_EMPIRICAL_SCALE).astype(np.float32)

    # Confirm first poly image: parse its (cv, il) from channel order in
    # _parse_poly_channels — use the same code path to rebuild labels.
    import pyexr
    exr = pyexr.open(f'{DATA_ROOT}/mat{mat_id:04d}_poly.exr')
    ch = exr.channel_map['all']
    poly_imgs = bonn_mod._parse_poly_channels(ch)
    assert len(poly_imgs) == K_poly
    for ki in range(min(5, K_poly)):
        cv, il = poly_imgs[ki]['camera'], poly_imgs[ki]['led']
        if (cv, il) in w['per_il']:
            expected = w['per_il'][(cv, il)]  # calibrated filter LED
        else:
            expected = global_avg             # other poly LEDs → global fallback
        got = mat['pan_weights'][ki]
        assert np.allclose(got, expected, atol=1e-6), \
            f'poly img {ki} ({cv}, {il}) pan_weights mismatch: got {got}, expected {expected}'

    # Confirm pan rays use global_avg, lls rays use global_avg × scale
    if K_gray > 0:
        dtypes = mat['data_type']
        pan_inds = np.where(dtypes == DTYPE_PAN)[0]
        if pan_inds.size > 0:
            ki = int(pan_inds[0])
            assert np.allclose(mat['pan_weights'][ki], global_avg, atol=1e-6), \
                f'pan img {ki} weights {mat["pan_weights"][ki]} != global_avg {global_avg}'
        lls_inds = np.where(dtypes == DTYPE_LLS)[0]
        if lls_inds.size > 0:
            ki = int(lls_inds[0])
            assert np.allclose(mat['pan_weights'][ki], lls_expected, atol=1e-6), \
                f'lls img {ki} weights {mat["pan_weights"][ki]} != global_avg*{_LLS_EMPIRICAL_SCALE} = {lls_expected}'
    print('[PASS] test_bonn_dataset_loads_pan_weights')


# ----------------------------------------------------------------------
# Test 4 — sample_batch propagates pan_weights
# ----------------------------------------------------------------------
def test_sample_batch_has_pan_weights():
    cfg = _build_cfg([1], use_pan=True, use_lls=True)
    ds = BonnDataset(cfg, DATA_ROOT, split='train')
    batch = ds._sample_batch(ds._all_data, 64)
    assert 'pan_weights' in batch, batch.keys()
    pw = batch['pan_weights']
    assert pw.shape == (64, 3), pw.shape
    assert pw.dtype == torch.float32
    assert torch.all(pw > 0) and torch.all(pw < 1)
    print('[PASS] test_sample_batch_has_pan_weights')


# ----------------------------------------------------------------------
# Test 5 — BonnValDataset also attaches pan_weights per item
# ----------------------------------------------------------------------
def test_val_dataset_has_pan_weights():
    cfg = _build_cfg([1], use_pan=True, use_lls=True)
    ds = BonnValDataset(cfg, DATA_ROOT)
    item = ds[0]
    assert 'pan_weights' in item
    pw = item['pan_weights']
    assert pw.ndim == 2 and pw.shape[1] == 3
    assert pw.shape[0] == ds.n_pixels
    # All pixels in one image share the same weights
    assert torch.allclose(pw[0], pw[-1])
    print('[PASS] test_val_dataset_has_pan_weights')


# ----------------------------------------------------------------------
# Test 6 — single-material dataset & its val
# ----------------------------------------------------------------------
def test_single_material_datasets():
    cfg = _build_cfg([1], use_pan=True, use_lls=True, overfit_id=1)
    ds = BonnSingleMaterialDataset(cfg, DATA_ROOT, split='train')
    assert ds.pan_weights.shape == (ds.n_images, 3), ds.pan_weights.shape
    batch = ds._sample_batch(32)
    assert 'pan_weights' in batch and batch['pan_weights'].shape == (32, 3)

    val_ds = BonnSingleMaterialValDataset(cfg, DATA_ROOT)
    item = val_ds[0]
    assert 'pan_weights' in item
    assert item['pan_weights'].shape == (val_ds.n_pixels, 3)
    print('[PASS] test_single_material_datasets')


# ----------------------------------------------------------------------
# Test 7 — _pan_weights_for_image lookup logic
# ----------------------------------------------------------------------
def test_lookup_helper():
    raw = spio.loadmat(f'{DATA_ROOT}/mat0001_calibration.mat')
    w = _parse_poly2pan_weights(raw)
    calib = {'poly2pan_per_il':     w['per_il'],
             'poly2pan_per_cv':     w['per_cv'],
             'poly2pan_global_avg': w['global_avg']}
    global_avg = w['global_avg']

    # Poly lookup with calibrated filter LED — exact per-(cv, il) vector
    got = _pan_weights_for_image(calib, 'rot000', 'cv02', 'il027', is_poly=True)
    exp = w['per_il'][('cv02', 'il027')]
    assert np.allclose(got, exp, atol=1e-6)

    # Pan lookup — falls back to global_avg (NOT per-cv)
    got = _pan_weights_for_image(calib, 'rot000', 'cv03', led=None, is_poly=False)
    assert np.allclose(got, global_avg, atol=1e-6), (got, global_avg)

    # LLS lookup — global_avg × empirical scale
    got = _pan_weights_for_image(calib, 'rot000', 'cv03', led=None,
                                 is_poly=False, is_lls=True)
    exp = (global_avg * _LLS_EMPIRICAL_SCALE).astype(np.float32)
    assert np.allclose(got, exp, atol=1e-6), (got, exp)

    # Poly lookup with uncalibrated LED — fallback to global_avg
    got = _pan_weights_for_image(calib, 'rot000', 'cv01', 'il999', is_poly=True)
    assert np.allclose(got, global_avg, atol=1e-6), (got, global_avg)

    # Empty calib — returns global default for pan
    got = _pan_weights_for_image({}, 'rot000', 'cv99', led=None, is_poly=False)
    assert np.allclose(got, _DEFAULT_PAN_WEIGHTS, atol=1e-6)

    # Empty calib — LLS returns default × scale
    got = _pan_weights_for_image({}, 'rot000', 'cv99', led=None,
                                 is_poly=False, is_lls=True)
    exp = (_DEFAULT_PAN_WEIGHTS * _LLS_EMPIRICAL_SCALE).astype(np.float32)
    assert np.allclose(got, exp, atol=1e-6)
    print('[PASS] test_lookup_helper')


# ----------------------------------------------------------------------
if __name__ == '__main__':
    test_parser_structure()
    test_cross_material_identity()
    test_lookup_helper()
    test_bonn_dataset_loads_pan_weights()
    test_sample_batch_has_pan_weights()
    test_val_dataset_has_pan_weights()
    test_single_material_datasets()
    print('\n[ALL TESTS PASSED]')
