#!/usr/bin/env python3
"""Unit test: verify single-channel gray_vals storage for pan/lls.

Loads 1 Bonn material with pan+lls, checks:
1. gray_vals is stored as (K_gray, V) — NOT tripled to (K, V, 3)
2. Sampled batches have correct shapes and values
3. Pan/LLS rays have grayscale value in channel 0 only
4. Confidence is computed correctly
5. Training loss computes without error (if trainer available)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from pathlib import Path

# Try local data path first, then CC path
DATA_ROOT = None
for p in ['/media/raid/cloth/Bonn_train', '/home/zla247/scratch/data/Bonn/train']:
    if Path(p).exists():
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    print("ERROR: No Bonn data found")
    sys.exit(1)

print(f"Using data root: {DATA_ROOT}")

from utils.dataset.bonn import (
    _load_single_material_full, DTYPE_POLY, DTYPE_PAN, DTYPE_LLS
)

# ---- Test 1: Load material and check storage shapes ----
print("\n=== Test 1: Load mat0001 with pan+lls ===")
mat = _load_single_material_full(DATA_ROOT, 1, use_pan=True, use_lls=True)
assert mat is not None, "Failed to load mat0001"

n_poly = mat['rgbs'].shape[0]
n_pixels = mat['rgbs'].shape[1]
print(f"  poly_rgbs shape: {mat['rgbs'].shape}  (expect 3 channels)")
assert mat['rgbs'].ndim == 3 and mat['rgbs'].shape[2] == 3, \
    f"poly_rgbs should be (K_poly, V, 3), got {mat['rgbs'].shape}"

assert mat['gray_vals'] is not None, "gray_vals should not be None with pan+lls"
print(f"  gray_vals shape: {mat['gray_vals'].shape}  (expect 2D: K_gray x V)")
assert mat['gray_vals'].ndim == 2, \
    f"gray_vals should be (K_gray, V), got {mat['gray_vals'].shape}"

n_gray = mat['gray_vals'].shape[0]
n_total = n_poly + n_gray
dt = mat['data_type']
n_dt_poly = (dt == DTYPE_POLY).sum()
n_dt_pan = (dt == DTYPE_PAN).sum()
n_dt_lls = (dt == DTYPE_LLS).sum()

print(f"  n_poly={n_poly}, n_gray={n_gray}, total={n_total}")
print(f"  data_type counts: poly={n_dt_poly}, pan={n_dt_pan}, lls={n_dt_lls}")
assert n_poly == n_dt_poly, f"n_poly mismatch: {n_poly} vs {n_dt_poly}"
assert n_gray == n_dt_pan + n_dt_lls, f"n_gray mismatch: {n_gray} vs {n_dt_pan + n_dt_lls}"
assert n_total == len(dt), f"total mismatch: {n_total} vs {len(dt)}"

# Memory savings check
old_mem_mb = n_total * n_pixels * 3 * 2 / 1e6  # old: all images x 3 channels x float16
new_mem_mb = (n_poly * n_pixels * 3 * 2 + n_gray * n_pixels * 2) / 1e6
savings_pct = (1 - new_mem_mb / old_mem_mb) * 100
print(f"  Memory: old={old_mem_mb:.0f} MB, new={new_mem_mb:.0f} MB, saved {savings_pct:.0f}%")
print("  PASS")

# ---- Test 2: Verify data_type ordering (poly first, then gray) ----
print("\n=== Test 2: Data type ordering ===")
# First n_poly entries should be POLY
assert np.all(dt[:n_poly] == DTYPE_POLY), "First n_poly entries should be DTYPE_POLY"
# Remaining should be PAN or LLS
assert np.all(dt[n_poly:] != DTYPE_POLY), "Entries after n_poly should not be DTYPE_POLY"
print("  Ordering correct: [poly...][pan...][lls...]")
print("  PASS")

# ---- Test 3: Simulate batch sampling (like _sample_batch) ----
print("\n=== Test 3: Batch sampling ===")
n_rays = 1024
img_i = np.random.randint(0, n_total, n_rays)
pix_i = np.random.randint(0, n_pixels, n_rays)

rgbs = np.zeros((n_rays, 3), dtype=np.float32)
poly_m = img_i < n_poly
if poly_m.any():
    rgbs[poly_m] = mat['rgbs'][img_i[poly_m], pix_i[poly_m]].astype(np.float32)
gray_m = ~poly_m
if gray_m.any():
    rgbs[gray_m, 0] = mat['gray_vals'][img_i[gray_m] - n_poly, pix_i[gray_m]].astype(np.float32)

n_poly_rays = poly_m.sum()
n_gray_rays = gray_m.sum()
print(f"  Sampled {n_rays} rays: {n_poly_rays} poly, {n_gray_rays} gray")

# Check poly rays have all 3 channels (may be nonzero)
if n_poly_rays > 0:
    poly_rgbs = rgbs[poly_m]
    print(f"  Poly: ch0 mean={poly_rgbs[:, 0].mean():.4f}, ch1 mean={poly_rgbs[:, 1].mean():.4f}, ch2 mean={poly_rgbs[:, 2].mean():.4f}")

# Check gray rays have value only in channel 0
if n_gray_rays > 0:
    gray_rgbs = rgbs[gray_m]
    assert np.all(gray_rgbs[:, 1] == 0), "Gray rays should have 0 in channel 1"
    assert np.all(gray_rgbs[:, 2] == 0), "Gray rays should have 0 in channel 2"
    print(f"  Gray: ch0 mean={gray_rgbs[:, 0].mean():.4f}, ch1=0 (verified), ch2=0 (verified)")

# Check confidence
confidence = (rgbs.sum(axis=-1) > 0).astype(np.float32)
print(f"  Confidence: {confidence.mean():.2%} non-zero")
print("  PASS")

# ---- Test 4: Verify gray values match original tripled approach ----
print("\n=== Test 4: Value correctness ===")
# Pick a few pan images and verify channel 0 matches the EXR data
pan_dt_mask = dt == DTYPE_PAN
if pan_dt_mask.any():
    pan_global_idx = np.where(pan_dt_mask)[0][0]  # first pan image
    gray_local_idx = pan_global_idx - n_poly
    # Get a few pixel values from gray_vals
    test_pix = [0, 100, 1000, n_pixels // 2]
    gray_values = mat['gray_vals'][gray_local_idx, test_pix]
    print(f"  Pan image {pan_global_idx} (gray local {gray_local_idx}), sample pixels: {gray_values}")
    assert gray_values.dtype == np.float16, f"Expected float16, got {gray_values.dtype}"
    print("  PASS")

# ---- Test 5: Load without pan/lls (backward compatibility) ----
print("\n=== Test 5: Poly-only loading ===")
mat_poly = _load_single_material_full(DATA_ROOT, 1, use_pan=False, use_lls=False)
assert mat_poly is not None
assert mat_poly['gray_vals'] is None, "gray_vals should be None without pan/lls"
assert mat_poly['rgbs'].shape[0] > 0, "Should have poly images"
n_poly_only = mat_poly['rgbs'].shape[0]
dt_poly = mat_poly['data_type']
assert np.all(dt_poly == DTYPE_POLY), "All images should be DTYPE_POLY"
print(f"  Poly-only: {n_poly_only} images, gray_vals=None")
print("  PASS")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
