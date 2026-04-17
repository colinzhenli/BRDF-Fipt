#!/usr/bin/env python3
"""Compare radiometric scales of poly vs pan vs lls for mat0001.

If the radiometric calibration differs across modalities, the network will
learn biased BRDFs when we mix them.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pyexr

DATA_ROOT = '/media/raid/cloth/Bonn_train'
mat_id = 1

# Load poly
poly = pyexr.open(f'{DATA_ROOT}/mat{mat_id:04d}_poly.exr')
poly_data = poly.get(group='all', precision=pyexr.HALF)
poly_ch = poly.channel_map['all']
print(f"POLY: shape {poly_data.shape}, channels: {poly_ch[:6]} ... (first 6 of {len(poly_ch)})")
# Find only channels with ch ending _R (red)
print(f"  mean all: {poly_data.astype(np.float32).mean():.4f}")
print(f"  max  all: {poly_data.astype(np.float32).max():.4f}")
print(f"  99th pct all: {np.percentile(poly_data.astype(np.float32).flatten(), 99):.4f}")

# Load pan
pan = pyexr.open(f'{DATA_ROOT}/mat{mat_id:04d}_pan.exr')
pan_data = pan.get(group='all', precision=pyexr.HALF)
pan_ch = pan.channel_map['all']
print(f"\nPAN: shape {pan_data.shape}, {len(pan_ch)} channels")
print(f"  mean all: {pan_data.astype(np.float32).mean():.4f}")
print(f"  max  all: {pan_data.astype(np.float32).max():.4f}")
print(f"  99th pct all: {np.percentile(pan_data.astype(np.float32).flatten(), 99):.4f}")

# Load lls
lls = pyexr.open(f'{DATA_ROOT}/mat{mat_id:04d}_lls.exr')
lls_data = lls.get(group='all', precision=pyexr.HALF)
lls_ch = lls.channel_map['all']
print(f"\nLLS: shape {lls_data.shape}, {len(lls_ch)} channels")
print(f"  mean all: {lls_data.astype(np.float32).mean():.4f}")
print(f"  max  all: {lls_data.astype(np.float32).max():.4f}")
print(f"  99th pct all: {np.percentile(lls_data.astype(np.float32).flatten(), 99):.4f}")

# Compare: For same LED, does pan match a weighted sum of poly?
# Poly uses il025-il029 (color-filtered LEDs), pan uses il001-il024 (unfiltered).
# So we can't directly compare. Compare means of comparable LEDs.
# Pan LED 1 at rot000 should be ~luminance(poly at rot000) IF poly LEDs emitted white light
# and pan LED 1 had the same spectrum. But they don't, so the scales might differ.

# A better comparison: predicted gray from BRDF should match observed gray.
# For now, just check that scales are similar order of magnitude.

# Also: is LLS data in the same units as poly/pan? The MC formula pi*weighted_avg predicts
# something in BRDF units (albedo * cos / pi ~ albedo/pi). If LLS data is in reflected
# radiance units (not BRDF), there's a missing factor.

# Check non-zero pixel stats — per-image means
print("\n=== Per-image mean brightness (valid pixels only) ===")
valid_poly = poly_data.sum(axis=-1) > 0  # per-pixel valid mask
poly_per_img = []
n_poly_img = len(poly_ch) // 3
print(f"  Poly (n={n_poly_img}): {poly_data.reshape(-1, 3, n_poly_img)[valid_poly.reshape(-1)].mean(axis=(0,1))[:5]}")

# Per-modality simple summary
def _summary(data, ch_names, tag):
    d = data.astype(np.float32)
    if d.ndim == 3 and d.shape[-1] > 1:
        flat = d.reshape(-1, d.shape[-1])
    else:
        flat = d.reshape(-1, 1)
    non_zero = (flat.sum(axis=-1) > 0)
    if non_zero.sum() > 0:
        valid = flat[non_zero]
        print(f"  {tag}: non-zero pix={non_zero.sum()}  mean={valid.mean():.4f}  median={np.median(valid):.4f}  99p={np.percentile(valid, 99):.4f}")

# Just look at channel 0 per modality
print("\n=== Channel-0 stats (picks out one image) ===")
_summary(poly_data[..., 0:1], None, 'poly ch0')
_summary(pan_data[..., 0:1], None, 'pan  ch0')
_summary(lls_data[..., 0:1], None, 'lls  ch0')

# check the poly2pan weights metadata in calibration file
import scipy.io as spio
raw = spio.loadmat(f'{DATA_ROOT}/mat{mat_id:04d}_calibration.mat')
if 'poly2panIndices' in raw:
    print(f"\n=== poly2pan mapping (for pan->poly conversion) ===")
    p2p_idx = raw['poly2panIndices']
    p2p_w = raw['poly2panWeights']
    print(f"  poly2panIndices shape: {p2p_idx.shape}, dtype: {p2p_idx.dtype}")
    print(f"  poly2panWeights shape: {p2p_w.shape}")
    print(f"  weights sample: {p2p_w.flatten()[:6]}")
