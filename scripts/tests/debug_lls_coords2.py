#!/usr/bin/env python3
"""Continue LLS coord investigation: check _xyz_others.exr structure."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pyexr
from pathlib import Path

DATA_ROOT = '/media/raid/cloth/Bonn_train'
mat_id = 1

print("=== xyz_rot000.exr ===")
exr = pyexr.open(f'{DATA_ROOT}/mat{mat_id:04d}_xyz_rot000.exr')
print(f"  channels: {exr.channel_map['all']}")
data = exr.get(group='all')
print(f"  shape: {data.shape}, dtype: {data.dtype}")
print(f"  sample middle: {data[data.shape[0]//2, data.shape[1]//2]}")

print("\n=== xyz_others.exr ===")
exr = pyexr.open(f'{DATA_ROOT}/mat{mat_id:04d}_xyz_others.exr')
print(f"  channels: {exr.channel_map['all']}")
data2 = exr.get(group='all')
print(f"  shape: {data2.shape}, dtype: {data2.dtype}")

# The xyz_others likely has more than 3 channels (one set per rotation)
# Channel 0,1,2 may be rot045; 3,4,5 rot090; etc.
# OR it stores the unrotated-then-reprojected coords
print(f"  sample middle (first 12 channels): {data2[data2.shape[0]//2, data2.shape[1]//2, :min(12, data2.shape[2])]}")

# Compare x,y,z of rot000 vs first 3 channels of others
if data2.shape[2] >= 3:
    diff = data2[..., :3] - data[..., :3]
    print(f"  max diff (others_ch0:3 - rot000): {np.abs(diff).max()}")

# Also check xyz_refined_others
print("\n=== xyz_refined_others.exr ===")
exr = pyexr.open(f'{DATA_ROOT}/mat{mat_id:04d}_xyz_refined_others.exr')
print(f"  channels: {exr.channel_map['all']}")
data3 = exr.get(group='all')
print(f"  shape: {data3.shape}")
print(f"  sample middle (first 12): {data3[data3.shape[0]//2, data3.shape[1]//2, :min(12, data3.shape[2])]}")
