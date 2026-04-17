#!/usr/bin/env python3
"""Drill into `poly2panWeights` / `poly2panIndices` to understand the structure."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import scipy.io as spio
from pathlib import Path

DATA_ROOTS = [
    '/media/raid/cloth/Bonn_train',
    '/home/zla247/scratch/data/Bonn/train',
]
DATA_ROOT = next((p for p in DATA_ROOTS if Path(p).exists()), None)
assert DATA_ROOT

for mat_id in [1, 2, 3]:
    path = f'{DATA_ROOT}/mat{mat_id:04d}_calibration.mat'
    raw = spio.loadmat(path)
    print(f'\n===== mat{mat_id:04d} =====')

    idx = raw['poly2panIndices'].flatten()
    print(f'poly2panIndices: shape={idx.shape}  dtype={idx.dtype}  '
          f'range=[{idx.min()}, {idx.max()}]  first15={idx[:15].tolist()}')

    wstruct = raw['poly2panWeights'][0, 0]
    print(f'poly2panWeights fields: {wstruct.dtype.names}')
    for field in wstruct.dtype.names:
        v = wstruct[field]
        v_np = np.array(v).astype(np.float64)
        print(f'  {field}: shape={v_np.shape}  dtype={v.dtype}  '
              f'sum={v_np.sum():.4f}  min={v_np.min():.4f}  max={v_np.max():.4f}  '
              f'first6={v_np.flatten()[:6]}')

    # Is there an il025 in the rot000 struct?  Confirm poly LED IDs.
    rd = raw['rot000'][0, 0]
    print(f'\n  rot000 LED fields present: {[f for f in rd.dtype.names if f.startswith("il")]}')

    # Extract ALL weight vectors
    print('\n  === All weight summaries ===')
    all_sum = {}
    for field in wstruct.dtype.names:
        v = np.array(wstruct[field]).astype(np.float64).flatten()
        all_sum[field] = v.sum()
        # Also print if this is a 3-vector (per-channel weights) or spectrum
        if v.size <= 6:
            print(f'    {field:20s}: {v}')
        else:
            # spectrum form; show spectral concentration
            top_inds = np.argsort(v)[-5:]
            print(f'    {field:20s}: shape={v.shape}  top5 inds={top_inds.tolist()}  '
                  f'top5 vals={v[top_inds]}')

    # Is there a global weight somewhere else? Check all top-level keys
    print('\n  === Top-level arrays ===')
    for k in raw.keys():
        if k.startswith('__'):
            continue
        v = raw[k]
        if isinstance(v, np.ndarray):
            if v.dtype.fields is None:
                # plain array
                print(f'    {k}: shape={v.shape}  dtype={v.dtype}')
