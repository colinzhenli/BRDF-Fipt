#!/usr/bin/env python3
"""Inspect a Bonn calibration .mat file to discover per-image color-conversion
coefficient structure.

Bonn_dataset.py (official reference) uses, per-image:
  - poly2pan[cv][il]  (3,)   poly RGB -> pan scalar
  - poly2lls[cv]       (3,)   poly RGB -> lls scalar   (depends on cv only)
  - poly2poly[cv][il]  (3,)   poly RGB -> poly RGB     (gain/bias)
  - pan2pan[cv][il]   scalar
  - pan2lls[cv]       scalar

This script dumps every top-level key and every per-rotation sub-field of the
.mat file so we can figure out where those coefficients live and how to load
them in our bonn.py.
"""
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
assert DATA_ROOT, f'no Bonn data root found: {DATA_ROOTS}'
print(f'DATA_ROOT = {DATA_ROOT}')

MAT_IDS = [1, 2, 3]  # inspect a few to confirm shapes are consistent


def describe(v, indent=2):
    prefix = ' ' * indent
    if isinstance(v, np.ndarray):
        dt = str(v.dtype)
        shp = v.shape
        # summarize value
        try:
            if v.size <= 12:
                flat = v.astype(np.float64).flatten()
                sample = np.array2string(flat, precision=4, suppress_small=True)
            else:
                flat = v.astype(np.float64).flatten()
                sample = (
                    f'first6={np.array2string(flat[:6], precision=4, suppress_small=True)} '
                    f'min={flat.min():.4f} max={flat.max():.4f}'
                )
        except (TypeError, ValueError):
            sample = '<non-numeric>'
        print(f'{prefix}shape={shp}  dtype={dt}  {sample}')
    else:
        print(f'{prefix}{type(v).__name__}  {v!r}')


for mat_id in MAT_IDS:
    path = f'{DATA_ROOT}/mat{mat_id:04d}_calibration.mat'
    print('\n' + '=' * 72)
    print(f'FILE: {path}')
    print('=' * 72)
    raw = spio.loadmat(path)

    top_keys = [k for k in raw.keys() if not k.startswith('__')]
    print('\n[Top-level keys]')
    for k in top_keys:
        v = raw[k]
        print(f'  {k}:')
        describe(v, indent=4)

    # Drill into each rot* struct if present
    for rot_key in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
        if rot_key not in raw:
            continue
        rd = raw[rot_key][0, 0]
        print(f'\n[{rot_key}] fields: {rd.dtype.names}')
        for field in rd.dtype.names:
            v = rd[field]
            print(f'  {field}:')
            describe(v, indent=4)

    # Specific focus: look for poly2pan / poly2lls / pan2lls / weights
    print('\n[Candidate conversion keys found]')
    for k in top_keys:
        low = k.lower()
        if any(s in low for s in ('poly2', 'pan2', 'lls2', 'weight', 'convert', 'coeff', 'panchromatic', 'colortransform')):
            print(f'  {k}:')
            describe(raw[k], indent=6)

    # Also scan per-rotation fields for conversion-like names
    rot_key = 'rot000'
    if rot_key in raw:
        rd = raw[rot_key][0, 0]
        print(f'\n[rot000 conversion-like fields]')
        for field in rd.dtype.names:
            low = field.lower()
            if any(s in low for s in ('poly2', 'pan2', 'lls2', 'weight', 'convert', 'coeff')):
                print(f'  {field}:')
                describe(rd[field], indent=6)

    if mat_id == MAT_IDS[0]:
        # extra: dump everything we haven't already covered
        print('\n[Full nested dump (rot000 only)]')
        rd = raw['rot000'][0, 0]
        for field in rd.dtype.names:
            v = rd[field]
            print(f'  {field}:  type={type(v).__name__}  shape={getattr(v, "shape", None)}')
