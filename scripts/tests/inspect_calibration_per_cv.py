#!/usr/bin/env python3
"""Summarize poly2panWeights per camera (averaged over poly LEDs).

Also confirms (a) weights are instrument-specific (identical across materials)
and (b) per-cv variation is small but non-zero.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import scipy.io as spio
from pathlib import Path
from glob import glob

DATA_ROOTS = [
    '/media/raid/cloth/Bonn_train',
    '/home/zla247/scratch/data/Bonn/train',
]
DATA_ROOT = next((p for p in DATA_ROOTS if Path(p).exists()), None)
assert DATA_ROOT, f'no Bonn data root found: {DATA_ROOTS}'

calib_files = sorted(glob(os.path.join(DATA_ROOT, 'mat*_calibration.mat')))[:8]
print(f'Checking {len(calib_files)} calibration files')

all_per_cv = {}  # mat_id -> {cv: (3,)}
for f in calib_files:
    mat_id = int(Path(f).stem.split('_')[0][3:])
    raw = spio.loadmat(f)
    w = raw['poly2panWeights'][0, 0]
    per_cv = {}
    for cv in ['cv01', 'cv02', 'cv03', 'cv04']:
        vecs = []
        for il in ['il026', 'il027', 'il028', 'il031', 'il032']:
            key = f'{cv}_{il}'
            if key in w.dtype.names:
                vecs.append(np.array(w[key]).astype(np.float64).flatten())
        per_cv[cv] = np.mean(vecs, axis=0)
    all_per_cv[mat_id] = per_cv

# Check mat-to-mat identity
print('\n=== mat_id → per-cv averaged poly2pan weights ===')
print(f'{"mat":>4} {"cv01":>28} {"cv02":>28} {"cv03":>28} {"cv04":>28}')
for mat_id, per_cv in all_per_cv.items():
    row = f'{mat_id:>4} '
    for cv in ['cv01', 'cv02', 'cv03', 'cv04']:
        row += f'{str(np.round(per_cv[cv], 4)):>28} '
    print(row)

# Are weights identical across all mats?
first = all_per_cv[list(all_per_cv.keys())[0]]
all_same = True
for mid, per_cv in all_per_cv.items():
    for cv in ['cv01', 'cv02', 'cv03', 'cv04']:
        if not np.allclose(per_cv[cv], first[cv], atol=1e-6):
            all_same = False
            print(f'  DIFFER: mat{mid} cv={cv}')
print(f'\nAll materials have identical poly2panWeights? {all_same}')

# Show per-cv std across LEDs (to see intra-cv variation)
print('\n=== Intra-cv variation (std across 5 poly LEDs) ===')
raw = spio.loadmat(calib_files[0])
w = raw['poly2panWeights'][0, 0]
for cv in ['cv01', 'cv02', 'cv03', 'cv04']:
    vecs = []
    for il in ['il026', 'il027', 'il028', 'il031', 'il032']:
        key = f'{cv}_{il}'
        vecs.append(np.array(w[key]).astype(np.float64).flatten())
    vecs = np.stack(vecs, axis=0)  # (5, 3)
    print(f'  {cv}: mean={vecs.mean(0).round(4)}  std={vecs.std(0).round(5)}')
