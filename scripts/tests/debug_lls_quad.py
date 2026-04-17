#!/usr/bin/env python3
"""Inspect LLS quad corner geometry (corner ordering, strip axis)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import scipy.io as spio

DATA_ROOT = '/media/raid/cloth/Bonn_train'
mat_id = 1
calib_path = f'{DATA_ROOT}/mat{mat_id:04d}_calibration.mat'
raw = spio.loadmat(calib_path)

rd = raw['rot000'][0, 0]
corners_all = np.array(rd['llsCorners'], dtype=np.float32)  # (3, 4, 14)
print(f"llsCorners shape: {corners_all.shape}  (3, 4, n_angles)")

angles = raw['llsAnglesDegrees'].flatten()
print(f"Angles: {angles}")

# Inspect corners at each angle
print("\n=== Corners at angle index 0 (la=-2) ===")
c = corners_all[:, :, 0].T   # (4, 3)
for i in range(4):
    print(f"  c{i}: {c[i]}  |dist from origin| = {np.linalg.norm(c[i]):.2f}")
print(f"  center: {c.mean(axis=0)}")
print(f"  edge0->1: |{np.linalg.norm(c[1]-c[0]):.3f}|   {c[1]-c[0]}")
print(f"  edge1->2: |{np.linalg.norm(c[2]-c[1]):.3f}|   {c[2]-c[1]}")
print(f"  edge2->3: |{np.linalg.norm(c[3]-c[2]):.3f}|   {c[3]-c[2]}")
print(f"  edge3->0: |{np.linalg.norm(c[0]-c[3]):.3f}|   {c[0]-c[3]}")
print(f"  diag0->2: |{np.linalg.norm(c[2]-c[0]):.3f}|")
print(f"  diag1->3: |{np.linalg.norm(c[3]-c[1]):.3f}|")

# Check orthogonality of edges (valid quad should have nearly-orthogonal adjacent edges)
e01 = c[1]-c[0]; e12 = c[2]-c[1]; e23 = c[3]-c[2]; e30 = c[0]-c[3]
def cos_ang(a, b):
    return float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
print(f"\n  cos(e01, e12) = {cos_ang(e01, e12):.4f}  (0 = perpendicular)")
print(f"  cos(e12, e23) = {cos_ang(e12, e23):.4f}")
print(f"  cos(e01, e23) = {cos_ang(e01, e23):.4f}  (should be -1 for opposite edges)")

# Check all angles
print("\n=== Edge lengths and quad area across all angles ===")
print(f"{'angle':>7}  {'len01':>8}  {'len12':>8}  {'diag02':>8}  {'diag13':>8}")
for ai, ang in enumerate(angles):
    c = corners_all[:, :, ai].T   # (4, 3)
    len01 = np.linalg.norm(c[1]-c[0])
    len12 = np.linalg.norm(c[2]-c[1])
    d02 = np.linalg.norm(c[2]-c[0])
    d13 = np.linalg.norm(c[3]-c[1])
    print(f"  {ang:>5.1f}   {len01:>8.3f}  {len12:>8.3f}  {d02:>8.3f}  {d13:>8.3f}")

# Sample 100 points with the code's sample_quad_uniform
print("\n=== sample_quad_uniform sanity check ===")
c = corners_all[:, :, 0].T   # (4, 3) at angle 0
u = np.random.rand(1000)
v = np.random.rand(1000)
samples = ((1-u)*(1-v))[:,None]*c[0] + (u*(1-v))[:,None]*c[1] + (u*v)[:,None]*c[2] + ((1-u)*v)[:,None]*c[3]
print(f"  sample bbox: {samples.min(axis=0)} to {samples.max(axis=0)}")
print(f"  quad bbox:   {c.min(axis=0)} to {c.max(axis=0)}")
in_bounds = np.all((samples >= c.min(axis=0)-1e-3) & (samples <= c.max(axis=0)+1e-3), axis=1)
print(f"  samples in bounds: {in_bounds.sum()}/1000")
