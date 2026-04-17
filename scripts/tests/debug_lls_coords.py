#!/usr/bin/env python3
"""Diagnose LLS coordinate/rotation convention vs poly/pan LED positions.

Checks whether `llsCorners` are stored in the same convention as
`il001`/`cv01` per-rotation. If the conventions differ, computed wi in the
Monte-Carlo integrator will be in the wrong frame.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from pathlib import Path

DATA_ROOT = None
for p in ['/media/raid/cloth/Bonn_train', '/home/zla247/scratch/data/Bonn/train']:
    if Path(p).exists():
        DATA_ROOT = p
        break
assert DATA_ROOT, "Bonn data not found"
print(f"DATA_ROOT = {DATA_ROOT}")

import scipy.io as spio

mat_id = 1
calib_path = f'{DATA_ROOT}/mat{mat_id:04d}_calibration.mat'
raw = spio.loadmat(calib_path)

print("\n=== Top-level keys ===")
print([k for k in raw.keys() if not k.startswith('__')])

print("\n=== Per-rotation fields ===")
for rot_key in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
    rd = raw[rot_key][0, 0]
    print(f"  {rot_key}: {rd.dtype.names}")

print("\n=== cv01 (camera) across rotations ===")
for rot_key in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
    rd = raw[rot_key][0, 0]
    cv01 = np.array(rd['cv01'], dtype=np.float32).flatten()
    print(f"  {rot_key}: cv01 = {cv01}")

print("\n=== il001 (LED) across rotations ===")
for rot_key in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
    rd = raw[rot_key][0, 0]
    il01 = np.array(rd['il01'], dtype=np.float32).flatten() if 'il01' in rd.dtype.names else np.array(rd['il001'], dtype=np.float32).flatten()
    print(f"  {rot_key}: il01 = {il01}")

print("\n=== llsCorners shape across rotations ===")
for rot_key in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
    rd = raw[rot_key][0, 0]
    corners = np.array(rd['llsCorners'], dtype=np.float32)
    print(f"  {rot_key}: shape {corners.shape}")

print("\n=== llsCorners at angle index 0 (lla[0] = {}) across rotations ===".format(
    raw['llsAnglesDegrees'].flatten()[0]))
for rot_key in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
    rd = raw[rot_key][0, 0]
    corners = np.array(rd['llsCorners'], dtype=np.float32)
    c0 = corners[:, :, 0].T   # (4, 3)  first angle
    center = c0.mean(axis=0)
    print(f"  {rot_key}: center = {center}  corner0 = {c0[0]}")

print("\n=== All LLS angles ===")
print(f"  {raw['llsAnglesDegrees'].flatten()}")

# --- Check magnitude scale vs LED
print("\n=== Magnitude comparison (should be same order of magnitude) ===")
rd0 = raw['rot000'][0, 0]
il01 = np.array(rd0['il01'], dtype=np.float32).flatten()
corners = np.array(rd0['llsCorners'], dtype=np.float32)
c_center = corners[:, :, 0].T.mean(axis=0)
print(f"  |il01|     = {np.linalg.norm(il01):.4f}")
print(f"  |llsMid|   = {np.linalg.norm(c_center):.4f}")

# --- Check xyz sample magnitudes
import pyexr
xyz_path = f'{DATA_ROOT}/mat{mat_id:04d}_xyz_rot000.exr'
xyz = pyexr.read(xyz_path)  # (H, W, 3)
xyz_flat = xyz.reshape(-1, 3)
valid = np.linalg.norm(xyz_flat, axis=1) > 1e-6
xyz_v = xyz_flat[valid]
print(f"\n  xyz  min/max/mean magnitude: "
      f"{np.linalg.norm(xyz_v, axis=1).min():.4f} / "
      f"{np.linalg.norm(xyz_v, axis=1).max():.4f} / "
      f"{np.linalg.norm(xyz_v, axis=1).mean():.4f}")
print(f"  xyz  bbox min = {xyz_v.min(axis=0)}")
print(f"  xyz  bbox max = {xyz_v.max(axis=0)}")

# --- Compute wi from LED and from LLS center for a middle pixel
mid_idx = len(xyz_v) // 2
p = xyz_v[mid_idx]
wi_led = il01 - p
wi_led = wi_led / np.linalg.norm(wi_led)
wi_lls = c_center - p
wi_lls = wi_lls / np.linalg.norm(wi_lls)
print(f"\n  Sample pixel at {p}")
print(f"  wi_from_LED   = {wi_led}")
print(f"  wi_from_LLS(la=0, rot0) = {wi_lls}")

# --- Check if llsCorners are rotation-equivariant (would indicate per-rotation frame
#     that mirrors il01 convention)
print("\n=== Rotation equivariance check ===")
print("  If llsCorners at rot045 is rot_z(45)-related to rot000 corners, that's consistent")
print("  with per-rotation sample-frame convention (same as il01/cv01).")

def rotz(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

# compare il01
il01_rot000 = np.array(raw['rot000'][0, 0]['il01'], dtype=np.float32).flatten()
il01_rot045 = np.array(raw['rot045'][0, 0]['il01'], dtype=np.float32).flatten()
pred_il01_rot045_fwd  = rotz( 45) @ il01_rot000  # forward rotate
pred_il01_rot045_inv  = rotz(-45) @ il01_rot000  # inverse rotate
print(f"  il01 rot000              = {il01_rot000}")
print(f"  il01 rot045 (stored)     = {il01_rot045}")
print(f"  il01 rot000 via +45° rot = {pred_il01_rot045_fwd}")
print(f"  il01 rot000 via -45° rot = {pred_il01_rot045_inv}")

lls_rot000 = np.array(raw['rot000'][0, 0]['llsCorners'], dtype=np.float32)[:, :, 0].T  # (4, 3)
lls_rot045 = np.array(raw['rot045'][0, 0]['llsCorners'], dtype=np.float32)[:, :, 0].T
pred_lls_rot045_fwd = (rotz( 45) @ lls_rot000.T).T
pred_lls_rot045_inv = (rotz(-45) @ lls_rot000.T).T
print(f"\n  lls[angle0] rot000 center      = {lls_rot000.mean(0)}")
print(f"  lls[angle0] rot045 (stored)    = {lls_rot045.mean(0)}")
print(f"  lls[angle0] rot000 via +45°    = {pred_lls_rot045_fwd.mean(0)}")
print(f"  lls[angle0] rot000 via -45°    = {pred_lls_rot045_inv.mean(0)}")
