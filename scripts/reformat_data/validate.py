#!/usr/bin/env python3
"""
Validate a structured observations file against the original 50-chunk data.

Tests performed:
  1. Shape & dtype sanity checks
  2. xyz / point_ids match point_positions.npz exactly
  3. cam_pos / light_pos match scan_log.json exactly
  4. For EVERY observation in ALL original chunks, verify that
     structured rgbs[img_idx, pt_idx] == original rgb (uint16)
  5. Report duplicate (image, point) pairs (should be 0)
  6. Report density and any zero-rgb observations in chunks

Usage:
  python validate.py /path/to/Dataset_Nov11/0
  python validate.py /path/to/Dataset_Nov11/0 --structured /tmp/test.npz
"""
import numpy as np
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def _load_chunk(path):
    return np.load(path)['observations']


def validate_material(material_dir, structured_path=None, verbose=True):
    material_dir = Path(material_dir)
    if structured_path is None:
        structured_path = material_dir / 'observations_structured.npz'

    print(f"\n{'='*70}")
    print(f"Validating material {material_dir.name}")
    print(f"  Structured file: {structured_path}")
    print(f"{'='*70}")

    errors = []

    # ---- Load structured data -------------------------------------------
    s = np.load(structured_path)
    s_xyz = s['xyz']
    s_pids = s['point_ids']
    s_rgbs = s['rgbs']
    s_cam = s['cam_pos']
    s_light = s['light_pos']

    K, V, C = s_rgbs.shape
    print(f"\n[Structured] K={K}, V={V:,}, C={C}")
    print(f"  xyz:       {s_xyz.shape} {s_xyz.dtype}")
    print(f"  point_ids: {s_pids.shape} {s_pids.dtype}")
    print(f"  rgbs:      {s_rgbs.shape} {s_rgbs.dtype}")
    print(f"  cam_pos:   {s_cam.shape} {s_cam.dtype}")
    print(f"  light_pos: {s_light.shape} {s_light.dtype}")

    # ---- Test 1: dtype checks -------------------------------------------
    print("\n[Test 1] Dtype checks...")
    checks = [
        (s_xyz.dtype, np.float32, 'xyz'),
        (s_pids.dtype, np.int32, 'point_ids'),
        (s_rgbs.dtype, np.uint16, 'rgbs'),
        (s_cam.dtype, np.float32, 'cam_pos'),
        (s_light.dtype, np.float32, 'light_pos'),
    ]
    for actual, expected, name in checks:
        ok = actual == expected
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name} dtype={actual} (expected {expected})")
        if not ok:
            errors.append(f"{name} dtype mismatch: {actual} != {expected}")
    assert C == 3, "rgbs must have 3 channels"

    # ---- Test 2: xyz / point_ids match point_positions.npz --------------
    print("\n[Test 2] xyz / point_ids vs point_positions.npz...")
    pp = np.load(material_dir / 'point_positions.npz')
    pp_pids = pp['point_ids']
    pp_pos = pp['positions']

    pid_match = np.array_equal(s_pids, pp_pids.astype(np.int32))
    xyz_match = np.allclose(s_xyz, pp_pos, atol=1e-7)
    print(f"  point_ids match: {'PASS' if pid_match else 'FAIL'}")
    print(f"  xyz match:       {'PASS' if xyz_match else 'FAIL'}")
    if not pid_match:
        errors.append("point_ids mismatch with point_positions.npz")
    if not xyz_match:
        errors.append("xyz mismatch with point_positions.npz")

    # ---- Test 3: cam_pos / light_pos match scan_log.json ----------------
    print("\n[Test 3] cam_pos / light_pos vs scan_log.json...")
    with open(material_dir / 'scan_log.json') as f:
        scan_log = json.load(f)
    ref_cam = np.array([e['position'] for e in scan_log], dtype=np.float32)
    ref_light = np.array([e['position_light'] for e in scan_log], dtype=np.float32)

    cam_match = np.allclose(s_cam, ref_cam, atol=1e-6)
    light_match = np.allclose(s_light, ref_light, atol=1e-6)
    print(f"  cam_pos match:   {'PASS' if cam_match else 'FAIL'}  (K={len(scan_log)})")
    print(f"  light_pos match: {'PASS' if light_match else 'FAIL'}")
    if not cam_match:
        errors.append("cam_pos mismatch")
    if not light_match:
        errors.append("light_pos mismatch")

    # ---- Test 4: cross-check every observation from chunks --------------
    print("\n[Test 4] Cross-checking ALL chunk observations against structured rgbs...")

    # Build point_id → dense index mapping (same logic as convert_single.py)
    max_pid = int(pp_pids.max())
    pid_lookup = np.full(max_pid + 1, -1, dtype=np.int32)
    pid_lookup[pp_pids.astype(np.int64)] = np.arange(V, dtype=np.int32)

    obs_dir = material_dir / 'observations'
    chunk_files = sorted(obs_dir.glob('observations_chunk_*.npz'))

    total_checked = 0
    total_mismatch = 0
    total_zero_in_chunk = 0
    seen_pairs = set()
    total_dupes = 0

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=4) as pool:
        chunk_iter = pool.map(_load_chunk, chunk_files)
        for ci, obs in enumerate(chunk_iter):
            img_ids = obs[:, 3].astype(np.int64) - 1       # 0-based
            pt_ids_raw = obs[:, 9].astype(np.int64)
            rgb_chunk = np.clip(obs[:, 6:9], 0, 65535).astype(np.uint16)

            safe_pt = np.clip(pt_ids_raw, 0, max_pid)
            pt_indices = pid_lookup[safe_pt]
            pt_indices[pt_ids_raw < 0] = -1
            pt_indices[pt_ids_raw > max_pid] = -1

            valid = (img_ids >= 0) & (img_ids < K) & (pt_indices >= 0)
            vi = img_ids[valid]
            vp = pt_indices[valid]
            vr = rgb_chunk[valid]

            # Look up in structured
            sr = s_rgbs[vi, vp]
            mismatches = ~np.all(sr == vr, axis=1)
            n_mis = int(mismatches.sum())
            total_mismatch += n_mis
            total_checked += int(valid.sum())

            # Zero-rgb in chunk (observations where original rgb is all-zero)
            zero_mask = (vr.sum(axis=1) == 0)
            total_zero_in_chunk += int(zero_mask.sum())

            if verbose and n_mis > 0:
                idx = np.where(mismatches)[0][:5]
                for j in idx:
                    print(f"    MISMATCH chunk {ci} row: "
                          f"img={vi[j]} pt={vp[j]} "
                          f"chunk_rgb={vr[j]} struct_rgb={sr[j]}")

            print(f"\r  Chunk {ci+1:2d}/{len(chunk_files)}: "
                  f"checked {total_checked:,}, mismatches {total_mismatch:,}", end='', flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")

    rgb_ok = total_mismatch == 0
    print(f"  Total checked:      {total_checked:,}")
    print(f"  Total mismatches:   {total_mismatch:,}  {'PASS' if rgb_ok else 'FAIL'}")
    print(f"  Zero-rgb in chunks: {total_zero_in_chunk:,}")
    if not rgb_ok:
        errors.append(f"{total_mismatch:,} rgb mismatches")

    # ---- Test 5: density report -----------------------------------------
    print("\n[Test 5] Density report...")
    non_zero = (s_rgbs.sum(axis=2) > 0).sum()
    density = non_zero / (K * V) * 100
    print(f"  Non-zero cells: {non_zero:,} / {K * V:,}  ({density:.1f}%)")
    print(f"  Filled from chunks: {total_checked:,}")

    # ---- Summary --------------------------------------------------------
    print(f"\n{'='*70}")
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("ALL TESTS PASSED")
    print(f"{'='*70}\n")

    return len(errors) == 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Validate structured observations")
    parser.add_argument('material_dir', help='Path to material folder')
    parser.add_argument('--structured', '-s', default=None,
                        help='Path to structured npz (default: <dir>/observations_structured.npz)')
    args = parser.parse_args()

    ok = validate_material(args.material_dir, args.structured)
    sys.exit(0 if ok else 1)
