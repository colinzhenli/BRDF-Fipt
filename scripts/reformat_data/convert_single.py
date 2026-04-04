#!/usr/bin/env python3
"""
Convert a single material from 50 NPZ chunks to structured dense format.

Input (per material folder):
  - point_positions.npz : point_ids (V,) int64, positions (V, 3) float32
  - scan_log.json       : K entries with 'position' (cam) and 'position_light'
  - observations/observations_chunk_*.npz : flat (N, 10) float64
    columns: [x, y, z, image_id, pixel_x, pixel_y, r, g, b, point_id]
    image_id is 1-based COLMAP ID; point_id is 0-based local index.

Output (single file):
  observations_structured.npz:
    xyz        (V, 3)    float32
    point_ids  (V,)      int32
    rgbs       (K, V, 3) uint16   ← raw sensor values, CCM applied at train time
    cam_pos    (K, 3)    float32  ← in mm (matches scan_log)
    light_pos  (K, 3)    float32  ← in mm (matches scan_log)

Usage:
  python convert_single.py /path/to/Dataset_Nov11/0
  python convert_single.py /path/to/Dataset_Nov11/0 --output /tmp/test_structured.npz
"""
import numpy as np
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def _load_chunk(chunk_path):
    """Load one NPZ chunk, return raw observations array."""
    return np.load(chunk_path)['observations']


def convert_material(material_dir, output_path=None, verbose=True, parallel_io=4):
    """
    Convert one material from chunked flat observations to dense structured format.

    Returns dict with conversion statistics.
    """
    material_dir = Path(material_dir)
    if output_path is None:
        output_path = material_dir / 'observations_structured.npz'

    t0 = time.time()

    # --- 1. Load point positions -----------------------------------------
    pp = np.load(material_dir / 'point_positions.npz')
    point_ids_arr = pp['point_ids']   # (V,) unique local point indices
    positions = pp['positions']       # (V, 3) float32
    V = len(point_ids_arr)

    # Vectorised point_id → dense-row lookup
    max_pid = int(point_ids_arr.max())
    pid_lookup = np.full(max_pid + 1, -1, dtype=np.int32)
    pid_lookup[point_ids_arr.astype(np.int64)] = np.arange(V, dtype=np.int32)

    # --- 2. Load scan_log for cam / light positions ----------------------
    with open(material_dir / 'scan_log.json') as f:
        scan_log = json.load(f)
    K = len(scan_log)

    cam_pos = np.array([e['position'] for e in scan_log], dtype=np.float32)
    light_pos = np.array([e['position_light'] for e in scan_log], dtype=np.float32)

    if verbose:
        mem_gb = K * V * 3 * 2 / 1e9
        print(f"Material {material_dir.name}: V={V:,} points, K={K} images")
        print(f"  Dense rgbs will be ({K}, {V}, 3) uint16 = {mem_gb:.2f} GB")

    # --- 3. Allocate dense RGB array -------------------------------------
    rgbs = np.zeros((K, V, 3), dtype=np.uint16)

    # --- 4. Stream chunks and fill the dense matrix ----------------------
    obs_dir = material_dir / 'observations'
    chunk_files = sorted(obs_dir.glob('observations_chunk_*.npz'))

    total_obs = 0
    total_filled = 0
    total_dupes = 0

    t_io = time.time()

    # Load chunks with thread-parallel I/O (NFS benefits from pipelining)
    with ThreadPoolExecutor(max_workers=parallel_io) as pool:
        chunk_iter = pool.map(_load_chunk, chunk_files)
        for ci, obs in enumerate(chunk_iter):
            n = len(obs)
            total_obs += n

            img_ids = obs[:, 3].astype(np.int64) - 1          # 1-based → 0-based
            pt_ids_raw = obs[:, 9].astype(np.int64)
            rgb_vals = np.clip(obs[:, 6:9], 0, 65535).astype(np.uint16)

            # Vectorised point_id → row index
            safe_pt = np.clip(pt_ids_raw, 0, max_pid)
            pt_indices = pid_lookup[safe_pt]
            pt_indices[pt_ids_raw < 0] = -1
            pt_indices[pt_ids_raw > max_pid] = -1

            valid = (img_ids >= 0) & (img_ids < K) & (pt_indices >= 0)
            vi = img_ids[valid]
            vp = pt_indices[valid]
            vr = rgb_vals[valid]

            # Count duplicates (same image+point written twice → should be 0)
            already_set = rgbs[vi, vp].sum(axis=1) > 0
            total_dupes += int(already_set.sum())

            rgbs[vi, vp] = vr
            total_filled += int(valid.sum())

            if verbose:
                print(f"\r  Chunk {ci+1:2d}/{len(chunk_files)}: "
                      f"{total_filled:,} filled ({total_obs:,} read)", end='', flush=True)

    t_io_done = time.time()
    if verbose:
        print()

    # --- 5. Stats --------------------------------------------------------
    non_zero = (rgbs.sum(axis=2) > 0).sum()
    density = non_zero / (K * V) * 100

    if verbose:
        print(f"  I/O + fill: {t_io_done - t_io:.1f}s")
        print(f"  Total obs read:   {total_obs:,}")
        print(f"  Filled cells:     {total_filled:,}")
        print(f"  Duplicate writes: {total_dupes:,}")
        print(f"  Non-zero cells:   {non_zero:,} / {K * V:,}  ({density:.1f}%)")

    # --- 6. Save ---------------------------------------------------------
    t_save = time.time()
    np.savez(
        output_path,
        xyz=positions,
        point_ids=point_ids_arr.astype(np.int32),
        rgbs=rgbs,
        cam_pos=cam_pos,
        light_pos=light_pos,
    )
    file_size = os.path.getsize(output_path) / 1e9
    t_end = time.time()

    if verbose:
        print(f"  Save: {t_end - t_save:.1f}s  ({file_size:.2f} GB)")
        print(f"  Total: {t_end - t0:.1f}s")

    return {
        'V': V, 'K': K,
        'total_obs': total_obs,
        'filled': total_filled,
        'dupes': total_dupes,
        'density_pct': density,
        'elapsed_s': t_end - t0,
        'file_size_gb': file_size,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert one material to structured format")
    parser.add_argument('material_dir', help='Path to material folder')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path (default: <material_dir>/observations_structured.npz)')
    parser.add_argument('--parallel-io', type=int, default=4,
                        help='Number of threads for chunk I/O (default: 4)')
    args = parser.parse_args()

    result = convert_material(args.material_dir, args.output, parallel_io=args.parallel_io)
    print(f"\nResult: {json.dumps(result, indent=2)}")
