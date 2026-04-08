#!/usr/bin/env python3
"""
Generate point_positions.npz for materials that are missing it,
by extracting unique (point_id, xyz) from observation chunks.

This mirrors the logic in shape_matching.py lines 850-856:
    unique_pids, first_idx = np.unique(point_ids, return_index=True)
    unique_xyz = observations[first_idx, :3].astype(np.float32)

Usage:
  # Find and fix all missing materials in a dataset:
  python generate_point_positions.py /media/raid/cloth/capture_data/Dataset_Nov11

  # Process specific materials:
  python generate_point_positions.py /media/raid/cloth/capture_data/Dataset_Nov11 --mat-ids 1 3 4 5

  # Dry run (just list missing materials):
  python generate_point_positions.py /media/raid/cloth/capture_data/Dataset_Nov11 --dry-run
"""
import numpy as np
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse


def _load_chunk(chunk_path):
    """Load one NPZ chunk, return raw observations array."""
    return np.load(chunk_path)['observations']


def generate_point_positions(material_dir, parallel_io=4, verbose=True):
    """
    Extract unique (point_id, x, y, z) from observation chunks and save as point_positions.npz.

    Args:
        material_dir: path to material folder containing observations/
        parallel_io: number of threads for parallel chunk loading
        verbose: print progress

    Returns:
        dict with stats: V (num points), elapsed_s
    """
    material_dir = Path(material_dir)
    obs_dir = material_dir / 'observations'
    chunk_files = sorted(obs_dir.glob('observations_chunk_*.npz'))

    if not chunk_files:
        if verbose:
            print(f"  Material {material_dir.name}: No observation chunks found, skipping")
        return None

    t0 = time.time()
    if verbose:
        print(f"  Material {material_dir.name}: Extracting point positions from {len(chunk_files)} chunks...")

    # Collect unique point_ids and their first-seen xyz
    pid_to_xyz = {}
    with ThreadPoolExecutor(max_workers=parallel_io) as pool:
        chunk_iter = pool.map(_load_chunk, chunk_files)
        for ci, obs in enumerate(chunk_iter):
            pt_ids = obs[:, 9].astype(np.int64)
            xyz = obs[:, :3].astype(np.float32)
            unique_pids, first_idx = np.unique(pt_ids, return_index=True)
            for pid, idx in zip(unique_pids, first_idx):
                pid = int(pid)
                if pid not in pid_to_xyz:
                    pid_to_xyz[pid] = xyz[idx]
            if verbose:
                print(f"\r    Chunk {ci+1}/{len(chunk_files)}: "
                      f"{len(pid_to_xyz):,} unique points", end='', flush=True)
    if verbose:
        print()

    # Sort by point_id (matches shape_matching.py behavior via np.unique)
    sorted_pids = sorted(pid_to_xyz.keys())
    point_ids_arr = np.array(sorted_pids, dtype=np.int64)
    positions = np.stack([pid_to_xyz[pid] for pid in sorted_pids]).astype(np.float32)

    # Save
    output_path = material_dir / 'point_positions.npz'
    np.savez(output_path, point_ids=point_ids_arr, positions=positions)

    elapsed = time.time() - t0
    if verbose:
        print(f"    Saved {output_path}: {len(point_ids_arr):,} points, {elapsed:.1f}s")

    return {'V': len(point_ids_arr), 'elapsed_s': elapsed}


def find_missing_materials(dataset_root):
    """Find all material folders missing point_positions.npz but having observations."""
    dataset_root = Path(dataset_root)
    missing = []
    for d in sorted(dataset_root.iterdir()):
        if not d.is_dir() or not d.name.isdigit():
            continue
        obs_dir = d / 'observations'
        pp_path = d / 'point_positions.npz'
        if obs_dir.exists() and not pp_path.exists():
            n_chunks = len(list(obs_dir.glob('observations_chunk_*.npz')))
            if n_chunks > 0:
                missing.append((d, n_chunks))
    return missing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate point_positions.npz from observation chunks for materials missing it")
    parser.add_argument('dataset_root', help='Path to dataset root (e.g., Dataset_Nov11)')
    parser.add_argument('--mat-ids', nargs='+', type=int, default=None,
                        help='Specific material IDs to process (default: auto-detect missing)')
    parser.add_argument('--parallel-io', type=int, default=4,
                        help='Threads for chunk I/O (default: 4)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just list missing materials, do not process')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    if args.mat_ids:
        # Process specific materials
        materials = [(dataset_root / str(mid), -1) for mid in args.mat_ids]
    else:
        # Auto-detect missing
        materials = find_missing_materials(dataset_root)

    if not materials:
        print("All materials already have point_positions.npz!")
        exit(0)

    print(f"Found {len(materials)} materials missing point_positions.npz:")
    for mdir, n_chunks in materials:
        chunks_str = f" ({n_chunks} chunks)" if n_chunks > 0 else ""
        print(f"  {mdir.name}{chunks_str}")

    if args.dry_run:
        exit(0)

    print(f"\nProcessing {len(materials)} materials...")
    t_total = time.time()
    for mdir, _ in materials:
        generate_point_positions(mdir, parallel_io=args.parallel_io)

    print(f"\nDone! Total time: {time.time() - t_total:.1f}s")
