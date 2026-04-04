#!/usr/bin/env python3
"""
Batch-convert multiple materials with speed benchmarking.

Strategies tested:
  - Sequential (baseline)
  - Parallel I/O threads within each material (--parallel-io N)
  - Multiple materials in parallel via multiprocessing (--workers N)

Usage:
  # Convert 10 materials, 4 workers, 4 I/O threads each:
  python convert_batch.py /path/to/Dataset_Nov11 --mat-ids 0 48 71 95 112 121 145 169 192 215 --workers 4

  # Benchmark different configs:
  python convert_batch.py /path/to/Dataset_Nov11 --mat-ids 0 112 --benchmark
"""
import numpy as np
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool, set_start_method
from convert_single import convert_material


def _convert_one(args):
    """Wrapper for multiprocessing Pool."""
    mat_dir, output_path, parallel_io = args
    try:
        result = convert_material(mat_dir, output_path, verbose=False, parallel_io=parallel_io)
        result['material'] = Path(mat_dir).name
        result['status'] = 'ok'
        print(f"  Material {result['material']}: {result['elapsed_s']:.1f}s, "
              f"{result['density_pct']:.1f}% dense, {result['file_size_gb']:.2f} GB")
        return result
    except Exception as e:
        print(f"  Material {Path(mat_dir).name}: FAILED — {e}")
        return {'material': Path(mat_dir).name, 'status': 'error', 'error': str(e)}


def run_batch(dataset_root, mat_ids, workers=1, parallel_io=4, output_suffix='observations_structured.npz'):
    dataset_root = Path(dataset_root)
    print(f"\n{'='*70}")
    print(f"Batch conversion: {len(mat_ids)} materials, {workers} worker(s), {parallel_io} I/O threads each")
    print(f"{'='*70}")

    args_list = []
    for mid in mat_ids:
        mat_dir = dataset_root / str(mid)
        out_path = mat_dir / output_suffix
        args_list.append((str(mat_dir), str(out_path), parallel_io))

    t0 = time.time()

    if workers <= 1:
        results = [_convert_one(a) for a in args_list]
    else:
        with Pool(processes=workers) as pool:
            results = pool.map(_convert_one, args_list)

    elapsed = time.time() - t0
    ok_results = [r for r in results if r.get('status') == 'ok']
    failed = [r for r in results if r.get('status') != 'ok']

    print(f"\n{'='*70}")
    print(f"Batch complete: {len(ok_results)} succeeded, {len(failed)} failed")
    if ok_results:
        times = [r['elapsed_s'] for r in ok_results]
        sizes = [r['file_size_gb'] for r in ok_results]
        print(f"  Per-material time: min={min(times):.1f}s, max={max(times):.1f}s, "
              f"avg={sum(times)/len(times):.1f}s")
        print(f"  Per-material size: avg={sum(sizes)/len(sizes):.2f} GB")
        print(f"  Wall-clock total: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
        print(f"  Wall-clock per material: {elapsed/len(mat_ids):.1f}s")
    if failed:
        print(f"\n  Failed materials:")
        for r in failed:
            print(f"    {r['material']}: {r.get('error', 'unknown')}")
    print(f"{'='*70}\n")

    return results


def run_benchmark(dataset_root, mat_ids):
    """Try different worker/IO configs and report timings."""
    configs = [
        (1, 1, "Sequential, no thread I/O"),
        (1, 4, "Sequential, 4 I/O threads"),
        (4, 4, "4 workers, 4 I/O threads"),
    ]

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {len(mat_ids)} materials, testing {len(configs)} configurations")
    print(f"{'='*70}\n")

    all_results = []
    for workers, pio, label in configs:
        print(f"\n--- Config: {label} (workers={workers}, parallel_io={pio}) ---")

        # Clean up any previous output
        for mid in mat_ids:
            p = Path(dataset_root) / str(mid) / 'observations_structured.npz'
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

        t0 = time.time()
        results = run_batch(dataset_root, mat_ids, workers=workers, parallel_io=pio)
        wall = time.time() - t0

        ok = [r for r in results if r.get('status') == 'ok']
        avg_per_mat = wall / len(mat_ids) if mat_ids else 0

        entry = {
            'label': label,
            'workers': workers,
            'parallel_io': pio,
            'wall_s': wall,
            'per_mat_s': avg_per_mat,
            'n_ok': len(ok),
        }
        all_results.append(entry)
        print(f"  → Wall: {wall:.1f}s, Per-material: {avg_per_mat:.1f}s")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Config':<42} {'Wall':>8} {'Per-mat':>8} {'OK':>4}")
    print(f"{'-'*70}")
    for r in all_results:
        print(f"{r['label']:<42} {r['wall_s']:>7.1f}s {r['per_mat_s']:>7.1f}s {r['n_ok']:>4}")
    print(f"{'='*70}\n")

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Batch convert materials to structured format")
    parser.add_argument('dataset_root', help='Path to Dataset_Nov11 root')
    parser.add_argument('--mat-ids', nargs='+', type=int, required=True,
                        help='Material IDs to process')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Number of parallel material workers')
    parser.add_argument('--parallel-io', type=int, default=4,
                        help='Threads for chunk I/O per material')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark across multiple configurations')
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args.dataset_root, args.mat_ids)
    else:
        run_batch(args.dataset_root, args.mat_ids,
                  workers=args.workers, parallel_io=args.parallel_io)
