"""Benchmark MultiMaterialDenseDataset cold-load: serial vs ProcessPoolExecutor.

Mirrors scripts/test_bonn_loader_speed.py for the dense (points_dense) loader.
Constructs MultiMaterialDenseDataset twice on a small material subset — once with
num_load_workers=0 (serial) and once with num_load_workers=N — and reports the
speedup. Also spot-checks that the parallel loader produces matching per-material
shapes + rgbs checksums so a refactor regression would surface.

Usage:
  conda run -n fipt_copy python scripts/test_dense_loader_speed.py \
      list_size=8 num_workers=8 point_subsample_ratio=0.1
"""
import gc
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from hydra import initialize_config_dir, compose

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.dataset.points import MultiMaterialDenseDataset


def _make_subset_list(full_list_path, list_size, out_path):
    """Write the first `list_size` material ids from `full_list_path` to `out_path`."""
    with open(full_list_path) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    ids = ids[:list_size]
    with open(out_path, 'w') as f:
        f.write('\n'.join(ids) + '\n')
    return ids


def _build_cfg(dataset_folder, training_list_path, point_subsample_ratio, num_load_workers):
    """Compose the same Hydra cfg main.py uses, with overrides for this test."""
    config_dir = str(REPO / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=points_dense",
                "renderer=multiarea_emitter",
                "material=multi_material_latent",
                f"dataset_folder={dataset_folder}",
                f"data.training_list_path={training_list_path}",
                f"data.point_subsample_ratio={point_subsample_ratio}",
                f"data.num_load_workers={num_load_workers}",
                "data.rays_num=500000",
                "data.filter_observations=False",
                "data.legacy_swap_indexing=False",
            ],
        )
    return cfg


def _summarize(ds):
    """Compact per-material fingerprint: shape, rgbs sum, V."""
    out = {}
    for i, mid in enumerate(ds.mat_material_ids):
        rgbs = ds.mat_rgbs[i]
        # Use sum-of-uint32 to avoid float precision drift on the checksum
        out[int(mid)] = (
            tuple(rgbs.shape),
            int(rgbs.astype(np.uint64).sum()),
            int(ds.mat_xyz[i].shape[0]),
            int(ds.mat_num_valid_obs[i]),
        )
    return out


def _time_load(label, dataset_folder, training_list_path,
               point_subsample_ratio, num_load_workers):
    print(f"\n{'#'*70}\n# {label}: num_load_workers={num_load_workers}, "
          f"point_subsample_ratio={point_subsample_ratio}\n{'#'*70}")
    cfg = _build_cfg(dataset_folder, training_list_path,
                     point_subsample_ratio, num_load_workers)
    t0 = time.perf_counter()
    ds = MultiMaterialDenseDataset(
        cfg, root_folder=cfg.dataset_folder, split='train')
    elapsed = time.perf_counter() - t0
    n_mats = len(ds.mat_material_ids)
    total_obs = sum(ds.mat_num_valid_obs)
    print(f"\n[{label}] init took {elapsed:.1f}s for {n_mats} materials "
          f"({total_obs:,} observations)")
    sig = _summarize(ds)
    del ds
    gc.collect()
    return elapsed, sig


def main():
    overrides = {
        'list_size': 4,
        'num_workers': 8,
        'point_subsample_ratio': 0.1,
        'dataset_folder': '/media/raid/cloth/capture_data/Dataset_Nov11',
        'full_list': '/media/raid/cloth/capture_data/Dataset_Nov11/training_list_500.txt',
        # 'serial_first' (default): serial-cold, parallel-warm, serial-warm.
        #   Highlights cold-startup speedup (most realistic for training).
        # 'parallel_first': parallel-cold, serial-warm.
        #   Pessimistic for parallel (it pays the full cold-cache cost), so
        #   any speedup vs serial-warm is a true lower bound.
        'order': 'serial_first',
    }
    for arg in sys.argv[1:]:
        if '=' not in arg:
            continue
        k, v = arg.split('=', 1)
        k = k.split('.')[-1]
        if k in overrides:
            if isinstance(overrides[k], bool):
                overrides[k] = v.lower() in ('1', 'true', 'yes')
            elif isinstance(overrides[k], int):
                overrides[k] = int(v)
            elif isinstance(overrides[k], float):
                overrides[k] = float(v)
            else:
                overrides[k] = v

    list_size = int(overrides['list_size'])
    n_workers = int(overrides['num_workers'])
    psr = float(overrides['point_subsample_ratio'])
    dataset_folder = str(overrides['dataset_folder'])
    full_list = str(overrides['full_list'])

    # Build a temp training list with the first `list_size` ids
    tmp_dir = tempfile.mkdtemp(prefix='dense_speed_test_')
    sub_list = os.path.join(tmp_dir, f'subset_{list_size}.txt')
    ids = _make_subset_list(full_list, list_size, sub_list)

    print(f"=== Dense loader speed test ===")
    print(f"  dataset_folder       : {dataset_folder}")
    print(f"  full_list            : {full_list}")
    print(f"  list_size            : {list_size}  (mat ids: {ids})")
    print(f"  point_subsample_ratio: {psr}")
    print(f"  parallel workers     : {n_workers}")
    print(f"  subset list          : {sub_list}")

    order = str(overrides['order'])
    if order == 'parallel_first':
        parallel_cold_t, parallel_sig = _time_load(
            "PARALLEL_COLD", dataset_folder, sub_list, psr, n_workers)
        serial_warm_t, serial_warm_sig = _time_load(
            "SERIAL_WARM", dataset_folder, sub_list, psr, 0)
        ref_sig = parallel_sig
        other_sig = serial_warm_sig
        other_label = 'serial_warm'
        print(f"\n{'='*70}")
        print(f"RESULT  parallel(cold)={parallel_cold_t:.1f}s  "
              f"serial(warm)={serial_warm_t:.1f}s")
        if parallel_cold_t < serial_warm_t:
            print(f"  Parallel cold beat serial warm by "
                  f"{serial_warm_t/parallel_cold_t:.2f}x — true speedup.")
        else:
            print(f"  Serial warm beat parallel cold by "
                  f"{parallel_cold_t/serial_warm_t:.2f}x — IPC/fork "
                  f"overhead dominates here.")
        print(f"  workers={n_workers}, mats={list_size}")
        print(f"{'='*70}")
    else:
        serial_cold_t, serial_cold_sig = _time_load(
            "SERIAL_COLD", dataset_folder, sub_list, psr, 0)
        parallel_t, parallel_sig = _time_load(
            "PARALLEL", dataset_folder, sub_list, psr, n_workers)
        serial_t, serial_sig = _time_load(
            "SERIAL_WARM", dataset_folder, sub_list, psr, 0)
        ref_sig = serial_cold_sig
        other_sig = parallel_sig
        other_label = 'parallel'

        speedup = serial_t / parallel_t if parallel_t > 0 else float('inf')
        cold_speedup = serial_cold_t / parallel_t if parallel_t > 0 else float('inf')
        print(f"\n{'='*70}")
        print(f"RESULT  serial_cold={serial_cold_t:.1f}s  "
              f"parallel(warm)={parallel_t:.1f}s  "
              f"serial(warm)={serial_t:.1f}s")
        print(f"  warm/warm speedup (apples-to-apples): {speedup:.2f}x")
        print(f"  cold/warm speedup (parallel hides cold I/O): {cold_speedup:.2f}x")
        print(f"  workers={n_workers}, mats={list_size}")
        print(f"{'='*70}")

    # Correctness spot check: shapes + rgbs sum + V match between the two
    # variants (most important — guards against worker drift).
    mismatched = []
    for mid, sig in ref_sig.items():
        if mid not in other_sig:
            mismatched.append((mid, f"missing in {other_label}"))
        elif other_sig[mid] != sig:
            mismatched.append(
                (mid, f"sig {sig} vs {other_sig[mid]}"))
    if mismatched:
        print("\n[FAIL] Parallel result differs from serial:")
        for mid, msg in mismatched:
            print(f"  mat{mid:>4}: {msg}")
        sys.exit(1)
    print("[OK] Parallel and serial produced matching per-material fingerprints.")


if __name__ == "__main__":
    main()
