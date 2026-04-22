"""Benchmark BonnDataset cold-load: serial vs ProcessPoolExecutor.

Constructs BonnDataset twice on the same material subset (debug mode) — once with
num_load_workers=0 (serial) and once with num_load_workers=N — and reports the
speedup. Also spot-checks that the parallel loader produces the same per-material
shapes and a matching rgbs checksum, so a refactor regression would surface.

Usage:
  conda run -n fipt_copy python scripts/test_bonn_loader_speed.py \
      data.debug_num=8 data.use_pan=True data.use_lls=True num_workers=16
"""
import sys
import time
from pathlib import Path

import numpy as np
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.dataset.bonn import BonnDataset


def _build_cfg(debug_num, use_pan, use_lls, num_load_workers, dataset_folder):
    """Compose the same Hydra cfg main.py uses, with overrides for this test."""
    config_dir = str(REPO / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=bonn",
                f"dataset_folder={dataset_folder}",
                "data.debug=True",
                f"data.debug_num={debug_num}",
                f"data.use_pan={use_pan}",
                f"data.use_lls={use_lls}",
                f"data.num_load_workers={num_load_workers}",
            ],
        )
    return cfg


def _summarize(materials):
    """Compact per-material fingerprint for cross-run sanity check."""
    out = {}
    for m in materials:
        rgbs = m['rgbs']
        out[m['mat_id']] = (
            tuple(rgbs.shape),
            float(rgbs.astype(np.float32).sum()),
            int(m['xyz'].shape[0]),
        )
    return out


def _time_load(label, debug_num, use_pan, use_lls, num_load_workers, dataset_folder):
    print(f"\n{'#'*70}\n# {label}: num_load_workers={num_load_workers}, "
          f"debug_num={debug_num}, use_pan={use_pan}, use_lls={use_lls}\n{'#'*70}")
    cfg = _build_cfg(debug_num, use_pan, use_lls, num_load_workers, dataset_folder)
    t0 = time.perf_counter()
    ds = BonnDataset(cfg, root_folder=cfg.dataset_folder, split='train')
    elapsed = time.perf_counter() - t0
    n_mats = len(ds._all_data.materials)
    total_obs = ds._all_data.total_obs
    print(f"\n[{label}] init took {elapsed:.1f}s for {n_mats} materials "
          f"({total_obs:,} observations)")
    return elapsed, _summarize(ds._all_data.materials)


def main():
    # Lightweight CLI: KEY=VAL overrides (mirrors Hydra style)
    overrides = {
        'debug_num': 4,
        'num_workers': 16,
        'use_pan': True,
        'use_lls': True,
        'dataset_folder': '/media/raid/cloth/Bonn_train',
    }
    for arg in sys.argv[1:]:
        if '=' not in arg:
            continue
        k, v = arg.split('=', 1)
        # accept both `data.debug_num=8` and `debug_num=8`
        k = k.split('.')[-1]
        if k in overrides:
            if isinstance(overrides[k], bool):
                overrides[k] = v.lower() in ('1', 'true', 'yes')
            elif isinstance(overrides[k], int):
                overrides[k] = int(v)
            else:
                overrides[k] = v

    debug_num = int(overrides['debug_num'])
    n_workers = int(overrides['num_workers'])
    use_pan = bool(overrides['use_pan'])
    use_lls = bool(overrides['use_lls'])
    dataset_folder = str(overrides['dataset_folder'])

    print(f"=== Bonn loader speed test ===")
    print(f"  dataset_folder : {dataset_folder}")
    print(f"  debug_num      : {debug_num}")
    print(f"  use_pan/use_lls: {use_pan}/{use_lls}")
    print(f"  parallel workers: {n_workers}")

    serial_t, serial_sig = _time_load(
        "SERIAL", debug_num, use_pan, use_lls, 0, dataset_folder)
    parallel_t, parallel_sig = _time_load(
        f"PARALLEL", debug_num, use_pan, use_lls, n_workers, dataset_folder)

    speedup = serial_t / parallel_t if parallel_t > 0 else float('inf')
    print(f"\n{'='*70}")
    print(f"RESULT  serial={serial_t:.1f}s  parallel={parallel_t:.1f}s  "
          f"speedup={speedup:.2f}x  (workers={n_workers}, mats={debug_num})")
    print(f"{'='*70}")

    # Correctness spot check: same material set, shapes, and rgbs sum
    mismatched = []
    for mid, sig in serial_sig.items():
        if mid not in parallel_sig:
            mismatched.append((mid, "missing in parallel"))
        elif parallel_sig[mid][0] != sig[0]:
            mismatched.append((mid, f"shape {sig[0]} vs {parallel_sig[mid][0]}"))
        else:
            ds_diff = abs(parallel_sig[mid][1] - sig[1])
            ratio = ds_diff / max(abs(sig[1]), 1e-6)
            if ratio > 1e-5:
                mismatched.append(
                    (mid, f"rgbs sum diff ratio={ratio:.2e}"))
    if mismatched:
        print("\n[FAIL] Parallel result differs from serial:")
        for mid, msg in mismatched:
            print(f"  mat{mid:04d}: {msg}")
        sys.exit(1)
    print("[OK] Parallel and serial produced matching per-material fingerprints.")


if __name__ == "__main__":
    main()
