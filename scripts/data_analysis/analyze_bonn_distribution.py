"""Streaming RGB value-distribution analysis for the Bonn poly dataset.

Reads `mat{ID:04d}_poly.exr` files one material at a time from
`/media/raid/cloth/Bonn_train/`, computes per-material distribution stats
(median / mean / geomean / quantiles), prints them as it goes, and finally
reports an aggregate distribution sampled from all materials.

The poly EXR layout follows utils/dataset/bonn.py: channels are sorted
alphabetically, every 3 form one (B, G, R) image, and reshape goes
`(H, W, 3K)` → `(V=H*W, K, 3)` → `(K, V, 3)`. We do NOT apply any CCM
because the Bonn radiometric calibration is already baked into the EXR.

Usage:
    python scripts/data_analysis/analyze_bonn_distribution.py            # full
    python scripts/data_analysis/analyze_bonn_distribution.py --max-materials 5
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

# Make `utils` importable so we reuse the loader's parser exactly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pyexr  # noqa: E402

from utils.dataset.bonn import _parse_poly_channels, _read_exr  # noqa: E402

BONN_DIR = Path("/media/raid/cloth/Bonn_train")

# Sample budget per material (post-positivity filter) for the global
# distribution. The full poly stack for one Bonn material is 4-12 GB, so we
# subsample to keep RAM bounded while still giving a representative pool.
GLOBAL_SAMPLES_PER_MAT = 200_000

# Default `logrel_ref` currently used in scripts/jobs/run_stage1_bonn_cc.sh
DEFAULT_REF = 0.05

QUANTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
ABOVE_THRESHOLDS = [1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]


def _ref_position(pool: np.ndarray, ref: float) -> dict:
    """Where the chosen ref sits inside the distribution."""
    pool = pool.astype(np.float64)
    pos = pool[pool > 0]
    pct_below = float((pool <= ref).mean()) * 100
    median = float(np.median(pool))
    geomean = float(np.exp(np.log(pos + 1e-12).mean())) if pos.size else 0.0
    return {
        "ref": ref,
        "percentile_in_pool": pct_below,
        "ref/median": ref / max(median, 1e-12),
        "ref/geomean": ref / max(geomean, 1e-12),
        "median": median,
        "geomean": geomean,
    }


def _summarize(label: str, values: np.ndarray) -> dict:
    """Print and return distribution stats for a flat float32 array."""
    if values.size == 0:
        print(f"  [{label}] no positive samples")
        return {}
    pos = values[values > 0]
    median = float(np.median(values))
    mean = float(values.mean())
    std = float(values.std())
    minv = float(values.min())
    maxv = float(values.max())
    geomean = float(np.exp(np.log(pos + 1e-12).mean())) if pos.size else 0.0
    qvals = np.percentile(values, QUANTILES)

    print(
        f"  [{label}] n={values.size:>10,d}  "
        f"min={minv:.4g}  max={maxv:.4g}  mean={mean:.4g}  "
        f"median={median:.4g}  std={std:.4g}  geomean(>0)={geomean:.4g}"
    )
    print("    quantiles:", "  ".join(
        f"p{q:>5g}={v:.4g}" for q, v in zip(QUANTILES, qvals)
    ))
    print("    above-thr:", "  ".join(
        f">{thr:g}={float((values > thr).mean()) * 100:5.2f}%"
        for thr in ABOVE_THRESHOLDS
    ))

    return {
        "n": int(values.size),
        "min": minv, "max": maxv, "mean": mean,
        "median": median, "std": std, "geomean_pos": geomean,
        "quantiles": dict(zip(QUANTILES, [float(x) for x in qvals])),
    }


def _list_material_ids(bonn_dir: Path) -> list[int]:
    pattern = re.compile(r"mat(\d{4})_poly\.exr$")
    ids = []
    for p in sorted(bonn_dir.glob("mat*_poly.exr")):
        m = pattern.search(p.name)
        if m:
            ids.append(int(m.group(1)))
    return ids


def _load_material_rgbs(mat_id: int) -> np.ndarray | None:
    """Returns (K, V, 3) float32, or None if the file is missing/empty."""
    prefix = BONN_DIR / f"mat{mat_id:04d}"
    poly_path = f"{prefix}_poly.exr"
    if not os.path.exists(poly_path):
        return None

    poly_data, ch_names, H, W = _read_exr(poly_path)  # (H, W, C) float16
    images = _parse_poly_channels(ch_names)
    n_poly = len(images)
    if n_poly == 0:
        return None

    n_pixels = H * W
    # Match the loader: (H,W,3K) → (V, K, 3) → (K, V, 3)
    rgbs = poly_data.reshape(n_pixels, n_poly, 3).transpose(1, 0, 2)
    return rgbs.astype(np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-materials", type=int, default=None,
                    help="Process at most this many materials (unit test mode).")
    ap.add_argument("--bonn-dir", type=Path, default=BONN_DIR,
                    help="Override the Bonn dataset directory.")
    ap.add_argument("--samples-per-mat", type=int, default=GLOBAL_SAMPLES_PER_MAT,
                    help="Per-material random subsample for the global pool.")
    ap.add_argument("--ref", type=float, default=DEFAULT_REF,
                    help=f"logrel_ref to evaluate (default {DEFAULT_REF}).")
    args = ap.parse_args()

    bonn_dir = args.bonn_dir
    if not bonn_dir.exists():
        print(f"ERROR: {bonn_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    mat_ids = _list_material_ids(bonn_dir)
    if args.max_materials is not None:
        mat_ids = mat_ids[: args.max_materials]
    print(f"Found {len(mat_ids)} materials in {bonn_dir}")
    print(f"Subsampling {args.samples_per_mat:,} positive values / material "
          f"for the global pool.\n")

    rng = np.random.default_rng(0)
    per_mat_medians: list[float] = []
    per_mat_means: list[float] = []
    per_mat_geomeans: list[float] = []
    global_pool: list[np.ndarray] = []
    global_total_count = 0
    global_positive_count = 0

    for i, mat_id in enumerate(mat_ids):
        print(f"[{i + 1}/{len(mat_ids)}] mat{mat_id:04d}")
        try:
            rgbs = _load_material_rgbs(mat_id)
        except Exception as exc:  # noqa: BLE001 -- want to log + continue
            print(f"  ERROR reading mat{mat_id:04d}: {exc}")
            continue
        if rgbs is None:
            print("  no poly data, skipping")
            continue

        K, V, _ = rgbs.shape
        flat = rgbs.reshape(-1)
        # Treat negatives as zero (occasional EXR noise) for distribution work.
        flat = np.clip(flat, a_min=0.0, a_max=None)
        global_total_count += int(flat.size)
        pos = flat[flat > 0]
        global_positive_count += int(pos.size)

        print(f"  shape: K={K} views, V={V} pixels, "
              f"size={flat.size:,d}, positive={pos.size:,d}")
        stats = _summarize("per-mat", flat)
        if stats:
            per_mat_medians.append(stats["median"])
            per_mat_means.append(stats["mean"])
            per_mat_geomeans.append(stats["geomean_pos"])

        # Subsample positive values for the global pool to bound RAM.
        if pos.size:
            n_keep = min(args.samples_per_mat, pos.size)
            sel = rng.choice(pos.size, size=n_keep, replace=False)
            global_pool.append(pos[sel])

        # Free the big array between materials.
        del rgbs, flat, pos

    print("\n" + "=" * 76)
    print("AGGREGATE STATISTICS")
    print("=" * 76)
    if not global_pool:
        print("No materials produced any data.")
        return

    print(f"Materials processed         : {len(per_mat_medians)}")
    print(f"Total values seen           : {global_total_count:,}")
    print(f"Total positive values seen  : {global_positive_count:,} "
          f"({global_positive_count / max(global_total_count, 1) * 100:.2f}%)")

    pmed = np.array(per_mat_medians, dtype=np.float64)
    pmean = np.array(per_mat_means, dtype=np.float64)
    pgeo = np.array(per_mat_geomeans, dtype=np.float64)
    print("\nDistribution of per-material medians:")
    _summarize("per-mat-medians", pmed.astype(np.float32))
    print("\nDistribution of per-material means:")
    _summarize("per-mat-means", pmean.astype(np.float32))
    print("\nDistribution of per-material geomeans (positive only):")
    _summarize("per-mat-geomeans", pgeo.astype(np.float32))

    pool = np.concatenate(global_pool)
    print(f"\nGlobal pool of positive values "
          f"({pool.size:,} samples, {len(global_pool)} mats):")
    _summarize("global-pool", pool)

    print("\nSuggested logrel_ref candidates (pick one matching loss intent):")
    print(f"  median(global pool)     = {float(np.median(pool)):.4g}    "
          "<- mass-balanced, good general default")
    print(f"  geomean(global pool)    = {float(np.exp(np.log(pool + 1e-12).mean())):.4g}    "
          "<- log-symmetric (matches log_mapping)")
    print(f"  mean(per-mat medians)   = {float(pmed.mean()):.4g}    "
          "<- robust to one giant material")

    # ----- where does the configured ref actually sit? ---------------------
    print("\n" + "-" * 76)
    print(f"REF POSITION  (current Bonn ref = {args.ref:g})")
    print("-" * 76)
    rp = _ref_position(pool, args.ref)
    print(f"  ref                          = {rp['ref']:.6g}")
    print(f"  percentile inside global pool= {rp['percentile_in_pool']:.2f}%   "
          f"(fraction of values <= ref)")
    print(f"  ref / global-median          = {rp['ref/median']:.3g}   "
          f"(median = {rp['median']:.4g})")
    print(f"  ref / global-geomean         = {rp['ref/geomean']:.3g}   "
          f"(geomean(>0) = {rp['geomean']:.4g})")
    if rp['ref/median'] >= 1:
        print("  -> ref >= median: most loss mass falls in the LINEAR regime "
              "(small-rel-error, gradient ~ 1/ref).")
    else:
        print("  -> ref <  median: most loss mass falls in the LOG regime "
              "(log_mapping ~ log(x/ref) for x >> ref).")


if __name__ == "__main__":
    main()
