"""Streaming RGB value-distribution analysis for Dataset_Nov11.

Reads `observations_structured.npz` per material from
`/media/raid/cloth/capture_data/Dataset_Nov11/`, applies the CCM exactly
like utils/dataset/points.py:507 does, and prints distribution stats per
material plus a final aggregate broken down by camera-factor cohort.

Cohorts (mirroring trainers/stage1_trainer.py):
    cohort1: material_id < 100        (linear_factor1)
    cohort2: 100 <= material_id < 352 (linear_factor2)
    cohort3: material_id >= 352       (linear_factor3 = factor2 * 8000/20000)

Materials without `observations_structured.npz` are skipped silently.

Usage:
    python scripts/data_analysis/analyze_real_distribution.py
    python scripts/data_analysis/analyze_real_distribution.py --max-materials 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path("/media/raid/cloth/capture_data/Dataset_Nov11")

# Original CCM from config/data/base.yaml (the one the trainer uses).
CCM = np.array([
    [3.6724617, -0.94800931, 0.08428962],
    [-0.44629176, 2.96095854, -1.17898539],
    [-0.47909694, -0.39991418, 2.10705124],
], dtype=np.float64)

GLOBAL_SAMPLES_PER_MAT = 200_000

# Default `logrel_ref` currently used in real-data Stage-1 jobs.
DEFAULT_REF = 30_000.0

QUANTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
ABOVE_THRESHOLDS = [10.0, 100.0, 1000.0, 5000.0, 10_000.0, 30_000.0, 50_000.0]


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
    if values.size == 0:
        print(f"  [{label}] no samples")
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
        f">{int(thr)}={float((values > thr).mean()) * 100:5.2f}%"
        for thr in ABOVE_THRESHOLDS
    ))
    return {
        "n": int(values.size),
        "min": minv, "max": maxv, "mean": mean,
        "median": median, "std": std, "geomean_pos": geomean,
        "quantiles": dict(zip(QUANTILES, [float(x) for x in qvals])),
    }


def _cohort_of(material_id: int) -> str:
    if material_id < 100:
        return "cohort1 (mat<100, factor1)"
    if material_id < 352:
        return "cohort2 (100<=mat<352, factor2)"
    return "cohort3 (mat>=352, factor3=factor2*0.4)"


def _list_material_ids(data_dir: Path) -> list[int]:
    """Return sorted integer-named material folders that contain the NPZ."""
    ids: list[int] = []
    for p in sorted(data_dir.iterdir()):
        if not p.is_dir():
            continue
        try:
            mat_id = int(p.name)
        except ValueError:
            continue  # skip e.g. "0_2", "10_2"
        if (p / "observations_structured.npz").exists():
            ids.append(mat_id)
    return sorted(ids)


def _load_material_rgbs(data_dir: Path, mat_id: int) -> np.ndarray | None:
    """Returns post-CCM (N, 3) float32 for *valid* observations, or None."""
    npz_path = data_dir / str(mat_id) / "observations_structured.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    if "rgbs" not in data:
        return None
    rgbs_raw = data["rgbs"]  # (K, V, 3) uint16
    if rgbs_raw.size == 0:
        return None
    K, V, _ = rgbs_raw.shape

    flat = rgbs_raw.reshape(-1, 3).astype(np.float64)
    valid_mask = flat.sum(axis=1) > 0
    if not valid_mask.any():
        return None
    flat_valid = flat[valid_mask]
    rgbs_ccm = (flat_valid @ CCM)
    rgbs_ccm = np.clip(rgbs_ccm, a_min=0.0, a_max=None).astype(np.float32)
    print(
        f"  shape: K={K}, V={V}, total={flat.shape[0]:,d}, "
        f"valid(non-zero)={valid_mask.sum():,d} ({valid_mask.mean() * 100:.1f}%)"
    )
    return rgbs_ccm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-materials", type=int, default=None)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--samples-per-mat", type=int, default=GLOBAL_SAMPLES_PER_MAT)
    ap.add_argument("--min-id", type=int, default=0,
                    help="Skip materials with id < min-id (inclusive of min-id).")
    ap.add_argument("--max-id", type=int, default=350,
                    help="Skip materials with id > max-id (inclusive of max-id). "
                         "Default 350 keeps cohorts 1+2 only.")
    ap.add_argument("--ref", type=float, default=DEFAULT_REF,
                    help=f"logrel_ref to evaluate (default {DEFAULT_REF}).")
    args = ap.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: {args.data_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    mat_ids = _list_material_ids(args.data_dir)
    mat_ids = [m for m in mat_ids if args.min_id <= m <= args.max_id]
    if args.max_materials is not None:
        mat_ids = mat_ids[: args.max_materials]
    print(f"Found {len(mat_ids)} materials with observations_structured.npz "
          f"in {args.data_dir} (id range {args.min_id}..{args.max_id})")
    print(f"Subsampling {args.samples_per_mat:,} values / material "
          f"for the global pool.\n")

    rng = np.random.default_rng(0)
    per_mat_summary: list[tuple[int, dict]] = []
    cohort_pools: dict[str, list[np.ndarray]] = {
        "cohort1 (mat<100, factor1)": [],
        "cohort2 (100<=mat<352, factor2)": [],
        "cohort3 (mat>=352, factor3=factor2*0.4)": [],
    }
    global_pool: list[np.ndarray] = []

    for i, mat_id in enumerate(mat_ids):
        cohort = _cohort_of(mat_id)
        print(f"[{i + 1}/{len(mat_ids)}] mat{mat_id:04d}  {cohort}")
        try:
            rgbs = _load_material_rgbs(args.data_dir, mat_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR loading mat{mat_id}: {exc}")
            continue
        if rgbs is None:
            print("  no usable data, skipping")
            continue

        flat = rgbs.reshape(-1)
        stats = _summarize("per-mat (post-CCM)", flat)
        if stats:
            per_mat_summary.append((mat_id, stats))

        if flat.size:
            n_keep = min(args.samples_per_mat, flat.size)
            sel = rng.choice(flat.size, size=n_keep, replace=False)
            sample = flat[sel]
            cohort_pools[cohort].append(sample)
            global_pool.append(sample)

        del rgbs, flat

    print("\n" + "=" * 76)
    print("AGGREGATE STATISTICS")
    print("=" * 76)
    print(f"Materials processed: {len(per_mat_summary)}")

    for cohort_name, parts in cohort_pools.items():
        if not parts:
            print(f"\n[{cohort_name}] no materials")
            continue
        pool = np.concatenate(parts)
        print(f"\n[{cohort_name}] {len(parts)} materials, {pool.size:,} samples")
        _summarize(cohort_name, pool)

    if not global_pool:
        print("\nNo data sampled; nothing to aggregate.")
        return

    pool_all = np.concatenate(global_pool)
    print(f"\n[ALL COHORTS] {pool_all.size:,} samples")
    _summarize("global-pool", pool_all)

    pmed = np.array([s["median"] for _, s in per_mat_summary], dtype=np.float64)
    pmean = np.array([s["mean"] for _, s in per_mat_summary], dtype=np.float64)
    pgeo = np.array([s["geomean_pos"] for _, s in per_mat_summary], dtype=np.float64)
    print("\nDistribution of per-material medians (post-CCM):")
    _summarize("per-mat-medians", pmed.astype(np.float32))
    print("\nDistribution of per-material means:")
    _summarize("per-mat-means", pmean.astype(np.float32))
    print("\nDistribution of per-material geomeans (positive only):")
    _summarize("per-mat-geomeans", pgeo.astype(np.float32))

    print("\nSuggested logrel_ref candidates (pick one matching loss intent):")
    print(f"  median(global pool)        = {float(np.median(pool_all)):.4g}    "
          "<- mass-balanced default")
    print(f"  geomean(global pool)       = "
          f"{float(np.exp(np.log(pool_all[pool_all > 0] + 1e-12).mean())):.4g}    "
          "<- log-symmetric (matches log_mapping)")
    print(f"  mean(per-mat medians)      = {float(pmed.mean()):.4g}    "
          "<- robust to one outlier material")
    cohort1_meds = [s["median"] for mid, s in per_mat_summary if mid < 100]
    if cohort1_meds:
        print(f"  mean(per-mat medians, mat<100) = "
              f"{float(np.mean(cohort1_meds)):.4g}    "
              "<- aligned with current factor1 reference cohort")

    # ----- where does the configured ref actually sit? ---------------------
    print("\n" + "-" * 76)
    print(f"REF POSITION  (current real ref = {args.ref:g})")
    print("-" * 76)
    rp_global = _ref_position(pool_all, args.ref)
    print(f"  ref                                = {rp_global['ref']:.6g}")
    print(f"  percentile inside global pool      = "
          f"{rp_global['percentile_in_pool']:.2f}%   (fraction of values <= ref)")
    print(f"  ref / global-median                = "
          f"{rp_global['ref/median']:.3g}   (median = {rp_global['median']:.4g})")
    print(f"  ref / global-geomean               = "
          f"{rp_global['ref/geomean']:.3g}   (geomean(>0) = {rp_global['geomean']:.4g})")
    if rp_global['ref/median'] >= 1:
        print("  -> ref >= median: most loss mass falls in the LINEAR regime "
              "(small-rel-error, gradient ~ 1/ref).")
    else:
        print("  -> ref <  median: most loss mass falls in the LOG regime "
              "(log_mapping ~ log(x/ref) for x >> ref).")

    # Per-cohort ref positioning (since ref is scaled by camera_factor at
    # train time -- this prints the effective ref location per cohort).
    print("\n  per-cohort ref position (raw ref, before per-cohort scaling):")
    for cohort_name, parts in cohort_pools.items():
        if not parts:
            continue
        cpool = np.concatenate(parts)
        rp = _ref_position(cpool, args.ref)
        print(f"    {cohort_name:42s}  "
              f"pct={rp['percentile_in_pool']:6.2f}%  "
              f"ref/med={rp['ref/median']:.3g}  "
              f"ref/geo={rp['ref/geomean']:.3g}")


if __name__ == "__main__":
    main()
