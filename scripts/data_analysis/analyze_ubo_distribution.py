"""Streaming RGB value-distribution analysis for UBO2014 BTF dataset.

Iterates ``*_W400xH400_L151xV151.btf`` files in ``/media/raid/cloth/BTF/``,
samples a random subset of angles per material, applies BGR->RGB + clip
(matching scripts/debugging/ubo_stats.py), and prints distribution stats
per material plus aggregate breakdowns by material category (carpet,
fabric, felt, leather, ...).

Note: UBO2014 stores ``BRDF * cos(theta_l)`` (see project memory) — the
stats are over the raw stored values, no cosine correction.

Usage:
    python scripts/data_analysis/analyze_ubo_distribution.py
    python scripts/data_analysis/analyze_ubo_distribution.py --max-materials 5
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

BTF_DIR = Path("/media/raid/cloth/BTF")

# Per-material angle subsample. Each angle image is 400*400*3 = 480k scalars,
# so 20 angles -> ~9.6M scalars per material before subsampling.
ANGLES_PER_MAT = 20
GLOBAL_SAMPLES_PER_MAT = 200_000

DEFAULT_REF = 0.05

QUANTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
ABOVE_THRESHOLDS = [1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1.0]


def _ref_position(pool: np.ndarray, ref: float) -> dict:
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
        f">{thr:g}={float((values > thr).mean()) * 100:5.2f}%"
        for thr in ABOVE_THRESHOLDS
    ))
    return {
        "n": int(values.size),
        "min": minv, "max": maxv, "mean": mean,
        "median": median, "std": std, "geomean_pos": geomean,
        "quantiles": dict(zip(QUANTILES, [float(x) for x in qvals])),
    }


def _category_of(name: str) -> str:
    m = re.match(r"([a-zA-Z]+)\d+", name)
    return m.group(1) if m else "other"


def _list_btfs(btf_dir: Path) -> list[Path]:
    return sorted(btf_dir.glob("*_W400xH400_L151xV151.btf"))


def _load_material_rgbs(btf_path: Path, n_angles: int,
                        rng: np.random.Generator) -> np.ndarray | None:
    """Return (n_kept, H, W, 3) float32 over a random angle subset."""
    from btf_extractor import Ubo2014
    btf = Ubo2014(str(btf_path))
    angles = sorted(btf.angles_set)
    if not angles:
        return None
    n_total = len(angles)
    n_pick = min(n_angles, n_total)
    idx = rng.choice(n_total, size=n_pick, replace=False)

    imgs = []
    for i in idx:
        a = angles[i]
        img = btf.angles_to_image(*a)
        img = img[:, :, ::-1].copy()   # BGR -> RGB
        np.clip(img, 0.0, None, out=img)
        imgs.append(img.astype(np.float32, copy=False))
    return np.stack(imgs, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-materials", type=int, default=None)
    ap.add_argument("--btf-dir", type=Path, default=BTF_DIR)
    ap.add_argument("--angles-per-mat", type=int, default=ANGLES_PER_MAT)
    ap.add_argument("--samples-per-mat", type=int, default=GLOBAL_SAMPLES_PER_MAT)
    ap.add_argument("--ref", type=float, default=DEFAULT_REF)
    args = ap.parse_args()

    if not args.btf_dir.exists():
        print(f"ERROR: {args.btf_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    btfs = _list_btfs(args.btf_dir)
    if args.max_materials is not None:
        btfs = btfs[: args.max_materials]
    print(f"Found {len(btfs)} BTF files in {args.btf_dir}")
    print(f"Sampling {args.angles_per_mat} angles per material, "
          f"{args.samples_per_mat:,} scalars per material.\n")

    rng = np.random.default_rng(0)
    per_mat_summary: list[tuple[str, dict]] = []
    cat_pools: dict[str, list[np.ndarray]] = {}
    global_pool: list[np.ndarray] = []

    for i, btf_path in enumerate(btfs):
        name = btf_path.stem.split("_W")[0]
        cat = _category_of(name)
        print(f"[{i + 1}/{len(btfs)}] {name}  category={cat}")
        try:
            stack = _load_material_rgbs(btf_path, args.angles_per_mat, rng)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR loading {name}: {exc}")
            continue
        if stack is None:
            print("  no angles, skipping")
            continue

        flat = stack.reshape(-1)
        stats = _summarize("per-mat", flat)
        if stats:
            per_mat_summary.append((name, stats))

        if flat.size:
            n_keep = min(args.samples_per_mat, flat.size)
            sel = rng.choice(flat.size, size=n_keep, replace=False)
            sample = flat[sel]
            cat_pools.setdefault(cat, []).append(sample)
            global_pool.append(sample)

        del stack, flat

    print("\n" + "=" * 76)
    print("AGGREGATE STATISTICS")
    print("=" * 76)
    print(f"Materials processed: {len(per_mat_summary)}")

    for cat, parts in sorted(cat_pools.items()):
        if not parts:
            continue
        pool = np.concatenate(parts)
        print(f"\n[{cat}] {len(parts)} materials, {pool.size:,} samples")
        _summarize(cat, pool)

    if not global_pool:
        print("\nNo data sampled; nothing to aggregate.")
        return

    pool_all = np.concatenate(global_pool)
    print(f"\n[ALL CATEGORIES] {pool_all.size:,} samples")
    _summarize("global-pool", pool_all)

    pmed = np.array([s["median"] for _, s in per_mat_summary], dtype=np.float64)
    pmean = np.array([s["mean"] for _, s in per_mat_summary], dtype=np.float64)
    pgeo = np.array([s["geomean_pos"] for _, s in per_mat_summary], dtype=np.float64)
    print("\nDistribution of per-material medians:")
    _summarize("per-mat-medians", pmed.astype(np.float32))
    print("\nDistribution of per-material means:")
    _summarize("per-mat-means", pmean.astype(np.float32))
    print("\nDistribution of per-material geomeans (positive only):")
    _summarize("per-mat-geomeans", pgeo.astype(np.float32))

    print("\nSuggested logrel_ref candidates:")
    print(f"  median(global pool)     = {float(np.median(pool_all)):.4g}")
    print(f"  geomean(global pool)    = "
          f"{float(np.exp(np.log(pool_all[pool_all > 0] + 1e-12).mean())):.4g}")
    print(f"  mean(per-mat medians)   = {float(pmed.mean()):.4g}")

    print("\n" + "-" * 76)
    print(f"REF POSITION  (eval ref = {args.ref:g})")
    print("-" * 76)
    rp = _ref_position(pool_all, args.ref)
    print(f"  ref                          = {rp['ref']:.6g}")
    print(f"  percentile inside global pool= {rp['percentile_in_pool']:.2f}%   "
          f"(fraction of values <= ref)")
    print(f"  ref / global-median          = {rp['ref/median']:.3g}   "
          f"(median = {rp['median']:.4g})")
    print(f"  ref / global-geomean         = {rp['ref/geomean']:.3g}   "
          f"(geomean(>0) = {rp['geomean']:.4g})")


if __name__ == "__main__":
    main()
