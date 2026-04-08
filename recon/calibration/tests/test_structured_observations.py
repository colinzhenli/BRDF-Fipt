"""
Unit test for build_and_save_structured_observations() in shape_matching.py.

Validates that the inline structured-npz writer produces output that is
bit-for-bit identical to scripts/reformat_data/convert_single.py running over
the same chunks. We use a real material from the live dataset (134, the
healthy reference) so the test exercises the full code path including the
duplicate-write counting and dense-row mapping.

Run:
    python recon/calibration/tests/test_structured_observations.py
or
    python recon/calibration/tests/test_structured_observations.py --material 134
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make shape_matching importable
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "recon" / "calibration"))

from shape_matching import build_and_save_structured_observations  # noqa: E402


def load_observations_from_chunks(material_dir: Path) -> np.ndarray:
    """Concatenate all observations_chunk_*.npz back into one (M, 10) array."""
    obs_dir = material_dir / "observations"
    chunk_files = sorted(obs_dir.glob("observations_chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunks in {obs_dir}")
    parts = [np.load(c)["observations"] for c in chunk_files]
    return np.vstack(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        default="/media/raid/cloth/capture_data/Dataset_Nov11",
        help="Dataset root containing per-material subfolders",
    )
    ap.add_argument(
        "--material",
        default="134",
        help="Material id to test against (default 134, the healthy reference)",
    )
    args = ap.parse_args()

    material_dir = Path(args.dataset) / args.material
    if not material_dir.is_dir():
        sys.exit(f"material dir not found: {material_dir}")

    reference_npz = material_dir / "observations_structured.npz"
    if not reference_npz.exists():
        sys.exit(
            f"reference {reference_npz} not found — this test compares against "
            f"the existing convert_single.py output, so the material must already "
            f"have observations_structured.npz."
        )

    print(f"Test material: {material_dir}")

    # Reconstruct in-memory observations array from existing chunks
    print("Loading chunks...")
    observations = load_observations_from_chunks(material_dir)
    print(f"  observations: shape={observations.shape}, dtype={observations.dtype}")

    # Load point_positions.npz the same way convert_single.py does
    pp = np.load(material_dir / "point_positions.npz")
    unique_pids = pp["point_ids"]
    unique_xyz = pp["positions"]
    print(f"  point_positions: V={len(unique_pids)}")

    # Call the new inline function, writing to a temp path so the live dataset
    # is untouched
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "observations_structured.npz"
        stats = build_and_save_structured_observations(
            observations=observations,
            unique_pids=unique_pids,
            unique_xyz=unique_xyz,
            scan_log_path=str(material_dir / "scan_log.json"),
            output_path=str(out_path),
            verbose=True,
        )

        # Load both and compare bit-for-bit
        print("\nComparing against reference...")
        new_d = np.load(out_path)
        ref_d = np.load(reference_npz)

        keys_expected = {"xyz", "point_ids", "rgbs", "cam_pos", "light_pos"}
        keys_new = set(new_d.files)
        keys_ref = set(ref_d.files)
        if keys_new != keys_expected:
            sys.exit(f"FAIL: new keys {keys_new} != expected {keys_expected}")
        if keys_ref != keys_expected:
            sys.exit(f"FAIL: reference keys {keys_ref} != expected {keys_expected}")

        all_ok = True
        for k in sorted(keys_expected):
            a = new_d[k]
            b = ref_d[k]
            if a.shape != b.shape:
                print(f"  {k}: SHAPE MISMATCH new={a.shape} ref={b.shape}")
                all_ok = False
                continue
            if a.dtype != b.dtype:
                print(f"  {k}: DTYPE MISMATCH new={a.dtype} ref={b.dtype}")
                all_ok = False
                continue
            if not np.array_equal(a, b):
                # Quantify the diff so the failure is actionable
                diff = (a != b)
                if a.ndim >= 2:
                    n_diff = int(diff.any(axis=tuple(range(1, a.ndim))).sum())
                else:
                    n_diff = int(diff.sum())
                print(f"  {k}: VALUE MISMATCH ({n_diff} differing elements)")
                all_ok = False
            else:
                print(f"  {k}: OK  shape={a.shape} dtype={a.dtype}")

        if not all_ok:
            sys.exit("\nFAIL: arrays do not match reference bit-for-bit")

        # Sanity-check density vs the values reported in stats
        rgbs = new_d["rgbs"]
        non_zero = (rgbs.sum(axis=2) > 0).sum()
        density = non_zero / rgbs.size * 3 * 100  # rgbs.size = K*V*3
        # density above is the same as non_zero / (K*V) * 100
        K, V = rgbs.shape[0], rgbs.shape[1]
        density2 = non_zero / (K * V) * 100
        assert abs(density2 - stats["density_pct"]) < 1e-6, (
            f"density stat mismatch: {density2} vs {stats['density_pct']}"
        )

        print(f"\nPASS: {len(keys_expected)} arrays match the reference exactly")
        print(f"      density={stats['density_pct']:.2f}%, dupes={stats['dupes']}")


if __name__ == "__main__":
    main()
