"""
Unit test for verify_completed_materials() in job_scheduler.py.

Builds a fake state dict with three COMPLETED materials in different conditions
and asserts the sanity-check transitions are correct:

  1. Fully healthy (registration OK + structured npz exists)
     → stays COMPLETED
  2. Registration OK but observations_structured.npz missing
     → reset to COLMAP_DONE (will trigger shape_matching rerun)
  3. Low registration
     → reset to NOT_STARTED with colmap_variant="exhaustive",
        sparse/ archived to sparse_seq_failed/

Run:
    python recon/scheduler/tests/test_restart_sanity_check.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "recon" / "scheduler"))
sys.path.insert(0, str(REPO_ROOT / "recon" / "calibration"))

from read_write_model import Image, write_images_binary  # noqa: E402
from job_scheduler import verify_completed_materials, JobStatus  # noqa: E402


def make_material(
    parent: Path, name: str, n_registered: int, n_scans: int, with_structured: bool
) -> Path:
    """Create a fake material folder with optional observations_structured.npz."""
    mat = parent / name
    sparse = mat / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)

    images = {}
    for i in range(1, n_registered + 1):
        images[i] = Image(
            id=i,
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.array([0.0, 0.0, 0.0]),
            camera_id=1,
            name=f"scan-{i-1:04d}.png",
            xys=np.zeros((0, 2)),
            point3D_ids=np.array([], dtype=np.int64),
        )
    write_images_binary(images, str(sparse / "images.bin"))

    scan_log = [{"scan_id": i, "position": [0, 0, 0], "position_light": [0, 0, 0]}
                for i in range(n_scans)]
    with open(mat / "scan_log.json", "w") as f:
        json.dump(scan_log, f)

    if with_structured:
        # Just needs to exist and be non-empty; the helper only checks size.
        np.savez(mat / "observations_structured.npz", dummy=np.array([1, 2, 3]))

    return mat


def make_completed_info(folder_path: Path) -> dict:
    return {
        "status": JobStatus.COMPLETED,
        "folder_path": str(folder_path),
        "pid": None,
        "gpu": None,
        "workers": None,
        "colmap_start_time": "2026-01-01T00:00:00",
        "colmap_end_time": "2026-01-01T00:30:00",
        "shape_matching_start_time": "2026-01-01T00:30:00",
        "shape_matching_end_time": "2026-01-01T00:35:00",
        "error": None,
        "ready": True,
        "colmap_variant": "sequential",
        "colmap_attempts": 1,
    }


def assert_(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 3 fake materials, all currently marked COMPLETED
        good      = make_material(tmp, "good",      n_registered=590, n_scans=590, with_structured=True)
        no_struct = make_material(tmp, "no_struct", n_registered=585, n_scans=590, with_structured=False)
        bad_reg   = make_material(tmp, "bad_reg",   n_registered=10,  n_scans=590, with_structured=True)

        state = {
            "materials": {
                "good":      make_completed_info(good),
                "no_struct": make_completed_info(no_struct),
                "bad_reg":   make_completed_info(bad_reg),
            }
        }

        n_colmap, n_shape = verify_completed_materials(state, verbose=True)

        print()
        assert_(n_colmap == 1, f"n_colmap_reset == 1 (got {n_colmap})")
        assert_(n_shape == 1, f"n_shape_reset == 1 (got {n_shape})")

        # Material 'good' should be untouched
        g = state["materials"]["good"]
        assert_(g["status"] == JobStatus.COMPLETED, f"good still COMPLETED (got {g['status']})")
        assert_(g["colmap_variant"] == "sequential",
                f"good keeps sequential variant (got {g['colmap_variant']})")

        # Material 'no_struct' should be reset to COLMAP_DONE for shape rerun
        ns = state["materials"]["no_struct"]
        assert_(ns["status"] == JobStatus.COLMAP_DONE,
                f"no_struct → COLMAP_DONE (got {ns['status']})")
        assert_(ns["colmap_variant"] == "sequential",
                f"no_struct keeps sequential variant (got {ns['colmap_variant']})")
        assert_(ns["shape_matching_end_time"] is None,
                "no_struct shape_matching_end_time cleared")
        # COLMAP timestamps preserved (we don't want to rerun COLMAP)
        assert_(ns["colmap_end_time"] == "2026-01-01T00:30:00",
                "no_struct colmap_end_time preserved")
        # The sparse dir for no_struct should still exist (we did not archive it)
        assert_((no_struct / "sparse").exists(),
                "no_struct sparse/ kept (we only need shape_matching, not COLMAP)")

        # Material 'bad_reg' should be reset for exhaustive COLMAP
        br = state["materials"]["bad_reg"]
        assert_(br["status"] == JobStatus.NOT_STARTED,
                f"bad_reg → NOT_STARTED (got {br['status']})")
        assert_(br["colmap_variant"] == "exhaustive",
                f"bad_reg → exhaustive variant (got {br['colmap_variant']})")
        assert_(br["colmap_end_time"] is None, "bad_reg colmap_end_time cleared")
        assert_(br["shape_matching_end_time"] is None, "bad_reg shape_matching_end_time cleared")
        assert_(not (bad_reg / "sparse").exists(),
                "bad_reg sparse/ moved away")
        assert_((bad_reg / "sparse_seq_failed").exists(),
                "bad_reg sparse_seq_failed/ created")

        print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
