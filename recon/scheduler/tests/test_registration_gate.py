"""
Unit test for evaluate_colmap_registration() in job_scheduler.py.

Sets up fake material directories with stub COLMAP outputs (real binary
images.bin so the read path is exercised) and asserts that the gate makes
the right decision in three scenarios:

  1. Healthy registration (>= 95%) → proceed to COLMAP_DONE
  2. Low registration on sequential variant → archive sparse, return retry signal
  3. Low registration on exhaustive variant → return failure signal

Run:
    python recon/scheduler/tests/test_registration_gate.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "recon" / "scheduler"))
sys.path.insert(0, str(REPO_ROOT / "recon" / "calibration"))

from read_write_model import Image, write_images_binary  # noqa: E402
from job_scheduler import evaluate_colmap_registration  # noqa: E402
from registration_check import REGISTRATION_THRESHOLD  # noqa: E402


def make_fake_material(tmpdir: Path, n_registered: int, n_scans: int) -> Path:
    """Create a material folder with a stub sparse/0/images.bin and scan_log.json."""
    mat_dir = tmpdir / "fake_material"
    sparse_dir = mat_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Build a real images.bin with n_registered fake entries
    images = {}
    for i in range(1, n_registered + 1):
        images[i] = Image(
            id=i,
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.array([0.0, 0.0, 0.0]),
            camera_id=1,
            name=f"scan-{i-1:04d}.png",
            xys=np.zeros((0, 2)),  # no observations needed for this test
            point3D_ids=np.array([], dtype=np.int64),
        )
    write_images_binary(images, str(sparse_dir / "images.bin"))

    # scan_log with n_scans entries (any minimal valid JSON list)
    scan_log = [{"scan_id": i, "position": [0, 0, 0], "position_light": [0, 0, 0]}
                for i in range(n_scans)]
    with open(mat_dir / "scan_log.json", "w") as f:
        json.dump(scan_log, f)

    return mat_dir


def make_info(folder_path: Path, variant: str) -> dict:
    """Build the dict shape that update_job_status passes to the gate."""
    return {
        "folder_path": str(folder_path),
        "colmap_variant": variant,
    }


def assert_(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def test_healthy_registration():
    print("\n[Test 1] Healthy registration (590/590) on sequential variant")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mat = make_fake_material(tmp, n_registered=590, n_scans=590)
        info = make_info(mat, "sequential")

        healthy, msg = evaluate_colmap_registration("test", info)
        assert_(healthy is True, f"healthy=True (got {healthy}), msg={msg}")
        assert_("100.0%" in msg, f"msg mentions 100.0% (got: {msg})")
        # Sparse must NOT be archived for healthy materials
        assert_((mat / "sparse").exists(), "sparse/ still exists")
        assert_(not (mat / "sparse_seq_failed").exists(), "sparse_seq_failed/ NOT created")


def test_low_sequential_triggers_retry():
    print("\n[Test 2] Low registration (10/590 = 1.7%) on sequential variant")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mat = make_fake_material(tmp, n_registered=10, n_scans=590)
        info = make_info(mat, "sequential")

        healthy, msg = evaluate_colmap_registration("test", info)
        assert_(healthy is False, f"healthy=False (got {healthy}), msg={msg}")
        assert_("1.7%" in msg, f"msg mentions 1.7% (got: {msg})")
        assert_("exhaustive" in msg, f"msg mentions exhaustive retry (got: {msg})")
        # Sparse MUST be archived
        assert_(not (mat / "sparse").exists(), "sparse/ has been moved away")
        assert_((mat / "sparse_seq_failed").exists(), "sparse_seq_failed/ created")
        assert_((mat / "sparse_seq_failed" / "0" / "images.bin").exists(),
                "archived images.bin survives the move")


def test_low_exhaustive_marks_failed():
    print("\n[Test 3] Low registration (10/590) on exhaustive variant (final failure)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mat = make_fake_material(tmp, n_registered=10, n_scans=590)
        info = make_info(mat, "exhaustive")

        healthy, msg = evaluate_colmap_registration("test", info)
        assert_(healthy is False, f"healthy=False (got {healthy}), msg={msg}")
        assert_("exhaustive retry also failed" in msg,
                f"msg mentions terminal exhaustive failure (got: {msg})")
        # On exhaustive failure we do NOT archive again — keep the (bad) sparse so
        # a human can inspect it.
        assert_((mat / "sparse").exists(), "sparse/ kept after exhaustive failure")
        assert_(not (mat / "sparse_seq_failed").exists(),
                "no archive on exhaustive failure")


def test_threshold_just_below():
    print(f"\n[Test 4] Registration just below threshold ({REGISTRATION_THRESHOLD*100:.0f}%)")
    # 567/598 = 94.8% - exactly the marginal case from the live dataset (mat 1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mat = make_fake_material(tmp, n_registered=567, n_scans=598)
        info = make_info(mat, "sequential")

        healthy, msg = evaluate_colmap_registration("test", info)
        assert_(healthy is False, f"healthy=False (got {healthy}), msg={msg}")
        assert_("94.8%" in msg, f"msg mentions 94.8% (got: {msg})")


def test_threshold_just_above():
    print(f"\n[Test 5] Registration just above threshold ({REGISTRATION_THRESHOLD*100:.0f}%)")
    # 576/588 = 97.96% - mat 92's actual numbers (legit sparse cloud, healthy ratio)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mat = make_fake_material(tmp, n_registered=576, n_scans=588)
        info = make_info(mat, "sequential")

        healthy, msg = evaluate_colmap_registration("test", info)
        assert_(healthy is True, f"healthy=True (got {healthy}), msg={msg}")
        assert_("98.0%" in msg, f"msg mentions 98.0% (got: {msg})")


def main():
    test_healthy_registration()
    test_low_sequential_triggers_retry()
    test_low_exhaustive_marks_failed()
    test_threshold_just_below()
    test_threshold_just_above()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
