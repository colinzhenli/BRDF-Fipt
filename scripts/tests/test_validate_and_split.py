"""
Unit test for validate_and_split_data.py.

Builds a fake data folder with four material directories that exercise the
validation rules added in Step 4:

  1. "good"            — all required files + healthy registration → VALID
  2. "no_struct"       — missing observations_structured.npz       → INVALID
  3. "low_reg"         — has structured npz but registration below threshold → INVALID
  4. "missing_bbox"    — missing bbox.json                         → INVALID

Then asserts that:
  - validate_data_folder() returns exactly the one valid id ("good")
  - create_train_test_split() respects training_number and seed
  - check_registration_health() returns the expected ok/msg pairs

Run:
    python scripts/tests/test_validate_and_split.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "recon" / "calibration"))

from read_write_model import Image, write_images_binary  # noqa: E402
from validate_and_split_data import (  # noqa: E402
    REQUIRED_FILES,
    REQUIRED_FILES_CAN_BE_EMPTY,
    REQUIRED_FOLDERS,
    check_registration_health,
    create_train_test_split,
    validate_data_folder,
    validate_material_folder,
)


def make_material(
    parent: Path,
    name: str,
    n_registered: int,
    n_scans: int,
    *,
    with_structured: bool = True,
    skip_files: tuple[str, ...] = (),
) -> Path:
    """
    Create a fake material folder that satisfies validate_material_folder()
    by default. Use with_structured=False or skip_files=(...) to introduce
    specific failures.
    """
    mat = parent / name
    mat.mkdir(parents=True, exist_ok=True)

    # Required folders — populate them with a placeholder file so they are not empty.
    for folder in REQUIRED_FOLDERS:
        d = mat / folder
        d.mkdir(parents=True, exist_ok=True)
        (d / ".keep").write_text("")

    # sparse/0/images.bin with n_registered fake entries
    sparse0 = mat / "sparse" / "0"
    sparse0.mkdir(parents=True, exist_ok=True)
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
    write_images_binary(images, str(sparse0 / "images.bin"))

    # scan_log.json with n_scans entries
    scan_log = [
        {"scan_id": i, "position": [0, 0, 0], "position_light": [0, 0, 0]}
        for i in range(n_scans)
    ]
    (mat / "scan_log.json").write_text(json.dumps(scan_log))

    # Required files (must exist + non-empty); allow caller to skip some
    file_payloads = {
        "bbox.json": '{"min": [0,0,0], "max": [1,1,1]}',
        "colmap.log": "stub colmap log\n",
        "point_metadata.json": '{"n_points": 1}',
        "rotated_camera.json": '{"R": [[1,0,0],[0,1,0],[0,0,1]]}',
        "shape_matching.log": "stub shape_matching log\n",
    }
    for fname, payload in file_payloads.items():
        if fname in skip_files:
            continue
        (mat / fname).write_text(payload)

    if with_structured and "observations_structured.npz" not in skip_files:
        np.savez(
            mat / "observations_structured.npz",
            cam_pos=np.zeros((n_scans, 3), dtype=np.float32),
            light_pos=np.zeros((n_scans, 3), dtype=np.float32),
            point_ids=np.arange(4, dtype=np.int32),
            xyz=np.zeros((4, 3), dtype=np.float32),
            rgbs=np.zeros((n_scans, 4, 3), dtype=np.uint16),
        )

    # Files that are required to exist but may be empty
    for fname in REQUIRED_FILES_CAN_BE_EMPTY:
        if fname in skip_files:
            continue
        (mat / fname).write_text("")

    return mat


def assert_(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def test_check_registration_health():
    print("\n[Test 1] check_registration_health() returns the right verdicts")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        good = make_material(tmp, "good", n_registered=590, n_scans=590)
        low = make_material(tmp, "low", n_registered=10, n_scans=590)

        ok_g, msg_g = check_registration_health(good)
        assert_(ok_g is True, f"good ok=True (got {ok_g}, msg={msg_g})")
        assert_("100.0%" in msg_g, f"good msg has 100.0% (got: {msg_g})")

        ok_l, msg_l = check_registration_health(low)
        assert_(ok_l is False, f"low ok=False (got {ok_l}, msg={msg_l})")
        assert_("Low registration" in msg_l, f"low msg flagged Low (got: {msg_l})")
        assert_("1.7%" in msg_l, f"low msg has 1.7% (got: {msg_l})")


def test_validate_material_folder_passes_for_healthy():
    print("\n[Test 2] validate_material_folder() passes for a fully healthy material")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        good = make_material(tmp, "good", n_registered=590, n_scans=590)
        is_valid, issues = validate_material_folder(good)
        assert_(is_valid is True, f"good is_valid (got {is_valid}, issues={issues})")
        assert_(len(issues) == 0, f"good has no issues (got: {issues})")


def test_validate_material_folder_flags_missing_structured():
    print("\n[Test 3] missing observations_structured.npz fails validation")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ns = make_material(
            tmp, "no_struct", n_registered=590, n_scans=590, with_structured=False
        )
        is_valid, issues = validate_material_folder(ns)
        assert_(is_valid is False, f"no_struct is_valid=False (got {is_valid})")
        assert_(
            any("observations_structured.npz" in i for i in issues),
            f"issues mention observations_structured.npz (got: {issues})",
        )


def test_validate_material_folder_flags_low_registration():
    print("\n[Test 4] low registration fails validation even with all files present")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        low = make_material(tmp, "low_reg", n_registered=10, n_scans=590)
        is_valid, issues = validate_material_folder(low)
        assert_(is_valid is False, f"low_reg is_valid=False (got {is_valid})")
        assert_(
            any("Low registration" in i for i in issues),
            f"issues mention Low registration (got: {issues})",
        )


def test_validate_data_folder_filters_correctly():
    print("\n[Test 5] validate_data_folder() returns only the truly healthy ids")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Use integer-named folders so they're treated as material ids
        make_material(tmp, "1", n_registered=590, n_scans=590)
        make_material(tmp, "2", n_registered=590, n_scans=590, with_structured=False)
        make_material(tmp, "3", n_registered=10, n_scans=590)
        make_material(
            tmp, "4", n_registered=590, n_scans=590, skip_files=("bbox.json",)
        )

        valid_ids = validate_data_folder(tmp, verbose=False)
        assert_(valid_ids == [1], f"only material 1 valid (got {valid_ids})")


def test_create_train_test_split_with_training_number():
    print("\n[Test 6] create_train_test_split() honors training_number and seed")
    valid = list(range(20))
    train, test = create_train_test_split(valid, train_ratio=0.5, seed=42, training_number=5)
    assert_(len(train) == 5, f"train has 5 (got {len(train)})")
    assert_(len(test) == 15, f"test has 15 (got {len(test)})")
    assert_(set(train) | set(test) == set(valid), "train ∪ test == valid set")
    assert_(set(train) & set(test) == set(), "train ∩ test == ∅")
    # Reproducibility
    train2, test2 = create_train_test_split(valid, train_ratio=0.5, seed=42, training_number=5)
    assert_(train == train2 and test == test2, "same seed → same split")


def main():
    test_check_registration_health()
    test_validate_material_folder_passes_for_healthy()
    test_validate_material_folder_flags_missing_structured()
    test_validate_material_folder_flags_low_registration()
    test_validate_data_folder_filters_correctly()
    test_create_train_test_split_with_training_number()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
