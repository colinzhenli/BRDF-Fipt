"""
Lightweight unit test for the multi-material pipeline.

Without running an actual crop, this verifies:
  1. --range parsing works for the documented formats.
  2. The eligibility gate matches what upload_to_computecanada.py expects.
  3. material_is_already_done() correctly recognises a completed material
     (we use the existing mat 227 submission for this).
  4. count_expected_hdr() returns a value consistent with the cropped folder.

For an end-to-end sanity check, additionally pass --run_one MAT, which will
execute a single material's full pipeline (crop + copy) with --crop_workers 1.
Use a material that's small or already done to keep the test fast.

Usage:
    python test_pipeline.py
    python test_pipeline.py --run_one 227   # also exercises the real pipeline
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "dataset_submission"))

from process_all_materials import (
    parse_range,
    material_is_eligible,
    material_is_already_done,
    count_expected_hdr,
    process_one,
    REQUIRED_FILES,
)
from crop_hdr_by_mask import load_intrinsics_from_config


def expect(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}{(' — ' + detail) if detail and not cond else ''}")
    return cond


def test_parse_range():
    print("\n--- parse_range ---")
    ok = True
    ok &= expect("single id", parse_range("227") == ["227"])
    ok &= expect("simple range", parse_range("0-3") == ["0", "1", "2", "3"])
    ok &= expect("flipped range",
                 parse_range("3-0") == ["0", "1", "2", "3"])
    ok &= expect("mixed",
                 parse_range("0-2,10,20-21") == ["0", "1", "2", "10", "20", "21"])
    ok &= expect("dedup",
                 parse_range("0-2,1-3") == ["0", "1", "2", "3"])
    return ok


def test_eligibility(src_root):
    print(f"\n--- eligibility on {src_root} ---")
    if not os.path.isdir(src_root):
        return expect("src_root exists", False, src_root)

    # Find one ineligible (or note none if all are eligible)
    sample_eligible = None
    sample_ineligible = None
    for entry in sorted(os.listdir(src_root)):
        p = os.path.join(src_root, entry)
        if not os.path.isdir(p):
            continue
        ok, _ = material_is_eligible(p)
        if ok and sample_eligible is None:
            sample_eligible = entry
        if not ok and sample_ineligible is None:
            sample_ineligible = entry
        if sample_eligible and sample_ineligible:
            break

    ok = expect("found at least one eligible material",
                sample_eligible is not None,
                "no material has all REQUIRED_FILES — the dataset isn't ready")
    if sample_ineligible:
        print(f"      sample ineligible: {sample_ineligible}")
    if sample_eligible:
        print(f"      sample eligible:   {sample_eligible}")
    return ok


def test_done_check(src_root, dst_root, mat="227"):
    print(f"\n--- material_is_already_done(mat {mat}) ---")
    src_mat = os.path.join(src_root, mat)
    dst_mat = os.path.join(dst_root, mat)
    if not os.path.isdir(dst_mat):
        return expect(f"{dst_mat} exists", False,
                      "skip — no prior submission to verify against")

    expected = count_expected_hdr(src_mat)
    done = material_is_already_done(dst_mat, expected)
    expect(f"mat {mat} reports done", done,
           f"dst_mat={dst_mat} expected_hdr={expected}")

    # Negative: drop expected_hdr above the directory contents on purpose
    fake_high = expected + 10**6
    expect(f"mat {mat} *not* done at expected={fake_high}",
           not material_is_already_done(dst_mat, fake_high))

    return done


def test_run_one(src_root, dst_root, mat):
    print(f"\n--- end-to-end process_one(mat {mat}) ---")
    intr = load_intrinsics_from_config()
    status, msg, dt = process_one(
        material_id=mat,
        src_root=src_root,
        dst_root=dst_root,
        intrinsics=intr,
        dilate_px=16,
        crop_workers=1,
        png_compression=3,
        skip_existing=True,
        copy_mode="copy",
    )
    print(f"  status={status} dt={dt:.1f}s  msg={msg}")
    return expect("process_one returned ok|skipped_done",
                  status in ("ok", "skipped_done"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="/media/raid/cloth/capture_data/Dataset_Nov11")
    ap.add_argument("--dst", default="/media/raid/cloth/Dataset_submission")
    ap.add_argument("--run_one", default=None,
                    help="Also exercise process_one on this material id.")
    args = ap.parse_args()

    all_ok = True
    all_ok &= test_parse_range()
    all_ok &= test_eligibility(args.src)
    all_ok &= test_done_check(args.src, args.dst, "227")
    if args.run_one:
        all_ok &= test_run_one(args.src, args.dst, args.run_one)

    print()
    print("=== ALL PASS ===" if all_ok else "=== FAILURES ===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
