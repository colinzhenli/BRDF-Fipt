"""
Assemble a single material's submission folder under
  /media/raid/cloth/Dataset_submission/<material_id>/

Per-material layout produced (the minimum used by stage 1 dense + stage 2 from-Real
training, plus a few small files that downstream scripts expect):
    bbox.json
    point_metadata.json
    point_positions.npz
    rotated_camera.json
    scan_log.json
    unmatched_scan_ids.json
    observations_structured.npz
    hdr/                       # cropped (written by crop_hdr_by_mask.py first)
    hdr_crop_bboxes.json       # written by crop_hdr_by_mask.py

Skipped on purpose (large or only used by preprocessing, not training):
    ldr/                  ~5.3 GB, not used by stage 1/2 training
    sparse/               ~700 MB, only used by preprocessing
    colmap.log, shape_matching.log
    light_distribution_*.png  (visualizations)

Global files (copied once into the dataset root):
    emitter_calibration.json
    sample_size.json
    training_list.txt, test_list.txt   (whichever the user picks via --lists)

Run:
    python crop_hdr_by_mask.py --material 227   # write hdr/ first
    python build_submission.py  --material 227   # then copy the rest
"""

import argparse
import json
import os
import shutil
import sys


PER_MATERIAL_FILES = [
    "bbox.json",
    "point_metadata.json",
    "point_positions.npz",
    "rotated_camera.json",
    "scan_log.json",
    "unmatched_scan_ids.json",
    "observations_structured.npz",
]

GLOBAL_FILES = [
    "emitter_calibration.json",
    "sample_size.json",
]


def safe_copy(src, dst, mode):
    if not os.path.exists(src):
        print(f"  [skip] missing source: {src}")
        return False
    if os.path.exists(dst):
        # Already populated; assume the prior run was correct.
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:  # copy
        shutil.copy2(src, dst)
    return True


def build_material(src_root, dst_root, material_id, mode="copy"):
    src = os.path.join(src_root, str(material_id))
    dst = os.path.join(dst_root, str(material_id))
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Source material folder not found: {src}")

    hdr_dst = os.path.join(dst, "hdr")
    if not os.path.isdir(hdr_dst):
        raise FileNotFoundError(
            f"Cropped HDR folder missing: {hdr_dst}. "
            "Run crop_hdr_by_mask.py first."
        )

    print(f"[mat {material_id}] populating {dst} (mode={mode})")
    for name in PER_MATERIAL_FILES:
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        ok = safe_copy(s, d, mode)
        print(f"  {'ok' if ok else 'skip'}: {name}")

    print(f"[mat {material_id}] done.")


def copy_global(src_root, dst_root, lists, mode="copy"):
    for name in GLOBAL_FILES:
        s = os.path.join(src_root, name)
        d = os.path.join(dst_root, name)
        ok = safe_copy(s, d, mode)
        print(f"  {'ok' if ok else 'skip'}: {name}")
    for lname in lists:
        s = os.path.join(src_root, lname)
        d = os.path.join(dst_root, lname)
        ok = safe_copy(s, d, mode)
        print(f"  {'ok' if ok else 'skip'}: {lname}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="/media/raid/cloth/capture_data/Dataset_Nov11")
    p.add_argument("--dst", default="/media/raid/cloth/Dataset_submission")
    p.add_argument("--material", required=True)
    p.add_argument("--mode", default="copy", choices=["copy", "symlink", "hardlink"],
                   help="copy preserves a fully self-contained submission folder; "
                        "symlink/hardlink save space when the source filesystem persists.")
    p.add_argument("--lists", nargs="*",
                   default=["training_list.txt", "test_list.txt"],
                   help="Global list files to copy alongside emitter_calibration.json")
    p.add_argument("--copy_globals", action="store_true",
                   help="Also copy global calibration/list files (run once per submission build)")
    return p.parse_args()


def main():
    args = parse_args()
    build_material(args.src, args.dst, args.material, mode=args.mode)
    if args.copy_globals:
        copy_global(args.src, args.dst, args.lists, mode=args.mode)


if __name__ == "__main__":
    main()
