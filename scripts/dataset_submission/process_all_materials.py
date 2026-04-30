"""
Build the submission dataset for a range of materials, end-to-end.

For each material id under SRC_ROOT/<id>/ that has all REQUIRED_FILES, this:
  1. crops every HDR image by the projected sample-rectangle polygon mask
     (with --dilate_px safety buffer), writing to DST_ROOT/<id>/hdr/
  2. copies the small per-material metadata files into DST_ROOT/<id>/
  3. once, at the end, copies the global calibration / list files into DST_ROOT/

Sequential by design — meant to run on the file server itself (no NFS), so
adding workers is straightforward later. For the NFS path keep `--crop_workers`
small (≤ 3); larger values stall the writeback pipeline and starve other jobs
on the same mount (this happened to us with --workers 12 earlier).

Usage:
    # Smoke test on one material
    python process_all_materials.py --range 227 --skip_existing

    # First batch on file server
    python process_all_materials.py --range 0-99 --crop_workers 8

    # Resume after interruption
    python process_all_materials.py --range 0-484 --skip_existing

The --range argument accepts:
    227                       single material
    0-99                      inclusive numeric range
    0-99,200-227,300          mixed ranges + singles, comma separated
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "dataset_submission"))

from crop_hdr_by_mask import (
    crop_material,
    load_intrinsics_from_config,
)
from build_submission import (
    build_material,
    copy_global,
    PER_MATERIAL_FILES,
    GLOBAL_FILES,
)


# A material is processed only if all of these exist non-empty under its folder.
# (Mirrors REQUIRED_FILES in scripts/upload_to_computecanada.py — same gate.)
REQUIRED_FILES = [
    "scan_log.json",
    "rotated_camera.json",
    "bbox.json",
    "point_metadata.json",
    "observations_structured.npz",
]


def parse_range(spec):
    """'0-99,200-227,300' -> sorted unique list of strings ['0', '1', ..., '300']."""
    out = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            out.update(str(i) for i in range(lo, hi + 1))
        else:
            int(chunk)  # validate
            out.add(chunk)
    return sorted(out, key=lambda s: int(s))


def material_is_eligible(mat_dir):
    """All REQUIRED_FILES exist and are non-empty."""
    for name in REQUIRED_FILES:
        p = os.path.join(mat_dir, name)
        if not os.path.exists(p):
            return False, f"missing {name}"
        if os.path.getsize(p) == 0:
            return False, f"empty {name}"
    return True, ""


def material_is_already_done(dst_mat_dir, expected_hdr_count):
    """Treat a material as done iff (a) hdr folder exists with at least
    `expected_hdr_count` PNGs, (b) all PER_MATERIAL_FILES are present,
    and (c) hdr_crop_bboxes.json (with polygons) is present.

    We only sample a quick file-count check for hdr/ — full bit-checks belong
    to the unit-test scripts, which run separately.
    """
    if not os.path.isdir(dst_mat_dir):
        return False
    bbox_json = os.path.join(dst_mat_dir, "hdr_crop_bboxes.json")
    if not os.path.exists(bbox_json):
        return False
    try:
        with open(bbox_json) as f:
            meta = json.load(f)
        if "polygons" not in meta or not meta["polygons"]:
            return False
    except Exception:
        return False
    for name in PER_MATERIAL_FILES:
        if not os.path.exists(os.path.join(dst_mat_dir, name)):
            return False
    hdr_dir = os.path.join(dst_mat_dir, "hdr")
    if not os.path.isdir(hdr_dir):
        return False
    n_png = sum(1 for f in os.listdir(hdr_dir) if f.endswith(".png"))
    return n_png >= expected_hdr_count


def count_expected_hdr(src_mat_dir):
    """How many HDR files we expect to write for this material — same logic the
    cropper uses (skip unmatched scan ids and scans whose camera has no entry
    in rotated_camera.json)."""
    with open(os.path.join(src_mat_dir, "scan_log.json")) as f:
        scan_log = json.load(f)
    with open(os.path.join(src_mat_dir, "rotated_camera.json")) as f:
        rc = {entry["camera_id"] for entry in json.load(f)}
    unmatched_path = os.path.join(src_mat_dir, "unmatched_scan_ids.json")
    if os.path.exists(unmatched_path):
        with open(unmatched_path) as f:
            unmatched = set(json.load(f))
    else:
        unmatched = set()
    n = 0
    hdr_dir = os.path.join(src_mat_dir, "hdr")
    for s in scan_log:
        if int(s.get("id", -1)) in unmatched:
            continue
        if s["camera_id"] not in rc:
            continue
        if not os.path.exists(os.path.join(hdr_dir, s["filename"])):
            continue
        n += 1
    return n


def process_one(material_id, src_root, dst_root, intrinsics, dilate_px,
                crop_workers, png_compression, skip_existing, copy_mode):
    src_mat = os.path.join(src_root, material_id)
    dst_mat = os.path.join(dst_root, material_id)

    ok, reason = material_is_eligible(src_mat)
    if not ok:
        return "skipped_ineligible", reason, 0.0

    expected = count_expected_hdr(src_mat)
    if skip_existing and material_is_already_done(dst_mat, expected):
        return "skipped_done", f"already has {expected} cropped HDRs + metadata", 0.0

    t0 = time.time()
    crop_material(
        src_root=src_root,
        dst_root=dst_root,
        material_id=material_id,
        intrinsics=intrinsics,
        dilate_px=dilate_px,
        workers=crop_workers,
        png_compression=png_compression,
        overwrite=False,  # resume-safe; partial runs continue from where they stopped
    )
    build_material(src_root=src_root, dst_root=dst_root,
                   material_id=material_id, mode=copy_mode)
    return "ok", f"{expected} HDRs cropped + metadata copied", time.time() - t0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="/media/raid/cloth/capture_data/Dataset_Nov11",
                   help="Source dataset root (contains <material>/hdr/...)")
    p.add_argument("--dst", default="/media/raid/cloth/Dataset_submission",
                   help="Destination submission root")
    p.add_argument("--range", required=True,
                   help="Material range, e.g. '0-99', '227', or '0-99,200-227,300'")
    p.add_argument("--dilate_px", type=int, default=16,
                   help="Pixel dilation around projected rectangle (~1.2mm at typical depth).")
    p.add_argument("--crop_workers", type=int, default=1,
                   help="Workers within a single material's HDR crop. Use 1 for "
                        "the sequential sanity run; bump to 8-12 once running on "
                        "local disk. Keep <=3 on NFS.")
    p.add_argument("--png_compression", type=int, default=3)
    p.add_argument("--copy_mode", default="copy",
                   choices=["copy", "symlink", "hardlink"],
                   help="How to populate small per-material files. 'copy' makes "
                        "a fully self-contained submission; 'hardlink' is the "
                        "fastest and uses no extra space on the same FS.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip materials that already have a complete submission folder.")
    p.add_argument("--copy_globals", action="store_true",
                   help="Also copy emitter_calibration.json + sample_size.json + "
                        "training_list/test_list at the end of the run.")
    p.add_argument("--lists", nargs="*",
                   default=["training_list.txt", "test_list.txt"])
    p.add_argument("--log_file", default=None,
                   help="Append per-material status to this file.")
    p.add_argument("--continue_on_error", action="store_true",
                   help="Don't abort the batch if one material fails.")
    p.add_argument("--dry_run", action="store_true",
                   help="Don't crop/copy anything — just print which materials "
                        "are eligible, already done, or would be processed.")
    return p.parse_args()


def main():
    args = parse_args()
    materials = parse_range(args.range)
    print(f"Source:      {args.src}")
    print(f"Destination: {args.dst}")
    print(f"Materials:   {len(materials)} ids in range '{args.range}'")
    print(f"dilate_px={args.dilate_px}  crop_workers={args.crop_workers}  "
          f"copy_mode={args.copy_mode}  skip_existing={args.skip_existing}")

    intrinsics = load_intrinsics_from_config()
    print(f"Intrinsics:  {intrinsics}")

    if args.dry_run:
        print("\n=== DRY RUN — no files will be cropped or copied ===")
        plan = {"would_process": [], "already_done": [], "ineligible": []}
        for mid in materials:
            src_mat = os.path.join(args.src, mid)
            ok, reason = material_is_eligible(src_mat)
            if not ok:
                plan["ineligible"].append((mid, reason))
                continue
            expected = count_expected_hdr(src_mat)
            dst_mat = os.path.join(args.dst, mid)
            if args.skip_existing and material_is_already_done(dst_mat, expected):
                plan["already_done"].append((mid, expected))
            else:
                plan["would_process"].append((mid, expected))
        print(f"  would_process: {len(plan['would_process'])}  "
              f"already_done: {len(plan['already_done'])}  "
              f"ineligible: {len(plan['ineligible'])}")
        if plan["would_process"][:10]:
            print("  first 10 would_process:")
            for mid, n in plan["would_process"][:10]:
                print(f"    mat {mid}: {n} HDR files to crop")
        if plan["ineligible"][:10]:
            print("  first 10 ineligible:")
            for mid, reason in plan["ineligible"][:10]:
                print(f"    mat {mid}: {reason}")
        sys.exit(0)

    log_fh = open(args.log_file, "a") if args.log_file else None
    counts = {"ok": 0, "skipped_done": 0, "skipped_ineligible": 0, "error": 0}
    timings = []
    failures = []

    t_global = time.time()
    for i, mid in enumerate(materials, 1):
        line_prefix = f"[{i:>4d}/{len(materials)}] mat {mid}"
        try:
            status, msg, dt = process_one(
                material_id=mid,
                src_root=args.src,
                dst_root=args.dst,
                intrinsics=intrinsics,
                dilate_px=args.dilate_px,
                crop_workers=args.crop_workers,
                png_compression=args.png_compression,
                skip_existing=args.skip_existing,
                copy_mode=args.copy_mode,
            )
            counts[status] += 1
            if status == "ok":
                timings.append(dt)
                print(f"{line_prefix}  OK   ({dt:.1f}s)  {msg}")
            else:
                print(f"{line_prefix}  {status}: {msg}")
            if log_fh:
                log_fh.write(f"{mid}\t{status}\t{dt:.2f}\t{msg}\n")
                log_fh.flush()
        except Exception as e:  # noqa: BLE001
            counts["error"] += 1
            tb = traceback.format_exc(limit=4)
            print(f"{line_prefix}  ERROR: {e}\n{tb}")
            failures.append((mid, str(e)))
            if log_fh:
                log_fh.write(f"{mid}\terror\t0\t{e}\n")
                log_fh.flush()
            if not args.continue_on_error:
                break

    if args.copy_globals:
        print("\n=== Copying global calibration / list files ===")
        copy_global(src_root=args.src, dst_root=args.dst,
                    lists=args.lists, mode=args.copy_mode)

    elapsed = time.time() - t_global
    print("\n=== Summary ===")
    print(f"  ok={counts['ok']}  skipped_done={counts['skipped_done']}  "
          f"skipped_ineligible={counts['skipped_ineligible']}  error={counts['error']}")
    if timings:
        avg = sum(timings) / len(timings)
        print(f"  avg time / processed material: {avg:.1f}s "
              f"(min {min(timings):.1f}s, max {max(timings):.1f}s)")
    print(f"  total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if failures:
        print(f"  {len(failures)} failures:")
        for mid, err in failures[:20]:
            print(f"    mat {mid}: {err}")

    if log_fh:
        log_fh.close()
    sys.exit(1 if counts["error"] > 0 else 0)


if __name__ == "__main__":
    main()
