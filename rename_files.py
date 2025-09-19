#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm  # pip install tqdm

# ==== ONLY CHANGE THIS ====
ROOT = Path("/media/raid/cloth/New_turntable_Sep15/BRDF_recon")  # contains scans_0915/ and scan_log_0915.json
# ==========================

SCANS_DIR = ROOT / "scans_0915"
LOG_IN = ROOT / "scan_log_0915.json"
LOG_OUT = ROOT / "scan_log_0915_reindexed.json"

# Old filename pattern example: scan-22-8-phi0.598_theta0.000.exr
CUR_PATTERN = re.compile(
    r"^scan-(?P<light>\d+)-(?P<cam>\d+)-phi(?P<phi>[0-9]+\.[0-9]{3})_theta(?P<theta>[0-9]+\.[0-9]{3})\.(?P<ext>[^.]+)$"
)

def f3(x):
    return f"{float(x):.3f}"

def scan_id_str(i):
    return f"{i:04d}"

def parse_files(scan_dir: Path):
    files = []
    for p in scan_dir.iterdir():
        if not p.is_file():
            continue
        m = CUR_PATTERN.match(p.name)
        if not m:
            continue
        files.append({
            "path": p,
            "light": int(m.group("light")),
            "cam": int(m.group("cam")),
            "phi_s": m.group("phi"),
            "theta_s": m.group("theta"),
            "ext": m.group("ext").lower(),
        })
    if not files:
        raise FileNotFoundError(f"No files matched pattern in {scan_dir}")
    return files

def main():
    if not SCANS_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {SCANS_DIR}")
    if not LOG_IN.exists():
        raise FileNotFoundError(f"Missing JSON log: {LOG_IN}")

    # 1) Load JSON
    with open(LOG_IN, "r") as f:
        entries = json.load(f)

    # 2) Index entries by (light, phi_s, theta_s)
    entry_by_key = {}
    for idx, e in enumerate(entries):
        light = int(e["light_id"])
        phi_s = f3(e.get("phi", 0.0))
        theta_s = f3(e.get("theta", 0.0))
        key = (light, phi_s, theta_s)
        if key in entry_by_key:
            # If duplicates exist, we can't disambiguate—fail loudly.
            raise RuntimeError(f"Duplicate JSON key {key} at entries {entry_by_key[key]['__idx__']} and {idx}")
        entry_by_key[key] = {**e, "__idx__": idx}

    # 3) Parse files & sort order to assign scan_id
    files = parse_files(SCANS_DIR)
    files_sorted = sorted(files, key=lambda d: (d["light"], d["cam"]))
    # Map (light, phi, theta) -> scan_id string
    key_to_scanid = {}
    # Also map by filename identity for renaming plan
    rename_plans = []  # list of (src_path, dst_path)
    dst_names_set = set()

    for new_i, fi in enumerate(files_sorted):
        sid = scan_id_str(new_i)
        key = (fi["light"], fi["phi_s"], fi["theta_s"])
        key_to_scanid[key] = sid

        new_name = f"scan-{sid}_light-{fi['light']}_camera-{sid}_phi{fi['phi_s']}_theta{fi['theta_s']}.{fi['ext']}"
        dst_path = fi["path"].with_name(new_name)
        if dst_path.name in dst_names_set:
            raise RuntimeError(f"Planned destination collision: {dst_path.name}")
        dst_names_set.add(dst_path.name)
        if fi["path"] != dst_path:
            rename_plans.append((fi["path"], dst_path))

    # 4) Build new JSON (same entry order as original), attach scan_id/camera_id/filename
    new_entries = [None] * len(entries)
    for e in entries:
        light = int(e["light_id"])
        phi_s = f3(e.get("phi", 0.0))
        theta_s = f3(e.get("theta", 0.0))
        key = (light, phi_s, theta_s)
        if key not in key_to_scanid:
            raise FileNotFoundError(
                f"No image file for JSON entry key {key} "
                f"(light_id={light}, phi={phi_s}, theta={theta_s})."
            )
        sid = key_to_scanid[key]
        # Extension for filename: find any src file for this key (unique by design)
        # Reconstruct extension via reverse lookup in files list:
        ext = next(fi["ext"] for fi in files if (fi["light"], fi["phi_s"], fi["theta_s"]) == key)
        new_name = f"scan-{sid}_light-{light}_camera-{sid}_phi{phi_s}_theta{theta_s}.{ext}"

        new_e = {
            **e,  # keep everything
            "scan_id": sid,              # new zero-padded string
            "camera_id": sid,            # set camera_id equal to scan_id (string)
            "filename": new_name,        # updated filename
        }
        idx = entry_by_key[key]["__idx__"]
        new_entries[idx] = new_e

    # 5) Rename concurrently with progress bar
    #    (skips if src == dst; we only queued when different)
    if rename_plans:
        max_workers = min(32, (os_cpu_count() or 8) * 2)
        with ThreadPoolExecutor(max_workers=max_workers) as ex, tqdm(total=len(rename_plans), desc="Renaming", unit="file") as pbar:
            futures = [ex.submit(_rename_one, src, dst) for (src, dst) in rename_plans]
            for fut in as_completed(futures):
                # Raise exceptions immediately if any
                fut.result()
                pbar.update(1)

    # 6) Write out JSON
    with open(LOG_OUT, "w") as f:
        # Strip helper indices
        for e in new_entries:
            e.pop("__idx__", None)
        json.dump(new_entries, f, indent=2)

    print(f"\nDone. Renamed {len(rename_plans)} files.")
    print(f"Wrote updated log: {LOG_OUT}")

def _rename_one(src: Path, dst: Path):
    # Guard against accidental overwrite
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    src.rename(dst)

def os_cpu_count():
    try:
        import os
        return os.cpu_count()
    except Exception:
        return None

if __name__ == "__main__":
    main()
