#!/usr/bin/env python3
"""
Merge observation chunks into a single efficient file per material.

For each valid material folder:
  - Reads all observations_chunk_*.npz (50 chunks, float64) sequentially
  - Converts to structured array with mixed dtypes (32 bytes/row, 60% savings):
      xyz:          float32 (3,)    — 12 bytes
      image_id:     uint16          —  2 bytes
      pixel_coords: float32 (2,)    —  8 bytes
      rgb:          uint16  (3,)    —  6 bytes
      point_id:     int32           —  4 bytes
  - Saves as all_observations.npy (single raw binary file)

Original observation folders are NOT removed.

Usage:
    python scripts/process_observations.py /media/raid/cloth/capture_data/Dataset_Nov11 --materials 0
    python scripts/process_observations.py /media/raid/cloth/capture_data/Dataset_Nov11
"""

import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import sys

OBS_DTYPE = np.dtype([
    ('xyz', np.float32, (3,)),
    ('image_id', np.uint16),
    ('pixel_coords', np.float32, (2,)),
    ('rgb', np.uint16, (3,)),
    ('point_id', np.int32),
])


def process_material(material_path):
    material_path = Path(material_path)
    material_id = material_path.name
    obs_folder = material_path / "observations"
    output_path = material_path / "all_observations.npy"

    if not obs_folder.exists():
        return material_id, "NO_OBS_FOLDER", 0, 0.0

    chunk_files = sorted(obs_folder.glob("observations_chunk_*.npz"))
    if not chunk_files:
        return material_id, "NO_CHUNKS", 0, 0.0

    try:
        metadata_path = material_path / "point_metadata.json"
        expected_total = None
        if metadata_path.exists():
            with open(metadata_path) as f:
                expected_total = json.load(f).get("num_observations")

        if not expected_total or expected_total <= 0:
            return material_id, "NO_METADATA", 0, 0.0

        # Pre-allocate SEPARATE contiguous arrays (avoids cache thrashing)
        xyz = np.empty((expected_total, 3), dtype=np.float32)
        iid = np.empty(expected_total, dtype=np.uint16)
        px  = np.empty((expected_total, 2), dtype=np.float32)
        rgb = np.empty((expected_total, 3), dtype=np.uint16)
        pid = np.empty(expected_total, dtype=np.int32)

        offset = 0
        for chunk_path in tqdm(
            chunk_files,
            desc=f"  Mat {material_id:>3s} read",
            leave=False,
            unit="chunk",
        ):
            try:
                data = np.load(chunk_path)
                obs = data["observations"]
                n = min(obs.shape[0], expected_total - offset)
                end = offset + n

                xyz[offset:end] = obs[:n, :3].astype(np.float32)
                iid[offset:end] = obs[:n, 3].astype(np.uint16)
                px[offset:end]  = obs[:n, 4:6].astype(np.float32)
                rgb[offset:end] = obs[:n, 6:9].astype(np.uint16)
                pid[offset:end] = obs[:n, 9].astype(np.int32)

                offset += n
                data.close()
                del obs
            except (EOFError, IOError, ValueError, KeyError) as e:
                tqdm.write(f"    [Warning] {chunk_path.name}: {e}")
                continue

        if offset == 0:
            return material_id, "EMPTY", 0, 0.0

        if offset < expected_total:
            xyz = xyz[:offset]
            iid = iid[:offset]
            px  = px[:offset]
            rgb = rgb[:offset]
            pid = pid[:offset]

        # Pack separate arrays into structured array
        print(f"  Mat {material_id:>3s}: packing {offset:,} obs ...", end=" ", flush=True)
        t0 = time.time()
        out = np.empty(offset, dtype=OBS_DTYPE)
        out['xyz'] = xyz
        out['image_id'] = iid
        out['pixel_coords'] = px
        out['rgb'] = rgb
        out['point_id'] = pid
        del xyz, iid, px, rgb, pid
        dt_pack = time.time() - t0

        # Save single file
        print(f"saving ...", end=" ", flush=True)
        t0 = time.time()
        np.save(output_path, out)
        dt_save = time.time() - t0
        del out

        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"done (pack {dt_pack:.1f}s + save {dt_save:.1f}s = {file_size_mb:.0f} MB)", flush=True)
        return material_id, "OK", offset, file_size_mb

    except Exception as e:
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        return material_id, f"ERROR: {type(e).__name__}: {e}", 0, 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Merge observation chunks into single all_observations.npy",
    )
    parser.add_argument("dataset_path", help="Path to Dataset_Nov11 root folder")
    parser.add_argument(
        "--materials",
        type=int,
        nargs="+",
        default=None,
        help="Specific material IDs to process (default: all)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: path does not exist: {dataset_path}")
        return 1

    if args.materials is not None:
        material_folders = []
        for mid in args.materials:
            folder = dataset_path / str(mid)
            if folder.is_dir() and (folder / "observations").exists():
                material_folders.append(str(folder))
            else:
                print(f"Warning: material {mid} not found or no observations, skipping")
    else:
        material_folders = []
        for item in sorted(
            dataset_path.iterdir(),
            key=lambda x: int(x.name) if x.name.isdigit() else -1,
        ):
            if item.is_dir() and item.name.isdigit():
                if (item / "observations").exists():
                    material_folders.append(str(item))

    print(f"{'='*60}")
    print(f"Observation Chunk Merger")
    print(f"{'='*60}")
    print(f"Dataset:   {dataset_path}")
    print(f"Materials: {len(material_folders)}")
    print(f"Output:    <material>/all_observations.npy")
    print(f"Dtype:     xyz=f32(3), image_id=u16, pixel=f32(2), rgb=u16(3), point_id=i32")
    print(f"Per row:   32 bytes (vs 80 bytes original = 60% savings)")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    results = []
    start_time = time.time()

    for folder in tqdm(material_folders, desc="Materials", unit="mat"):
        result = process_material(folder)
        results.append(result)
        mid, status, nobs, size_mb = result
        if status.startswith("ERROR"):
            tqdm.write(f"  Material {mid:>3s}: {status}")

    elapsed = time.time() - start_time

    ok_count = sum(1 for _, s, _, _ in results if s == "OK")
    err_count = sum(1 for _, s, _, _ in results if s.startswith("ERROR"))
    total_obs = sum(n for _, s, n, _ in results if s == "OK")
    total_size = sum(sz for _, s, _, sz in results if s == "OK")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Processed:    {ok_count} / {len(results)}")
    print(f"Errors:       {err_count}")
    print(f"Total obs:    {total_obs:,}")
    print(f"Total output: {total_size / 1024:.2f} GB")
    print(f"Time:         {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"{'='*60}")

    if err_count > 0:
        print(f"\nFailed materials:")
        for mid, status, _, _ in sorted(
            results,
            key=lambda x: int(x[0]) if x[0].isdigit() else -1,
        ):
            if status.startswith("ERROR"):
                print(f"  {mid}: {status}")

    return 0


if __name__ == "__main__":
    exit(main() or 0)
