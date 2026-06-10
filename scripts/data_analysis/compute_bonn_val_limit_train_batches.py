#!/usr/bin/env python
"""Compute the one-epoch `limit_train_batches` for every Bonn_val material.

For a Stage-2 single-material overfit run (BonnSingleMaterialDataset), the
training ray pool is:

    total_train_obs = n_pixels * n_train_images

where
    n_pixels       = H * W   (all pixels, no subsampling in the full loader)
    n_images       = n_poly + n_pan + n_lls   (use_pan=use_lls=True)
    n_val          = max(1, int(n_images * VAL_VIEW_RATIO))   # 80/20 split
    n_train_images = n_images - n_val

`limit_train_batches` for ONE full pass over the training observations is then
    ceil(total_train_obs / RAYS_NUM)

Only EXR *headers* are read (channel names + data window) — no pixel data —
so this is fast and light on NFS. The poly/pan/lls counting mirrors
utils/dataset/bonn.py exactly (pan keeps il<=24; poly = 3 channels/image).
"""
import os
import re
import csv
import math
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyexr

# ---- config (matches scripts/jobs/run_stage2_bonn_from_Real.sh) -----------
ROOT = "/media/raid/cloth/Bonn_val"
RAYS_NUM = 500_000
VAL_VIEW_RATIO = 0.2        # data.val_view_ratio
USE_PAN = True              # bonn.yaml default (not overridden by the script)
USE_LLS = True
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bonn_val_limit_train_batches.csv")

# ---- channel parsers copied verbatim from utils/dataset/bonn.py -----------
_POLY = re.compile(r"poly_(cv\d+)_(il\d+)_(rot\d+)_[BGR]")
_PAN = re.compile(r"pan_(cv\d+)_(il\d+)_(rot\d+)")
_LLS = re.compile(r"lls_(cv\d+)_lls\d+_la([+-]?\d+\.?\d*)_(rot\d+)")


def n_poly_images(ch):
    n = 0
    for i in range(0, len(ch), 3):
        if i + 2 >= len(ch):
            break
        if _POLY.match(ch[i]):
            n += 1
    return n


def n_pan_images(ch):
    # pan keeps only il001..il024
    return sum(1 for c in ch
               if _PAN.match(c) and int(_PAN.match(c).group(2)[2:]) <= 24)


def n_lls_images(ch):
    return sum(1 for c in ch if _LLS.match(c))


def channels(path):
    """Header-only channel-name list (pyexr.open does not read pixels)."""
    return pyexr.open(path).channel_map["all"]


def process(mat_id):
    prefix = os.path.join(ROOT, f"mat{mat_id:04d}")
    fp = pyexr.open(f"{prefix}_poly.exr")           # header only
    H, W = fp.height, fp.width
    n_poly = n_poly_images(fp.channel_map["all"])
    n_pan = n_pan_images(channels(f"{prefix}_pan.exr")) if USE_PAN else 0
    n_lls = n_lls_images(channels(f"{prefix}_lls.exr")) if USE_LLS else 0

    n_images = n_poly + n_pan + n_lls
    n_val = max(1, int(n_images * VAL_VIEW_RATIO))
    n_train = n_images - n_val
    n_pixels = H * W
    total_train_obs = n_pixels * n_train
    limit = math.ceil(total_train_obs / RAYS_NUM)

    return dict(mat_id=mat_id, H=H, W=W, n_pixels=n_pixels,
                n_poly=n_poly, n_pan=n_pan, n_lls=n_lls, n_images=n_images,
                n_val=n_val, n_train_imgs=n_train,
                total_train_obs=total_train_obs, limit_train_batches=limit)


def main():
    poly_files = sorted(glob.glob(os.path.join(ROOT, "mat*_poly.exr")))
    mat_ids = [int(os.path.basename(p).split("_")[0][3:]) for p in poly_files]
    print(f"Discovered {len(mat_ids)} materials in {ROOT}")
    print(f"Config: rays_num={RAYS_NUM:,}  val_view_ratio={VAL_VIEW_RATIO}  "
          f"use_pan={USE_PAN}  use_lls={USE_LLS}\n")

    rows, errors = {}, {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut = {ex.submit(process, m): m for m in mat_ids}
        for f in as_completed(fut):
            m = fut[f]
            try:
                r = rows[m] = f.result()
                print(f"  mat{m:04d}: {r['n_pixels']:>9,}px x "
                      f"{r['n_train_imgs']:>3} train img "
                      f"(poly={r['n_poly']},pan={r['n_pan']},lls={r['n_lls']}) "
                      f"-> {r['total_train_obs']:>14,} obs  limit={r['limit_train_batches']:>5}")
            except Exception as e:
                errors[m] = str(e)
                print(f"  mat{m:04d}: FAILED {e}")

    ordered = [rows[m] for m in mat_ids if m in rows]
    fields = ["mat_id", "H", "W", "n_pixels", "n_poly", "n_pan", "n_lls",
              "n_images", "n_val", "n_train_imgs", "total_train_obs",
              "limit_train_batches"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ordered)

    if ordered:
        limits = [r["limit_train_batches"] for r in ordered]
        s = sorted(limits)
        median = s[len(s) // 2]
        print(f"\n{'='*70}")
        print(f"Saved {len(ordered)} materials -> {OUT_CSV}")
        if errors:
            print(f"Failed: {sorted(errors)}")
        print(f"limit_train_batches  min={min(limits)}  median={median}  "
              f"max={max(limits)}  (rays_num={RAYS_NUM:,})")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
