"""
Unit test: verify that cropping HDR images by the projected sample mask preserves
Stage-2-from-Real training equivalence.

Equivalence criteria
--------------------
The Stage-2 trainer:
  1. reads HDR PNGs at full HxW (utils/dataset/real.py, lines ~496-510),
  2. applies CCM and a luminance > 1e-7 filter,
  3. forwards rays through the renderer, which sets `vis=True` only for rays that
     hit the sample geometry,
  4. computes loss as `(rgbs - rgbs_gt)[vis].mean()`.

So, for the cropped HDR to produce identical training results we need:
  (A) Bit-identical pixel values inside the projected polygon mask + buffer
      (these are the only pixels whose rays can reach the sample).
  (B) Pixels we drop (outside the polygon mask) would have had `vis=False` anyway.

(A) is checked exactly by comparing pixels inside the rasterised polygon mask.
(B) is checked indirectly: every dropped pixel that *was* bright in the original
     must lie outside the projection of the sample rectangle + buffer. We
     additionally report what fraction of dropped pixels were bright so the user
     can tighten/loosen the buffer if needed.

Usage:
    python test_crop_equivalence.py \\
        --orig /media/raid/cloth/capture_data/Dataset_Nov11/227 \\
        --cropped /media/raid/cloth/Dataset_submission/227 \\
        --num_samples 12

Exits with non-zero if any in-mask pixel differs (criterion A).
"""

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np


def luminance(img_uint16, ccm):
    """Match real.py: img = (uint16 -> float) @ ccm; clip(0); luminance from RGB."""
    f = img_uint16.astype(np.float32)
    f = f @ ccm
    f = np.clip(f, 0, None)
    lum = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
    return lum, f


def load_default_ccm():
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "data", "base.yaml",
    )
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return np.array(cfg["ccm"], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orig", required=True, help="Original material dir (Dataset_Nov11/<id>)")
    ap.add_argument("--cropped", required=True, help="Cropped material dir (Dataset_submission/<id>)")
    ap.add_argument("--num_samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lum_threshold", type=float, default=1e-7,
                    help="Same threshold real.py uses to drop low-luminance rays.")
    args = ap.parse_args()

    bbox_json = os.path.join(args.cropped, "hdr_crop_bboxes.json")
    if not os.path.exists(bbox_json):
        print(f"FATAL: missing {bbox_json}. Run crop_hdr_by_mask.py first.")
        sys.exit(2)
    with open(bbox_json) as f:
        bbox_meta = json.load(f)
    polygons = bbox_meta.get("polygons", {})
    bboxes = bbox_meta.get("bboxes", {})
    dilate_px = int(bbox_meta.get("dilate_px", 0))
    if not polygons:
        print("FATAL: hdr_crop_bboxes.json has no 'polygons'. Re-run crop_hdr_by_mask.py.")
        sys.exit(2)

    with open(os.path.join(args.orig, "scan_log.json")) as f:
        scan_log = json.load(f)
    ccm = load_default_ccm()

    rng = random.Random(args.seed)
    valid_ids = [str(i) for i in range(len(scan_log))
                 if str(i) in polygons and polygons[str(i)] is not None]
    if not valid_ids:
        print("FATAL: no usable polygons recorded in hdr_crop_bboxes.json")
        sys.exit(2)
    sample_ids = rng.sample(valid_ids, min(args.num_samples, len(valid_ids)))

    fails_A = 0
    total_dropped_bright = 0
    summary_rows = []

    for sid in sample_ids:
        idx = int(sid)
        scan = scan_log[idx]
        fname = scan["filename"]
        orig_path = os.path.join(args.orig, "hdr", fname)
        crop_path = os.path.join(args.cropped, "hdr", fname)
        if not (os.path.exists(orig_path) and os.path.exists(crop_path)):
            print(f"  [skip] missing files for scan {idx} ({fname})")
            continue

        orig = cv2.imread(orig_path, cv2.IMREAD_UNCHANGED)
        crop = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)
        assert orig.shape == crop.shape, f"shape mismatch on {fname}: {orig.shape} vs {crop.shape}"
        H, W = orig.shape[:2]

        # Reconstruct the polygon mask the cropper used.
        poly = np.round(np.array(polygons[sid])).astype(np.int32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly, 1)
        if dilate_px > 0:
            k = 2 * dilate_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel)
        mask_bool = mask.astype(bool)

        # ---- Criterion (A): inside-mask pixels are bit-identical -----------
        diff_pix = (orig != crop).any(axis=-1) & mask_bool
        ndiff = int(diff_pix.sum())
        if ndiff:
            fails_A += 1
            print(f"  [FAIL-A] scan {idx} ({fname}): {ndiff} pixels differ inside polygon mask")

        # ---- Criterion (B): dropped bright pixels (outside mask) -----------
        outside = ~mask_bool
        lum_orig, _ = luminance(orig, ccm)
        dropped_bright = outside & (lum_orig > args.lum_threshold)
        n_dropped_bright = int(dropped_bright.sum())
        total_dropped_bright += n_dropped_bright

        summary_rows.append({
            "scan": idx, "fname": fname,
            "bbox": bboxes.get(sid),
            "kept_pixels": int(mask_bool.sum()),
            "outside_pixels": int(outside.sum()),
            "outside_bright_pixels": n_dropped_bright,
            "outside_bright_ratio": n_dropped_bright / max(int(outside.sum()), 1),
        })

    print("\n=== Per-scan summary ===")
    print(f"{'scan':>6} {'kept':>10} {'out_total':>11} {'out_bright':>11} {'bright_ratio':>13}")
    for r in summary_rows:
        print(f"{r['scan']:>6d} {r['kept_pixels']:>10d} {r['outside_pixels']:>11d} "
              f"{r['outside_bright_pixels']:>11d} {r['outside_bright_ratio']:>13.4%}")

    print("\n=== Verdict ===")
    if fails_A == 0:
        print(f"  Criterion A (inside polygon mask bit-identical): PASS on all {len(summary_rows)} samples")
    else:
        print(f"  Criterion A FAILED on {fails_A}/{len(summary_rows)} samples")

    print(f"  Total dropped *bright* pixels across samples: {total_dropped_bright}")
    print("  These are pixels whose rays *might* still hit the sample. Run a short")
    print("  Stage-2 training in --debug mode on both datasets to confirm the")
    print("  vis-masked loss is unchanged. If the ratio above is large, increase")
    print("  --dilate_px in crop_hdr_by_mask.py.")

    sys.exit(1 if fails_A > 0 else 0)


if __name__ == "__main__":
    main()
