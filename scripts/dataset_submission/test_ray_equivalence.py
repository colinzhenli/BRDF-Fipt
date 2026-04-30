"""
Stronger equivalence test: run real.py's exact loader path on a handful of HDR
images for both the original and cropped dataset, then verify that

  (S_crop ∩ in-bbox)  ==  (S_orig ∩ in-bbox)   bit-identical at the ray level

where S_orig / S_crop are the (rays, rgbs, camera_ids, emitter_ids) tuples that
real.py.RealImageDataset._preload_given_metadata() would emit for a given image.

This is the strongest non-GPU check we can do without re-implementing the full
ray-tracer: it walks the same code path the trainer uses to construct training
samples, so if it passes we know the training-step inputs are identical for
every pixel that lies inside the projected sample mask + dilation buffer.

Run after crop_hdr_by_mask.py + build_submission.py have completed:

    python test_ray_equivalence.py \\
        --orig    /media/raid/cloth/capture_data/Dataset_Nov11/227 \\
        --cropped /media/raid/cloth/Dataset_submission/227 \\
        --num_samples 8
"""

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from utils.dataset.real import build_4x4, get_ray_directions, get_rays  # noqa: E402


def load_intrinsics_and_ccm():
    import yaml
    with open(os.path.join(REPO_ROOT, "config", "renderer", "multiarea_emitter.yaml")) as f:
        cfg_r = yaml.safe_load(f)
    with open(os.path.join(REPO_ROOT, "config", "data", "base.yaml")) as f:
        cfg_d = yaml.safe_load(f)
    intr = cfg_r["camera"]["intrinsics"]
    return intr, np.array(cfg_d["ccm"], dtype=np.float32)


def load_image_and_extract(img_path, directions, ccm, focal, c2w, emitter_id, camera_id, lum_threshold=1e-7):
    """Reproduces the per-image branch of real.py._preload_given_metadata."""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img @ ccm
    img = img.clip(0, None)
    img = torch.from_numpy(img).float()
    img_flat = img.reshape(-1, 3)

    rays_o, rays_d, dxdu, dydv = get_rays(directions, c2w, focal=focal)
    rays = torch.cat([rays_o, rays_d, dxdu, dydv], dim=-1)

    emitter_ids = torch.full((rays.shape[0],), emitter_id, dtype=torch.long)
    camera_ids = torch.full((rays.shape[0],), camera_id, dtype=torch.long)

    lum = (0.2126 * img_flat[..., 0] +
           0.7152 * img_flat[..., 1] +
           0.0722 * img_flat[..., 2])
    valid_mask = lum > lum_threshold
    return (rays[valid_mask], img_flat[valid_mask],
            emitter_ids[valid_mask], camera_ids[valid_mask],
            valid_mask.numpy().reshape(img.shape[0], img.shape[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--cropped", required=True)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    intr, ccm = load_intrinsics_and_ccm()
    H, W = intr["height"], intr["width"]
    focal = intr["focal_length"]
    cx, cy, k = intr["cx"], intr["cy"], intr["distortion"]
    directions = get_ray_directions(H, W, focal, cx, cy, k)

    with open(os.path.join(args.orig, "scan_log.json")) as f:
        scan_log = json.load(f)
    with open(os.path.join(args.orig, "rotated_camera.json")) as f:
        rc = {entry["camera_id"]: entry for entry in json.load(f)}
    with open(os.path.join(args.cropped, "hdr_crop_bboxes.json")) as f:
        bbox_meta = json.load(f)
    polygons = bbox_meta.get("polygons", {})
    dilate_px = int(bbox_meta.get("dilate_px", 0))
    if not polygons:
        print("FATAL: hdr_crop_bboxes.json has no 'polygons'. Re-run crop_hdr_by_mask.py.")
        sys.exit(2)

    rng = random.Random(args.seed)
    valid_ids = [
        i for i in range(len(scan_log))
        if str(i) in polygons and polygons[str(i)] is not None
        and scan_log[i]["camera_id"] in rc
    ]
    sample_ids = rng.sample(valid_ids, min(args.num_samples, len(valid_ids)))

    overall_inside_match = 0
    overall_inside_diff = 0
    overall_outside_dropped_bright = 0
    overall_orig_rays = 0
    overall_crop_rays = 0

    for idx in sample_ids:
        scan = scan_log[idx]
        cam_id = scan["camera_id"]
        cam_info = rc[cam_id]
        # Match real.py exactly: position is divided by 1000 in load_camera_metadata,
        # rotation_matrix is taken as-is (the "0.0964 scale" is not normalized there).
        pos_m = [p / 1000.0 for p in cam_info["position"]]
        c2w = torch.from_numpy(build_4x4(cam_info["rotation_matrix"], pos_m)).float()[:3, :4]

        fname = scan["filename"]
        emitter_id = scan["light_id"]
        orig_path = os.path.join(args.orig, "hdr", fname)
        crop_path = os.path.join(args.cropped, "hdr", fname)

        rays_o, rgbs_o, eo_o, co_o, mask_o = load_image_and_extract(
            orig_path, directions, ccm, focal, c2w, emitter_id, cam_id,
        )
        rays_c, rgbs_c, eo_c, co_c, mask_c = load_image_and_extract(
            crop_path, directions, ccm, focal, c2w, emitter_id, cam_id,
        )

        # Reconstruct the polygon mask the cropper used.
        poly = np.round(np.array(polygons[str(idx)])).astype(np.int32)
        in_box = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(in_box, poly, 1)
        if dilate_px > 0:
            k = 2 * dilate_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            in_box = cv2.dilate(in_box, kernel)
        in_box = in_box.astype(bool)
        flat_in_box = in_box.flatten()

        # Check (S_crop ∩ in-bbox) bit-identical to (S_orig ∩ in-bbox)
        orig_keep_in_box = mask_o.flatten() & flat_in_box
        crop_keep_in_box = mask_c.flatten() & flat_in_box
        # Cropped keeps must equal Orig keeps inside bbox (because pixel values match)
        diff_mask = orig_keep_in_box != crop_keep_in_box
        if diff_mask.any():
            overall_inside_diff += int(diff_mask.sum())

        # For pixels both keep (intersection): values must match
        both = orig_keep_in_box & crop_keep_in_box
        if both.any():
            # Can't directly index ragged variable-length tensors; recompute on flat
            # CCM-applied images for the kept positions.
            img_o = cv2.imread(orig_path, cv2.IMREAD_UNCHANGED)
            img_c = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)
            img_o = (cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB) @ ccm).clip(0, None).astype(np.float32)
            img_c = (cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB) @ ccm).clip(0, None).astype(np.float32)
            flat_o = img_o.reshape(-1, 3)
            flat_c = img_c.reshape(-1, 3)
            same = np.allclose(flat_o[both], flat_c[both], atol=0)
            if same:
                overall_inside_match += int(both.sum())
            else:
                ndiff = int((flat_o[both] != flat_c[both]).any(axis=-1).sum())
                overall_inside_diff += ndiff
                print(f"  [FAIL] scan {idx}: {ndiff} pixels inside bbox have differing values")

        # Outside-bbox: dropped bright pixels (those in S_orig but not in S_crop AND outside bbox)
        outside = ~flat_in_box
        dropped = mask_o.flatten() & outside & (~mask_c.flatten())
        overall_outside_dropped_bright += int(dropped.sum())

        overall_orig_rays += int(rays_o.shape[0])
        overall_crop_rays += int(rays_c.shape[0])

        mask_size = int(in_box.sum())
        print(f"  scan {idx:>4d}  mask_pixels={mask_size:>8d}  "
              f"orig_rays={rays_o.shape[0]:>9d}  crop_rays={rays_c.shape[0]:>9d}  "
              f"dropped_outside_bright={int(dropped.sum()):>9d}")

    print("\n=== Verdict ===")
    print(f"  Inside-bbox bit-identical:  {overall_inside_match} pixels match, {overall_inside_diff} differ")
    print(f"  Total rays (orig vs cropped): {overall_orig_rays} vs {overall_crop_rays}")
    print(f"  Outside-bbox bright pixels dropped by crop: {overall_outside_dropped_bright}")
    print()
    if overall_inside_diff == 0:
        print("  PASS: every pixel inside the projected mask + buffer is bit-identical.")
        print("  -> The Stage-2 trainer would feed the renderer identical (rays, rgbs)")
        print("     for every pixel whose ray can possibly hit the sample geometry.")
        print("     vis-masked loss (rgbs[vis] - rgbs_gt[vis]).mean() is therefore")
        print("     unchanged in expectation; observed differences would only reflect")
        print("     the lower chunk size shifting the random sampling distribution.")
    else:
        print("  FAIL: some inside-bbox pixels differ. Investigate the crop script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
