"""
Crop HDR images using projected sample mask + safety buffer.

For each HDR scan, project the sample point cloud (point_positions.npz, in canonical
"0-angle world" frame) into the image using the camera's rotated_camera.json pose
and the renderer's calibrated intrinsics, compute the axis-aligned 2D bounding box
of the in-image projected pixels, dilate by `--dilate_px` pixels for SPP edge
safety, and write a new PNG that keeps the original HxW dimensions but zeros every
pixel outside the dilated bbox.

Why this preserves Stage-2 training:
  - real.RealImageDataset filters rays by `luminance > 1e-7`, then renders them
    through the multi-area emitter and computes loss as `(rgbs - rgbs_gt)[vis]`,
    where `vis` comes from the renderer's intersection test against the sample
    surface. Pixels outside the sample's projected silhouette necessarily have
    `vis=False` (their rays do not hit the geometry), so zeroing them out drops
    rays that would have contributed nothing to the loss. Pixels inside the
    silhouette are kept bit-identical.

Usage:
    python crop_hdr_by_mask.py \\
        --src /media/raid/cloth/capture_data/Dataset_Nov11 \\
        --dst /media/raid/cloth/Dataset_submission \\
        --material 227 \\
        --dilate_px 8

The default --dilate_px=8 corresponds to roughly 0.6 mm at typical sample
distance (~0.5 m, focal 6721 px). The user spec asks for ~0.1 mm; 8 px is a
generous safety margin since the projected mask is the convex hull of sparse
COLMAP points and may underestimate the sample silhouette by a few pixels.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np


def load_intrinsics(intrinsics_json_path):
    """Load camera intrinsics used by the renderer.

    Expected schema (matches renderer.camera.intrinsics in the Hydra config):
        {"width": int, "height": int, "focal_length": float,
         "cx": float, "cy": float, "distortion": float}
    """
    with open(intrinsics_json_path) as f:
        d = json.load(f)
    return d


def normalize_rotation(R):
    """rotated_camera.json stores c2w rotation pre-multiplied by a uniform scale s
    (~0.0964 in this dataset). Recover the orthonormal rotation by dividing by
    the mean column norm."""
    R = np.asarray(R, dtype=np.float64)
    s = float(np.mean(np.linalg.norm(R, axis=0)))
    return R / s, s


def project_points_opengl(points_world, R_c2w_norm, t_c2w_m,
                          focal, cx, cy, distortion, W, H,
                          newton_iters=8):
    """Project (N, 3) world-frame points through an OpenGL c2w camera.

    Returns
    -------
    pix_i, pix_j : (N,) float64
        Pixel column / row coordinates (NaN where the point is behind the camera).
    in_image_mask : (N,) bool
        True for points that project inside [0, W) x [0, H).
    """
    points_world = np.asarray(points_world, dtype=np.float64)
    P_cam = (R_c2w_norm.T @ (points_world - t_c2w_m).T).T  # (N, 3)

    # OpenGL convention: forward is -z; in front of camera means z < 0.
    in_front = P_cam[:, 2] < 0
    pix_i = np.full(P_cam.shape[0], np.nan)
    pix_j = np.full(P_cam.shape[0], np.nan)

    if not in_front.any():
        return pix_i, pix_j, np.zeros(P_cam.shape[0], dtype=bool)

    Pf = P_cam[in_front]
    # Match real.py.get_ray_directions sign convention:
    #   direction = [x_corr, -y_corr, -1] with x_corr = -P.x/P.z, y_corr = P.y/P.z
    x_corr = -Pf[:, 0] / Pf[:, 2]
    y_corr =  Pf[:, 1] / Pf[:, 2]

    # Invert radial distortion (single-coefficient SIMPLE_RADIAL):
    #   x_corr = x_norm * (1 + k * (x_norm^2 + y_norm^2))
    r_corr = np.sqrt(x_corr * x_corr + y_corr * y_corr)
    r = r_corr.copy()
    for _ in range(newton_iters):
        r = r_corr / (1.0 + distortion * r * r)
    factor = 1.0 + distortion * r * r
    x_norm = x_corr / factor
    y_norm = y_corr / factor

    pix_i_f = x_norm * focal + cx
    pix_j_f = y_norm * focal + cy
    pix_i[in_front] = pix_i_f
    pix_j[in_front] = pix_j_f

    in_image = np.zeros(P_cam.shape[0], dtype=bool)
    in_image[in_front] = (
        (pix_i_f >= 0) & (pix_i_f < W) &
        (pix_j_f >= 0) & (pix_j_f < H)
    )
    return pix_i, pix_j, in_image


def project_rectangle_corners(bbox_center, bbox_size, R_norm, t_m, intr):
    """Project the 4 corners of the sample rectangle (the same one the renderer
    intersects against; see utils.path_tracing.ray_rectangle_intersect_TBN) into
    image space. The rectangle is axis-aligned in the canonical "0-angle world"
    frame: width along x, length along y, lying on z = center_z.

    Returns a (4, 2) float array of pixel coordinates (col, row) and a (4,) bool
    `in_front` mask. Caller should skip the image when any corner lies behind
    the camera (the projection equation is undefined there).
    """
    cx_, cy_, cz_ = bbox_center
    w, l = bbox_size[0], bbox_size[1]
    hw, hl = w / 2.0, l / 2.0
    # CCW order in xy when viewed from +z. The renderer uses normal=+z, so this
    # matches the surface that vis=True tests against.
    corners = np.array([
        [cx_ - hw, cy_ - hl, cz_],
        [cx_ + hw, cy_ - hl, cz_],
        [cx_ + hw, cy_ + hl, cz_],
        [cx_ - hw, cy_ + hl, cz_],
    ], dtype=np.float64)
    pix_i, pix_j, in_image = project_points_opengl(
        corners, R_norm, t_m,
        intr["focal_length"], intr["cx"], intr["cy"], intr["distortion"],
        intr["width"], intr["height"],
    )
    # in_image only checks bounds; "in_front" is needed for projection validity.
    P_cam = (R_norm.T @ (corners - t_m).T).T
    in_front = P_cam[:, 2] < 0
    poly = np.stack([pix_i, pix_j], axis=-1)
    return poly, in_front


def build_polygon_mask(poly_xy, H, W, dilate_px):
    """Rasterize the projected rectangle as a binary mask, dilated by
    `dilate_px` pixels for SPP edge safety. cv2.fillConvexPoly handles the
    perspective-induced quadrilateral (still convex)."""
    pts = np.round(poly_xy).astype(np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 1)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


def mask_image_bbox_from_mask(mask):
    """Tight axis-aligned bbox of a True region in `mask`, used only for
    bookkeeping (kept_pixels stat, json record). Returns None if mask is empty."""
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))


def crop_hdr_image_with_mask(src_path, dst_path, mask, png_compression=3):
    """Read HDR PNG, zero pixels where `mask` is False, write to dst_path."""
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read {src_path}")
    H, W = img.shape[:2]
    out = np.zeros_like(img)
    if mask is not None and mask.any():
        # broadcast (H,W) bool -> (H,W,3)
        out[mask] = img[mask]

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    ok = cv2.imwrite(dst_path, out, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
    if not ok:
        raise RuntimeError(f"Failed to write {dst_path}")

    orig_bytes = os.path.getsize(src_path)
    new_bytes = os.path.getsize(dst_path)
    kept = int(mask.sum()) if mask is not None else 0
    return orig_bytes, new_bytes, kept, H * W


def _process_one(args):
    (idx, src_path, dst_path, bbox_center, bbox_size, cam, intr,
     dilate_px, png_compression, overwrite) = args
    R_norm, _scale = normalize_rotation(cam["rotation_matrix"])
    t_m = np.array(cam["position"], dtype=np.float64) / 1000.0  # mm -> m
    poly_xy, in_front = project_rectangle_corners(bbox_center, bbox_size, R_norm, t_m, intr)
    H = intr["height"]
    W = intr["width"]
    if not in_front.all():
        # At least one corner is behind camera; projection is undefined.
        # Fall back to an empty mask (image becomes fully zeroed).
        mask = np.zeros((H, W), dtype=bool)
    else:
        mask = build_polygon_mask(poly_xy, H, W, dilate_px)
    bbox = mask_image_bbox_from_mask(mask)
    poly_record = poly_xy.tolist()

    # Resume support: if dst already has a non-empty PNG, skip the write but still
    # report stats so the bbox JSON is fully populated.
    if not overwrite and os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        orig = os.path.getsize(src_path)
        new = os.path.getsize(dst_path)
        kept = int(mask.sum())
        total = W * H
        return idx, bbox, poly_record, orig, new, kept, total
    orig, new, kept, total = crop_hdr_image_with_mask(src_path, dst_path, mask, png_compression)
    return idx, bbox, poly_record, orig, new, kept, total


def crop_material(src_root, dst_root, material_id, intrinsics, dilate_px,
                  workers=4, png_compression=3, limit=None,
                  hdr_subdir="hdr", json_name="hdr_crop_bboxes.json",
                  scan_indices=None, overwrite=False):
    mat_src = os.path.join(src_root, str(material_id))
    mat_dst = os.path.join(dst_root, str(material_id))
    hdr_src = os.path.join(mat_src, "hdr")
    hdr_dst = os.path.join(mat_dst, hdr_subdir)
    os.makedirs(hdr_dst, exist_ok=True)

    with open(os.path.join(mat_src, "scan_log.json")) as f:
        scan_log = json.load(f)
    with open(os.path.join(mat_src, "rotated_camera.json")) as f:
        rc_list = json.load(f)
    rc_by_cam = {entry["camera_id"]: entry for entry in rc_list}

    unmatched_path = os.path.join(mat_src, "unmatched_scan_ids.json")
    if os.path.exists(unmatched_path):
        with open(unmatched_path) as f:
            unmatched = set(json.load(f))
    else:
        unmatched = set()

    with open(os.path.join(mat_src, "bbox.json")) as f:
        bbox_data = json.load(f)
    bbox_center = bbox_data["bbox_center"]
    bbox_size = bbox_data["bbox_size"]

    # Build task list
    indices_filter = None if scan_indices is None else set(scan_indices)
    tasks = []
    skipped_no_camera = 0
    skipped_unmatched = 0
    skipped_missing_src = 0
    for idx, scan in enumerate(scan_log):
        if indices_filter is not None and idx not in indices_filter:
            continue
        if int(scan.get("id", -1)) in unmatched:
            skipped_unmatched += 1
            continue
        cam_id = scan["camera_id"]
        if cam_id not in rc_by_cam:
            skipped_no_camera += 1
            continue
        fname = scan["filename"]
        src_path = os.path.join(hdr_src, fname)
        if not os.path.exists(src_path):
            skipped_missing_src += 1
            continue
        dst_path = os.path.join(hdr_dst, fname)
        tasks.append((idx, src_path, dst_path, bbox_center, bbox_size,
                      rc_by_cam[cam_id], intrinsics, dilate_px, png_compression,
                      overwrite))
        if limit is not None and len(tasks) >= limit:
            break

    print(f"[mat {material_id}] scans={len(scan_log)} -> tasks={len(tasks)}  "
          f"(unmatched={skipped_unmatched}, no_camera={skipped_no_camera}, "
          f"missing_src={skipped_missing_src}, hdr_dst={hdr_dst})")

    total_orig = 0
    total_new = 0
    total_kept = 0
    total_pix = 0
    bboxes = {}
    polygons = {}

    def _accumulate(result):
        nonlocal total_orig, total_new, total_kept, total_pix
        idx, bbox, poly, orig, new, kept, total = result
        total_orig += orig
        total_new += new
        total_kept += kept
        total_pix += total
        bboxes[idx] = bbox
        polygons[idx] = poly

    if workers <= 1:
        for t in tasks:
            _accumulate(_process_one(t))
            if (len(bboxes) % 25) == 0:
                print(f"  [{len(bboxes)}/{len(tasks)}] orig={total_orig/1e6:.0f}MB new={total_new/1e6:.0f}MB ratio={total_new/max(total_orig,1):.2f}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_one, t) for t in tasks]
            for n_done, fut in enumerate(as_completed(futures), 1):
                _accumulate(fut.result())
                if (n_done % 50) == 0:
                    print(f"  [{n_done}/{len(tasks)}] orig={total_orig/1e6:.0f}MB new={total_new/1e6:.0f}MB ratio={total_new/max(total_orig,1):.2f}")

    bbox_path = os.path.join(mat_dst, json_name)
    bboxes_serial = {str(k): list(v) if v is not None else None for k, v in bboxes.items()}
    poly_serial = {str(k): v for k, v in polygons.items()}
    with open(bbox_path, "w") as f:
        json.dump({
            "intrinsics": intrinsics,
            "dilate_px": dilate_px,
            "bbox_center": bbox_center,
            "bbox_size": bbox_size,
            "bboxes": bboxes_serial,
            "polygons": poly_serial,
        }, f)

    print(f"[mat {material_id}] DONE: {len(tasks)} files, "
          f"{total_orig/1e9:.2f} GB -> {total_new/1e9:.2f} GB "
          f"({100.0 * total_new / max(total_orig, 1):.1f}% of original); "
          f"kept_pixels {total_kept/total_pix*100:.1f}% inside polygon mask")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="/media/raid/cloth/capture_data/Dataset_Nov11",
                   help="Source dataset root (contains <material>/hdr/...)")
    p.add_argument("--dst", default="/media/raid/cloth/Dataset_submission",
                   help="Destination submission root")
    p.add_argument("--material", required=True, type=str,
                   help="Material id (folder name under --src)")
    p.add_argument("--dilate_px", type=int, default=8,
                   help="Pixel dilation around projected sample bbox (default 8 ~ 0.6mm)")
    p.add_argument("--workers", type=int, default=4,
                   help="NFS-friendly default; >4 saturates the writeback pipeline.")
    p.add_argument("--limit", type=int, default=None,
                   help="If set, only crop first N images (for smoke testing)")
    p.add_argument("--png_compression", type=int, default=3,
                   help="PNG zlib level [0-9]; 3 is a good speed/size tradeoff")
    p.add_argument("--intrinsics_json", default=None,
                   help="Optional override for camera intrinsics. If unset, reads from "
                        "config/renderer/multiarea_emitter.yaml.")
    p.add_argument("--hdr_subdir", default="hdr",
                   help="Subfolder under <DST>/<material> to write cropped images to. "
                        "Use 'hdr_debug' to stage a visual review without overwriting.")
    p.add_argument("--json_name", default="hdr_crop_bboxes.json",
                   help="Name of the bbox+polygon JSON written next to the HDR folder.")
    p.add_argument("--scan_indices", type=str, default=None,
                   help="Comma-separated list of scan_log indices to crop "
                        "(e.g. '0,50,100,200'). Useful for the debug subset.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing PNGs in the destination instead of skipping.")
    return p.parse_args()


def load_intrinsics_from_config():
    """Read renderer.camera.intrinsics from the project's Hydra config."""
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "renderer", "multiarea_emitter.yaml",
    )
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    intr = cfg["camera"]["intrinsics"]
    return {
        "width": int(intr["width"]),
        "height": int(intr["height"]),
        "focal_length": float(intr["focal_length"]),
        "cx": float(intr["cx"]),
        "cy": float(intr["cy"]),
        "distortion": float(intr["distortion"]),
    }


def main():
    args = parse_args()
    if args.intrinsics_json:
        with open(args.intrinsics_json) as f:
            intrinsics = json.load(f)
    else:
        intrinsics = load_intrinsics_from_config()
    print(f"Using intrinsics: {intrinsics}")
    scan_indices = None
    if args.scan_indices:
        scan_indices = [int(s) for s in args.scan_indices.split(",") if s.strip()]
    crop_material(
        args.src, args.dst, args.material,
        intrinsics=intrinsics,
        dilate_px=args.dilate_px,
        workers=args.workers,
        png_compression=args.png_compression,
        limit=args.limit,
        hdr_subdir=args.hdr_subdir,
        json_name=args.json_name,
        scan_indices=scan_indices,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
