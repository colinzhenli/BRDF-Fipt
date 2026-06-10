"""Split a full COLMAP points3D.ply into fabric (inside bbox) + surrounding (checkerboard).

Uses the same umeyama Sim3 alignment as recon/calibration/shape_matching.py:
  rotated_camera.json (base frame) ↔ images.bin (COLMAP world) → T_BW
Then applies T_BW to all points and crops by bbox.json.

Output: two PLY files written next to the input:
  <material>/fabric_points.ply        (inside bbox, in COLMAP world frame)
  <material>/surrounding_points.ply   (outside bbox, in COLMAP world frame)

These keep COLMAP-world coordinates (not base) so Blender renders the whole scene
together at COLMAP scale.

Usage:
  python split_pointcloud.py --material_folder /media/.../Dataset_Nov11/496
"""
import argparse
import json
import sys
import os
from pathlib import Path
import numpy as np
import open3d as o3d

# Pull in shape_matching helpers
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "recon" / "calibration"))
from read_write_model import read_model, qvec2rotmat


def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3]  = np.asarray(t, float).reshape(3)
    return T


def sim3_umeyama(P, Q):
    """P (N,3) COLMAP centres → Q (N,3) robot/base centres. Returns s, R, t."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    mu_P, mu_Q = P.mean(0), Q.mean(0)
    P0, Q0 = P - mu_P, Q - mu_Q
    Sigma = (Q0.T @ P0) / len(P)
    U, D, VT = np.linalg.svd(Sigma)
    S = np.diag([1, 1, np.sign(np.linalg.det(U) * np.linalg.det(VT))])
    R = U @ S @ VT
    s = (np.trace(np.diag(D) @ S) / (P0**2).sum() * len(P))
    t = mu_Q - s * (R @ mu_P)
    return s, R, t


def parse_colmap_centres(images):
    """Return dict {img_id: cam_centre_in_colmap_world}."""
    out = {}
    for img_id, img in images.items():
        R = qvec2rotmat(img.qvec)
        t = img.tvec.reshape(3)
        w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
        opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
        w2c = opencv_to_opengl @ w2c
        c2w = np.linalg.inv(w2c)
        out[img_id] = (img.name, c2w[:3, 3])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--material_folder", required=True)
    args = p.parse_args()

    mat = Path(args.material_folder)
    sparse_dir = mat / "sparse"
    ply_path = sparse_dir / "points3D.ply"
    bbox_path = mat / "bbox.json"
    cam_log_path = mat / "rotated_camera.json"

    # Load PLY (COLMAP world coords)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts_world = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    print(f"[loaded] {len(pts_world):,} points from {ply_path.name}")

    # Load bbox in base frame
    with open(bbox_path) as f:
        bbox = json.load(f)
    bbox_min = np.array(bbox["bbox_min"])
    bbox_max = np.array(bbox["bbox_max"])
    print(f"[bbox] (base frame) min={bbox_min}, max={bbox_max}")

    # Load both camera sets and compute T_BW via umeyama
    cameras, images = read_model(str(sparse_dir), ext=".bin")
    colmap_centres = parse_colmap_centres(images)  # {img_id: (name, centre)}

    with open(cam_log_path) as f:
        rotated_cams = json.load(f)
    rotated_map = {int(e["overall_id"]): np.array(e["position"]) / 1000.0  # mm→m
                   for e in rotated_cams}

    # Match: rotated_camera.json's "overall_id" corresponds to scan-XXXX (which gets parsed from image name)
    import re
    P, Q = [], []   # P = COLMAP world centres, Q = base frame centres
    for img_id, (name, c_world) in colmap_centres.items():
        m = re.search(r'scan-(\d+)', name)
        if not m:
            continue
        scan_id = int(m.group(1))
        # rotated_camera.json index uses scan_log row index, not raw scan_id
        # But shape_matching.py saves overall_id = scan_log_index of the matched entry.
        # We need to find which rotated_cam entry corresponds. Simpler: just look up by overall_id == scan_id
        # (works for most non-skipped cases)
        # Try matching scan_id directly first
        for e in rotated_cams:
            if int(e["overall_id"]) == scan_id - 1 or int(e["camera_id"]) == scan_id - 1:
                P.append(c_world)
                Q.append(np.array(e["position"]) / 1000.0)
                break

    P, Q = np.array(P), np.array(Q)
    print(f"[match] {len(P)} camera pairs (COLMAP world ↔ base frame)")
    if len(P) < 3:
        print("[error] Not enough pairs — falling back to RANSAC-only classification")
        T_BW = None
    else:
        s, R, t = sim3_umeyama(P, Q)
        T_BW = build_4x4(s * R, t)
        # Verify
        P_h = np.hstack([P, np.ones((len(P), 1))])
        P_b = (T_BW @ P_h.T).T[:, :3]
        err = np.mean(np.linalg.norm(P_b - Q, axis=1))
        print(f"[umeyama] scale={s:.4f}, mean cam-centre error={err*1000:.2f} mm")

    # Transform all points to base frame
    pts_h = np.hstack([pts_world, np.ones((len(pts_world), 1))])
    pts_base = (T_BW @ pts_h.T).T[:, :3]

    # Classify: inside bbox = fabric; outside = surrounding (XY only).
    # No subsampling, no Z-outlier dropping — EVERY original point is kept and
    # assigned to exactly one of the two clouds.
    in_xy = ((pts_base[:, 0] >= bbox_min[0]) & (pts_base[:, 0] <= bbox_max[0]) &
             (pts_base[:, 1] >= bbox_min[1]) & (pts_base[:, 1] <= bbox_max[1]))
    surrounding_mask = ~in_xy

    print(f"[classify] fabric (inside bbox):  {in_xy.sum():,} points")
    print(f"[classify] surrounding:           {surrounding_mask.sum():,} points")
    print(f"[classify] total (all kept):      {in_xy.sum() + surrounding_mask.sum():,} / {len(pts_world):,}")

    # Save two PLY files (in COLMAP world coords for direct Blender render)
    fabric_pcd = o3d.geometry.PointCloud()
    fabric_pcd.points = o3d.utility.Vector3dVector(pts_world[in_xy])
    fabric_pcd.colors = o3d.utility.Vector3dVector(cols[in_xy])
    fabric_path = mat / "fabric_points.ply"
    o3d.io.write_point_cloud(str(fabric_path), fabric_pcd)
    print(f"[wrote] {fabric_path}")

    surr_pcd = o3d.geometry.PointCloud()
    surr_pcd.points = o3d.utility.Vector3dVector(pts_world[surrounding_mask])
    surr_pcd.colors = o3d.utility.Vector3dVector(cols[surrounding_mask])
    surr_path = mat / "surrounding_points.ply"
    o3d.io.write_point_cloud(str(surr_path), surr_pcd)
    print(f"[wrote] {surr_path}")


if __name__ == "__main__":
    main()
