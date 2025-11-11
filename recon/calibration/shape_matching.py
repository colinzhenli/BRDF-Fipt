#!/usr/bin/env python3
import os
import re
import json
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

# COLMAP helpers (same ones you already use)
from read_write_model import read_model, qvec2rotmat, read_points3D_binary, read_points3D_text, detect_model_format

# -------------------- small linear-algebra helpers --------------------

def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3]  = np.asarray(t, float).reshape(3)
    return T

def sim3_umeyama(P, Q, with_scale=True):
    """
    P (N,3): COLMAP camera centres
    Q (N,3): robot (base) camera centres (or targets)
    Returns: s, R, t  such that  Q ≈ s R P + t
    """
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    mu_P, mu_Q = P.mean(0), Q.mean(0)
    P0, Q0 = P - mu_P, Q - mu_Q
    Sigma = (Q0.T @ P0) / len(P)
    U, D, VT = np.linalg.svd(Sigma)
    S = np.diag([1, 1, np.sign(np.linalg.det(U) * np.linalg.det(VT))])
    R = U @ S @ VT
    s = (np.trace(np.diag(D) @ S) / (P0**2).sum() * len(P)) if with_scale else 1.0
    t = mu_Q - s * (R @ mu_P)
    return s, R, t

def rodrigues_axis_angle(n: np.ndarray, degrees: float | np.ndarray) -> np.ndarray:
    """
    Right-handed CCW rotation about axis n by angle_rad (rad).
    Supports scalar or vector of angles → returns (..,3,3).
    """
    th = np.deg2rad(degrees)
    n = np.asarray(n, float)
    n = n / (np.linalg.norm(n) + 1e-15)
    nx, ny, nz = n
    K = np.array([[0.0, -nz,  ny],
                  [nz,  0.0, -nx],
                  [-ny,  nx,  0.0]], dtype=float)
    I = np.eye(3)
    angle = np.asarray(th, float)[..., None, None]
    Sa = np.sin(angle); Ca = np.cos(angle)
    return I + Sa * K + (1.0 - Ca) * (K @ K)

def T_about_point(R, p):
    """Homogeneous transform that rotates by R about world point p (3,)."""
    p = np.asarray(p, float).reshape(3)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = (np.eye(3) - R) @ p
    return T

# -------------------- your existing helpers (kept) --------------------

def parse_colmap_images_txt(images):
    """
    images (read_model output) → dict[name] = 4x4 c2w
    """
    cam_c2w_dict = {}
    for img in images.values():
        rotation = qvec2rotmat(img.qvec)
        translation = img.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([0, 0, 0, 1])[None]], 0)
        opencv_to_opengl = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=np.float64)
        w2c = opencv_to_opengl @ w2c
        c2w = np.linalg.inv(w2c)
        cam_c2w_dict[img.name] = c2w
    return cam_c2w_dict

def find_matching_entry(fname, scan_log):
    """
    Expect filenames like ...scan-<id>....*  → match scan_log entry with e['id']==<id>
    """
    base = fname.replace(".png", "").replace(".jpg", "")
    m = re.search(r'scan-(\d+)', base)
    if not m:
        raise ValueError(f"No 'scan-<id>' in filename: {fname}")
    scan_id = int(m.group(1))
    for i, e in enumerate(scan_log):
        if int(e["scan_id"]) == scan_id:
            return i
    raise ValueError(f"No match for scan_id={scan_id}")

def print_unmatched_scan_ids(colmap_c2w_dict, scan_log):
    """
    Print scan_ids that are in scan_log but not found in any COLMAP filename.
    """
    # Extract all scan_ids from filenames
    matched_scan_ids = set()
    for fname in colmap_c2w_dict.keys():
        base = fname.replace(".png", "").replace(".jpg", "")
        m = re.search(r'scan-(\d+)', base)
        if m:
            matched_scan_ids.add(int(m.group(1)))
    
    # Find scan_ids in scan_log that weren't matched
    scan_log_ids = set(int(e["scan_id"]) for e in scan_log)
    unmatched_ids = scan_log_ids - matched_scan_ids
    
    if unmatched_ids:
        print(f"Scan IDs in scan_log but not in COLMAP filenames: {sorted(unmatched_ids)}")
    else:
        print("All scan_log IDs were matched in COLMAP filenames.")

# -------------------- rotation undo and pose building --------------------

def rotated_c2w(json_entry, R_c2g, t_c2g, rotation_center, rotation_axis):
    """
    Compute T_{c->w0}: camera pose in the 0-angle world by undoing table rotation.
    """
    # gripper->base from robot
    R_g2b = np.asarray(json_entry["rotation_matrix"], dtype=float)
    t_g2b = np.asarray(json_entry["position"], dtype=float) / 1000.0  # mm -> m
    T_g2b = build_4x4(R_g2b, t_g2b)

    # camera->gripper (given)
    T_c2g = build_4x4(R_c2g, t_c2g)

    # camera->base at this frame
    T_c2b = T_g2b @ T_c2g

    theta = float(json_entry.get("turn_angle", 0.0))

    R_undo = rodrigues_axis_angle(rotation_axis, -theta)
    T_b2w0 = T_about_point(R_undo, rotation_center)

    # camera->0-angle world
    return T_b2w0 @ T_c2b

# -------------------- your matching function (unchanged) --------------------

def load_robot_poses_c2w0(scan_log_path, images, R_c2g, t_c2g, rotation_center, rotation_axis):
    """
    Load robot poses and match to COLMAP poses from images.txt via φ/θ in the filename.
    (Matching uses your scan-(light)-(camera) scheme; not θ.)
    """
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)

    colmap_c2w_dict = parse_colmap_images_txt(images)

    robot_poses, cam_c2w, scan_id = [], [], []
    for fname, c2w in colmap_c2w_dict.items():

        idx = find_matching_entry(fname, scan_log)
        if idx is None:
            continue

        # COLMAP camera pose
        cam_c2w.append(c2w)

        # Robot camera pose in 0-angle world (using current TURNTABLE_CENTER)
        entry = scan_log[idx]
        T = rotated_c2w(entry, R_c2g, t_c2g, rotation_center, rotation_axis)
        robot_poses.append(T)
        scan_id.append(idx)

    print_unmatched_scan_ids(colmap_c2w_dict, scan_log)
    
    print(f"Matched {len(robot_poses)}/{len(colmap_c2w_dict)} frames.")
    return robot_poses, cam_c2w, scan_id


# -------------------- world->base estimation (your pipeline kept) --------------------

def estimate_world2base(scan_log_path, images, R_c2g, t_c2g, rotation_center, rotation_axis, solve_center_xy=True):
    """
    Returns T_BW: world->base Sim3 mapping COLMAP world to robot base.
    If solve_center_xy=True, first refines ROTATION_CENTER[0:2] from data.
    """
    # Now build your matched robot & colmap poses using the (possibly) updated center
    robot_T, cam_c2w, scan_id = load_robot_poses_c2w0(scan_log_path, images, R_c2g, t_c2g, rotation_center, rotation_axis)

    # collect centres
    cam_centres_base, cam_centres_world = [], []
    for T_c2b, c2w in zip(robot_T, cam_c2w):
        cam_centres_base.append(T_c2b[:3, 3])
        cam_centres_world.append(c2w[:3, 3])
    cam_centres_base  = np.vstack(cam_centres_base)
    cam_centres_world = np.vstack(cam_centres_world)

    # single Sim3 world->base
    s, R, t = sim3_umeyama(cam_centres_world, cam_centres_base)
    T_BW = build_4x4(s * R, t)

    # debug
    print(f"[umeyama] scale: {s:.6f}")
    print(f"[umeyama] rotation matrix:\n{R}")
    print(f"[umeyama] translation:\n{t}")
    W_h = np.hstack([cam_centres_world, np.ones((len(cam_centres_world), 1))])
    W2B = (T_BW @ W_h.T).T[:, :3]
    mean_err = np.mean(np.linalg.norm(W2B - cam_centres_base, axis=1))
    print(f"[umeyama] mean error: {mean_err:.6f} m  (N={len(cam_centres_world)})")

    return T_BW, cam_c2w, scan_id

# -------------------- mesh transform & camera log save (kept) --------------------

def transform_mesh_to_base(mesh_path, T_BW, output_path=None):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh from {mesh_path}")
    mesh.transform(T_BW)
    if output_path is not None:
        ok = o3d.io.write_triangle_mesh(str(output_path), mesh)
        if not ok:
            raise RuntimeError(f"Failed to save mesh to {output_path}")
        print(f"Transformed mesh saved to: {output_path}")
    return mesh

def process_pointcloud_to_base(pointcloud_path, T_BW, cfg, output_path=None, z_outlier_percentile=5.0):
    """
    Transform Colmap sparse pointcloud to base frame, crop by XY rectangle, remove Z outliers,
    and compute axis-aligned bounding box.
    
    Args:
        pointcloud_path: Path to point3d.ply file from Colmap
        T_BW: Transformation matrix from world to base frame
        cfg: Config dict containing mesh.rectangle parameters
        output_path: Path to save filtered pointcloud (optional)
        z_outlier_percentile: Percentage of points to remove as outliers from top and bottom (default 5%)
    
    Returns:
        tuple: (filtered_pcd, bbox_min, bbox_max, bbox_center, bbox_size, filtered_indices)
            filtered_indices: numpy array of indices of filtered points in the original pointcloud
    """
    import open3d as o3d
    
    # Load pointcloud
    pcd = o3d.io.read_point_cloud(str(pointcloud_path))
    if len(pcd.points) == 0:
        raise ValueError(f"Failed to load pointcloud from {pointcloud_path}")
    
    print(f"Loaded {len(pcd.points)} points from {pointcloud_path}")
    
    # Transform to base frame
    pcd.transform(T_BW)
    print(f"Transformed pointcloud to base frame")
    
    # Get rectangle parameters from config
    center = np.array(cfg.mesh.rectangle.center)
    width = cfg.mesh.rectangle.width  # x direction
    length = cfg.mesh.rectangle.length  # y direction
    
    # Calculate XY bounds
    x_min = center[0] - width / 2.0
    x_max = center[0] + width / 2.0
    y_min = center[1] - length / 2.0
    y_max = center[1] + length / 2.0
    
    print(f"Cropping rectangle: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]")
    
    # Crop by XY coordinates
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    # Track indices through the filtering process
    original_indices = np.arange(len(points))
    
    xy_mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
              (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    
    points_cropped = points[xy_mask]
    colors_cropped = colors[xy_mask] if colors is not None else None
    indices_after_xy = original_indices[xy_mask]
    
    print(f"After XY cropping: {len(points_cropped)} points ({100*len(points_cropped)/len(points):.1f}%)")
    
    if len(points_cropped) == 0:
        raise ValueError("No points remain after XY cropping. Check rectangle parameters.")
    
    # Remove Z outliers by percentile
    z_values = points_cropped[:, 2]
    z_lower = np.percentile(z_values, z_outlier_percentile)
    z_upper = np.percentile(z_values, 100 - z_outlier_percentile)
    
    z_mask = (z_values >= z_lower) & (z_values <= z_upper)
    points_filtered = points_cropped[z_mask]
    colors_filtered = colors_cropped[z_mask] if colors_cropped is not None else None
    filtered_indices = indices_after_xy[z_mask]
    
    print(f"After Z outlier removal ({z_outlier_percentile}% top/bottom): {len(points_filtered)} points "
          f"({100*len(points_filtered)/len(points_cropped):.1f}% of cropped)")
    print(f"Z range: [{z_lower:.6f}, {z_upper:.6f}]")
    
    # Create filtered pointcloud
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(points_filtered)
    if colors_filtered is not None:
        pcd_filtered.colors = o3d.utility.Vector3dVector(colors_filtered)
    
    # Compute axis-aligned bounding box
    bbox_min = points_filtered.min(axis=0)
    bbox_max = points_filtered.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0
    bbox_size = bbox_max - bbox_min
    
    print(f"\n{'='*60}")
    print(f"Axis-Aligned Bounding Box (Base Frame):")
    print(f"{'='*60}")
    print(f"Min:    [{bbox_min[0]:9.6f}, {bbox_min[1]:9.6f}, {bbox_min[2]:9.6f}]")
    print(f"Max:    [{bbox_max[0]:9.6f}, {bbox_max[1]:9.6f}, {bbox_max[2]:9.6f}]")
    print(f"Center: [{bbox_center[0]:9.6f}, {bbox_center[1]:9.6f}, {bbox_center[2]:9.6f}]")
    print(f"Size:   [{bbox_size[0]:9.6f}, {bbox_size[1]:9.6f}, {bbox_size[2]:9.6f}]")
    print(f"{'='*60}\n")
    
    # Save filtered pointcloud if output path specified
    if output_path is not None:
        ok = o3d.io.write_point_cloud(str(output_path), pcd_filtered)
        if not ok:
            raise RuntimeError(f"Failed to save pointcloud to {output_path}")
        print(f"Filtered pointcloud saved to: {output_path}")
    
    return pcd_filtered, bbox_min, bbox_max, bbox_center, bbox_size, filtered_indices

def save_camera_log_from_colmap(camera_c2w, T_BW, scan_id, output_path):
    camera_log = []
    for i, C2W in enumerate(camera_c2w):
        C2B = T_BW @ np.asarray(C2W)
        position = C2B[:3, 3] * 1000.0  # m->mm to match scan_log
        rotation_matrix = C2B[:3, :3]
        camera_entry = {
            "overall_id": scan_id[i],
            "camera_id": scan_id[i],
            "position": position.tolist(),
            "rotation_matrix": rotation_matrix.tolist()
        }
        camera_log.append(camera_entry)
    with open(output_path, 'w') as f:
        json.dump(camera_log, f, indent=2)
    print(f"Camera log saved to: {output_path}")
    print(f"Saved {len(camera_log)} camera poses")
    return camera_log

def save_points_pixel_data(pcd_filtered, filtered_indices, images, points3D, hdr_path, output_path):
    """
    Save point-ray observation data from COLMAP sparse reconstruction with HDR RGB values.
    
    Args:
        pcd_filtered: Open3D pointcloud (filtered and transformed to base frame)
        filtered_indices: numpy array of indices mapping filtered points to original COLMAP points
        images: dict of COLMAP Image objects (from read_images_binary/text)
        points3D: dict of COLMAP Point3D objects (from read_points3D_binary/text)
        hdr_path: Path to folder containing HDR images (filenames match COLMAP image names)
        output_path: Path to save the compressed .npz file
    
    Saves:
        observations.npz containing:
            - observations: (N, 9) array with [x, y, z, image_id, pixel_x, pixel_y, r, g, b]
            - filtered_indices: array mapping to original COLMAP point IDs
    """
    import cv2
    import os
    from tqdm import tqdm
    
    print(f"\n{'='*60}")
    print(f"Extracting Point-Ray Observations with HDR RGB")
    print(f"{'='*60}")
    
    # 1. Create mapping from PLY index to COLMAP point ID
    colmap_point_ids = list(points3D.keys())  # ordered list
    print(f"Total COLMAP points: {len(colmap_point_ids)}")
    print(f"Filtered points: {len(filtered_indices)}")
    
    # 2. Count total observations for filtered points
    total_observations = 0
    for filt_idx in filtered_indices:
        colmap_point_id = colmap_point_ids[filt_idx]
        point = points3D[colmap_point_id]
        total_observations += len(point.image_ids)
    
    print(f"Total observations to extract: {total_observations:,}")
    
    # 3. Allocate observations array
    observations = np.zeros((total_observations, 9), dtype=np.float32)
    
    # 4. Pre-load ALL HDR images into memory
    print("\nPre-loading HDR images...")
    image_cache = {}
    for img_id, image in tqdm(images.items(), desc="Loading HDR images"):
        hdr_image_path = os.path.join(hdr_path, image.name)
        # Remove '_max' and everything after it, but keep .png extension
        # Find image file starting with 'scan-{img_id}'
        prefix = f'scan-{img_id}'
        matching_files = [f for f in os.listdir(hdr_path) if f.startswith(prefix)]
        if matching_files:
            hdr_image_path = os.path.join(hdr_path, matching_files[0])
        hdr_img = cv2.imread(hdr_image_path, cv2.IMREAD_UNCHANGED)
        hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
        image_cache[img_id] = hdr_img
    
    print(f"Loaded {len(image_cache)} HDR images")
    
    # 5. Build observation data using vectorized operations
    print("\nBuilding observation arrays...")
    
    # Pre-allocate lists for vectorized assembly
    xyz_list = []
    img_id_list = []
    pixel_xy_list = []
    point_indices_list = []  # Track which filtered point each observation belongs to
    
    # First pass: collect all metadata (fast, no RGB sampling yet)
    for i, filt_idx in enumerate(filtered_indices):
        colmap_point_id = colmap_point_ids[filt_idx]
        point = points3D[colmap_point_id]
        xyz = np.asarray(pcd_filtered.points[i])
        
        for img_id, point2D_idx in zip(point.image_ids, point.point2D_idxs):
            if img_id not in images:
                continue
            
            image = images[img_id]
            pixel_xy = image.xys[point2D_idx]
            
            xyz_list.append(xyz)
            img_id_list.append(img_id)
            pixel_xy_list.append(pixel_xy)
            point_indices_list.append(i)
    
    # Convert to numpy arrays
    xyz_array = np.array(xyz_list, dtype=np.float32)  # (N, 3)
    img_id_array = np.array(img_id_list, dtype=np.int32)  # (N,)
    pixel_xy_array = np.array(pixel_xy_list, dtype=np.float32)  # (N, 2)
    
    actual_observations = len(xyz_array)
    print(f"Collected {actual_observations:,} valid observations")
    
    # 6. Vectorized RGB sampling
    print("Sampling RGB values from HDR images...")
    rgb_array = np.zeros((actual_observations, 3), dtype=np.float32)
    
    # Group observations by image for efficient sampling
    from collections import defaultdict
    obs_by_image = defaultdict(list)
    for obs_idx, img_id in enumerate(img_id_array):
        obs_by_image[img_id].append(obs_idx)
    
    # Sample RGB for each image's observations in batch
    for img_id, obs_indices in tqdm(obs_by_image.items(), desc="Sampling RGB"):
        hdr_img = image_cache[img_id]
        obs_indices = np.array(obs_indices)
        
        # Get pixel coordinates for this image's observations
        pixels = pixel_xy_array[obs_indices]  # (M, 2)
        
        # Round and clip coordinates
        x_coords = np.round(pixels[:, 0]).astype(np.int32)
        y_coords = np.round(pixels[:, 1]).astype(np.int32)
        x_coords = np.clip(x_coords, 0, hdr_img.shape[1] - 1)
        y_coords = np.clip(y_coords, 0, hdr_img.shape[0] - 1)
        
        # Vectorized RGB sampling
        rgb_array[obs_indices] = hdr_img[y_coords, x_coords]
    
    # 7. Assemble final observations array
    observations = np.column_stack([
        xyz_array,           # (N, 3) - x, y, z
        img_id_array.reshape(-1, 1).astype(np.float32),  # (N, 1) - image_id
        pixel_xy_array,      # (N, 2) - pixel_x, pixel_y
        rgb_array            # (N, 3) - r, g, b
    ])  # Final shape: (N, 9)
    
    skipped_count = total_observations - actual_observations
    
    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} observations (image not in images dict)")
    
    print(f"\nFinal observations extracted: {actual_observations:,}")
    print(f"Unique images loaded: {len(image_cache)}")
    
    # 7. Calculate storage size
    storage_size_mb = (observations.nbytes + filtered_indices.nbytes) / (1024 * 1024)
    print(f"Uncompressed size: {storage_size_mb:.2f} MB")
    
    # 8. Save compressed
    print(f"\nSaving to: {output_path}")
    np.savez_compressed(output_path,
                       observations=observations,
                       filtered_indices=filtered_indices)
    
    # Check actual file size
    actual_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Compressed file size: {actual_size_mb:.2f} MB")
    print(f"Compression ratio: {storage_size_mb/actual_size_mb:.2f}x")
    print(f"{'='*60}\n")

def _process_camera_batch(args):
    """
    Worker function to process a batch of cameras.
    Returns list of observation arrays for the batch.
    """
    import cv2
    img_ids, images, cameras, points_world, points_base, hdr_path = args
    
    batch_observations = []
    
    for img_id in img_ids:
        image = images[img_id]
        camera = cameras[image.camera_id]
        
        # Load HDR image
        hdr_image_path = os.path.join(hdr_path, image.name)
        prefix = f'scan-{img_id:04d}'
        matching_files = [f for f in os.listdir(hdr_path) if f.startswith(prefix)]
        if matching_files:
            hdr_image_path = os.path.join(hdr_path, matching_files[0])
        hdr_img = cv2.imread(hdr_image_path, cv2.IMREAD_UNCHANGED)
        hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
        
        # Get camera extrinsics (world to camera)
        R_cam = qvec2rotmat(image.qvec)  # 3x3
        t_cam = image.tvec  # 3,
        
        # Transform all points to camera frame
        points_cam = (R_cam @ points_world.T).T + t_cam  # (N, 3)
        
        # Filter points in front of camera
        valid_mask = points_cam[:, 2] > 0
        
        if not valid_mask.any():
            continue
        
        # Get camera intrinsics for SIMPLE_RADIAL
        f = camera.params[0]
        cx = camera.params[1]
        cy = camera.params[2]
        k = camera.params[3] if len(camera.params) > 3 else 0.0
        
        # Project to image plane with distortion (vectorized)
        X = points_cam[:, 0]
        Y = points_cam[:, 1]
        Z = points_cam[:, 2]
        
        # Normalized coordinates
        x_norm = X / Z
        y_norm = Y / Z
        
        # Apply radial distortion
        r2 = x_norm**2 + y_norm**2
        distortion = 1 + k * r2
        
        # Distorted pixel coordinates
        pixels_x = f * distortion * x_norm + cx
        pixels_y = f * distortion * y_norm + cy
        
        # Check image bounds
        valid_mask &= (pixels_x >= 0) & (pixels_x < camera.width)
        valid_mask &= (pixels_y >= 0) & (pixels_y < camera.height)
        
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            continue
        
        # Sample RGB for valid pixels (vectorized)
        x_coords = np.round(pixels_x[valid_indices]).astype(np.int32)
        y_coords = np.round(pixels_y[valid_indices]).astype(np.int32)
        x_coords = np.clip(x_coords, 0, hdr_img.shape[1] - 1)
        y_coords = np.clip(y_coords, 0, hdr_img.shape[0] - 1)
        rgb_values = hdr_img[y_coords, x_coords]  # (M, 3)
        
        # Build observations for this camera (vectorized - NO FOR LOOP!)
        camera_observations = np.column_stack([
            points_base[valid_indices],           # xyz in base frame (M, 3)
            np.full(len(valid_indices), float(img_id)),  # image_id (M,)
            pixels_x[valid_indices],              # pixel x (M,)
            pixels_y[valid_indices],              # pixel y (M,)
            rgb_values                            # rgb (M, 3)
        ])  # Shape: (M, 9)
        batch_observations.append(camera_observations)
    
    return batch_observations


def save_points_pixel_data_reprojection(pcd_filtered, filtered_indices, cameras, images, points3D, hdr_path, observations_folder, num_workers=32):
    """
    Reproject filtered 3D points to all camera views with SIMPLE_RADIAL distortion.
    Uses multiprocessing to speed up processing.
    
    Args:
        pcd_filtered: Open3D pointcloud (filtered and transformed to base frame) - used for getting base frame coordinates
        filtered_indices: numpy array of indices mapping filtered points to original COLMAP points
        cameras: dict of COLMAP Camera objects (from read_cameras_binary/text)
        images: dict of COLMAP Image objects (from read_images_binary/text)
        points3D: dict of COLMAP Point3D objects (from read_points3D_binary/text)
        hdr_path: Path to folder containing HDR images
        observations_folder: Path to folder where observation chunks will be saved
        num_workers: Number of parallel workers (default: 32)
    
    Saves:
        50 npz files in observations_folder, each containing a chunk of shuffled observations
    """
    import cv2
    import os
    from tqdm import tqdm
    from multiprocessing import Pool
    
    print(f"\n{'='*60}")
    print(f"Reprojecting Points to All Camera Views (Multiprocessing)")
    print(f"{'='*60}")
    
    # Get filtered points from COLMAP (in COLMAP world frame)
    colmap_point_ids = list(points3D.keys())  # ordered list
    points_base = np.asarray(pcd_filtered.points)  # (N, 3) in base frame for final output
    
    # Extract COLMAP 3D coordinates for filtered points
    points_world = []
    for filt_idx in filtered_indices:
        colmap_point_id = colmap_point_ids[filt_idx]
        point = points3D[colmap_point_id]
        points_world.append(point.xyz)
    
    points_world = np.array(points_world, dtype=np.float32)  # (N, 3) in COLMAP world frame
    num_points = len(points_world)
    
    print(f"Filtered points to reproject: {num_points:,}")
    print(f"Total cameras: {len(images)}")
    print(f"Using {num_workers} parallel workers")
    
    # Split camera IDs into batches of 10 for better progress tracking
    img_ids = list(images.keys())
    chunk_size = 10  # Process 10 cameras per batch
    batches = [img_ids[i:i+chunk_size] for i in range(0, len(img_ids), chunk_size)]
    
    print(f"Split into {len(batches)} batches ({chunk_size} cameras each)")
    
    # Prepare arguments for each batch
    batch_args = [
        (batch, images, cameras, points_world, points_base, hdr_path)
        for batch in batches
    ]
    
    # Process batches in parallel
    print("\nProcessing cameras in parallel...")
    all_observations = []
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(_process_camera_batch, batch_args),
            total=len(batch_args),
            desc="Processing batches"
        ))
    
    # Combine results from all batches (concatenate numpy arrays)
    for batch_result in results:
        all_observations.extend(batch_result)
    
    # Convert to numpy array by stacking all observation arrays
    if all_observations:
        observations = np.vstack(all_observations)  # (M, 9)
    else:
        observations = np.array([], dtype=np.float32).reshape(0, 9)
    total_obs = len(observations)
    
    print(f"\nTotal valid observations: {total_obs:,}")
    print(f"Average observations per point: {total_obs / num_points:.2f}")
    print(f"Average observations per camera: {total_obs / len(images):.2f}")
    
    # Calculate storage size
    storage_size_mb = observations.nbytes / (1024 * 1024)
    print(f"Uncompressed size: {storage_size_mb:.2f} MB")
    
    # Save to chunks
    save_observations_to_chunks(observations, observations_folder, num_chunks=50)


def _save_chunk(args):
    """
    Worker function to save a single chunk.
    """
    import os
    chunk_data, chunk_idx, output_folder = args
    
    # Save chunk
    output_path = os.path.join(output_folder, f"observations_chunk_{chunk_idx:02d}.npz")
    np.savez_compressed(output_path, observations=chunk_data)
    
    # Return stats
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return chunk_idx, len(chunk_data), file_size_mb


def save_observations_to_chunks(observations, output_folder, num_chunks=50, num_workers=32):
    """
    Shuffle observations via index permutation and save to multiple npz files using multiprocessing.
    
    Args:
        observations: (N, 9) numpy array
        output_folder: Path to output folder
        num_chunks: Number of files to split into
        num_workers: Number of parallel workers for saving (default: 32)
    """
    import os
    from tqdm import tqdm
    from multiprocessing import Pool
    
    print(f"\n{'='*60}")
    print(f"Saving Observations to Chunks (Multiprocessing)")
    print(f"{'='*60}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Randomly shuffle indices instead of the whole array (much faster!)
    total_obs = len(observations)
    print(f"Total observations: {total_obs:,}")
    print(f"Number of chunks: {num_chunks}")
    print(f"Using {num_workers} parallel workers")
    
    print("Shuffling via index permutation...")
    shuffled_indices = np.random.permutation(total_obs)
    
    # Split shuffled indices into chunks
    chunk_size = int(np.ceil(total_obs / num_chunks))
    print(f"Approximate observations per chunk: {chunk_size:,}")
    
    # Prepare chunk data and arguments for parallel processing
    print("\nPreparing chunks for parallel saving...")
    chunk_args = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_obs)
        
        if start_idx >= total_obs:
            break
        
        # Get chunk using shuffled indices (already shuffled, no need to shuffle again!)
        chunk_indices = shuffled_indices[start_idx:end_idx]
        chunk_data = observations[chunk_indices].copy()  # Make a copy for each worker
        
        chunk_args.append((chunk_data, i, output_folder))
    
    # Save chunks in parallel
    print(f"\nSaving {len(chunk_args)} chunks in parallel...")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(_save_chunk, chunk_args),
            total=len(chunk_args),
            desc="Saving chunks"
        ))
    
    # Print results
    print("\nChunk Summary:")
    for chunk_idx, chunk_len, file_size_mb in sorted(results):
        print(f"  Chunk {chunk_idx:02d}: {chunk_len:,} observations, {file_size_mb:.2f} MB")
    
    print(f"\nSaved {len(results)} chunks to: {output_folder}")
    print(f"{'='*60}\n")
    
# -------------------- main --------------------

def main_process(cfg, cameras, images, points3D=None, scan_log_path=None, mesh_path=None, camera_log_path=None, hdr_path=None, observations_folder=None, pointcloud_path=None, bbox_json_path=None, z_outlier_percentile=5.0, num_workers=32):
    # Extract parameters from config
    R_c2g = np.array(cfg.camera.R_c2g)
    t_c2g = np.array(cfg.camera.t_c2g)
    rotation_center = np.array(cfg.emitter.turntable.center)
    rotation_axis = np.array(cfg.emitter.turntable.axis)
    
    T_BW, cam_c2w, scan_id = estimate_world2base(scan_log_path, images, R_c2g, t_c2g, rotation_center, rotation_axis)
    if mesh_path is not None:
        transform_mesh_to_base(mesh_path, T_BW, output_path=str(Path(mesh_path).with_name(Path(mesh_path).stem + "_transformed.ply")))
    
    # Process pointcloud if path provided
    if pointcloud_path is not None:
        output_pcd_path = str(Path(pointcloud_path).with_name(Path(pointcloud_path).stem + "_transformed_filtered.ply"))
        pcd_filtered, bbox_min, bbox_max, bbox_center, bbox_size, filtered_indices = process_pointcloud_to_base(
            pointcloud_path, T_BW, cfg, output_path=output_pcd_path, z_outlier_percentile=z_outlier_percentile
        )
        
        save_points_pixel_data_reprojection(pcd_filtered, filtered_indices, cameras, images, points3D, hdr_path, observations_folder, num_workers=num_workers)

        # Save bounding box info to a JSON file
        bbox_info = {
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "bbox_center": bbox_center.tolist(),
            "bbox_size": bbox_size.tolist(),
            "num_points": len(pcd_filtered.points)
        }
        with open(bbox_json_path, 'w') as f:
            json.dump(bbox_info, f, indent=2)
        print(f"Bounding box info saved to: {bbox_json_path}")
    
    save_camera_log_from_colmap(cam_c2w, T_BW, scan_id, camera_log_path)

@hydra.main(version_base=None, config_path="../../config/renderer", config_name="realcapture_area_emitter")
def main(cfg: DictConfig) -> None:
    # Get parameters from Hydra config (can be overridden via command line)
    folder_path = cfg.shape_matching.folder_path
    num_workers = cfg.shape_matching.num_workers
    z_outlier_percentile = cfg.shape_matching.z_outlier_percentile
    
    # Build paths from folder_path
    scan_log_path = os.path.join(folder_path, "scan_log.json")
    mesh_path = None  # Optional mesh transformation
    model_path = os.path.join(folder_path, "sparse")
    pointcloud_path = os.path.join(model_path, "points3D.ply")
    points3D_path = os.path.join(model_path, "points3D.bin")
    hdr_path = os.path.join(folder_path, "hdr")
    observations_folder = os.path.join(folder_path, "observations")
    camera_log_path = os.path.join(folder_path, "rotated_camera.json")
    bbox_json_path = os.path.join(folder_path, "bbox.json")
    print(f"Processing folder: {folder_path}")
    print(f"Number of workers: {num_workers}")
    print(f"Z outlier percentile: {z_outlier_percentile}")

    # Load COLMAP data (cameras, images, points3D)
    cameras, images = read_model(model_path, ext=".bin")
    
    # Load points3D with format detection
    if detect_model_format(model_path, ".bin"):
        points3D = read_points3D_binary(points3D_path)
    elif detect_model_format(model_path, ".txt"):
        points3D = read_points3D_text(points3D_path)
    else:
        raise ValueError(f"Could not detect COLMAP model format in {model_path}")
    
    print(f"Loaded {len(points3D)} 3D points from COLMAP")

    main_process(cfg, cameras, images, points3D=points3D, scan_log_path=scan_log_path, mesh_path=mesh_path, 
                 camera_log_path=camera_log_path, hdr_path=hdr_path, observations_folder=observations_folder, 
                 pointcloud_path=pointcloud_path, bbox_json_path=bbox_json_path, z_outlier_percentile=z_outlier_percentile, num_workers=num_workers)
    print(f"Finished shape matching for {folder_path}")

if __name__ == "__main__":
    main()
