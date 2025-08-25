import trimesh
import numpy as np
import argparse
import os

# Your first image's T_b2cCV matrix
T_b2cCV = np.array([
    [3.05632940e-05, -9.99989470e-01,  4.58831481e-03, -1.73842111e-1],
    [-5.78015280e-01,  3.72652000e-03,  8.16017436e-01,  1.46225606e-1],
    [-8.16025946e-01, -2.67706000e-03, -5.78009078e-01,  7.58373379e-1],
    [0.0, 0.0, 0.0, 1.0]
])

# Global bounding box limits (x_min, y_min, z_min, x_max, y_max, z_max)
# Adjust these values as needed for your specific use case
global_bbox = [0.0, -0.15, -0.12, 0.3, 0.15, 0.05-0.12]  # Example values in meters

def transform_mesh(mesh, T_w2c):
    """Apply a 4x4 transform to mesh vertices."""
    vertices_camera = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
    # Compute camera-to-world transform (inverse of world-to-camera)
    T_c2w = np.linalg.inv(T_w2c)
    # Transform vertices from camera coordinates to world coordinates
    vertices_world = (T_c2w @ vertices_camera.T).T
    mesh.vertices = vertices_world[:, :3]  # Keep only x, y, z coordinates
    return mesh

def crop_mesh_bbox(mesh: trimesh.Trimesh, bbox):
    """Crop mesh using an axis-aligned bounding box.
    
    Args:
        mesh (trimesh.Trimesh): The input mesh.
        bbox (list or array): [x_min, y_min, z_min, x_max, y_max, z_max]
        
    Returns:
        trimesh.Trimesh: Cropped submesh.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = bbox

    # Identify vertices inside the bbox
    in_bbox = (
        (mesh.vertices[:, 0] >= x_min) & (mesh.vertices[:, 0] <= x_max) &
        (mesh.vertices[:, 1] >= y_min) & (mesh.vertices[:, 1] <= y_max) &
        (mesh.vertices[:, 2] >= z_min) & (mesh.vertices[:, 2] <= z_max)
    )

    # Filter faces where all 3 vertices are in bbox
    face_mask = np.all(in_bbox[mesh.faces], axis=1)
    face_indices = np.nonzero(face_mask)[0]

    if len(face_indices) == 0:
        raise ValueError("No mesh part remains after cropping. Adjust the bounding box.")

    return mesh.submesh([face_indices], only_watertight=False)[0]

def crop_mesh(mesh: trimesh.Trimesh, crop_percent=0.8):
    """Crop mesh by percentage around the center AABB in XY plane only.

    Args:
        mesh (trimesh.Trimesh): Input mesh.
        crop_percent (float): Percentage to keep. E.g. 0.8 keeps center 80%.
    
    Returns:
        trimesh.Trimesh: Cropped mesh.
    """
    bounds_min = mesh.bounds[0]
    bounds_max = mesh.bounds[1]
    size = bounds_max - bounds_min  # OK: no ptp() used

    crop_margin = (1 - crop_percent) / 2.0
    
    # Only crop in XY plane, keep full Z range
    crop_min = bounds_min.copy()
    crop_max = bounds_max.copy()
    
    # Apply cropping only to X and Y dimensions
    crop_min[0] = bounds_min[0] + size[0] * crop_margin  # X min
    crop_min[1] = bounds_min[1] + size[1] * crop_margin  # Y min
    # Keep original Z min: crop_min[2] = bounds_min[2]
    
    crop_max[0] = bounds_max[0] - size[0] * crop_margin  # X max
    crop_max[1] = bounds_max[1] - size[1] * crop_margin  # Y max
    # Keep original Z max: crop_max[2] = bounds_max[2]

    bbox = [*crop_min, *crop_max]
    return crop_mesh_bbox(mesh, bbox)


def main(input_path, output_path, crop_percent):
    mesh = trimesh.load(input_path, force='mesh')

    # # Step 1: Apply T_b2cCV to transform mesh to base (world) coordinate frame
    # mesh = transform_mesh(mesh, T_b2cCV)

    # # Step 2: Optionally export full transformed mesh for visualization/debug
    # transformed_path = os.path.splitext(output_path)[0] + "_transformed.ply"
    # mesh.export(transformed_path)
    # print(f"Saved transformed mesh to: {transformed_path}")

    print("skipping transform")
    print("cropping by bbox")
    mesh = crop_mesh_bbox(mesh, global_bbox)
    # Save intermediate mesh after bbox cropping
    bbox_cropped_path = os.path.splitext(output_path)[0] + "_bbox_cropped.ply"
    mesh.export(bbox_cropped_path)
    print(f"Saved bbox cropped mesh to: {bbox_cropped_path}")
    print("cropping by percentage")
    
    cropped = crop_mesh(mesh, crop_percent)
    cropped.export(output_path)
    print(f"Saved cropped mesh to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input mesh (.ply, .obj, etc.)')
    parser.add_argument('--output', type=str, required=True, help='Path to save cropped mesh')
    parser.add_argument('--percent', type=float, default=0.8, help='Crop percent (e.g., 0.8 for center 80%)')
    args = parser.parse_args()

    main(args.input, args.output, args.percent)
