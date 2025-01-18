import os
import numpy as np
import json
import open3d as o3d
import shutil
import math
from argparse import ArgumentParser

# Parse arguments
parser = ArgumentParser()
parser.add_argument('--output', type=str, required=True, help='output path')
parser.add_argument('--scene', type=str, required=True, help='scene path')
parser.add_argument('--dataset', type=str, required=True, help='dataset type')
args = parser.parse_args()

# Create output directory structure 
output_dir = args.scene
splits = ["train", "val"]

for split in splits:
    os.makedirs(os.path.join(output_dir, split, "Image"), exist_ok=True)

# Create sphere mesh
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
sphere.compute_vertex_normals()

# Save mesh
o3d.io.write_triangle_mesh(os.path.join(output_dir, "scene.obj"), sphere)

# Camera parameters
fov = 30.0 # degrees
h = w = 800 # image resolution
focal = 0.5*w/np.tan(0.5*fov*np.pi/180)

# Camera transform - looking at origin from -2 units back
origin = np.array([0, 0, -2])
target = np.array([0, 0, 1]) 
up = np.array([0, 1, 0])

# Compute camera-to-world transform
forward = target - origin
forward = forward / np.linalg.norm(forward)
right = np.cross(forward, up)
right = right / np.linalg.norm(right)
up = np.cross(right, forward)
c2w = np.eye(4)
c2w[:3,:3] = np.stack([right, up, forward], axis=1)
c2w[:3,3] = origin

# Create transforms.json
transforms = {
    "camera_angle_x": fov * np.pi/180,
    "frames": [{
        "transform_matrix": c2w.tolist()
    }]
}

# Save transforms
for split in splits:
    with open(os.path.join(output_dir, split, "transforms.json"), "w") as f:
        json.dump(transforms, f, indent=4)

# Copy envmap.exr to Image folder for frame 000
for split in splits:
    shutil.copy2("envmap.exr", 
                os.path.join(output_dir, split, "Image", "000_0001.exr"))
