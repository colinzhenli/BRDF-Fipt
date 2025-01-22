import os
import open3d as o3d
from argparse import ArgumentParser

# Parse arguments
parser = ArgumentParser()
parser.add_argument('--output', type=str, default='/localhome/zla247/theia2_data/fipt_indoor_synthetic/sphere', help='output path')
args = parser.parse_args()

# Create sphere mesh
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2.5)
sphere.compute_vertex_normals()

# Save mesh
os.makedirs(args.output, exist_ok=True)
o3d.io.write_triangle_mesh(os.path.join(args.output, "scene.obj"), sphere)
