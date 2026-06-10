"""Render a COLMAP sparse point cloud as a paper-quality figure using Blender.

Usage:
    blender --background --python render_pointcloud.py -- \
        --ply /path/to/points3D_transformed_filtered.ply \
        --out /path/to/output_dir \
        --views 3 \
        --resolution 800 600 \
        --sphere_radius 0.0004

Renders N views of the point cloud with:
  * Per-point vertex-color shading on a diffuse cloth-like material
  * Ambient occlusion via Cycles path tracer
  * Transparent background (RGBA PNG)
  * Soft HDRI-style lighting

Each point is instanced as a small icosphere with a principled BSDF
(high roughness, subsurface for cloth feel).
"""

import bpy
import bmesh
import sys
import os
import math
import argparse
import numpy as np
from pathlib import Path
from mathutils import Vector, Matrix


# ============== arg parsing (after Blender's '--') ==============

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser(description="Render point cloud views")
    p.add_argument("--ply", required=True, help="Path to .ply point cloud")
    p.add_argument("--out", default="./renders", help="Output directory")
    p.add_argument("--views", type=int, default=3, help="Number of views to render")
    p.add_argument("--resolution", type=int, nargs=2, default=[800, 600], help="Width Height")
    p.add_argument("--sphere_radius", type=float, default=0.0004, help="Radius of each point sphere")
    p.add_argument("--samples", type=int, default=128, help="Cycles render samples")
    p.add_argument("--roughness", type=float, default=0.85, help="Material roughness (1.0 = fully diffuse)")
    p.add_argument("--subsurface", type=float, default=0.05, help="Subsurface scattering weight for cloth feel")
    return p.parse_args(argv)


# ============== scene setup ==============

def clear_scene():
    """Remove all default objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Also remove orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_renderer(args):
    """Configure Cycles renderer with transparent background."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'

    # Try to enable GPU
    prefs = bpy.context.preferences.addons.get('cycles')
    if prefs:
        cprefs = prefs.preferences
        cprefs.compute_device_type = 'CUDA'
        cprefs.get_devices()
        for dev in cprefs.devices:
            dev.use = True

    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True

    # Transparent background
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    # Resolution
    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]
    scene.render.resolution_percentage = 100


def create_cloth_material(name="ClothPointMaterial", roughness=0.85, subsurface=0.05):
    """Create a principled BSDF material that looks like diffuse cloth.

    Uses vertex color (Attribute node with 'Col') as the base color.
    High roughness + slight subsurface scattering gives a fabric feel.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear defaults
    for n in nodes:
        nodes.remove(n)

    # Nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Subsurface Weight'].default_value = subsurface
    bsdf.inputs['Subsurface Radius'].default_value = (0.01, 0.01, 0.01)
    # Sheen for cloth micro-fibre highlights
    bsdf.inputs['Sheen Weight'].default_value = 0.3
    bsdf.inputs['Sheen Roughness'].default_value = 0.5

    # Vertex color attribute → base color
    attr = nodes.new('ShaderNodeAttribute')
    attr.location = (-300, 0)
    attr.attribute_name = 'Col'  # default vertex color layer name from PLY import

    links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    return mat


def load_pointcloud(ply_path):
    """Import PLY and return the mesh object."""
    bpy.ops.wm.ply_import(filepath=ply_path)
    obj = bpy.context.selected_objects[0]
    return obj


def setup_geometry_nodes(obj, sphere_radius, material):
    """Add a Geometry Nodes modifier that instances small icospheres at each point."""
    # Create a small icosphere template
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=sphere_radius)
    sphere_template = bpy.context.active_object
    sphere_template.name = "PointSphereTemplate"
    sphere_template.data.materials.append(material)
    # Hide template from render
    sphere_template.hide_render = True
    sphere_template.hide_viewport = True

    # Select the point cloud object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Add Geometry Nodes modifier
    mod = obj.modifiers.new(name="InstancePoints", type='NODES')

    # Build the node tree
    tree = bpy.data.node_groups.new(name="PointCloudInstancer", type='GeometryNodeTree')
    mod.node_group = tree

    nodes = tree.nodes
    links = tree.links

    # Input and output
    input_node = nodes.new('NodeGroupInput')
    input_node.location = (-600, 0)
    output_node = nodes.new('NodeGroupOutput')
    output_node.location = (600, 0)

    # Create input/output sockets
    tree.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    # Mesh to Points (convert mesh vertices to points)
    m2p = nodes.new('GeometryNodeMeshToPoints')
    m2p.location = (-300, 0)
    m2p.inputs['Radius'].default_value = sphere_radius

    # Instance on Points
    iop = nodes.new('GeometryNodeInstanceOnPoints')
    iop.location = (0, 0)

    # Object Info (reference to the sphere template)
    obj_info = nodes.new('GeometryNodeObjectInfo')
    obj_info.location = (-300, -200)
    obj_info.inputs['Object'].default_value = sphere_template
    obj_info.transform_space = 'RELATIVE'

    # Realize instances (so vertex colors transfer)
    realize = nodes.new('GeometryNodeRealizeInstances')
    realize.location = (300, 0)

    # Link
    links.new(input_node.outputs['Geometry'], m2p.inputs['Mesh'])
    links.new(m2p.outputs['Points'], iop.inputs['Points'])
    links.new(obj_info.outputs['Geometry'], iop.inputs['Instance'])
    links.new(iop.outputs['Instances'], realize.inputs['Geometry'])
    links.new(realize.outputs['Geometry'], output_node.inputs['Geometry'])

    # Assign material to the point cloud object too
    if len(obj.data.materials) == 0:
        obj.data.materials.append(material)
    else:
        obj.data.materials[0] = material

    return sphere_template


def setup_lighting():
    """Set up soft 3-point lighting for a clean paper figure look."""
    # Key light (warm, slightly above and to the right)
    bpy.ops.object.light_add(type='AREA', location=(0.3, -0.3, 0.2))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = 8
    key.data.size = 0.4
    key.data.color = (1.0, 0.95, 0.9)

    # Fill light (cooler, from the left, softer)
    bpy.ops.object.light_add(type='AREA', location=(-0.3, -0.2, 0.15))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 4
    fill.data.size = 0.6
    fill.data.color = (0.9, 0.93, 1.0)

    # Rim/back light (from behind, for edge definition)
    bpy.ops.object.light_add(type='AREA', location=(0.0, 0.3, 0.15))
    rim = bpy.context.active_object
    rim.name = "RimLight"
    rim.data.energy = 3
    rim.data.size = 0.3
    rim.data.color = (1.0, 1.0, 1.0)

    # Ambient: use world background with very faint environment
    world = bpy.data.worlds.get('World')
    if world is None:
        world = bpy.data.worlds.new('World')
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Strength'].default_value = 0.3
        bg.inputs['Color'].default_value = (0.95, 0.95, 0.95, 1.0)


def setup_camera(center, size, view_index, total_views):
    """Position camera for a specific view around the point cloud.

    Views are distributed around the point cloud at slightly different
    elevation and azimuth angles for variety.
    """
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Camera settings
    cam_data.lens = 85  # portrait-ish focal length for nice perspective
    cam_data.clip_start = 0.001
    cam_data.clip_end = 10.0

    # Orbit around the center
    # Different azimuth per view, slight elevation change
    azimuth_base = -30  # degrees, starting angle
    azimuth_step = 50   # degrees between views
    elevation = 55       # degrees from horizontal (looking down)

    if total_views == 1:
        azimuth = 0
    else:
        azimuth = azimuth_base + view_index * azimuth_step

    # Camera distance (based on bounding box diagonal)
    diag = math.sqrt(size[0]**2 + size[1]**2 + size[2]**2)
    distance = diag * 2.2

    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    cam_x = center[0] + distance * math.cos(el_rad) * math.cos(az_rad)
    cam_y = center[1] + distance * math.cos(el_rad) * math.sin(az_rad)
    cam_z = center[2] + distance * math.sin(el_rad)

    cam_obj.location = (cam_x, cam_y, cam_z)

    # Point camera at center
    direction = Vector(center) - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    return cam_obj


# ============== main ==============

def main():
    args = parse_args()
    ply_path = args.ply
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Point Cloud Renderer")
    print(f"{'='*60}")
    print(f"PLY: {ply_path}")
    print(f"Output: {out_dir}")
    print(f"Views: {args.views}")
    print(f"Resolution: {args.resolution}")
    print(f"Sphere radius: {args.sphere_radius}")
    print(f"Samples: {args.samples}")
    print(f"Roughness: {args.roughness}")
    print(f"{'='*60}\n")

    # Setup
    clear_scene()
    setup_renderer(args)

    # Load point cloud
    print("Loading point cloud...")
    pc_obj = load_pointcloud(ply_path)
    print(f"  Loaded: {len(pc_obj.data.vertices):,} vertices")

    # Compute center and size
    verts = np.array([v.co for v in pc_obj.data.vertices])
    center = verts.mean(axis=0).tolist()
    size = (verts.max(axis=0) - verts.min(axis=0)).tolist()
    print(f"  Center: {center}")
    print(f"  Size: {size}")

    # Create cloth material
    mat = create_cloth_material(
        roughness=args.roughness,
        subsurface=args.subsurface,
    )

    # Setup geometry nodes for instancing
    print("Setting up geometry nodes...")
    sphere_template = setup_geometry_nodes(pc_obj, args.sphere_radius, mat)

    # Lighting
    setup_lighting()

    # Render each view
    for i in range(args.views):
        print(f"\nRendering view {i + 1}/{args.views}...")

        # Remove previous camera
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA':
                bpy.data.objects.remove(obj, do_unlink=True)

        cam = setup_camera(center, size, i, args.views)

        # Output path
        out_path = str(out_dir / f"pointcloud_view_{i + 1}.png")
        bpy.context.scene.render.filepath = out_path

        # Render
        bpy.ops.render.render(write_still=True)
        print(f"  Saved: {out_path}")

    # Also save the .blend file for manual tweaking
    blend_path = str(out_dir / "pointcloud_scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"\nSaved Blender file: {blend_path}")
    print(f"{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
