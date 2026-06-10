"""Render two point clouds together with different materials.

Usage:
    blender --background --python render_two_pointclouds.py -- \
        --ply_fabric /path/to/fabric_points.ply \
        --ply_surrounding /path/to/surrounding_points.ply \
        --out /path/to/output_dir \
        --views 3

Fabric points get a fuzzy cloth-like material (high roughness, sheen, slight subsurface)
keeping their per-point colors. Surrounding (checkerboard region) points get a lighter,
smoother material with a neutral tint.

Both are rendered together in one Blender scene at their COLMAP-world coordinates.
"""

import bpy
import sys
import os
import math
import argparse
import numpy as np
from pathlib import Path
from mathutils import Vector


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser()
    p.add_argument("--ply_fabric", required=True)
    p.add_argument("--ply_surrounding", required=True)
    p.add_argument("--out", default="./renders")
    p.add_argument("--views", type=int, default=3)
    p.add_argument("--resolution", type=int, nargs=2, default=[1000, 800])
    p.add_argument("--sphere_radius_fabric", type=float, default=0.05,
                   help="Radius of fabric point spheres (COLMAP world units)")
    p.add_argument("--sphere_radius_surround", type=float, default=0.04,
                   help="Radius of surrounding point spheres")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--view_indices", type=int, nargs="*", default=None,
                   help="Specific 1-based view indices to render (e.g. 3). Default: all 1..views")
    return p.parse_args(argv)


# ============== scene setup ==============

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_renderer(args):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'

    prefs = bpy.context.preferences.addons.get('cycles')
    if prefs:
        cprefs = prefs.preferences
        cprefs.compute_device_type = 'CUDA'
        cprefs.get_devices()
        for dev in cprefs.devices:
            dev.use = True

    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True

    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]
    scene.render.resolution_percentage = 100


# ============== materials ==============

def create_fuzzy_cloth_material(name, color_mode='vertex', tint=(1.0, 1.0, 1.0, 1.0)):
    """Fuzzy cloth material: very rough, sheen, subsurface.

    color_mode='vertex' → use PLY vertex colors
    color_mode='tint'   → use the tint RGBA as base color
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for n in nodes:
        nodes.remove(n)

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    # Fuzzy fabric look
    bsdf.inputs['Roughness'].default_value = 0.95           # near-fully diffuse
    bsdf.inputs['Subsurface Weight'].default_value = 0.08
    bsdf.inputs['Subsurface Radius'].default_value = (0.005, 0.005, 0.005)
    bsdf.inputs['Sheen Weight'].default_value = 0.6         # strong fabric sheen
    bsdf.inputs['Sheen Roughness'].default_value = 0.35
    bsdf.inputs['Specular IOR Level'].default_value = 0.2

    if color_mode == 'vertex':
        attr = nodes.new('ShaderNodeAttribute')
        attr.location = (-300, 0)
        attr.attribute_name = 'Col'
        links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
    elif color_mode == 'tint_modulated':
        # Golden hue × per-point luminance → keeps the weave pattern while
        # giving the fabric a uniform golden colour.
        attr = nodes.new('ShaderNodeAttribute')
        attr.location = (-700, 0)
        attr.attribute_name = 'Col'

        rgb2bw = nodes.new('ShaderNodeRGBToBW')   # luminance of the original colour
        rgb2bw.location = (-500, 0)
        links.new(attr.outputs['Color'], rgb2bw.inputs['Color'])

        # Remap the (dark) fabric luminance into a brighter modulation range with
        # contrast, so light threads → bright gold, dark gaps → dark gold.
        maprange = nodes.new('ShaderNodeMapRange')
        maprange.location = (-300, 0)
        maprange.inputs['From Min'].default_value = 0.03
        maprange.inputs['From Max'].default_value = 0.30
        maprange.inputs['To Min'].default_value = 0.45
        maprange.inputs['To Max'].default_value = 1.7
        maprange.clamp = True
        links.new(rgb2bw.outputs['Val'], maprange.inputs['Value'])

        # Multiply the golden tint by the scalar luminance (scalar broadcasts to RGB).
        mix = nodes.new('ShaderNodeMixRGB')
        mix.location = (-100, 0)
        mix.blend_type = 'MULTIPLY'
        mix.inputs['Fac'].default_value = 1.0
        mix.inputs['Color1'].default_value = tint
        links.new(maprange.outputs['Result'], mix.inputs['Color2'])
        links.new(mix.outputs['Color'], bsdf.inputs['Base Color'])
    else:  # 'tint'
        bsdf.inputs['Base Color'].default_value = tint

    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat


def create_surrounding_material(name, mode='cream', base_color=(0.06, 0.06, 0.07, 1.0)):
    """Material for the surrounding checkerboard region.

    mode='cream'  → ORIGINAL look: per-point vertex colors brightened + cream
                    tint, so the checkerboard keeps its natural black/white feel.
    mode='solid'  → uniform `base_color` (e.g. dark gray) ignoring vertex colors.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for n in nodes:
        nodes.remove(n)

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (200, 0)
    bsdf.inputs['Roughness'].default_value = 0.75
    bsdf.inputs['Subsurface Weight'].default_value = 0.0
    bsdf.inputs['Sheen Weight'].default_value = 0.1
    bsdf.inputs['Specular IOR Level'].default_value = 0.3

    if mode == 'solid':
        bsdf.inputs['Base Color'].default_value = base_color
    else:  # 'cream' — original natural look
        attr = nodes.new('ShaderNodeAttribute')
        attr.location = (-400, 100)
        attr.attribute_name = 'Col'

        bright = nodes.new('ShaderNodeBrightContrast')
        bright.location = (-200, 200)
        bright.inputs['Bright'].default_value = 0.2
        bright.inputs['Contrast'].default_value = 0.1

        mix = nodes.new('ShaderNodeMixRGB')
        mix.location = (-100, 50)
        mix.blend_type = 'MULTIPLY'
        mix.inputs['Fac'].default_value = 0.6
        mix.inputs['Color2'].default_value = (1.5, 1.5, 1.4, 1.0)  # cream brighten

        links.new(attr.outputs['Color'], bright.inputs['Color'])
        links.new(bright.outputs['Color'], mix.inputs['Color1'])
        links.new(mix.outputs['Color'], bsdf.inputs['Base Color'])

    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat


# ============== point cloud import + instancing ==============

def load_pointcloud(ply_path, name=None):
    bpy.ops.wm.ply_import(filepath=ply_path)
    obj = bpy.context.selected_objects[0]
    if name:
        obj.name = name
    return obj


def setup_geometry_nodes(obj, sphere_radius, material, template_name):
    # Sphere template
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=sphere_radius)
    sphere = bpy.context.active_object
    sphere.name = template_name
    sphere.data.materials.append(material)
    sphere.hide_render = True
    sphere.hide_viewport = True

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mod = obj.modifiers.new(name="InstancePoints", type='NODES')
    tree = bpy.data.node_groups.new(name=f"PCInst_{template_name}", type='GeometryNodeTree')
    mod.node_group = tree

    nodes = tree.nodes
    links = tree.links

    input_node = nodes.new('NodeGroupInput'); input_node.location = (-600, 0)
    output_node = nodes.new('NodeGroupOutput'); output_node.location = (600, 0)
    tree.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    m2p = nodes.new('GeometryNodeMeshToPoints'); m2p.location = (-300, 0)
    m2p.inputs['Radius'].default_value = sphere_radius

    iop = nodes.new('GeometryNodeInstanceOnPoints'); iop.location = (0, 0)

    obj_info = nodes.new('GeometryNodeObjectInfo'); obj_info.location = (-300, -200)
    obj_info.inputs['Object'].default_value = sphere
    obj_info.transform_space = 'RELATIVE'

    realize = nodes.new('GeometryNodeRealizeInstances'); realize.location = (300, 0)

    links.new(input_node.outputs['Geometry'], m2p.inputs['Mesh'])
    links.new(m2p.outputs['Points'], iop.inputs['Points'])
    links.new(obj_info.outputs['Geometry'], iop.inputs['Instance'])
    links.new(iop.outputs['Instances'], realize.inputs['Geometry'])
    links.new(realize.outputs['Geometry'], output_node.inputs['Geometry'])

    if len(obj.data.materials) == 0:
        obj.data.materials.append(material)
    else:
        obj.data.materials[0] = material

    return sphere


# ============== lighting + camera ==============

def setup_lighting():
    """Three-point lighting for paper figure look."""
    bpy.ops.object.light_add(type='AREA', location=(15, -15, 25))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = 18000
    key.data.size = 8
    key.data.color = (1.0, 0.97, 0.92)

    bpy.ops.object.light_add(type='AREA', location=(-15, -10, 15))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 9000
    fill.data.size = 12
    fill.data.color = (0.88, 0.92, 1.0)

    bpy.ops.object.light_add(type='AREA', location=(0, 20, 15))
    rim = bpy.context.active_object
    rim.name = "RimLight"
    rim.data.energy = 7000
    rim.data.size = 6
    rim.data.color = (1.0, 1.0, 1.0)

    world = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Strength'].default_value = 0.35
        bg.inputs['Color'].default_value = (0.95, 0.95, 0.95, 1.0)


def setup_camera(center, radius, view_index, total_views, plane_basis):
    """Orbit camera around the fabric, using its plane frame.

    plane_basis = (u, v, n): two in-plane orthonormal axes (u, v) and the plane
    normal (n). The camera looks at `center` from a direction that is mostly along
    +n (top-down on the fabric) tilted by `elevation` toward an in-plane azimuth,
    so the fabric is always well-framed regardless of its orientation in COLMAP world.
    """
    u, v, n = plane_basis

    cam_data = bpy.data.cameras.new(name="Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_data.lens = 45   # wider FOV to capture the full checkerboard
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0

    # Different azimuth per view (rotation within the fabric plane),
    # fixed tilt from the plane normal.
    azimuth_base = -35
    azimuth_step = 45
    tilt_deg = 35   # angle away from the plane normal

    azimuth = math.radians(azimuth_base + view_index * azimuth_step)
    tilt = math.radians(tilt_deg)

    # In-plane direction for this azimuth
    inplane = math.cos(azimuth) * u + math.sin(azimuth) * v
    # View direction = mostly along normal, tilted toward inplane
    view_dir = math.cos(tilt) * n + math.sin(tilt) * inplane
    view_dir = view_dir / np.linalg.norm(view_dir)

    distance = radius * 2.0   # radius already spans the checkerboard; keep it framed
    cam_pos = np.array(center) + distance * view_dir
    cam_obj.location = tuple(cam_pos)

    # Look at center; use plane normal as a stable up-ish reference
    direction = Vector(center) - Vector(tuple(cam_pos))
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    return cam_obj


# ============== main ==============

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Two-Point-Cloud Renderer")
    print(f"{'='*60}")
    print(f"Fabric PLY:      {args.ply_fabric}")
    print(f"Surrounding PLY: {args.ply_surrounding}")
    print(f"Output:          {out_dir}")
    print(f"{'='*60}\n")

    clear_scene()
    setup_renderer(args)

    # Load both point clouds
    print("Loading fabric PLY...")
    fabric_obj = load_pointcloud(args.ply_fabric, name="FabricPoints")
    print(f"  {len(fabric_obj.data.vertices):,} fabric points")

    print("Loading surrounding PLY...")
    surr_obj = load_pointcloud(args.ply_surrounding, name="SurroundingPoints")
    print(f"  {len(surr_obj.data.vertices):,} surrounding points")

    # Compute center via median (robust). Use PCA on FABRIC points to find the
    # plane frame: principal axes 0,1 span the plane (u, v); axis 2 is normal (n).
    fabric_verts = np.array([v.co for v in fabric_obj.data.vertices])
    fabric_center = np.median(fabric_verts, axis=0)

    centered = fabric_verts - fabric_center
    U_, S_, Vt_ = np.linalg.svd(centered, full_matrices=False)
    u_axis = Vt_[0]
    v_axis = Vt_[1]
    n_axis = Vt_[2]
    plane_basis = (u_axis, v_axis, n_axis)

    # Framing radius: base it on the SURROUNDING (checkerboard) points so the whole
    # checkerboard fits in frame, not just the fabric. Use a percentile of the
    # surrounding points' in-plane distance to ignore far COLMAP outliers.
    surr_verts = np.array([v.co for v in surr_obj.data.vertices])
    surr_centered = surr_verts - fabric_center
    surr_inplane = surr_centered @ np.column_stack([u_axis, v_axis])
    surr_radial = np.linalg.norm(surr_inplane, axis=1)
    fabric_radius = np.percentile(surr_radial, 80) * 1.1   # 80th pct → most of the checkerboard

    print(f"Fabric center: {fabric_center}")
    print(f"Plane normal:  {n_axis}")
    print(f"Frame radius (from surrounding 80th pct):  {fabric_radius:.3f}\n")

    # Materials — original natural look (preview 2):
    # Fabric keeps its real per-point colors; checkerboard is the cream-tinted
    # vertex version.
    mat_fabric = create_fuzzy_cloth_material("FuzzyCloth", color_mode='vertex')
    mat_surr   = create_surrounding_material("Checkerboard", mode='cream')

    # Geometry-nodes instancing
    print("Setting up geometry nodes for fabric...")
    setup_geometry_nodes(fabric_obj, args.sphere_radius_fabric, mat_fabric, "FabricSphereTpl")

    print("Setting up geometry nodes for surrounding...")
    setup_geometry_nodes(surr_obj, args.sphere_radius_surround, mat_surr, "SurrSphereTpl")

    # Lighting
    setup_lighting()

    # Render views (optionally only a subset specified via --view_indices, 1-based)
    if args.view_indices:
        view_list = [v - 1 for v in args.view_indices]   # to 0-based
    else:
        view_list = list(range(args.views))

    for i in view_list:
        print(f"\nRendering view {i + 1}...")

        for obj in list(bpy.data.objects):
            if obj.type == 'CAMERA':
                bpy.data.objects.remove(obj, do_unlink=True)

        setup_camera(tuple(fabric_center), fabric_radius, i, args.views, plane_basis)
        out_path = str(out_dir / f"pointcloud_view_{i + 1}.png")
        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print(f"  saved {out_path}")

    blend_path = str(out_dir / "two_pointcloud_scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"\nSaved blend file: {blend_path}")
    print("Done!")


if __name__ == "__main__":
    main()
