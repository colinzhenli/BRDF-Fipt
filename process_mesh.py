import bpy
import bmesh
import numpy as np

def uv_unwrap_and_compute_TBN(obj_name, angle_limit=89.0, island_margin=0.02):
    # Get the object
    obj = bpy.data.objects.get(obj_name)
    if not obj or obj.type != 'MESH':
        print(f"❌ Mesh object named {obj_name} not found")
        return None, None

    # Ensure we're in object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Deselect all, select target object and make it active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Enter edit mode and perform UV unwrapping
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    # Check if UV map already exists
    if obj.data.uv_layers.active is None:
        print("No UV map found, creating one...")
        bpy.ops.uv.smart_project(angle_limit=angle_limit, island_margin=island_margin)
    else:
        print(f"UV map '{obj.data.uv_layers.active.name}' already exists, skipping unwrapping")
    
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Calculate tangents (must have UV and normals first)
    uvmap_name = obj.data.uv_layers.active.name
    #for uv in obj.data.uv_layers:
    #    print(uv.name)
    obj.data.calc_tangents(uvmap=uvmap_name)

    # Get UV, Tangent, Bitangent, Normal (per loop)
    uv_map = []
    tbn_list = []



    uv_layer = obj.data.uv_layers.active.data
    loops = obj.data.loops
    tangents = obj.data.loops

    for loop in obj.data.loops:
        loop_index = loop.index
        vertex_index = loop.vertex_index
        uv = uv_layer[loop_index].uv
        tangent = loop.tangent
        normal = loop.normal
        bitangent = loop.bitangent_sign * normal.cross(tangent)
        #print(tangent,normal,bitangent)

        uv_map.append((uv.x, uv.y))
        tbn_list.append((
            (tangent.x, tangent.y, tangent.z),
            (bitangent.x, bitangent.y, bitangent.z),
            (normal.x, normal.y, normal.z),
        ))

    print(f"✅ UV unwrapping and TBN calculation completed, total {len(uv_map)} loops")
    return uv_map, tbn_list

# Example usage
uvs, tbn = uv_unwrap_and_compute_TBN("fuse_post")