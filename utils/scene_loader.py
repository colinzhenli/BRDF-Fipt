import mitsuba as mi
import os
import numpy as np
import torch

mi.set_variant("cuda_ad_rgb")  # 或 "scalar_rgb" / "llvm_ad_rgb"

def load_uv_obj_to_mitsuba_scene(obj_path="mesh_test/cube_with_uv.obj"):
    """
    Load an OBJ file with UV coordinates and construct a Mitsuba Scene.

    Args:
        obj_path (str): Path to the .obj file, defaults to "mesh_test/cube_with_uv.obj".

    Returns:
        scene (mi.Scene): Mitsuba scene object containing the loaded geometry with default diffuse material.
        
    """
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"OBJ 文件不存在: {obj_path}")
    
    # obj_path = load_and_transform_mesh_trimesh(obj_path)
    # obj_path = obj_path.replace(".obj", "_uv.ply")

    # 构建形状加载字典
    shape_dict = {
        "type": "ply",
        "filename": obj_path,
        # "to_world": transform
    }


    shape_dict["bsdf"] = {
        "type": "diffuse",
        "reflectance": {
            "type": "rgb",
            "value": [0.8, 0.8, 0.8]
        }
    }

    # 构造场景
    scene_dict = {
        "type": "scene",
        "shape": shape_dict,
    }

    #scene = mi.load_dict(scene_dict)
    return scene_dict

def create_rectangle_scene(center=[0, 0, 0], width=0.4, length=0.4):
    #center=[100,100,100]
    to_world = (
        mi.ScalarTransform4f.translate(mi.ScalarPoint3f(*center)) @
        mi.ScalarTransform4f.scale(mi.ScalarVector3f(width/2, length/2, 1.0))
    )

    scene_dict = {
        "type": "scene",
        "shape_id": {
            "type": "rectangle",
            "to_world": to_world,
            "flip_normals": False,
        }
    }
    #return mi.load_dict(scene_dict)
    return scene_dict

def add_circle_to_scene_dict(scene_dict, pose_matrix, radius):
    """
    Add a circular disk to an existing Mitsuba scene dictionary.

    Args:
        scene_dict (dict): Scene dictionary created for Mitsuba.
        pose_matrix (array-like or torch.Tensor): 4x4 transformation matrix (world pose).
        radius (float): Disk radius.
    """
    # Convert to Mitsuba ScalarTransform4f
    #pose_matrix[0,3]+=0.1
    #pose_matrix[1,3]+=0.1

    l2g=torch.tensor([[1.0,  0.0,  0.0],
          [0.0,  0.0,  -1.0],
          [0.0,  1.0,  0.0]], dtype=torch.float32,device=pose_matrix.device)
    pose_trans=pose_matrix.clone()
    pose_trans[:3, :3] = pose_matrix[:3, :3] @ l2g
    print("pose_trans",pose_trans)
    if isinstance(pose_trans, torch.Tensor):
        pose_trans = pose_trans.detach().cpu().numpy()
    
    # ScalarTransform4f expects a numpy array directly
    pose_transform = mi.ScalarTransform4f(pose_trans)

    # Apply scaling for the radius
    circle_to_world = pose_transform @ mi.ScalarTransform4f.scale([radius, radius, 1.0])

    # Add disk entry (unique key to avoid overwriting)
    #circle_id = f"circle_{len([k for k in scene_dict.keys() if 'circle' in k])}"
    scene_dict["circle"] = {
        "type": "disk",
        "to_world": circle_to_world,
        "flip_normals": False,
    }
    return scene_dict

def load_and_transform_mesh_trimesh(obj_path):
    """
    Load mesh from obj_path, scale and transform it to the origin,( scale it to 0.2, and computing the average y axis and moving it toward -y direction), then save the transformed mesh.
    
    Args:
        obj_path (str): Path to the input OBJ file
        
    Returns:
        str: Path to the transformed mesh file
    """
    import trimesh
    
    # Load the mesh
    mesh = trimesh.load(obj_path)
    
    # Scale the mesh to 0.2
    mesh.apply_scale(0.2)
    
    # Compute the average y coordinate
    vertices = mesh.vertices
    avg_y = np.mean(vertices[:, 1])
    
    # Move the mesh toward -y direction to center it at origin
    translation = np.array([0, -avg_y, 0])
    mesh.apply_translation(translation)
    
    # Generate output path
    output_path = obj_path.replace('.obj', '_transformed.obj')
    
    # Save the transformed mesh
    mesh.export(output_path)
    
    return output_path
