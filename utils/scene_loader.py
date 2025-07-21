import mitsuba as mi
import os
import numpy as np

mi.set_variant("cuda_ad_rgb")  # 或 "scalar_rgb" / "llvm_ad_rgb"

def load_uv_obj_to_mitsuba_scene(obj_path="mesh_test/cube_with_uv.obj"):
    """
    读取带 UV 的 OBJ 文件，并构造一个 Mitsuba Scene。

    参数:
        obj_path (str): .obj 文件路径，需包含 UV 信息。
        texture_path (str or None): 可选，贴图图片路径（如 .jpg/.png）。
        scale (float): 对几何体的缩放比例。
        rotate_x90 (bool): 是否将几何绕 X 轴旋转 -90°，将其从 Blender 导出的坐标转换到 Mitsuba 的默认方向。

    返回:
        scene (mi.Scene): Mitsuba 场景对象。
    """
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"OBJ 文件不存在: {obj_path}")

    swap_yz = mi.ScalarTransform4f.rotate([1, 0, 0], -90)

    # 然后缩放 x 和 y 轴到 0.2 (注意缩放顺序和应用顺序)
    scale_xy = mi.ScalarTransform4f.scale([0.2, 0.2, -0.2])

    # 先旋转再缩放（矩阵乘法是右乘先执行）
    #transform = scale_xy @ swap_yz
    transform = scale_xy
    
    # 构建形状加载字典
    shape_dict = {
        "type": "obj",
        "filename": obj_path,
        "to_world": transform
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
        "shape": shape_dict
    }

    scene = mi.load_dict(scene_dict)
    return scene
