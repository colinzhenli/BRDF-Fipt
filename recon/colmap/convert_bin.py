#!/usr/bin/env python3
import struct
import numpy as np

# --- Camera Model Info ---
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12)
}

# --- Reading binary files ---
def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        num_cams = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cams):
            cam_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            model_name, num_params = CAMERA_MODELS[model_id]
            params = struct.unpack("<" + "d" * num_params, f.read(8 * num_params))
            cameras[cam_id] = (model_name, width, height, params)
    return cameras

def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        num_imgs = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_imgs):
            img_id_and_data = struct.unpack("<idddddddi", f.read(4 + 8*7 + 4))
            img_id = img_id_and_data[0]
            qvec = img_id_and_data[1:5]
            tvec = img_id_and_data[5:8]
            cam_id = img_id_and_data[8]
            # null-terminated image name
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8")
            # 2D points
            num_points = struct.unpack("<Q", f.read(8))[0]
            xys = []
            point3D_ids = []
            for _ in range(num_points):
                x, y, pid = struct.unpack("<ddq", f.read(8+8+8))
                xys.append((x, y))
                point3D_ids.append(pid)
            images[img_id] = (qvec, tvec, cam_id, name, xys, point3D_ids)
    return images

def read_points3D_bin(path):
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            pid, x, y, z, r, g, b, error = struct.unpack("<QdddBBBd", f.read(8+8*3+1*3+8))
            track_len = struct.unpack("<Q", f.read(8))[0]
            image_ids = []
            point2D_idxs = []
            for _ in range(track_len):
                img_id, idx = struct.unpack("<ii", f.read(4+4))
                image_ids.append(img_id)
                point2D_idxs.append(idx)
            points[pid] = ((x, y, z), (r, g, b), error, image_ids, point2D_idxs)
    return points

# --- Writing txt files ---
def write_cameras_txt(cameras, path):
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(cameras)))
        for cam_id, (model_name, width, height, params) in cameras.items():
            f.write(f"{cam_id} {model_name} {width} {height} {' '.join(map(str, params))}\n")

def write_images_txt(images, path):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}\n".format(len(images)))
        for img_id, (qvec, tvec, cam_id, name, xys, point3D_ids) in images.items():
            f.write(f"{img_id} {' '.join(map(str, qvec))} {' '.join(map(str, tvec))} {cam_id} {name}\n")
            pts_line = " ".join(f"{x} {y} {pid}" for (x, y), pid in zip(xys, point3D_ids))
            f.write(pts_line + "\n")

def write_points3D_txt(points, path):
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: {}\n".format(len(points)))
        for pid, (xyz, rgb, error, image_ids, point2D_idxs) in points.items():
            track_str = " ".join(f"{iid} {idx}" for iid, idx in zip(image_ids, point2D_idxs))
            f.write(f"{pid} {' '.join(map(str, xyz))} {' '.join(map(str, rgb))} {error} {track_str}\n")

# --- Main ---
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python convert_bin.py <sparse_path>")
        sys.exit(1)
    
    sparse_path = sys.argv[1]

    cams = read_cameras_bin(f"{sparse_path}/cameras.bin")
    imgs = read_images_bin(f"{sparse_path}/images.bin")
    # pts  = read_points3D_bin(f"{sparse_path}/points3D.bin")

    write_cameras_txt(cams, f"{sparse_path}/cameras.txt")
    write_images_txt(imgs, f"{sparse_path}/images.txt")
    # write_points3D_txt(pts, f"{sparse_path}/points3D.txt")

    print(f"✅ Converted to text: {len(cams)} cameras, {len(imgs)} images")
