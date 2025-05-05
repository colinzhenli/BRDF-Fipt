import torch
import torch.nn.functional as NF
from torch.utils.data import Dataset, IterableDataset
import json
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from PIL import Image
from torchvision import transforms as T
import cv2
import math
import matplotlib.pyplot as plt

def get_ray_directions(H, W, focal):
    """ get camera ray direction
    Args:
        H,W: height and width
        focal: focal length
    """
    x_coords = torch.linspace(0.5, W - 0.5, W)
    y_coords = torch.linspace(0.5, H - 0.5, H)
    j, i = torch.meshgrid([y_coords, x_coords])
    directions = \
        torch.stack([-(i-W/2)/focal, -(j-H/2)/focal, torch.ones_like(i)], -1) 

    return directions

def get_rays(directions, c2w, focal=None):
    """ world space camera ray
    Args:
        directions: camera ray direction (local)
        c2w: 3x4 camera to world matrix
        focal: if not None, return ray differentials as well
    """
    R = c2w[:,:3]
    rays_d = directions @ R.T
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    if focal is not None:
        dxdu = torch.tensor([1.0/focal,0,0])[None,None].expand_as(directions)@R.T
        dydv = torch.tensor([0,1.0/focal,0])[None,None].expand_as(directions)@R.T
        dxdu = dxdu.view(-1,3)
        dydv = dydv.view(-1,3)
        return rays_o, rays_d, dxdu, dydv
    else:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        return rays_o, rays_d

def read_image(path, img_hw):
    img = plt.imread(path)[...,:3]
    assert img.shape[0] == img_hw[0]
    assert img.shape[1] == img_hw[1]
    return torch.from_numpy(img.astype(np.float32))

def open_exr(file,img_hw):
    """ open image exr file """
    img = cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
    assert img.shape[0] == img_hw[0]
    assert img.shape[1] == img_hw[1]
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = img[...,[2,1,0]]
    img = torch.from_numpy(img.astype(np.float32))
    return img

def get_c2w(camera):
    position = torch.tensor(camera['position'], dtype=torch.float32)
    target = torch.tensor(camera['look_at'], dtype=torch.float32)
    up = torch.tensor(camera.get('up', [0,1,0]), dtype=torch.float32)
    
    forward = target - position
    forward = forward / torch.norm(forward)
    # Ensure `up` is not parallel to `forward`
    if torch.abs(torch.dot(forward, up)) > 0.99:  # Too parallel, adjust up
        up = torch.tensor([1.0, 0.0, 0.0]) if torch.abs(forward[0]) < 0.99 else torch.tensor([0.0, 1.0, 0.0])
    """ right hand coordinate system """
    right = torch.cross(up, forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)
    
    c2w = torch.eye(4)
    c2w[:3,:3] = torch.stack([right, up, forward], dim=1)
    c2w[:3,3] = position
    c2w = c2w[:3,:4]
    return c2w

class SphereTrainIterableDataset(IterableDataset):
    def __init__(self, cfg, gt_folder):
        self.cfg = cfg
        self.pixel = True
        self.rays_num = cfg.data.rays_num
        self.num_view_batch = 4

        # Load metadata
        metadata_path = "metadata/train.txt"
        self.metadata = []
        with open(metadata_path, 'r') as f:
            for line in f:
                if line.strip():
                    roughness, metallic = map(float, line.strip().split())
                    self.metadata.append((roughness, metallic))

        self.gt_folder = gt_folder
        self.img_hw = cfg.renderer.resolution
        h, w = self.img_hw
        self.camera_angle_x = cfg.renderer.camera.camera_angle_x
        self.focal = (0.5 * w / np.tan(0.5 * self.camera_angle_x)).item()
        self.directions = get_ray_directions(h, w, self.focal)
        self.number_of_views = cfg.renderer.camera.number_of_views
        self.distance = cfg.renderer.camera.distance
        self.camera_dict = self.get_camera_rotation_dicts()
        self.all_rays, self.all_rgbs = self.preload_rays_and_rgbs()

    def get_camera_rotation_dicts(self):
        camera_dicts = []
        look_at = self.cfg.renderer.camera.look_at
        up = self.cfg.renderer.camera.up
        dist = self.distance
        n_steps = self.number_of_views
        phi = np.pi
        thetas = np.linspace(0, 2*np.pi, n_steps, endpoint=False)
        for theta in thetas:
            x = dist * np.sin(theta) * np.cos(phi)
            y = dist * np.sin(theta) * np.sin(phi)
            z = dist * np.cos(theta)
            camera_dicts.append({"position": [x, y, z], "look_at": look_at, "up": up})
        return camera_dicts

    def preload_rays_and_rgbs(self):
        all_rays = []
        all_rgbs = []
        for i, cam in enumerate(self.camera_dict):
            c2w = get_c2w(cam)
            rays_o, rays_d, dxdu, dydv = get_rays(self.directions, c2w, focal=self.focal)
            all_rays.append(torch.cat([rays_o, rays_d, dxdu, dydv], dim=-1))
            if self.gt_folder is not None:
                img = open_exr(os.path.join(self.gt_folder, f'output_view_{i}.exr'), self.img_hw).reshape(-1, 3)
                all_rgbs.append(img)
        return torch.cat(all_rays), torch.cat(all_rgbs) if all_rgbs else None

    def __iter__(self):
        rays_per_view = self.img_hw[0] * self.img_hw[1]

        while True:
            for idx in torch.randperm(len(self.metadata)):
                view_indices = torch.randint(0, self.number_of_views, (self.num_view_batch,))
                total_indices = []
                for view_idx in view_indices:
                    base = view_idx.item() * rays_per_view
                    rand_idx = torch.randint(base, base + rays_per_view, (self.rays_num // self.num_view_batch,))
                    # rand_idx = torch.randint(base, base + rays_per_view, (self.rays_num // self.num_view_batch,))
                    total_indices.append(rand_idx)

                ray_idx = torch.cat(total_indices, dim=0)

                rays = self.all_rays[ray_idx]
                if self.all_rgbs is not None:
                    rgbs = self.all_rgbs[ray_idx]
                else:
                    rgbs = torch.zeros_like(rays[...,:3])

                yield {
                    'rays': rays,
                    'rgbs': rgbs,
                    'gt_params': {
                        'roughness': self.metadata[idx][0],
                        'metallic': self.metadata[idx][1]
                    }
                }


class SphereValDataset(Dataset):
    def __init__(self, cfg, gt_folder):
        self.cfg = cfg
        self.pixel = False
        self.gt_folder = gt_folder

        # Metadata
        """  use train data for validation """
        metadata_path = "metadata/train.txt"
        self.metadata = []
        with open(metadata_path, 'r') as f:
            for line in f:
                if line.strip():
                    roughness, metallic = map(float, line.strip().split())
                    self.metadata.append((roughness, metallic))

        self.img_hw = cfg.renderer.resolution
        h, w = self.img_hw
        self.camera_angle_x = cfg.renderer.camera.camera_angle_x
        self.focal = (0.5 * w / np.tan(0.5 * self.camera_angle_x)).item()
        self.directions = get_ray_directions(h, w, self.focal)
        self.distance = cfg.renderer.camera.distance
        self.camera_dict = self.get_camera_rotation_dicts()

    def get_camera_rotation_dicts(self):
        camera_dicts = []
        look_at = self.cfg.renderer.camera.look_at
        up = self.cfg.renderer.camera.up
        dist = self.distance
        n_steps = self.cfg.renderer.camera.number_of_views
        phi = np.pi
        thetas = np.linspace(0, 2*np.pi, n_steps, endpoint=False)
        for theta in thetas:
            x = dist * np.sin(theta) * np.cos(phi)
            y = dist * np.sin(theta) * np.sin(phi)
            z = dist * np.cos(theta)
            camera_dicts.append({"position": [x, y, z], "look_at": look_at, "up": up})
        return camera_dicts

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        c2w = get_c2w(self.camera_dict[0])
        rays_o, rays_d, dxdu, dydv = get_rays(self.directions, c2w, focal=self.focal)
        rays = torch.cat([rays_o, rays_d, dxdu, dydv], dim=-1)
        if self.gt_folder is not None:
            img = open_exr(os.path.join(self.gt_folder, f'output_view_{idx}.exr'), self.img_hw).reshape(-1, 3)
        else:
            img = torch.zeros_like(rays[...,:3])
        return {
            'rays': rays,
            'rgbs': img,
            'gt_params': {'roughness': self.metadata[idx][0], 'metallic': self.metadata[idx][1]}
        }