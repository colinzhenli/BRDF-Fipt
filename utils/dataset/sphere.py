import torch
import torch.nn.functional as NF
from torch.utils.data import Dataset
import json
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from PIL import Image
from torchvision import transforms as T
import cv2
import math
import matplotlib.pyplot as plt

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    c2w = c2w #@ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return c2w

trans_t = lambda t : np.asarray([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=np.float32)

rot_phi = lambda phi : np.asarray([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=np.float32)

rot_theta = lambda th : np.asarray([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=np.float32)


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

class SphereDataset(Dataset):
    """ Simple synthetic dataset for basic scenes like sphere
    Scene/
        {SPLIT}/ train or val split
            Image/{:03d}_0001.exr HDR images
            transforms.json c2w camera matrix file and fov
    """
    def __init__(self, cfg, gt_path, split='train'):
        """
        Args:
            root_dir: dataset root folder
            gt_path: path to ground truth RGB image
            split: train or val
            pixel: whether load every camera pixel
            ray_diff: whether load ray differentials
        """
        self.cfg = cfg
        self.pixel = cfg.data.pixel
        self.split = split
        self.ray_diff = cfg.data.ray_diff
        self.custom_c2w = cfg.data.custom_c2w
        self.gt_path = gt_path
        self.camera_dict = {
            "position": cfg.renderer.camera.position,
            "look_at": cfg.renderer.camera.look_at,
            "up": cfg.renderer.camera.up
        }

        self.img_hw = cfg.renderer.resolution
        if gt_path is not None:
            self.img = plt.imread(gt_path)[...,:3]
            assert self.img.shape[0] == self.img_hw[0]
            assert self.img.shape[1] == self.img_hw[1]
            self.img = torch.from_numpy(self.img.astype(np.float32))


        # camera focal length and ray directions
        h, w = self.img_hw
        self.camera_angle_x = cfg.renderer.camera.camera_angle_x
        self.focal = (0.5*w/np.tan(0.5*self.camera_angle_x)).item()
        self.directions = get_ray_directions(h, w, self.focal)

        # if self.pixel:
        #     self.all_rays = []
        #     self.all_rgbs = []
        #     self.get_render_poses()
        #     for cur_idx in range(self.total):
        #         c2w = torch.from_numpy(self.render_poses[cur_idx][:3, :4])
        #         img = open_exr(os.path.join(self.root_dir, 'Image', '{:03d}_0001.exr'.format(cur_idx)), self.img_hw).reshape(-1,3)
        #         self.all_rgbs += [img]
        #         rays_o, rays_d,dxdu,dydv = get_rays(self.directions, c2w, focal=self.focal) # both (h*w, 3)

        #         self.all_rays += [torch.cat([rays_o, rays_d,
        #                                      dxdu,
        #                                      dydv,
        #                                     ],1)] 
        #     self.all_rays = torch.cat(self.all_rays, 0)
        #     self.all_rgbs = torch.cat(self.all_rgbs, 0)
        #     # number of camera ray batches
        #     self.batch_num = math.ceil(len(self.all_rays)*1.0/self.batch_size)
        #     self.idxs = torch.randperm(len(self.all_rays)) 

    def get_render_poses(self):
        stride = 20 #self.opt.render_stride
        radius = 4 #self.opt.render_radius
        self.render_poses = np.stack([pose_spherical(angle, -30.0, radius) @ self.blender2opencv for angle in np.linspace(-180, 180, stride + 1)[:-1]], 0)
        self.total = len(self.render_poses)

    def __len__(self):
        if self.split == 'val' or self.split == 'test':
            return 1
        return self.cfg.data.batch_num

    def __getitem__(self, idx):          
        # Handle different ways of specifying custom camera transform
        if self.custom_c2w:
            if 'rotation' in self.camera_dict:
                # Compute c2w from position and rotation
                position = torch.tensor(self.camera_dict['position'])
                rotation = torch.tensor(self.camera_dict.get('rotation', [0,0,0]))
                
                # Convert euler angles to rotation matrix
                Rx = torch.tensor([[1, 0, 0],
                                    [0, torch.cos(rotation[0]), -torch.sin(rotation[0])],
                                    [0, torch.sin(rotation[0]), torch.cos(rotation[0])]])
                Ry = torch.tensor([[torch.cos(rotation[1]), 0, torch.sin(rotation[1])],
                                    [0, 1, 0],
                                    [-torch.sin(rotation[1]), 0, torch.cos(rotation[1])]])
                Rz = torch.tensor([[torch.cos(rotation[2]), -torch.sin(rotation[2]), 0],
                                    [torch.sin(rotation[2]), torch.cos(rotation[2]), 0],
                                    [0, 0, 1]])
                R = Rz @ Ry @ Rx
                
                c2w = torch.eye(4)
                c2w[:3,:3] = R
                c2w[:3,3] = position
                c2w = c2w[:3,:4]
                
            elif 'look_at' in self.camera_dict:
                # Compute c2w from look direction
                position = torch.tensor(self.camera_dict['position'])
                target = torch.tensor(self.camera_dict['look_at'])
                up = torch.tensor(self.camera_dict.get('up', [0,1,0]))
                
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
        
        
        if self.ray_diff:
            rays_o, rays_d, dxdu, dydv = get_rays(self.directions, c2w, focal=self.focal)
            rays = torch.cat([rays_o, rays_d, dxdu, dydv], 1)
        else:
            rays_o, rays_d = get_rays(self.directions, c2w)
            rays = torch.cat([rays_o, rays_d], 1)

        if self.gt_path is not None:
            sample = {
                'rays': rays,
                'c2w': c2w,
                'rgbs': self.img
            }
        else:
            sample = {
                'rays': rays,
                'c2w': c2w,
            }
        return sample