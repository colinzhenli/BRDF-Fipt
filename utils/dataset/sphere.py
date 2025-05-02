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

class SphereDataset(Dataset):
    """ Simple synthetic dataset for basic scenes like sphere
    Scene/
        {SPLIT}/ train or val split
            Image/{:03d}_0001.exr HDR images
            transforms.json c2w camera matrix file and fov
    """
    def __init__(self, cfg, gt_folder, split='train'):
        """
        Args:
            root_dir: dataset root folder
            gt_path: path to ground truth RGB image
            split: train or val
            pixel: whether load every camera pixel
            ray_diff: whether load ray differentials
        """
        self.cfg = cfg
        self.pixel = cfg.data.pixel if split == 'train' else False
        self.rays_num = cfg.data.rays_num
        self.num_view_batch = 4
        self.split = split
        # Load metadata from the appropriate split file
        metadata_path = f"metadata/{split}.txt"
        self.metadata = []
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                for line in f:
                    if line.strip():
                        roughness, metallic = map(float, line.strip().split())
                        self.metadata.append((roughness, metallic))
        else:
            print(f"Warning: Metadata file {metadata_path} not found.")
        self.custom_c2w = cfg.data.custom_c2w
        self.gt_folder = gt_folder  
        self.spiral_path = cfg.renderer.camera.spiral_path
        self.distance = cfg.renderer.camera.distance
        self.number_of_views = cfg.renderer.camera.number_of_views
        self.num_lights = cfg.renderer.emitter.num_lights
        self.initial_camera_dict = {
            "look_at": cfg.renderer.camera.look_at,
            "up": cfg.renderer.camera.up,
            "position": cfg.renderer.camera.position
        }

        self.img_hw = cfg.renderer.resolution

        # camera focal length and ray directions
        h, w = self.img_hw
        self.camera_angle_x = cfg.renderer.camera.camera_angle_x
        self.focal = (0.5*w/np.tan(0.5*self.camera_angle_x)).item()
        self.directions = get_ray_directions(h, w, self.focal)
        # self.get_render_poses()
        if self.spiral_path:
            # self.get_spiral_camera_dicts()
            self.get_camera_rotation_dicts()
        else:
            self.get_camera_dicts()
        

        if self.pixel:
            self.all_rays = []
            self.all_rgbs = []
            for cur_idx in range(self.total):
                c2w = get_c2w(self.camera_dict[cur_idx])
                rays_o, rays_d,dxdu,dydv = get_rays(self.directions, c2w, focal=self.focal) # both (h*w, 3)
                self.all_rays += [torch.cat([rays_o, rays_d,
                                                dxdu,
                                                dydv,
                                                ],1)] 
                if self.gt_folder is not None:
                    """ to be changed to remove light index """
                    img = open_exr(os.path.join(self.gt_folder, f'output_view_{cur_idx}.exr'), self.img_hw).reshape(-1,3)
                    self.all_rgbs += [img]

            self.all_rays = torch.cat(self.all_rays, 0)
            if self.gt_folder is not None:
                self.all_rgbs = torch.cat(self.all_rgbs, 0)
            else:
                self.all_rgbs = None


    def get_camera_dicts(self):
        # Initialize camera dicts list
        self.camera_dict = []
        
        # Keep look_at and up vectors fixed from initial camera settings
        look_at = self.initial_camera_dict["look_at"]
        up = self.initial_camera_dict["up"]
        dist = self.distance
        
        # Parameters to control sampling density
        n_theta = 7  # number of theta samples (excluding poles)
        n_phi = 6    # number of phi samples
        self.total = (n_theta-2) * n_phi + 2  # Add 2 for poles
        
        # Generate uniform samples for spherical coordinates, excluding poles
        thetas = np.linspace(0, np.pi, n_theta)  # Exclude 0 and pi
        thetas = thetas[1:-1]  # Remove the first and last elements
        phis = np.linspace(0, 2*np.pi, n_phi)
        
        # Add poles separately - they only need one phi value since they're at top/bottom
        
        # Create grid of angles
        theta_grid, phi_grid = np.meshgrid(thetas, phis)
        thetas_flat = theta_grid.flatten()
        phis_flat = phi_grid.flatten()
        
        # Convert spherical to cartesian coordinates
        for theta, phi in zip(thetas_flat, phis_flat):
            # Calculate camera position
            x = dist * np.sin(theta) * np.cos(phi)
            y = dist * np.sin(theta) * np.sin(phi) 
            z = dist * np.cos(theta)
            
            # Create camera dict for this position
            camera_dict = {
                "position": [x, y, z],
                "look_at": look_at,
                "up": up
            }
            
            self.camera_dict.append(camera_dict)
        north_pole = {
            "position": [0, 0, dist],  # x=0, y=0, z=dist
            "look_at": look_at,
            "up": up
        }
        self.camera_dict.append(north_pole)

        south_pole = {
            "position": [0, 0, -dist],  # x=0, y=0, z=-dist
            "look_at": look_at,
            "up": up
        }
        self.camera_dict.append(south_pole)


    def get_camera_rotation_dicts(self):
        # Initialize camera dicts list  
        self.camera_dict = []
        
        look_at = self.initial_camera_dict["look_at"]
        up = self.initial_camera_dict["up"]
        dist = self.distance
        phi = np.pi
        n_steps = self.number_of_views  # Can be adjusted for more/fewer views
        self.total = n_steps
        thetas = np.linspace(0, 2*np.pi, n_steps, endpoint=False)
        for theta in thetas:
            x = dist * np.sin(theta) * np.cos(phi)  # Using cos(0)=1 for fixed phi
            y = dist * np.sin(theta) * np.sin(phi)  # Using sin(0)=0 for fixed phi
            z = dist * np.cos(theta)
            
            # Create camera dict for this position
            camera_dict = {
                "position": [x, y, z],
                "look_at": look_at,
                "up": up
            }
            
            self.camera_dict.append(camera_dict)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):          
        # Handle different ways of specifying custom camera transform
        if self.pixel:
            # Randomly select num_view_batch views
            view_indices = torch.randperm(self.number_of_views)[:self.num_view_batch]
            
            # Get indices for all rays from selected views
            rays_per_view = self.img_hw[0] * self.img_hw[1]
            view_ray_indices = []
            for view_idx in view_indices:
                start_idx = view_idx * rays_per_view
                end_idx = start_idx + rays_per_view
                view_ray_indices.append(torch.arange(start_idx, end_idx))
            
            # Combine indices from all selected views
            view_ray_indices = torch.cat(view_ray_indices)
            
            # Randomly shuffle rays from selected views
            shuffled_indices = torch.randperm(len(view_ray_indices))
            self.idxs = view_ray_indices[shuffled_indices]
            
            # find camera ray indices in the batch
            ray_idx = self.idxs[:self.rays_num]
            tmp = self.all_rays[ray_idx]
            params = {'roughness': self.metadata[idx][0],
                      'metallic': self.metadata[idx][1]}
            
            # Use zero tensor instead of None for rgbs when all_rgbs is None
            rgbs = self.all_rgbs[ray_idx] if self.all_rgbs is not None else torch.zeros_like(tmp[...,:3])
            
            sample = {'rays': tmp[...,:12],
                      'rgbs': rgbs,
                      'gt_params': params}

        else:
            c2w = get_c2w(self.camera_dict[idx])
            rays_o,rays_d,dxdu,dydv = get_rays(self.directions, c2w, focal=self.focal)

            rays = torch.cat([rays_o, rays_d,
                              dxdu,
                              dydv],-1)
            params = {'roughness': self.metadata[idx][0], 
                      'metallic': self.metadata[idx][1]}
            if self.gt_folder is not None:
                img = open_exr(os.path.join(self.gt_folder, f'output_view_{idx}.exr'), self.img_hw).reshape(-1,3)

                sample = {'rays': rays,
                          'rgbs': img,
                          'gt_params': params}
            else:
                rgbs = torch.zeros_like(rays[...,:3])
                sample = {'rays': rays,
                          'rgbs': rgbs,
                          'gt_params': params}

        return sample