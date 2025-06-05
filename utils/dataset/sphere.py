import torch
import torch.nn.functional as NF
from torchvision import transforms as TF
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

def load_pbr_texture_stack(pbr_folder):
    def read_img(fname):
        path = os.path.join(pbr_folder, fname)
        if path.endswith(".exr"):
            exr = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # H × W × C, float32
            # Convert BGR to RGB for OpenCV
            exr = exr[..., ::-1]

            tensor = torch.from_numpy(exr.copy())         # torch.float32
            # Check if it's only two channels and unsqueeze if needed
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(2)  # Add channel dimension if missing
            return tensor.permute(2, 0, 1)                # [C,H,W]
        else:
            img = Image.open(path).convert('RGB')
            tensor = T.ToTensor()(img).float()
            return tensor if tensor.max() <= 1.0 else tensor / 255.0

    # Collect PBR texture data
    try:
        # Read all texture maps based on the image
        col_1 = read_img("fabric_pattern_07_col_1_4k.jpg")  # [3,H,W] - Color map
        ao = read_img("fabric_pattern_07_ao_4k.jpg")        # [3,H,W] - Ambient occlusion
        arm = read_img("fabric_pattern_07_arm_4k.jpg")      # [3,H,W] - (A, Roughness, Metalness)
        rough = read_img("fabric_pattern_07_rough_4k.exr")  # [1,H,W] - Roughness map (EXR)
        nor_dx = read_img("fabric_pattern_07_nor_dx_4k.exr") # [1,H,W] - Normal map X
        nor_gl = read_img("fabric_pattern_07_nor_gl_4k.exr") # [1,H,W] - Normal map GL
        
        
        # Combine all available channels into a texture
        # [Color (3) + AO (3) + ARM (3) + Roughness (1) + Normal DX (1) + Normal GL (1)] = 12 channels
        tex = torch.cat([
            col_1,                # RGB color (3 channels) 0:3
            ao,                   # Ambient occlusion (3 channels) 3:6
            arm,                  # ARM texture (3 channels) 6:9
            rough,                # Roughness map (1 channel) 9:10
            nor_dx,               # Normal map X (3 channel) 10:13
            nor_gl                # Normal map GL (3 channel) 13:16
        ], dim=0)  # [16,H,W]
        
        return tex.permute(1, 2, 0).contiguous()  # [H,W,6] => [U,V,6]
    except Exception as e:
        print(f"Error loading PBR textures: {e}")
        # Return a default texture if loading fails
        H, W = 1024, 1024
        return torch.ones(H, W, 6)

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

class SphereIterableDataset(IterableDataset):
    """ training dataset, return random view and pixel-level rays"""
    def __init__(self, cfg, gt_folder, split):
        self.cfg = cfg
        self.pixel = True
        self.rays_num = cfg.data.rays_num
        self.num_view_batch = 4

        # Load metadata
        self.gt_folder = gt_folder
        self.img_hw = cfg.renderer.resolution
        h, w = self.img_hw
        self.camera_angle_x = cfg.renderer.camera.camera_angle_x
        self.focal = (0.5 * w / np.tan(0.5 * self.camera_angle_x)).item()
        self.directions = get_ray_directions(h, w, self.focal)
        self.number_of_views = cfg.renderer.camera.number_of_views
        self.distance = cfg.renderer.camera.distance
        self.get_camera_dicts()
        self.all_rays, self.all_rgbs = self.preload_rays_and_rgbs()
        self.pbr_texture = load_pbr_texture_stack(self.cfg.data.pbr_path)

    
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

    def get_camera_dicts(self):
        # Initialize camera dicts list
        self.camera_dict = []
        
        # Keep look_at and up vectors fixed from initial camera settings
        look_at = self.cfg.renderer.camera.look_at
        up = self.cfg.renderer.camera.up
        dist = self.distance
        
        # Parameters to control sampling density based on number_of_views
        # We want approximately self.number_of_views total camera positions
        # Formula: total = (n_theta-2) * n_phi + 2 (for poles)
        # So: (n_theta-2) * n_phi = self.number_of_views - 2
        
        # Choose n_phi and calculate n_theta accordingly
        n_phi = max(4, int(np.sqrt(self.number_of_views - 2)))  # At least 4 phi samples
        n_theta = max(3, int((self.number_of_views - 2) / n_phi) + 2)  # At least 3 theta samples (excluding poles)
        
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
            # for idx in torch.randperm(len(self.metadata)):
            view_indices = torch.randint(0, self.total, (self.num_view_batch,))
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
                'gt_params': self.pbr_texture
            }

class SphereValDataset(Dataset):
    """ validation dataset, return random view """
    def __init__(self, cfg, gt_folder):
        self.cfg = cfg
        self.pixel = False
        self.gt_folder = gt_folder
        self.number_of_views = cfg.renderer.camera.number_of_view_test

        self.img_hw = cfg.renderer.resolution
        h, w = self.img_hw
        self.camera_angle_x = cfg.renderer.camera.camera_angle_x
        self.focal = (0.5 * w / np.tan(0.5 * self.camera_angle_x)).item()
        self.directions = get_ray_directions(h, w, self.focal)
        self.distance = cfg.renderer.camera.distance
        self.get_camera_rotation_dicts()
        self.pbr_texture = load_pbr_texture_stack(self.cfg.data.pbr_path)

    # def get_camera_dicts(self):
    #     # Initialize camera dicts list
    #     self.camera_dict = []
        
    #     # Keep look_at and up vectors fixed from initial camera settings
    #     look_at = self.cfg.renderer.camera.look_at
    #     up = self.cfg.renderer.camera.up
    #     dist = self.distance
        
    #     # Parameters to control sampling density
    #     # Choose n_phi and calculate n_theta accordingly
    #     n_phi = max(4, int(np.sqrt(self.number_of_views - 2)))  # At least 4 phi samples
    #     n_theta = max(3, int((self.number_of_views - 2) / n_phi) + 2)  # At least 3 theta samples (excluding poles)
        
    #     self.total = (n_theta-2) * n_phi + 2  # Add 2 for poles
        
    #     # Generate uniform samples for spherical coordinates, excluding poles
    #     thetas = np.linspace(0, np.pi, n_theta)  # Exclude 0 and pi
    #     thetas = thetas[1:-1]  # Remove the first and last elements
    #     phis = np.linspace(0, 2*np.pi, n_phi)
        
    #     # Add poles separately - they only need one phi value since they're at top/bottom
        
    #     # Create grid of angles
    #     theta_grid, phi_grid = np.meshgrid(thetas, phis)
    #     thetas_flat = theta_grid.flatten()
    #     phis_flat = phi_grid.flatten()
        
    #     # Convert spherical to cartesian coordinates
    #     for theta, phi in zip(thetas_flat, phis_flat):
    #         # Calculate camera position
    #         x = dist * np.sin(theta) * np.cos(phi)
    #         y = dist * np.sin(theta) * np.sin(phi) 
    #         z = dist * np.cos(theta)
            
    #         # Create camera dict for this position
    #         camera_dict = {
    #             "position": [x, y, z],
    #             "look_at": look_at,
    #             "up": up
    #         }
            
    #         self.camera_dict.append(camera_dict)
    #     north_pole = {
    #         "position": [0, 0, dist],  # x=0, y=0, z=dist
    #         "look_at": look_at,
    #         "up": up
    #     }
    #     self.camera_dict.append(north_pole)

    #     south_pole = {
    #         "position": [0, 0, -dist],  # x=0, y=0, z=-dist
    #         "look_at": look_at,
    #         "up": up
    #     }
    #     self.camera_dict.append(south_pole)

    def get_camera_rotation_dicts(self):
        # Initialize camera dicts list  
        self.camera_dict = []
        
        look_at = self.cfg.renderer.camera.look_at
        up = self.cfg.renderer.camera.up
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
        return len(self.camera_dict)

    def __getitem__(self, idx):
        c2w = get_c2w(self.camera_dict[idx])
        rays_o, rays_d, dxdu, dydv = get_rays(self.directions, c2w, focal=self.focal)
        rays = torch.cat([rays_o, rays_d, dxdu, dydv], dim=-1)
        if self.gt_folder is not None:
            img = open_exr(os.path.join(self.gt_folder, f'output_view_{idx}.exr'), self.img_hw).reshape(-1, 3)
        else:
            img = torch.zeros_like(rays[...,:3])
        return {
            'rays': rays,
            'rgbs': img,
            'gt_params': self.pbr_texture
        }
    
class SphereTestDataset(Dataset):
    """  test dataset, return all views """
    def __init__(self, cfg, gt_folder, split):
        self.cfg = cfg
        self.gt_folder = gt_folder
        self.img_hw = cfg.renderer.resolution
        h, w = self.img_hw
        self.number_of_views = cfg.renderer.camera.number_of_view_test
        self.distance = cfg.renderer.camera.distance
        self.camera_angle_x = cfg.renderer.camera.camera_angle_x
        self.focal = (0.5 * w / np.tan(0.5 * self.camera_angle_x)).item()
        self.directions = get_ray_directions(h, w, self.focal)
        self.get_camera_rotation_dicts()
        self.metadata = self._load_metadata(f"metadata/dual_{split}.txt")

    def _load_metadata(self, path):
        metadata = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    values = list(map(float, line.strip().split()))
                    if len(values) == 2:
                        r, m = values
                        metadata.append((r, m))
                    elif len(values) == 4:  # Support for dual parameter sets
                        r1, m1, r2, m2 = values
                        metadata.append((r1, m1, r2, m2))
        return metadata

    def get_camera_rotation_dicts(self):
        # Initialize camera dicts list  
        self.camera_dict = []
        
        look_at = self.cfg.renderer.camera.look_at
        up = self.cfg.renderer.camera.up
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
        params = self.metadata[idx]
        if len(params) == 2:
            r, m = params
            gt_params = {'roughness': torch.tensor(r, dtype=torch.float32),
                         'metallic': torch.tensor(m, dtype=torch.float32)}
        else:  # Handle dual parameter case
            r1, m1, r2, m2 = params
            gt_params = {'roughness': torch.tensor([r1, r2], dtype=torch.float32),
                         'metallic': torch.tensor([m1, m2], dtype=torch.float32)}
            
        all_views = []
        for view_idx, cam_dict in enumerate(self.camera_dict):
            c2w = get_c2w(cam_dict)
            rays_o, rays_d, dxdu, dydv = get_rays(self.directions, c2w, focal=self.focal)
            rays = torch.cat([rays_o, rays_d, dxdu, dydv], dim=-1)
            if self.gt_folder:
                img_path = os.path.join(self.gt_folder, f'output_{idx}_view_{view_idx}.exr')
                if os.path.exists(img_path):
                    img = open_exr(img_path, self.img_hw).reshape(-1, 3)
                else:
                    img = torch.zeros_like(rays[..., :3])
            else:
                img = torch.zeros_like(rays[..., :3])
            
            view_data = {
                'rays': rays,
                'rgbs': img,
                'gt_params': gt_params
            }
            all_views.append(view_data)
        
        return all_views
