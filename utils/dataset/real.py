import torch
import torch.nn.functional as NF
from torch.utils.data import Dataset
import json
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import math
from pathlib import Path
from torch.utils.data import IterableDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.io import load_camera_light_metadata, load_camera_metadata

def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T

def _cv_to_gl(cv):
    # convert to GL convention used in iNGP
    gl = cv * torch.tensor([1, -1, -1, 1])
    return gl

def get_ray_directions(H, W, focal, cx, cy, distortion):
    """ get camera ray direction with radial distortion correction, using opengl convention
    Args:
        H,W: height and width
        focal: focal length
        cx, cy: principal point coordinates
        distortion: radial distortion coefficient k1
    """
    x_coords = torch.linspace(0.5, W - 0.5, W)
    y_coords = torch.linspace(0.5, H - 0.5, H)
    j, i = torch.meshgrid([y_coords, x_coords])
    
    # Convert to normalized coordinates relative to principal point
    x_norm = (i - cx) / focal
    y_norm = (j - cy) / focal
    
    # Apply radial distortion correction
    r_squared = x_norm**2 + y_norm**2
    distortion_factor = 1 + distortion * r_squared
    
    x_corrected = x_norm * distortion_factor
    y_corrected = y_norm * distortion_factor
    
    directions = torch.stack([x_corrected, -y_corrected, -torch.ones_like(i)], -1)

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
    c2w = _cv_to_gl(c2w)
    c2w = c2w[:3,:4]
    return c2w

def get_c2w_from_robot_pose(camera_info, R_c2g, t_c2g):
    """ get camera to world matrix from robot pose """
    g2w = build_4x4(camera_info["rotation_matrix"], camera_info["position"])
    c2g = build_4x4(R_c2g, t_c2g)
    c2w = g2w @ c2g
    return torch.from_numpy(c2w[:3, :4]).float()

class RealImageDataset(IterableDataset):
    """ training dataset that loads images from metadata, returns sampled rays with emitter IDs"""
    def __init__(self, cfg, gt_folder, split):
        self.cfg = cfg
        self.pixel = True
        self.rays_num = cfg.data.rays_num
        self.num_view_batch = cfg.renderer.camera.views_per_batch
        self.importance_sampling = cfg.data.importance_sampling
        self.use_single_chunk_sampling = cfg.data.use_single_chunk_sampling
        self.multi_resolution = cfg.data.multi_resolution
        self.downsample_iter = cfg.data.downsample_iter
        self.gt_folder = gt_folder
        self.debug = cfg.data.debug
        self.debug_num = cfg.data.debug_num
        self.intrinsics = cfg.renderer.camera.intrinsics
        self.cx = self.intrinsics['cx']
        self.cy = self.intrinsics['cy']
        self.distortion = self.intrinsics['distortion']
        self.focal = self.intrinsics['focal_length']
        self.img_hw = (self.intrinsics['height'], self.intrinsics['width'])
        
        # get R_c2g and t_c2g from cfg
        self.R_c2g = cfg.renderer.camera.R_c2g
        self.t_c2g = cfg.renderer.camera.t_c2g
        
        # Load metadata from JSON file
        metadata_path = cfg.data.metadata_path
        camera_metadata_path = cfg.data.camera_metadata_path
        metadata, _, _= load_camera_light_metadata(metadata_path)
        camera_metadata = load_camera_metadata(camera_metadata_path)
        
        # Filter out metadata entries with non-existent image files
        valid_metadata = []
        for item in metadata:
            file_name = item["filename"]
            img_path = os.path.join(gt_folder, file_name)
            if os.path.exists(img_path):
                valid_metadata.append(item)
            else:
                print(f"Warning: Image file {img_path} does not exist, skipping from metadata...")
        
        metadata = valid_metadata
        print(f"Loaded {len(metadata)} valid images out of {len(metadata) + len([item for item in metadata if not os.path.exists(os.path.join(gt_folder, item['filename']))])} total metadata entries")
        self.camera_metadata = camera_metadata
        
        self.total_images = len(metadata)
        
        # Split metadata into training and validation sets with fixed random seed
        torch.manual_seed(42)  # Fixed seed for reproducible splits
        indices = torch.randperm(self.total_images)
        
        # Use 80% for training
        if self.debug:
            self.metadata = metadata[:self.debug_num]
        else:
            split_idx = int(0.8 * self.total_images)
            selected_indices = indices[:split_idx]
            
            # Filter metadata based on split
            self.metadata = [metadata[i] for i in selected_indices] 
        #self.metadata = [item for idx, item in enumerate(self.metadata) if idx % 10 == 0]#temporal modification
        self.set_step(0)
        # self.directions = get_ray_directions(self.img_hw[0], self.img_hw[1], self.focal, self.cx, self.cy, self.distortion)
        # self.all_rays, self.all_rgbs, self.all_emitter_ids, self.all_pdf = self.preload_rays_and_rgbs(downsample_scale=1)   
        
    def preload_rays_and_rgbs(self, downsample_scale=1):
        all_rays = []
        all_rgbs = []
        all_emitter_ids = []
        all_camera_ids = []
        for img_data in tqdm(self.metadata, desc="Loading images and rays"):
            # Get camera info using camera_id
            overall_id = img_data["overall_id"]
            camera_info = self.camera_metadata[str(overall_id)]
            camera_dict = {
                "position": camera_info["position"],
                "rotation_matrix": camera_info["rotation_matrix"],
            }
            
            # Generate rays for this camera
            # c2w = get_c2w_from_robot_pose(camera_dict, self.R_c2g, self.t_c2g)
            c2w = torch.from_numpy(build_4x4(camera_dict["rotation_matrix"], camera_dict["position"])).float()
            c2w = _cv_to_gl(c2w)[:3, :4]
            rays_o, rays_d, dxdu, dydv = get_rays(self.directions, c2w, focal=self.intrinsics['focal_length'])
            rays = torch.cat([rays_o, rays_d, dxdu, dydv], dim=-1)
            # Load original RGB image (without gamma correction)
            file_name = img_data["filename"]
            img_path = os.path.join(self.gt_folder, file_name)
            
            if img_path.endswith('.png'):
                # Load PNG image (original linear RGB)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Warning: Could not load image {img_path}, skipping...")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).float() / 255.0
            elif img_path.endswith('.exr'):
                # Load EXR image (already linear)
                img = open_exr(img_path, self.img_hw)
            else:
                raise ValueError(f"Unsupported image format: {img_path}")
            
            img_flat = img.reshape(-1, 3)
            # Downsample the image if needed
            if downsample_scale > 1:
                # Reshape to image format for downsampling
                img_reshaped = img_flat.reshape(self.img_hw[0], self.img_hw[1], 3)
                # Downsample using average pooling
                img_downsampled = torch.nn.functional.avg_pool2d(
                    img_reshaped.permute(2, 0, 1).unsqueeze(0),  # [1, 3, H, W]
                    kernel_size=downsample_scale,
                    stride=downsample_scale
                ).squeeze(0).permute(1, 2, 0)  # [H', W', 3]
                img_flat = img_downsampled.reshape(-1, 3)
            
            # Get emitter ID directly from metadata
            emitter_id = img_data["emitter_id"]
            emitter_ids = torch.full((rays.shape[0],), emitter_id, dtype=torch.long)
            camera_ids = torch.full((rays.shape[0],), int(overall_id), dtype=torch.long)
            # Filter out low luminance rays
            luminance = (0.2126 * img_flat[..., 0] +
                        0.7152 * img_flat[..., 1] +
                        0.0722 * img_flat[..., 2])
            luminance_threshold = 1e-7
            valid_mask = luminance > luminance_threshold
            
            # Apply mask to filter out low luminance rays
            rays = rays[valid_mask]
            img_flat = img_flat[valid_mask]
            emitter_ids = emitter_ids[valid_mask]
            camera_ids = camera_ids[valid_mask]
            all_rays.append(rays)
            all_rgbs.append(img_flat)
            all_emitter_ids.append(emitter_ids)    
            all_camera_ids.append(camera_ids)
        
        # Concatenate all data
        rays = torch.cat(all_rays)
        rgbs = torch.cat(all_rgbs)
        emitter_ids = torch.cat(all_emitter_ids)
        camera_ids = torch.cat(all_camera_ids)
        # Calculate luminance for importance sampling
        luminance = (0.2126 * rgbs[..., 0] +
                    0.7152 * rgbs[..., 1] +
                    0.0722 * rgbs[..., 2])         # (N_tot,)
        luminance = torch.abs(luminance)
        pdf = luminance.detach() / luminance.sum()               # p_i  (no grad)

        
        # Randomly permute the data
        perm_indices = torch.randperm(rays.shape[0])
        rays = rays[perm_indices]
        rgbs = rgbs[perm_indices]
        emitter_ids = emitter_ids[perm_indices]
        camera_ids = camera_ids[perm_indices]
        return (rays, rgbs, camera_ids, emitter_ids, pdf)

    def sampler(self, rgbs_gt):
        """Set the importance sampler to use for ray sampling"""
        if self.importance_sampling:
            # --------------------------------------------------------------
            # 5-A  Build luminance-pdf and draw importance samples
            # --------------------------------------------------------------
            # pdf = luminance.detach()
            pdf = self.all_pdf
            if not torch.isfinite(pdf).all():                               # all-black fallback
                pdf = torch.full_like(pdf, 1.0 / pdf.numel())

            N_sample = self.rays_num
            # Use chunked sampling to avoid memory issues with large datasets
            chunk_size = min(15000000, len(pdf))  # Process in chunks of 10M or less
            sample_idx = []
            
            if len(pdf) <= chunk_size:
                # Small enough to sample directly
                sample_idx = torch.multinomial(pdf, N_sample, replacement=True)
            else:
                # Option 1: Sample from all chunks (original behavior)
                # Option 2: Sample from one random chunk only (to reduce cost)
                use_single_chunk = getattr(self, 'use_single_chunk_sampling', False)
                
                if use_single_chunk:
                    # Randomly select one chunk and sample all rays from it
                    num_chunks = (len(pdf) + chunk_size - 1) // chunk_size  # Ceiling division
                    selected_chunk_idx = torch.randint(0, num_chunks, (1,)).item()
                    
                    start_idx = selected_chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(pdf))
                    chunk_pdf = pdf[start_idx:end_idx]
                    
                    # Skip if chunk has all zero pdf values
                    if chunk_pdf.sum() == 0:
                        # Fallback to uniform sampling from this chunk
                        chunk_indices = torch.randint(0, len(chunk_pdf), (N_sample,))
                    else:
                        chunk_indices = torch.multinomial(chunk_pdf, N_sample, replacement=True)
                    
                    # Adjust indices to global indexing
                    sample_idx = chunk_indices + start_idx
                else:
                    # Original behavior: sample from all chunks
                    num_chunks = (len(pdf) + chunk_size - 1) // chunk_size  # Ceiling division
                    samples_per_chunk = N_sample // num_chunks
                    remaining_samples = N_sample % num_chunks  # Extra samples for last chunk
                    
                    # Loop over each chunk
                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, len(pdf))
                        chunk_pdf = pdf[start_idx:end_idx]
                        
                        # Skip chunks where all pdf values are 0
                        if chunk_pdf.sum() == 0:
                            continue
                        
                        # Determine number of samples for this chunk
                        if chunk_idx == num_chunks - 1:
                            # Last chunk gets remaining samples
                            chunk_samples = samples_per_chunk + remaining_samples
                        else:
                            chunk_samples = samples_per_chunk
                        
                        if chunk_samples > 0:
                            chunk_indices = torch.multinomial(chunk_pdf, chunk_samples, replacement=True)
                            # Adjust indices to global indexing
                            global_indices = chunk_indices + start_idx
                            sample_idx.append(global_indices)
                    
                    sample_idx = torch.cat(sample_idx) if sample_idx else torch.empty(0, dtype=torch.long)
                    
                    # If we still need more samples, fill remaining with uniform sampling
                    if len(sample_idx) < N_sample:
                        remaining = N_sample - len(sample_idx)
                        uniform_indices = torch.randint(0, len(pdf), (remaining,))
                        sample_idx = torch.cat([sample_idx, uniform_indices])
            
            return sample_idx, pdf[sample_idx] * len(pdf)
        else:
            # Uniform random sampling
            N_sample = self.rays_num
            sample_idx = torch.randint(0, len(rgbs_gt), (N_sample,), dtype=torch.long)
            pdf = torch.full((N_sample,), 1.0)
            return sample_idx, pdf

    def set_step(self, step):
        a, b = self.downsample_iter[0], self.downsample_iter[1]
        
        # Determine downsample scale based on thresholds
        if a >= 0 and step < a:
            downsample_scale = 4  # Use scale 4 before threshold a
        elif b >= 0 and step < b:
            downsample_scale = 2  # Use scale 2 before threshold b
        else:
            downsample_scale = 1  # Use scale 1 after both thresholds
        
        # Reload data if downsample scale changed
        if not hasattr(self, '_current_downsample_scale') or self._current_downsample_scale != downsample_scale:
            self._current_downsample_scale = downsample_scale
            # Clear GPU memory if attributes exist
            if hasattr(self, 'directions'):
                del self.directions
            if hasattr(self, 'all_rays'):
                del self.all_rays
            if hasattr(self, 'all_rgbs'):
                del self.all_rgbs
            if hasattr(self, 'all_emitter_ids'):
                del self.all_emitter_ids
            if hasattr(self, 'all_pdf'):
                del self.all_pdf
            # Update directions with proper downsampling
            print(f"Loading rays with downsample scale {downsample_scale}...")
            h, w = self.img_hw
            h_down, w_down = h // downsample_scale, w // downsample_scale
            self.directions = get_ray_directions(h_down, w_down, self.focal, self.cx, self.cy, self.distortion) 
            self.all_rays, self.all_rgbs, self.all_camera_ids, self.all_emitter_ids, self.all_pdf = self.preload_rays_and_rgbs(downsample_scale=downsample_scale)

    def __iter__(self):
        while True:
            if self.sampler is not None:
            #if False:
                # Use importance sampler
                ray_indices, pdf = self.sampler(self.all_rgbs)
            else:
                # Fallback to uniform sampling
                ray_indices = torch.randint(0, len(self.all_rays), (self.rays_num,))
                pdf = torch.ones(self.rays_num) / len(self.all_rays)

            rays = self.all_rays[ray_indices]
            rgbs = self.all_rgbs[ray_indices]
            emitter_ids = self.all_emitter_ids[ray_indices]
            camera_ids = self.all_camera_ids[ray_indices]

            yield {
                'rays': rays,
                'rgbs': rgbs,
                'emitter_ids': emitter_ids,
                'pdf': pdf,
                'gt_params': torch.zeros(1),
                'camera_ids': camera_ids,
            }


class RealValDataset(Dataset):
    """ validation dataset that loads images from metadata, returns complete images """
    def __init__(self, cfg, gt_folder):
        self.cfg = cfg
        self.pixel = False
        self.gt_folder = gt_folder
        self.debug = cfg.data.debug
        self.debug_num = cfg.data.debug_num
        self.intrinsics = cfg.renderer.camera.intrinsics
        self.cx = self.intrinsics['cx']
        self.cy = self.intrinsics['cy']
        self.distortion = self.intrinsics['distortion']
        self.focal = self.intrinsics['focal_length']
        self.img_hw = (self.intrinsics['height'], self.intrinsics['width'])
        
        # get R_c2g and t_c2g from cfg
        self.R_c2g = cfg.renderer.camera.R_c2g
        self.t_c2g = cfg.renderer.camera.t_c2g
        
        # Load metadata from JSON file
        metadata_path = cfg.data.metadata_path
        camera_metadata_path = cfg.data.camera_metadata_path
        metadata, _, _ = load_camera_light_metadata(metadata_path)
        camera_metadata = load_camera_metadata(camera_metadata_path)
        # Filter out metadata entries with non-existent image files
        valid_metadata = []
        for item in metadata:
            file_name = item["filename"]
            img_path = os.path.join(gt_folder, file_name)
            if os.path.exists(img_path):
                valid_metadata.append(item)
            else:
                print(f"Warning: Image file {img_path} does not exist, skipping from metadata...")
        metadata = valid_metadata
        self.camera_metadata = camera_metadata    
        self.total_images = len(metadata)
        
        # Split metadata into training and validation sets with fixed random seed
        if self.debug:
            self.metadata = metadata[:self.debug_num]
        else:
            torch.manual_seed(42)  # Fixed seed for reproducible splits
            indices = torch.randperm(self.total_images)
            # Use 20% for validation
            split_idx = int(0.8 * self.total_images)
            selected_indices = indices[split_idx:]
        
            # Filter metadata based on split
            self.metadata = [metadata[i] for i in selected_indices]   
        self.directions = get_ray_directions(self.img_hw[0], self.img_hw[1], self.focal, self.cx, self.cy, self.distortion)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_data = self.metadata[idx]
        
        # Get camera info using camera_id
        overall_id = img_data["overall_id"]
        camera_info = self.camera_metadata[str(overall_id)]
        camera_dict = {
            "position": camera_info["position"],
            "rotation_matrix": camera_info["rotation_matrix"],
        }
        
        # Generate rays for this camera
        # c2w = get_c2w_from_robot_pose(camera_dict, self.R_c2g, self.t_c2g)
        c2w = torch.from_numpy(build_4x4(camera_dict["rotation_matrix"], camera_dict["position"])).float()
        c2w = _cv_to_gl(c2w)[:3, :4]
        rays_o, rays_d, dxdu, dydv = get_rays(self.directions, c2w, focal=self.focal)
        rays = torch.cat([rays_o, rays_d, dxdu, dydv], dim=-1)
        
        # Load original RGB image (without gamma correction)
        img_path = os.path.join(self.gt_folder, img_data["filename"])
                
        if img_path.endswith('.png'):
            # Load PNG image (original linear RGB)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float() / 255.0
        elif img_path.endswith('.exr'):
            # Load EXR image (already linear)
            img = open_exr(img_path, self.img_hw)
        else:
            raise ValueError(f"Unsupported image format: {img_path}")
        
        img_flat = img.reshape(-1, 3)
        
        # Get emitter ID directly from metadata
        emitter_ids = torch.full((rays.shape[0],), img_data["emitter_id"], dtype=torch.long)
        camera_ids = torch.full((rays.shape[0],), int(overall_id), dtype=torch.long)
        return {
            'rays': rays,
            'rgbs': img_flat,
            'emitter_ids': emitter_ids,
            'gt_params': torch.zeros(1),
            'camera_ids': camera_ids,
        }