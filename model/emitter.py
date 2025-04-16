import torch
import torch.nn as nn
import torch.nn.functional as NF
import numpy as np
import math
import imageio
import cv2
from .slf import VoxelSLF
from openexr_numpy import imread, imwrite

class AreaEmitter(nn.Module):
    """ triangle mesh emitters """
    def __init__(self,emitter_path):
        """ emitter_path file 
        is_emitter: B indicator of whether a triangle is emitter
        emitter_vertices: Kx3x3 triangle vertices of emitters
        emitter_area: K surface areas of emitters
        emitter_radiance: Bx3x3 emitter radiance
        """
        super(AreaEmitter,self).__init__()
        
        weight = torch.load(emitter_path,map_location='cpu')
        
        is_emitter = weight['is_emitter']
        emitter_vertices = weight['emitter_vertices']
        emitter_area = weight['emitter_area']
        emitter_radiance = weight['emitter_radiance']

        self.register_buffer('is_emitter',is_emitter)
        self.register_buffer('emitter_vertices',emitter_vertices)
        self.register_buffer('emitter_area',emitter_area)
        self.register_buffer('radiance',emitter_radiance)
        
        # emitter idx mapping, -1 indicates not an emitter
        emitter_idx = torch.full((len(is_emitter),),-1,device=is_emitter.device,dtype=torch.long)
        emitter_idx[is_emitter] = torch.arange(is_emitter.sum(),device=is_emitter.device)
        self.register_buffer('emitter_idx',emitter_idx)
        
        # emitter idx to triangle idx
        triangle_idx = torch.arange(len(is_emitter))[is_emitter]
        self.register_buffer('triangle_idx',triangle_idx)
        
        # sample emitters uniformly
        emitter_pdf = NF.normalize(torch.ones_like(emitter_area),dim=-1,p=1)
        emitter_cdf = emitter_pdf.cumsum(-1).contiguous()
        self.register_buffer('emitter_pdf',emitter_pdf)
        self.register_buffer('emitter_cdf',emitter_cdf)
    
    def forward(self,triangle_idx):
        """ get emitter radiance
        triangle_idx: B triangle indices
        """
        vis = triangle_idx != -1 # whether a valid triangle

        is_area = self.is_emitter[triangle_idx]&vis
        Le = torch.zeros(position.shape[0],3,device=position.device)
        if is_area.any():
            e_idx = self.emitter_idx[triangle_idx[is_area]]
            Le[is_area] = self.radiance[e_idx]
        
        # assume zero background lighting
        Le = Le*vis[...,None]
        return Le
    
    def eval_emitter(self, position,light_dir,triangle_idx,*args):
        """ evaluate surface emission and pdf
        Args:
            position: Bx3 intersection location
            light_dir: Bx3 emission direction
            triangle_idx: B intersected triangle id
        Return:
            Le: Bx3 radiance
            emit_pdf: Bx1 emitter pdf
            valid_next: B valid surface
        """
        # whether valid intersection
        vis = triangle_idx != -1

        # get area light
        is_area = self.is_emitter[triangle_idx]&vis

        Le = torch.zeros(position.shape[0],3,device=position.device)
        emit_pdf = torch.zeros(position.shape[0],device=position.device)
        if is_area.any():
            e_idx = self.emitter_idx[triangle_idx[is_area]]
            emit_pdf[is_area] = self.emitter_pdf[e_idx]/self.emitter_area[e_idx].clamp_min(1e-12)
            Le[is_area] = self.radiance[e_idx]

        # assume zero background lighting
        Le = Le*vis[...,None]

        # next: not area light or background
        valid_next = (~is_area)&vis
        return Le,emit_pdf.unsqueeze(-1),valid_next
    
    def sample_emitter(self,sample1,sample2,position):
        """ importance sampling emitters
        Args:
            sample1: B uniform samples
            sample2: Bx2 uniform samples
            position: Bx3 surfae location
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1 the sampling pdf (in area space)
            triangle_idx: B the sampled triangle id
        """
        # pick an emitter
        emitter_idx = torch.searchsorted(self.emitter_cdf,sample1.clamp_min(1e-12))
        pdf0 = self.emitter_pdf[emitter_idx]

        # unifromly sample points on triangles
        xi1 = sample2[...,0].sqrt()
        u = (1-xi1).unsqueeze(-1)
        v = (xi1*sample2[...,1]).unsqueeze(-1)
        w = 1-u-v

        # emitter area
        A1 = self.emitter_area[emitter_idx]
        # sampled location on triangle
        p1 = self.emitter_vertices[emitter_idx]
        p1 = p1[:,0]*u + p1[:,1]*v + p1[:,2]*w
        wi = NF.normalize(p1-position,dim=-1)
        triangle_idx = self.triangle_idx[emitter_idx]
        
        # pdf in area space
        pdf = pdf0/A1.clamp_min(1e-12)
        return wi,pdf.unsqueeze(-1),triangle_idx
    

class SLFEmitter(nn.Module):
    """ triangle emitters with diffuse radiance cache """
    def __init__(self,emitter_path,slf_path):
        """ 
        emitter_path: emitter parameter file
        slf_path: surface light field paramter file
        """
        super(SLFEmitter,self).__init__()
        
        # load surface light field
        state_dict = torch.load(slf_path,map_location='cpu') 
        self.slf = VoxelSLF(state_dict['mask'],
                            state_dict['voxel_min'],state_dict['voxel_max'])
        self.slf.load_state_dict(state_dict['weight'])
        
        # load emitters
        weight = torch.load(emitter_path,map_location='cpu')
        is_emitter = weight['is_emitter']
        emitter_vertices = weight['emitter_vertices']
        emitter_area = weight['emitter_area']
        emitter_radiance = weight['emitter_radiance']
        self.register_buffer('is_emitter',is_emitter)
        self.register_buffer('emitter_vertices',emitter_vertices)
        self.register_buffer('emitter_area',emitter_area)
        self.register_buffer('radiance',emitter_radiance)
        
        # emitter idx mapping, -1 indicates not an emitter
        emitter_idx = torch.full((len(is_emitter),),-1,device=is_emitter.device,dtype=torch.long)
        emitter_idx[is_emitter] = torch.arange(is_emitter.sum(),device=is_emitter.device)
        self.register_buffer('emitter_idx',emitter_idx)
        
        # emitter idx to triangle idx
        triangle_idx = torch.arange(len(is_emitter))[is_emitter]
        self.register_buffer('triangle_idx',triangle_idx)
        
        # randomly select a emitter
        emitter_pdf = NF.normalize(torch.ones_like(emitter_area),dim=-1,p=1)
        emitter_cdf = emitter_pdf.cumsum(-1).contiguous()
        self.register_buffer('emitter_pdf',emitter_pdf)
        self.register_buffer('emitter_cdf',emitter_cdf)
    
    def forward(self,position):
        """ surface light field from queried location """
        Le = self.slf(position)['rgb']
        return Le
    
    def eval_emitter(self, position,light_dir,triangle_idx,roughness=None):
        """ evaluate surface emission and pdf return radiance cache if diffuse
        Args:
            position: Bx3 intersection location
            light_dir: Bx3 emission direction
            triangle_idx: B intersected triangle id
            roughness: Bx1 surface roughness if not None
        Return:
            Le: Bx3 radiance
            emit_pdf: Bx1 emitter pdf
            valid_next: B valid surface
        """
        # whether valid intersection
        vis = triangle_idx != -1
        
        Le = torch.zeros(position.shape[0],3,device=position.device)
        emit_pdf = torch.zeros(position.shape[0],device=position.device)
        
        # get area light
        is_area = self.is_emitter[triangle_idx]&vis
        if is_area.any():
            e_idx = self.emitter_idx[triangle_idx[is_area]]
            emit_pdf[is_area] = self.emitter_pdf[e_idx]/self.emitter_area[e_idx].clamp_min(1e-12)
            Le[is_area] = self.radiance[e_idx]
        
        # assume zero background lighting
        Le = Le*vis[...,None]
        valid_next = (~is_area)&vis
        
        # check diffuse radiance cache
        if roughness is not None:
            # query the radiance cache and terminate for diffuse and non emissive surface 
            is_diffuse = (~is_area) & vis & (roughness.squeeze(-1)>0.6)
            if is_diffuse.any():
                diffuse_slf = self.slf(position[is_diffuse])['rgb']
                Le[is_diffuse] = diffuse_slf
                is_diffuse[is_diffuse.clone()] = diffuse_slf.sum(-1)>0 # diffuse radiance need to > 0
                valid_next &= (~is_diffuse) # terminate path 

        return Le,emit_pdf.unsqueeze(-1),valid_next
    

    def sample_emitter(self,sample1,sample2,position):
        """ importance sampling emitters
        Args:
            sample1: B uniform samples
            sample2: Bx2 uniform samples
            position: Bx3 surfae location
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1 the sampling pdf (in area space)
            triangle_idx: B the sampled triangle id
        """
        # pick an emitter
        emitter_idx = torch.searchsorted(self.emitter_cdf,sample1.clamp_min(1e-12))
        pdf0 = self.emitter_pdf[emitter_idx]

        # unifromly sample points on triangles
        xi1 = sample2[...,0].sqrt()
        u = (1-xi1).unsqueeze(-1)
        v = (xi1*sample2[...,1]).unsqueeze(-1)
        w = 1-u-v

        # emitter area
        A1 = self.emitter_area[emitter_idx]
        # sampled location on triangle
        p1 = self.emitter_vertices[emitter_idx]
        p1 = p1[:,0]*u + p1[:,1]*v + p1[:,2]*w
        wi = NF.normalize(p1-position,dim=-1)
        triangle_idx = self.triangle_idx[emitter_idx]
        
        # pdf in area space
        pdf = pdf0/A1.clamp_min(1e-12)
        return wi,pdf.unsqueeze(-1),triangle_idx


class PointEmitter(nn.Module):
    """ Point light emitter """
    def __init__(self, position, intensity, radius=0.1):
        """ 
        Args:
            position: 3-element tensor for light position
            intensity: 3-element tensor for RGB intensity
            radius: radius of sphere representing point light
        """
        super(PointEmitter, self).__init__()
        self.register_buffer('position', torch.tensor(position))
        self.register_buffer('intensity', torch.tensor(intensity))
        self.radius = radius

    def ray_sphere_intersect(self, ray_o, ray_d):
        """ Ray-sphere intersection test
        Args:
            ray_o: Bx3 ray origins
            ray_d: Bx3 ray directions (normalized)
        Returns:
            hit_pos: Bx3 intersection points
            normals: Bx3 surface normals
            valid: B whether ray hits sphere
        """
        # Solve quadratic equation for ray-sphere intersection
        oc = ray_o - self.position
        a = (ray_d * ray_d).sum(-1)
        b = 2.0 * (oc * ray_d).sum(-1)
        c = (oc * oc).sum(-1) - self.radius * self.radius
        disc = b * b - 4 * a * c
        
        valid = disc > 0
        t = torch.zeros_like(disc)
        t[valid] = (-b[valid] - torch.sqrt(disc[valid])) / (2.0 * a[valid])
        valid = valid & (t > 0)

        # Compute intersection points and normals
        hit_pos = ray_o + ray_d * t.unsqueeze(-1)
        normals = NF.normalize(hit_pos - self.position, dim=-1)
        
        return hit_pos, normals, valid

    def eval_emitter(self, position, light_dir, *args):
        """ Evaluate point light emission
        Args:
            position: Bx3 intersection points
            light_dir: Bx3 light directions
        Returns:
            Le: Bx3 radiance
            pdf: Bx1 pdf (unused for point light)
            valid: B valid intersections
        """
        # Check if ray hits sphere
        hit_pos, normals, valid = self.ray_sphere_intersect(position, light_dir)
        
        # Compute radiance falloff with distance
        Le = torch.zeros_like(position)
        if valid.any():
            dist2 = (position[valid] - self.position).pow(2).sum(-1, keepdim=True)
            Le[valid] = self.intensity / (4 * math.pi * dist2)
        
        # Sphere lights have uniform pdf over surface area
        pdf = torch.full((position.shape[0], 1), 1.0/(4*math.pi*self.radius*self.radius), device=position.device)
        pdf[~valid] = 0  # Zero pdf for rays that miss the sphere
        
        return Le, pdf, ~valid # Return ~valid since we want to continue tracing for misses

    def sample_emitter(self, sample1, sample2, position):
        """ Sample point on sphere surface
        Args:
            sample1: B uniform samples (unused)
            sample2: Bx2 uniform samples for sphere surface
            position: Bx3 surface positions
        Returns:
            wi: Bx3 sampled directions
            pdf: Bx1 sampling pdf
            idx: B dummy indices (-1)
        """
        # Sample uniform direction from sphere surface
        theta = 2 * math.pi * sample2[...,0]
        phi = torch.arccos(1 - 2 * sample2[...,1])
        
        sin_phi = torch.sin(phi)
        x = sin_phi * torch.cos(theta) 
        y = sin_phi * torch.sin(theta)
        z = torch.cos(phi)
        
        # Point on sphere surface
        sphere_point = self.position + self.radius * torch.stack([x,y,z], dim=-1)
        
        # Direction from position to sampled point
        wi = NF.normalize(sphere_point - position, dim=-1)
        
        # Uniform sampling pdf for sphere surface
        pdf = torch.full((position.shape[0], 1), 1.0/(4*math.pi*self.radius*self.radius), 
                        device=position.device)
        
        # Dummy triangle indices since we don't use mesh
        idx = torch.full((position.shape[0],), -1, dtype=torch.long, device=position.device)
        
        return wi, pdf, idx

class EnvMapEmitter(nn.Module):
    """ Environment map emitter using HDRI """
    def __init__(self, envmap_path):
        """
        Args:
            envmap_path: Path to the .exr or .hdr environment map
        """
        super(EnvMapEmitter, self).__init__()
        
        # Load environment map (assumed to be in lat-long format)
        envmap = imageio.imread(envmap_path).astype('float32')[:,:,:3]  # Shape: (H, W, 3)
        test = imread(envmap_path).astype('float32')
        envmap = torch.from_numpy(envmap).permute(2, 0, 1)  # Convert to (3, H, W)
        
        self.register_buffer('envmap', envmap)
        self.H, self.W = envmap.shape[1:]  # Get resolution
        self.save_envmap_as_image(self.envmap, 'envmap.png')
        
        # Compute importance sampling weights (optional)
        self.importance_weights = self.compute_importance_weights()

    def get_emitter_dicts(self):
        # Initialize camera dicts list
        self.emitter_dict = []
        
        # Keep look_at and up vectors fixed from initial camera settings
        look_at = self.initial_camera_dict["look_at"]
        up = self.initial_camera_dict["up"]
        dist = self.distance
        
        # Parameters to control sampling density
        n_theta = 8  # number of theta samples 
        n_phi = 4    # number of phi samples
        self.total = n_theta * n_phi
        
        # Generate uniform samples for spherical coordinates
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2*np.pi, n_phi)
        
        # Create grid of angles
        theta_grid, phi_grid = np.meshgrid(thetas, phis)
        thetas_flat = theta_grid.flatten()
        phis_flat = phi_grid.flatten()
    
    def save_envmap_as_image(self, envmap_tensor, filename="envmap.png"):
        """
        Convert HDR environment map to an 8-bit RGB image and save it.

        Args:
            envmap_tensor: (3, H, W) PyTorch tensor in HDR format
            filename: Output image file name
        """
        envmap_np = envmap_tensor.cpu().numpy()  # Convert to NumPy (3, H, W)

        # Normalize to [0,1] using simple exposure adjustment
        # envmap_np = envmap_np / (envmap_np.max() + 1e-6)  # Avoid division by zero
        
        # Apply gamma correction (optional, gamma = 2.2 for display)
        env_map = np.clip(envmap_np, 0, 1)
        envmap_np = env_map ** (1 / 2.2)

        # Convert to 8-bit (0-255)
        envmap_8bit = (envmap_np * 255).astype(np.uint8)

        # Transpose from (3, H, W) to (H, W, 3) for image saving
        envmap_8bit = np.transpose(envmap_8bit, (1, 2, 0))

        # Save the image
        imageio.imwrite(filename, envmap_8bit)
        print(f"Saved environment map as {filename}")

    def compute_importance_weights(self):
        """ Compute importance sampling weights from HDR map luminance """
        # Convert RGB to luminance (simple approximation)
        luminance = 0.2126 * self.envmap[0] + 0.7152 * self.envmap[1] + 0.0722 * self.envmap[2]
        return luminance / luminance.sum()  # Normalize

    def sample_emitter(self, sample1, sample2, position):
        """
        Sample a direction from the environment map
        Args:
            sample1: B uniform samples (unused)
            sample2: Bx2 uniform samples for spherical sampling
            position: Bx3 surface positions (unused)
        Returns:
            wi: Bx3 sampled directions
            pdf: Bx1 sampling pdf
            idx: B dummy indices (-1)
        """
        # Convert uniform samples to spherical coordinates
        phi = 2 * math.pi * sample2[..., 0]  # Azimuth
        theta = torch.acos(1 - 2 * sample2[..., 1])  # Elevation
        
        sin_theta = torch.sin(theta)
        x = sin_theta * torch.cos(phi)
        y = sin_theta * torch.sin(phi)
        z = torch.cos(theta)
        
        wi = torch.stack([x, y, z], dim=-1)  # Direction vectors
        
        # Compute PDF (uniform for now)
        pdf = torch.full((position.shape[0], 1), 1.0 / (4 * math.pi), device=position.device)
        
        idx = torch.full((position.shape[0],), -1, dtype=torch.long, device=position.device)
        
        return wi, pdf, idx
    def importance_sample_emitter(self, sample1, sample2, position):
        """
        Sample a direction from the environment map using weighted sampling.
        """
        B = position.shape[0]
        
        # Flatten PDF and get CDF
        flat_pdf = self.envmap_pdf.flatten()
        flat_cdf = torch.cumsum(flat_pdf, dim=0)
        
        # Invert CDF sampling using sample2[..., 0]
        u = sample2[..., 0].clamp(0, 1 - 1e-6)
        indices = torch.searchsorted(flat_cdf, u)

        # Convert 1D index to 2D pixel (v, u)
        v_idx = indices // self.W
        u_idx = indices % self.W

        # Convert to [0, 1] continuous UV (center of pixel)
        u = (u_idx.float() + 0.5) / self.W
        v = (v_idx.float() + 0.5) / self.H

        # Map [u,v] in [0,1]² to 3D direction using Equal-Area Octahedral Mapping
        # Shift to [-1, 1]
        u_ = 2 * u - 1
        v_ = 2 * v - 1

        abs_u = u_.abs()
        abs_v = v_.abs()
        signed_dist = 1 - (abs_u + abs_v)
        d = signed_dist.abs()
        r = 1 - d
        phi = (v_ - u_) / (r + 1e-6) + 1
        phi = phi * math.pi / 4
        z = torch.sign(signed_dist) * (1 - r * r)

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        x = cos_phi * r * torch.sqrt(torch.clamp(2 - r * r, min=1e-6))
        y = sin_phi * r * torch.sqrt(torch.clamp(2 - r * r, min=1e-6))
        
        wi = torch.stack([x, y, z], dim=-1)  # (B, 3)
        wi = torch.nn.functional.normalize(wi, dim=-1)

        # PDF from envmap
        pdf_vals = flat_pdf[indices].unsqueeze(-1)  # (B, 1)

        idx = torch.full((B,), -1, dtype=torch.long, device=position.device)
        return wi, pdf_vals, idx

    def ochmap_eval_emitter(self, position, light_dir, *args):
        """
        Evaluate environment map radiance along a given direction using Octahedral Mapping.
        """
        B = light_dir.shape[0]

        v = light_dir / (light_dir.abs().sum(dim=-1, keepdim=True) + 1e-6)

        is_upper = v[..., 2] >= 0

        x = torch.where(is_upper, v[..., 0], (1 - v[..., 1].abs()) * v[..., 0].sign())
        y = torch.where(is_upper, v[..., 1], (1 - v[..., 0].abs()) * v[..., 1].sign())

        # Map from [-1,1] to [0,1]
        u = ((x + 1) * 0.5 * (self.W - 1)).clamp(0, self.W - 1).long()
        v = ((y + 1) * 0.5 * (self.H - 1)).clamp(0, self.H - 1).long()

        # Fetch radiance from envmap
        Le = self.envmap[:, v, u].permute(1, 0)  # (B, 3)

        # PDF from envmap_pdf
        pdf = self.envmap_pdf[v, u].unsqueeze(-1)  # (B, 1)

        return Le, pdf, torch.ones_like(pdf, dtype=torch.bool)


    def eval_emitter(self, position, light_dir, *args):
        """
        Evaluate environment map radiance along given directions
        Args:
            position: Bx3 intersection points (unused)
            light_dir: Bx3 light directions
        Returns:
            Le: Bx3 radiance
            pdf: Bx1 pdf
            valid: B valid samples (always True for envmap)
        """
        # Convert direction to lat-long coordinates
        phi = torch.atan2(light_dir[..., 2], light_dir[..., 0])  # [-π, π]
        theta = torch.asin(-light_dir[..., 1])  # [-π/2, π/2]

        # Normalize to [0, 1] texture coordinates
        u = (phi / (2 * math.pi)) + 0.5
        v = theta / math.pi + 0.5

        # Convert to pixel indices
        u_idx = (u * (self.W - 1)).long().clamp(0, self.W - 1)
        v_idx = (v * (self.H - 1)).long().clamp(0, self.H - 1)

        # Sample radiance from environment map
        Le = self.envmap[:, v_idx, u_idx].permute(1, 0)  # (B, 3)

        # Compute PDF (assuming uniform distribution for now)
        pdf = torch.full((position.shape[0], 1), 1.0 / (4 * math.pi), device=position.device)

        return Le, pdf, torch.ones_like(pdf, dtype=torch.bool)  # Always valid

class MultiPointsEmitter(nn.Module):
    def __init__(self, dist=4.0, n_theta=8, n_phi=4):
        """
        Args:
            dist: Radius of the sphere
            n_theta: Number of samples in theta (latitude)
            n_phi: Number of samples in phi (longitude)
        """
        super(MultiPointsEmitter, self).__init__()

        self.dist = dist
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.total = n_theta * n_phi

        # Generate light positions on sphere
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi)
        theta_grid, phi_grid = np.meshgrid(thetas, phis)
        thetas_flat = theta_grid.flatten()
        phis_flat = phi_grid.flatten()

        positions = []
        intensities = []

        # Set fixed random seed for reproducible intensities
        np.random.seed(42)
        fixed_intensities = np.random.uniform(1.0, 20.0, size=len(thetas_flat))

        for i, (theta, phi) in enumerate(zip(thetas_flat, phis_flat)):
            x = dist * np.sin(theta) * np.cos(phi)
            y = dist * np.sin(theta) * np.sin(phi)
            z = dist * np.cos(theta)
            positions.append([x, y, z])
            intensities.append(fixed_intensities[i])  # Use pre-generated intensity

        # Add some manually specified light positions and intensities
        manual_positions = [
            [4.0, 0.0, 0.0],     # Right
            [-4.0, 0.0, 0.0],    # Left
            [0.0, 4.0, 0.0],     # Top 
            [0.0, -4.0, 0.0],    # Bottom
            [0.0, 0.0, 4.0],     # Front
            [0.0, 0.0, -4.0],    # Back
            [2.3, 2.3, 2.3],     # Top-Front-Right diagonal
            [-2.3, -2.3, -2.3],  # Bottom-Back-Left diagonal
        ]
        manually_intensities = [10, 20, 30, 40, 50, 60, 70, 80]
        # Define different colors for each light
        light_colors = [
            [1.0, 0.2, 0.2],  # Red
            [0.2, 1.0, 0.2],  # Green
            [0.2, 0.2, 1.0],  # Blue
            [1.0, 1.0, 0.2],  # Yellow
            [1.0, 0.2, 1.0],  # Magenta
            [0.2, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
        ]

        # Convert intensities to colored intensities by multiplying with colors
        manually_colored_intensities = []
        for intensity, color in zip(manually_intensities, light_colors):
            colored_intensity = [intensity * c for c in color]
            manually_colored_intensities.append(colored_intensity)
        manually_intensities = manually_colored_intensities
        
        # Extend the positions and intensities lists
        self.register_buffer('light_positions', torch.tensor(manual_positions[0:8], dtype=torch.float32))  # [N, 3]
        self.register_buffer('light_intensities', torch.tensor(manually_intensities[0:8], dtype=torch.float32).unsqueeze(-1))  # [N, 1]

    def sample_emitter(self, sample1, sample2, position):
        """
        Sample one of the point lights and compute direction to it.
        Args:
            sample1: B (unused)
            sample2: Bx2 (unused)
            position: Bx3 surface positions
        Returns:
            wi: Bx3 directions toward lights
            pdf: Bx1 uniform sampling pdf (1/N)
            idx: B indices of selected lights
        """
        B = position.shape[0]
        N = self.light_positions.shape[0]

        # Randomly select a point light per ray
        idx = torch.randint(0, N, (B,), device=position.device)
        light_pos = self.light_positions[idx]  # [B, 3]

        # Compute direction
        vec = light_pos - position  # [B, 3]
        wi = nn.functional.normalize(vec, dim=-1)

        pdf = torch.full((B, 1), 1.0 / N, device=position.device)

        return wi, pdf, light_pos, idx

    def eval_emitter(self, position, idx):
        """
        Evaluate radiance from point lights in given directions.
        Args:
            position: Bx3 surface points
            light_dir: Bx3 incoming light directions
            idx: B indices of selected lights
        Returns:
            Le: Bx3 radiance
            pdf: Bx1 pdf (1/N)
            valid: B boolean mask (True if a match found)
        """
        B = position.shape[0]
        N = self.light_positions.shape[0]

        # Get selected light intensities
        Le = self.light_intensities[idx].squeeze(-1)   # [B, 3]
        pdf = torch.full((B, 1), 1.0 / N, device=position.device)
        valid = torch.ones(B, dtype=torch.bool, device=position.device)

        return Le, pdf, valid