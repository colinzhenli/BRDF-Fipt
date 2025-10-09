import torch
import torch.nn as nn
import torch.nn.functional as NF
import numpy as np
import math
import imageio
import cv2
from openexr_numpy import imread, imwrite
import json
import os

from utils.io import read_light_transforms

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
        envmap = torch.from_numpy(envmap).permute(2, 0, 1)  # Convert to (3, H, W)
        
        self.register_buffer('envmap', envmap)
        self.H, self.W = envmap.shape[1:]  # Get resolution

    def sample_emitter(self, sample, position):
        """
        Sample a direction from the environment map
        Args:
            sample: Bx2 uniform samples for spherical sampling
            position: Bx3 surface positions (unused)
        Returns:
            wi: Bx3 sampled directions
            pdf: Bx1 sampling pdf
            idx: B dummy indices (-1)
        """
        # Convert uniform samples to spherical coordinates
        phi = 2 * math.pi * sample[..., 0]  # Azimuth
        theta = torch.acos(1 - 2 * sample[..., 1])  # Elevation
        
        sin_theta = torch.sin(theta)
        x = sin_theta * torch.cos(phi)
        y = sin_theta * torch.sin(phi)
        z = torch.cos(theta)
        
        wi = torch.stack([x, y, z], dim=-1)  # Direction vectors
        
        # Compute PDF (uniform for now)
        pdf = torch.full((position.shape[0], 1), 1.0 / (4 * math.pi), device=position.device)
        
        idx = torch.full((position.shape[0],), -1, dtype=torch.long, device=position.device)
        
        return wi, pdf, idx

    def eval_emitter(self, position, light_dir):
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

    
class DynamicPointEmitter(nn.Module):
    def __init__(self, ray_num, dist=4.0, num_lights=8, camera_phi=None, theta_angle=60.0, random_positions=True, random_intensities=False, different_per_point=False):
        """
        Args:
            dist: Radius of the sphere
            num_lights: Number of lights to sample
            fix_seed: Whether to fix the seed
        """
        super(DynamicPointEmitter, self).__init__()

        self.dist = dist
        self.num_lights = num_lights
        self.camera_phi = camera_phi
        self.theta_angle = theta_angle
        self.random_positions = random_positions
        self.ray_num = ray_num
        self.random_intensities = random_intensities
        self.different_per_point = different_per_point
        if self.different_per_point:
            theta = torch.pi/2 * torch.rand(ray_num, num_lights, device="cuda")
            phi = 2 * torch.pi * torch.rand(ray_num, num_lights, device="cuda")
            x = dist * torch.sin(theta) * torch.cos(phi)
            z = dist * torch.sin(theta) * torch.sin(phi)
            y = dist * torch.cos(theta)
            light_positions = torch.stack([x, y, z], dim=-1)
            if self.random_intensities:
                light_intensities = torch.rand(num_lights, 1, device="cuda") * 49.0 + 1.0  # Uniform [1.0, 50.0]
            else:
                light_intensities = torch.full((num_lights, 1), 50.0/num_lights, device="cuda")  # Fixed maximum intensity
            self.register_buffer('light_positions', light_positions)  # [B, N, 3]
            self.register_buffer('light_intensities', light_intensities)  # [B, N, 1]
            return

        if self.camera_phi is None:
            self.fixed_theta = False
        else:
            self.fixed_theta = True
        if self.random_positions:
            # randomly sample during training
            theta = torch.pi/2 * torch.rand(num_lights, device="cuda")
            phi   = 2 * torch.pi * torch.rand(num_lights, device="cuda")
        else:
            if self.fixed_theta:
                theta = np.deg2rad(theta_angle)
                theta = torch.full((num_lights,), theta, device='cuda')  # Fixed theta for all lights
                
                if num_lights == 1:
                    phi = torch.tensor([torch.pi], device='cuda')
                    phi = phi + camera_phi
                else:
                    # Uniformly distribute phi values for multiple lights
                    phi = torch.linspace(0, 2 * torch.pi, num_lights, endpoint=False, device='cuda')
                    phi = phi + camera_phi
            else:
                torch.manual_seed(0)
                # θ ∼ uniform on (0, π/2) to ensure y > 0 (cos(θ) > 0)
                theta = torch.pi/2 * torch.rand(num_lights, device="cuda")
                # φ uniform on (0, 2π)
                phi   = 2 * torch.pi * torch.rand(num_lights, device="cuda")

        x = dist * torch.sin(theta) * torch.cos(phi)
        z = dist * torch.sin(theta) * torch.sin(phi)
        y = dist * torch.cos(theta)   

        positions = torch.stack([x, y, z], dim=1)  # [N, 3]
        if self.random_intensities:
            intensities = torch.rand(num_lights, 1, device='cuda') * 49.0 + 1.0  # Uniform [1.0, 50.0]
        else:
            intensities = torch.full((num_lights, 1), 50.0/num_lights, device='cuda')  # Fixed maximum intensity

        self.register_buffer('light_positions', positions)  # [N, 3]
        self.register_buffer('light_intensities', intensities)  # [N, 1]

    def sample_emitter(self, position):
        """
        Deterministic sampling: For each position, sample toward every light.

        Args:
            sample1: (unused)
            sample2: (unused)
            position: (B, 3) surface positions

        Returns:
            wi: (B, N, 3) directions toward lights
            pdf: (B, N, 1) uniform pdf
            light_pos: (B, N, 3) selected light positions
            idx: (B, N) selected light indices
        """
        B = position.shape[0]
        if self.different_per_point:
            N = self.light_positions.shape[1]
        else:
            N = self.light_positions.shape[0]

        position_expand = position.unsqueeze(1).expand(B, N, 3)  # [B, N, 3]
        if self.different_per_point:
            # Select the first B positions from self.light_positions for per-point emitters
            light_pos_expand = self.light_positions[:B]  # [B, N, 3]
        else:
            light_pos_expand = self.light_positions.unsqueeze(0).expand(B, N, 3)  # [B, N, 3]

        vec = light_pos_expand - position_expand  # [B, N, 3]
        wi = NF.normalize(vec, dim=-1).reshape(-1, 3)  # [B*N, 3]

        pdf = torch.full((B*N, 1), 1.0 / N, device=position.device)

        idx = torch.arange(N, device=position.device).unsqueeze(0).expand(B, N).reshape(-1)  # [B*N]

        light_pos = light_pos_expand.reshape(-1, 3)

        return wi, pdf, light_pos, idx

    def eval_emitter(self, position, idx):
        """
        Evaluate radiance from selected lights.

        Args:
            position: (B, 3) surface points
            idx: (B,) selected light indices

        Returns:
            Le: (B, 3) radiance
            pdf: (B, 1) pdf
            valid: (B,) valid mask
        """
        B = position.shape[0]
        intensities = self.light_intensities.expand(-1, 3)  # [B, 3]
        pdf = torch.full((B, 1), 1.0 / self.light_positions.shape[0], device=position.device)
        valid = torch.ones(B, dtype=torch.bool, device=position.device)

        return intensities, pdf, valid
    
class PresetPointEmitter(nn.Module):
    def __init__(self, read_from_metadata=False, metadata_path=None, positions=None, intensities=None):
        """
        Args:
            positions: (N, 3) tensor of light positions
            intensities: (N, 1) tensor of light intensities
        """
        super(PresetPointEmitter, self).__init__()
        if read_from_metadata:
            # Read positions and intensities from metadata
            import json
            import os
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                # Extract emitter metadata
                positions = torch.tensor([em['position'] for em in metadata], device='cuda')
                intensities = torch.tensor([em['intensity'] for em in metadata], device='cuda')
                self.positions = positions
                self.intensities = intensities
            else:
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")  
        else:
            self.positions = positions
            self.intensities = intensities

        self.register_buffer('light_positions', positions)  # [N, 3]
        self.register_buffer('light_intensities', intensities)  # [N, 3]

    def sample_emitter(self, position, idx):
        """
        Deterministic sampling: For each position, sample toward one specific light using idx.

        Args:
            position: (B, 3) surface positions
            idx: (B,) selected light indices
        Returns:
            wi: (B, 3) directions toward selected lights
            pdf: (B, 1) uniform pdf
            light_pos: (B, 3) selected light positions
            idx: (B,) selected light indices
        """
        B = position.shape[0]
        
        # Select specific light positions based on idx
        light_pos = self.light_positions[idx]  # [B, 3]
        
        vec = light_pos - position  # [B, 3]
        wi = NF.normalize(vec, dim=-1)  # [B, 3]
        
        pdf = torch.full((B, 1), 1.0, device=position.device)
        
        return wi, pdf, light_pos, idx

    def eval_emitter(self, position, idx):
        """
        Evaluate radiance from selected lights.

        Args:
            position: (B, 3) surface points
            idx: (B,) selected light indices

        Returns:
            Le: (B, 3) radiance
            pdf: (B, 1) pdf
            valid: (B,) valid mask
        """
        B = position.shape[0]
        # Select intensities based on idx for each position
        intensities = self.light_intensities[idx]  #  [B, 3]
        pdf = torch.full((B, 1), 1.0 / self.light_positions.shape[0], device=position.device)
        valid = torch.ones(B, dtype=torch.bool, device=position.device)

        return intensities, pdf, valid
    
class RealAreaEmitter(nn.Module):
    def __init__(self, cfg, json_path):
        """
        Args:
            positions: (N, 3) light center
            radius: (N, 1) light radius
            radiance: (N, 1) largest radiance
        """
        super(RealAreaEmitter, self).__init__()
        # Extract configuration parameters
        radius = cfg.get('radius', 0.007)
        fwhm_deg = cfg.get('fwhm_deg', 115.0)
        self.light_radiance = nn.Parameter(torch.tensor(cfg.get('radiance'), dtype=torch.float32, device='cuda'))
        # self.register_buffer('light_radiance', torch.tensor(cfg.get('radiance'), dtype=torch.float32, device='cuda'))

        theta_half = math.radians(fwhm_deg * 0.5)
        m = math.log(0.5) / math.log(max(1e-8, math.cos(theta_half)))
        self.register_buffer('m', torch.tensor(m, dtype=torch.float32, device='cuda'))
        self.register_buffer('light_radius', torch.tensor(radius, dtype=torch.float32, device='cuda'))

        self.l2w = self._compute_l2w(cfg, json_path)
        self.register_buffer('light_positions', self.l2w[:, :3, 3])  # [N, 3] - translation part
        light_normal_local = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device='cuda')
        light_normals_world = torch.matmul(self.l2w[:, :3, :3], light_normal_local)  # [N, 3]
        light_normals_world = light_normals_world / (light_normals_world.norm(dim=-1, keepdim=True) + 1e-12)
        self.register_buffer('light_normal', light_normals_world)
        
    def _compute_l2w(self, cfg, json_path):
        """
        Compute the world transform for each light.
        """
        # Compute light transformation matrix
        R_l2g = torch.tensor(cfg.get('R_l2g'), dtype=torch.float32, device='cuda')
        t_l2g = torch.tensor(cfg.get('t_l2g'), dtype=torch.float32, device='cuda')
        base2_to_base1 = torch.tensor(cfg.get('base2_to_base1'), dtype=torch.float32, device='cuda')
        g2b0 = read_light_transforms(json_path, cfg.turntable.center, cfg.turntable.axis, base2_to_base1) # [N, 4, 4]
        # g2b0 = base2_to_base1.unsqueeze(0) @ g2b
        l2g = torch.eye(4, device='cuda')
        l2g[:3, :3] = R_l2g
        l2g[:3, 3] = t_l2g
        l2w = g2b0 @ l2g
        return l2w
    
    def _update_poses_for_vis(self, turntable_center, steps):
        turntable_center = turntable_center.to(self.l2w.device)
        light_normal_local = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device='cuda')
        p0_world = self.l2w[:3, 3]   # [3]
        R0_world = self.l2w[:3, :3]  # [3,3]
        n0_world = (R0_world @ light_normal_local)  # [3]
        n0_world = n0_world / (n0_world.norm() + 1e-12)

        # Clockwise in right-handed (+Z out) means negative angles
        angles = torch.arange(steps, device='cuda', dtype=torch.float32) * (-2.0 * math.pi / steps)  # [60]

        c = torch.cos(angles)
        s = torch.sin(angles)
        # Batch of Rz(θ): shape [60, 3, 3]
        Rz = torch.zeros(steps, 3, 3, device='cuda', dtype=torch.float32)
        Rz[:, 0, 0] =  c
        Rz[:, 0, 1] = -s
        Rz[:, 1, 0] =  s
        Rz[:, 1, 1] =  c
        Rz[:, 2, 2] =  1.0

        # Rotate the position around the center: p' = Rz*(p0 - center) + center
        rel = p0_world - turntable_center  # [3]
        rel = rel.unsqueeze(-1)            # [3,1] for batch matmul
        pos_rot = (Rz @ rel).squeeze(-1) + turntable_center  # [60,3]

        # Rotate the normal as a direction: n' = Rz * n0
        n0 = n0_world.unsqueeze(-1)  # [3,1]
        nor_rot = (Rz @ n0).squeeze(-1)  # [60,3]
        nor_rot = nor_rot / (nor_rot.norm(dim=-1, keepdim=True) + 1e-12)

        # ---- Register buffers ----
        with torch.no_grad():
            self.light_positions = pos_rot
            self.light_normal = nor_rot

                
    def _directional_distribution(self, light_dir,light_id):
        """
        Compute directional radiance L(θ) for rays headed from the light to the surface.
        Args:
            light_dir: (B, 3) directions from surface -> light (so emission dir is -light_dir)

        Returns:
            radiance: (B, 3) directional radiance (RGB) following L = L0 * cos^m(theta).
                      Clamped to zero for back-facing directions.
        """
        # Emission direction is from light -> surface
        v = -light_dir  # (B,3)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)

        # cos(theta) between light normal and emission direction
        cos_theta = torch.clamp((v * self.light_normal[light_id]).sum(dim=-1, keepdim=True), min=0.0)

        # Cosine-power lobe
        Lshape = cos_theta.pow(self.m)  # (B,1)

        # Use the first light's L0 as the color (assumes one emitter or shared spectrum)
        L0 = self.light_radiance     # (B, 3)
        radiance = Lshape * L0          # (B,3)
        return radiance
    
    def sample_emitter(self, sample, position, light_id):
        """
        Sample a direction(position) from the area light (circular disk).
        Args:
            sample: Bx2 uniform samples for disk sampling
            position: Bx3 surface positions
            light_id: B light indices
        Returns:
            wi: Bx3 sampled directions
            pdf: Bx1 sampling pdf (area pdf)
            emit_position: Bx3 sampled positions
            emitter_normal: Bx3 sampled normals
        """
        B = position.shape[0]
        
        # Get light properties for the specified light indices
        light_pos = self.light_positions[light_id]  # (B, 3)
        light_r = self.light_radius.expand(B)  # (B,)
        light_n = self.light_normal[light_id]  # (B, 3)
        
        # Uniform sampling on disk using polar coordinates
        r_sample = torch.sqrt(sample[..., 0]) * light_r   # (B,)
        theta = 2.0 * math.pi * sample[..., 1]           # (B,)
        disk_x = r_sample * torch.cos(theta)             # (B,)
        disk_y = r_sample * torch.sin(theta)             # (B,)
        
        # Create orthonormal basis for each light plane
        up = torch.tensor([0.0, 1.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        up = up.unsqueeze(0).expand(B, 3)  # (B, 3)
        
        # Check for near-parallel cases and use alternative up vector
        parallel_mask = torch.abs((light_n * up).sum(dim=-1)) > 0.9  # (B,)
        alt_up = torch.tensor([1.0, 0.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        alt_up = alt_up.unsqueeze(0).expand(B, 3)  # (B, 3)
        up = torch.where(parallel_mask.unsqueeze(-1), alt_up, up)  # (B, 3)
        
        # Compute u_axis for each light
        u_axis = torch.cross(light_n, up, dim=-1)  # (B, 3)
        u_axis = u_axis / (torch.norm(u_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute v_axis for each light
        v_axis = torch.cross(light_n, u_axis, dim=-1)  # (B, 3)
        v_axis = v_axis / (torch.norm(v_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute sampled position on light disk
        emit_position = (
            light_pos 
            + disk_x.unsqueeze(-1) * u_axis 
            + disk_y.unsqueeze(-1) * v_axis
        )  # (B, 3)
        
        # Calculate direction from surface to light sample
        wi = emit_position - position  # (B, 3)
        distance = torch.norm(wi, dim=-1, keepdim=True)  # (B, 1)
        wi = wi / distance  # (B, 3)
        
        # Calculate area-based PDF
        # PDF = 1 / Area = 1 / (π r^2)
        light_area = math.pi * light_r * light_r  # (B,)
        pdf = 1.0 / light_area  # (B,)
        pdf = pdf.unsqueeze(-1)  # (B, 1)
        
        # Emitter normal (same for all sampled points)
        emitter_normal = light_n  # (B, 3)
        
        return wi, pdf, emit_position, emitter_normal

    def intersect(self, position, light_dir, light_id):
        """
        Intersect a ray with the area light (circular disk)
        Args:
            position: (B, 3) ray origins
            light_dir: (B, 3) ray directions (normalized)
            light_id: (B,) light indices
        Returns:
            t: (B,) intersection distances (negative if no intersection)
            hit: (B,) boolean mask for valid intersections
            hit_pos: (B, 3) intersection positions
            light_idx: (B,) light indices
        """
        B = position.shape[0]
        
        # Get light properties for the specified light indices
        light_pos = self.light_positions[light_id]  # (B, 3)
        light_r = self.light_radius.expand(B)  # (B,)
        light_n = self.light_normal[light_id]  # (B, 3)
        
        # Ray-plane intersection
        # Ray: p(t) = pos + t * dirs
        # Plane: (p - light_pos) · light_normal = 0
        # Substituting: (pos + t*dirs - light_pos) · light_normal = 0
        # Solving for t: t = (light_pos - pos) · light_normal / (dirs · light_normal)
        
        # Compute denominator (ray direction dot plane normal)
        denom = (light_dir * light_n).sum(dim=-1)  # (B,)

        # Check if ray is parallel to plane (denom ≈ 0)
        
        # Compute numerator
        to_light = light_pos - position  # (B, 3)
        numer = (to_light * light_n).sum(dim=-1)  # (B,)
        '''
        print("position",position)
        print("light_pos",light_pos)
        print("to_light",to_light)
        print("light_n",light_n)
        
        print("numer",numer)
        print("denom",denom)
        '''
        # Compute intersection distance
        t = numer / denom  # (B,)
        
        # Check if intersection is in front of ray origin
        
        # Compute intersection points
        hit_pos = position + t.unsqueeze(-1) * light_dir  # (B, 3)
        
        # Check if intersection point is within the circular disk
        # Create orthonormal basis for each light plane
        up = torch.tensor([0.0, 1.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        up = up.unsqueeze(0).expand(B, 3)  # (B, 3)
        
        # Check for near-parallel cases and use alternative up vector
        parallel_mask = torch.abs((light_n * up).sum(dim=-1)) > 0.9  # (B,)
        alt_up = torch.tensor([1.0, 0.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        alt_up = alt_up.unsqueeze(0).expand(B, 3)  # (B, 3)
        up = torch.where(parallel_mask.unsqueeze(-1), alt_up, up)  # (B, 3)
        
        # Compute u_axis for each light
        u_axis = torch.cross(light_n, up, dim=-1)  # (B, 3)
        u_axis = u_axis / (torch.norm(u_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute v_axis for each light
        v_axis = torch.cross(light_n, u_axis, dim=-1)  # (B, 3)
        v_axis = v_axis / (torch.norm(v_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Project intersection point onto the light plane coordinate system
        to_hit = hit_pos - light_pos  # (B, 3)
        u_coord = (to_hit * u_axis).sum(dim=-1)  # (B,)
        v_coord = (to_hit * v_axis).sum(dim=-1)  # (B,)
        
        # Check if within circular disk bounds (distance from center <= radius)
        dist_from_center = torch.sqrt(u_coord * u_coord + v_coord * v_coord)  # (B,)
        
        # Final hit mask: valid t, not parallel, and within disk
        
        # Set invalid distances to negative
        #t = torch.where(hit, t, torch.full_like(t, -1.0))
        hit=True
        
        # Light indices for each intersection
        light_idx = light_id  # (B,)
        
        return t, hit, hit_pos, light_idx
    
    def eval_emitter(self, position, light_dir, light_id):
        """
        Evaluate environment map radiance along given directions
        Args:
            position: Bx3 intersection points 
            light_dir: Bx3 light directions(from surface to light)
        Returns:
            Le: Bx3 radiance
            pdf: Bx1 pdf
            valid: B valid samples (always True for area light)
        """
        t, hit, hit_pos, light_idx=self.intersect(position,light_dir,light_id)
        '''
        print("t",t.shape)
        print("hit_pos",hit_pos.shape)
        print("light_dir",light_dir.shape)
        print("light_normal",self.light_normal[light_id].shape)
        '''
        B = position.shape[0]
        Le=self._directional_distribution(light_dir,light_id)*self.light_radiance.repeat(B,1)*(1.0/(t*t)).unsqueeze(-1)
        # dA_dw=((position-hit_pos)*(position-hit_pos)).sum(dim=-1)/((-light_dir)*self.light_normal[light_id]).sum(dim=-1)
        pdf=1.0/(self.light_radius.expand(B)*self.light_radius.expand(B)*torch.pi)
        pdf=pdf.unsqueeze(-1)

        if torch.isnan(Le).any():
            print("Le is nan")
        return Le, pdf, torch.ones_like(pdf, dtype=torch.bool)  # Always valid
    
class AreaEmitter(nn.Module):
    """ reference triangle mesh emitters from FIPT paper"""
    # def __init__(self,emitter_path):
    #     """ emitter_path file 
    #     is_emitter: B indicator of whether a triangle is emitter
    #     emitter_vertices: Kx3x3 triangle vertices of emitters
    #     emitter_area: K surface areas of emitters
    #     emitter_radiance: Bx3x3 emitter radiance
    #     """
    #     super(AreaEmitter,self).__init__()
        
    #     weight = torch.load(emitter_path,map_location='cpu')
        
    #     is_emitter = weight['is_emitter']
    #     emitter_vertices = weight['emitter_vertices']
    #     emitter_area = weight['emitter_area']
    #     emitter_radiance = weight['emitter_radiance']

    #     self.register_buffer('is_emitter',is_emitter)
    #     self.register_buffer('emitter_vertices',emitter_vertices)
    #     self.register_buffer('emitter_area',emitter_area)
    #     self.register_buffer('radiance',emitter_radiance)
        
    #     # emitter idx mapping, -1 indicates not an emitter
    #     emitter_idx = torch.full((len(is_emitter),),-1,device=is_emitter.device,dtype=torch.long)
    #     emitter_idx[is_emitter] = torch.arange(is_emitter.sum(),device=is_emitter.device)
    #     self.register_buffer('emitter_idx',emitter_idx)
        
    #     # emitter idx to triangle idx
    #     triangle_idx = torch.arange(len(is_emitter))[is_emitter]
    #     self.register_buffer('triangle_idx',triangle_idx)
        
    #     # sample emitters uniformly
    #     emitter_pdf = NF.normalize(torch.ones_like(emitter_area),dim=-1,p=1)
    #     emitter_cdf = emitter_pdf.cumsum(-1).contiguous()
    #     self.register_buffer('emitter_pdf',emitter_pdf)
    #     self.register_buffer('emitter_cdf',emitter_cdf)
    def __init__(self, emitter_path):
        """ emitter_path file 
        is_emitter: B indicator of whether a triangle is emitter
        emitter_vertices: Kx3x3 triangle vertices of emitters
        emitter_area: K surface areas of emitters
        emitter_radiance: Bx3x3 emitter radiance
        """
        super(AreaEmitter, self).__init__()

        weight = torch.load(emitter_path, map_location='cpu')

        is_emitter        = weight['is_emitter']
        emitter_vertices  = weight['emitter_vertices']   # [K, 3, 3]
        emitter_area      = weight['emitter_area']       # [K]
        emitter_radiance  = weight['emitter_radiance']   # [B, 3, 3]

        # ---- Register original buffers (unchanged) ----
        self.register_buffer('is_emitter', is_emitter)
        self.register_buffer('emitter_vertices', emitter_vertices)
        self.register_buffer('emitter_area', emitter_area)
        self.register_buffer('radiance', emitter_radiance)

        # Emitter idx mapping, -1 indicates not an emitter
        emitter_idx = torch.full((len(is_emitter),), -1, device=is_emitter.device, dtype=torch.long)
        emitter_idx[is_emitter] = torch.arange(is_emitter.sum(), device=is_emitter.device)
        self.register_buffer('emitter_idx', emitter_idx)

        # Emitter idx to triangle idx
        triangle_idx = torch.arange(len(is_emitter))[is_emitter]
        self.register_buffer('triangle_idx', triangle_idx)

        # Sample emitters uniformly
        emitter_pdf = NF.normalize(torch.ones_like(emitter_area), dim=-1, p=1)
        emitter_cdf = emitter_pdf.cumsum(-1).contiguous()
        self.register_buffer('emitter_pdf', emitter_pdf)
        self.register_buffer('emitter_cdf', emitter_cdf)

        # ---- New: generate 60 emitter positions along a 360° trajectory (clockwise) ----
        # Turntable center (world space)
        TURNTABLE_CENTER = torch.tensor([0.21056884, -0.1938618, 0.0], dtype=emitter_vertices.dtype)

        # Define a single "emitter position" as the overall centroid of all emitting triangles
        # centroid of a triangle = mean of its 3 vertices; overall position = area-weighted mean
        tri_centroids = emitter_vertices.mean(dim=1)                    # [K, 3]
        areas = emitter_area.clamp_min(1e-12)                           # avoid div-by-zero
        weighted_sum = (tri_centroids * areas.unsqueeze(-1)).sum(dim=0) # [3]
        total_area = areas.sum()
        base_pos = weighted_sum / total_area                            # [3] starting position

        # Build 60 angles from 0 to 2π (clockwise => negative angles)
        num_steps = 60
        thetas = torch.linspace(0.0, 2.0 * torch.pi, steps=num_steps, dtype=emitter_vertices.dtype)
        thetas = -thetas  # clockwise

        # Helper: rotate a 3D point around Z about a center
        def rotate_around_z(p, center, theta):
            # p, center: [3]; return: [3]
            c, s = torch.cos(theta), torch.sin(theta)
            Rz = torch.tensor([[c, -s, 0.0],
                            [s,  c, 0.0],
                            [0.0, 0.0, 1.0]], dtype=p.dtype, device=p.device)
            return (Rz @ (p - center)) + center

        # Generate trajectory positions
        # keep everything on the same device as emitter_vertices
        base_pos = base_pos.to(emitter_vertices.device)
        TURNTABLE_CENTER = TURNTABLE_CENTER.to(emitter_vertices.device)
        thetas = thetas.to(emitter_vertices.device)

        positions = []
        for th in thetas:
            positions.append(rotate_around_z(base_pos, TURNTABLE_CENTER, th))
        emitter_positions = torch.stack(positions, dim=0)  # [60, 3]

        # Store center & positions for downstream use
        self.register_buffer('turntable_center', TURNTABLE_CENTER)
        self.register_buffer('emitter_positions', emitter_positions)

    
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