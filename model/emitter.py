import torch
import torch.nn as nn
import torch.nn.functional as NF
import numpy as np
import math
import imageio
import cv2
from openexr_numpy import imread, imwrite

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
    def __init__(self, positions, intensities):
        """
        Args:
            positions: (N, 3) tensor of light positions
            intensities: (N, 1) tensor of light intensities
        """
        super(PresetPointEmitter, self).__init__()

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