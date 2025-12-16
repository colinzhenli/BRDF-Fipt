"""
Refactored Neural BRDF Classes
- Latent: Single latent code vector
- LatentTexture: 2D grid of latent codes with interpolation and blur
- NeuralGeometry: Neural UV offset prediction
- BRDFDecoder: MLP decoder for BRDF evaluation
- AnisotropicLatentTexturedModel: Combined material class with all APIs
"""

import torch
import torch.nn as nn
import torch.nn.functional as NF
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import math
# from nerfstudio.field_components import encoding
from utils.ops import components_from_spherical_harmonics, num_sh_bases


# ============================================================================
# 1. LATENT CLASS - Single latent code vector
# ============================================================================
class Latent(nn.Module):
    """
    Single latent code vector.
    This is a conceptual class - in practice, latents are stored as tensors,
    but this provides a clean interface for operations on individual latents.
    """
    def __init__(self, dim: int, init_std: float = 0.1):
        super().__init__()
        self.dim = dim
        self.code = nn.Parameter(torch.randn(dim) * init_std)
    
    def get_code(self):
        """Get the latent code vector"""
        return self.code
    
    def set_code(self, code: torch.Tensor):
        """Set the latent code vector"""
        assert code.shape[-1] == self.dim, f"Code dimension mismatch: {code.shape[-1]} vs {self.dim}"
        self.code.data = code


# ============================================================================
# 2. LATENT TEXTURE CLASS - 2D grid of latent codes
# ============================================================================
class LatentTexture(nn.Module):
    """
    2D texture grid storing latent codes with interpolation and Gaussian blur.
    Each texel contains a latent code vector.
    """
    def __init__(
        self, 
        resolution: int,
        latent_dim: int,
        predict_frame: bool = False,
        init_std: float = 0.1,
        blur_config: dict = None
    ):
        """
        Args:
            resolution: Texture resolution (HxW)
            latent_dim: Dimension of each latent code
            predict_frame: Whether to include normal+tangent (6D) in latent
            init_std: Standard deviation for initialization
            blur_config: Dict with blur_sigma0 and blur_half_life
        """
        super().__init__()
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.predict_frame = predict_frame
        
        # Initialize latent grid [1, latent_dim, H, W]
        latent_init = torch.randn(1, latent_dim, resolution, resolution) * init_std
        
        # Initialize frame components if needed
        if predict_frame:
            # Last 6 dimensions: normal (0,0,1) and tangent (0,1,0)
            latent_init[:, -6:-3, :, :] = torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1)
            latent_init[:, -3:, :, :] = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1)
        
        self.params = nn.Parameter(latent_init)
        
        # Gaussian blur parameters
        if blur_config is not None:
            self.blur_sigma0 = blur_config.get('blur_sigma0', 8.0)
            self.blur_half_life = blur_config.get('blur_half_life', 3333)
        else:
            self.blur_sigma0 = 8.0
            self.blur_half_life = 3333
    
    def _gaussian_kernel(self, sigma: float, channels: int):
        """Generate Gaussian kernel for depth-wise convolution."""
        if sigma < 0.5:
            return None
        
        radius = int(math.ceil(3 * sigma))
        ksize = 2 * radius + 1
        grid = torch.arange(-radius, radius + 1,
                           dtype=self.params.dtype,
                           device=self.params.device)
        g1d = torch.exp(-0.5 * (grid / sigma) ** 2)
        g1d = g1d / g1d.sum()
        g2d = (g1d[:, None] * g1d[None, :]).expand(channels, 1, ksize, ksize)
        return g2d
    
    def apply_gaussian_blur(self, step: int):
        """
        Apply progressive Gaussian blur based on training step.
        σ(t) = σ₀ · 2^{-t/half_life}
        
        Args:
            step: Current training step
        
        Returns:
            Blurred latent texture [1, total_dim, H, W]
        """
        sigma = self.blur_sigma0 * (0.5 ** (step / self.blur_half_life))
        kernel = self._gaussian_kernel(sigma, self.params.shape[1])
        
        if kernel is None:
            return self.params
        
        pad = kernel.shape[-1] // 2
        return NF.conv2d(self.params, kernel, padding=pad, groups=self.params.shape[1])
    
    def query(self, uv: torch.Tensor, blur_step: int = None):
        """
        Query latent codes at UV coordinates with bilinear interpolation.
        
        Args:
            uv: [B, 2] UV coordinates in [0, 1]
            blur_step: If not None, apply Gaussian blur for this training step
        
        Returns:
            latent: [B, total_dim] sampled latent codes
        """
        # Apply blur if training step is provided
        if blur_step is not None:
            texture = self.apply_gaussian_blur(blur_step)
        else:
            texture = self.params
        
        # Convert UV to grid coordinates for F.grid_sample
        # grid_sample expects coordinates in [-1, 1]
        grid_coords = uv * 2.0 - 1.0  # [B, 2] -> [-1, 1]
        grid_coords = grid_coords.unsqueeze(1).unsqueeze(0)  # [1, 1, B, 2]
        
        # Bilinear sampling
        latent = NF.grid_sample(
            texture,  # [1, D, H, W]
            grid_coords,  # [1, 1, B, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # [1, D, 1, B]
        
        # Reshape to [B, D]
        latent = latent.squeeze(0).squeeze(-1).transpose(0, 1)
        
        return latent
    
    def get_full_texture(self):
        """Get the full latent texture grid"""
        return self.params
    
    def save(self, path: str):
        """Save latent texture to file"""
        torch.save(self.params.data, path)
    
    def load(self, path: str):
        """Load latent texture from file"""
        loaded = torch.load(path)
        assert loaded.shape == self.params.shape, \
            f"Shape mismatch: {loaded.shape} vs {self.params.shape}"
        self.params.data = loaded


# ============================================================================
# 3. NEURAL GEOMETRY CLASS - UV offset prediction
# ============================================================================
class NeuralGeometry(nn.Module):
    """
    Neural network for predicting UV offsets based on viewing/lighting directions.
    This enables view-dependent displacement/parallax effects.
    """
    def __init__(
        self,
        cfg,
        geometry_latent_dim: int,
        use_local_wi_wo: bool = True,
        use_pos_enc: bool = True
    ):
        """
        Args:
            cfg: Configuration with hidden_layers, activation, output_channels
            geometry_latent_dim: Dimension of geometry-specific latent
            use_local_wi_wo: Use local space directions (vs world space)
            use_pos_enc: Use spherical harmonics encoding
        """
        super().__init__()
        self.geometry_latent_dim = geometry_latent_dim
        self.use_local_wi_wo = use_local_wi_wo
        self.use_pos_enc = use_pos_enc
        
        # Setup positional encoding if enabled
        if use_pos_enc:
            self.degree = 3
            self.sh_encoder = lambda x: components_from_spherical_harmonics(self.degree, x)
            sh_dim = num_sh_bases(self.degree)
            encoded_dim = sh_dim * 2  # wi and wo
        else:
            encoded_dim = 6  # wi (3) + wo (3)
        
        # Input: encoded directions (or geometry latent + directions)
        input_dim = encoded_dim + geometry_latent_dim if not use_pos_enc else encoded_dim
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        for hidden_dim in cfg.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if cfg.activation.lower() == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.LeakyReLU(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, cfg.output_channels))  # 2D UV offset
        layers.append(nn.Tanh())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        wi: torch.Tensor,
        wo: torch.Tensor,
        geometry_latent: torch.Tensor = None
    ):
        """
        Predict UV offset based on directions.
        
        Args:
            wi: [B, 3] incoming light direction (local or world space)
            wo: [B, 3] outgoing view direction (local or world space)
            geometry_latent: [B, geometry_latent_dim] geometry latent code
        
        Returns:
            uv_offset: [B, 2] UV offset
        """
        if self.use_pos_enc:
            wi_enc = self.sh_encoder(wi)
            wo_enc = self.sh_encoder(wo)
            mlp_input = torch.cat([wi_enc, wo_enc], dim=-1)
        else:
            if geometry_latent is not None:
                mlp_input = torch.cat([geometry_latent, wi, wo], dim=-1)
            else:
                mlp_input = torch.cat([wi, wo], dim=-1)
        
        return self.mlp(mlp_input)


# ============================================================================
# 4. BRDF DECODER CLASS - MLP decoder
# ============================================================================
class BRDFDecoder(nn.Module):
    """
    MLP decoder that maps (encoded_directions + latent) -> BRDF value.
    Can have separate decoders per RGB channel or a single shared decoder.
    """
    def __init__(
        self,
        cfg,
        latent_dim: int,
        use_pos_enc: bool = True,
        different_decoder: bool = False
    ):
        """
        Args:
            cfg: Configuration with hidden_layers, activation, output_channels
            latent_dim: Dimension of latent code input
            use_pos_enc: Use spherical harmonics encoding for directions
            different_decoder: Use separate MLPs for R, G, B channels
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.use_pos_enc = use_pos_enc
        self.different_decoder = different_decoder
        
        # Setup positional encoding
        if use_pos_enc:
            self.degree = getattr(cfg, 'degree', 3)
            use_nerfstudio_sh = getattr(cfg, 'use_nerfstudio_sh', False)
            
            if use_nerfstudio_sh:
                self.sh_encoder = encoding.SHEncoding(levels=self.degree + 1)
                sh_dim = (self.degree + 1) ** 2
            else:
                self.sh_encoder = lambda x: components_from_spherical_harmonics(self.degree, x)
                sh_dim = num_sh_bases(self.degree)
            
            encoded_input_dim = sh_dim * 3  # wi, wo, normal
        else:
            encoded_input_dim = cfg.input_channels  # 9 (wi + wo + normal)
        
        input_dim = encoded_input_dim + latent_dim
        
        # Build MLP(s)
        def build_mlp():
            layers = []
            prev_dim = input_dim
            for hidden_dim in cfg.hidden_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if cfg.activation.lower() == "relu":
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.LeakyReLU(0.2))
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, cfg.output_channels))
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)
        
        if different_decoder:
            self.mlp_r = build_mlp()
            self.mlp_g = build_mlp()
            self.mlp_b = build_mlp()
        else:
            self.mlp = build_mlp()
    
    def encode_directions(
        self,
        wi_local: torch.Tensor,
        wo_local: torch.Tensor,
        normal_local: torch.Tensor
    ):
        """
        Encode local-space directions with spherical harmonics.
        
        Args:
            wi_local: [B, 3] incoming light direction (local space)
            wo_local: [B, 3] outgoing view direction (local space)
            normal_local: [B, 3] normal (local space, typically [0,0,1])
        
        Returns:
            encoded: [B, encoded_dim] encoded directions
        """
        if self.use_pos_enc:
            wi_enc = self.sh_encoder(wi_local)
            wo_enc = self.sh_encoder(wo_local)
            normal_enc = self.sh_encoder(normal_local)
            return torch.cat([wi_enc, wo_enc, normal_enc], dim=-1)
        else:
            return torch.cat([wi_local, wo_local, normal_local], dim=-1)
    
    def forward(
        self,
        enc_dir: torch.Tensor,
        latent: torch.Tensor,
        channel: str = None
    ):
        """
        Decode BRDF from encoded directions and latent.
        
        Args:
            enc_dir: [B, encoded_dim] encoded directions
            latent: [B, latent_dim] latent code
            channel: 'r', 'g', or 'b' if using different decoders, else None
        
        Returns:
            brdf: [B, output_channels] BRDF value (1 or 3 channels)
        """
        mlp_input = torch.cat([enc_dir, latent], dim=-1)
        
        if self.different_decoder:
            if channel == 'r':
                return self.mlp_r(mlp_input)
            elif channel == 'g':
                return self.mlp_g(mlp_input)
            else:  # 'b'
                return self.mlp_b(mlp_input)
        else:
            return self.mlp(mlp_input)


# ============================================================================
# 5. PROXY BRDF CLASS (for importance sampling)
# ============================================================================
class ProxyPBRBRDF(nn.Module):
    """
    Proxy PBR BRDF model for importance sampling.
    This is a placeholder - the actual implementation should come from your codebase.
    """
    def __init__(self):
        super().__init__()
    
    def sample_brdf(self, pos, sample1, sample2, wo, normal, roughness, batch_mask):
        """
        Sample directions from proxy BRDF distribution.
        
        Returns:
            wi: [B, 3] sampled directions
            pdf: [B, 1] probability density
        """
        # Placeholder - implement actual proxy BRDF sampling
        raise NotImplementedError("Implement proxy BRDF sampling from your codebase")


# ============================================================================
# 6. ANISOTROPIC LATENT TEXTURED MODEL - Combined material class
# ============================================================================
class AnisotropicLatentTexturedModel(LightningModule):
    """
    Complete material model combining all components.
    Provides the same API as the original class for compatibility.
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Store configuration
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.colorful_texture = cfg.colorful_texture
        self.larger_latent_dim = cfg.larger_latent_dim
        self.different_decoder = cfg.different_decoder
        self.predict_frame = cfg.predict_frame
        self.gt_frame = cfg.gt_frame
        self.anisotropic = True
        self.Gaussian_blur = cfg.Gaussian_blur
        
        # Neural geometry settings
        self.neural_geometry_enabled = cfg.neural_geometry.enable
        self.geometry_latent_dim = cfg.neural_geometry.latent_dim if self.neural_geometry_enabled else 0
        self.local_wi_wo = cfg.neural_geometry.local_wi_wo if self.neural_geometry_enabled else False
        self.neural_geometry_pos_enc = cfg.neural_geometry.positional_encoding if self.neural_geometry_enabled else False
        self.recompute_frame = cfg.neural_geometry.recompute_frame if self.neural_geometry_enabled else False
        self.neural_geometry_factor = cfg.neural_geometry.factor if self.neural_geometry_enabled else 0.4
        # Calculate total latent dimension
        if self.colorful_texture and self.larger_latent_dim:
            brdf_latent_dim = self.latent_dim * 3
        else:
            brdf_latent_dim = self.latent_dim
        
        # Add frame dimensions if predicting TBN
        total_latent_dim = brdf_latent_dim
        if self.predict_frame:
            total_latent_dim += 6  # normal (3) + tangent (3)
        
        # Add geometry latent dimensions
        if self.neural_geometry_enabled:
            total_latent_dim += self.geometry_latent_dim
        
        # 1. Create LatentTexture
        self.texture_resolution = getattr(cfg, 'texture_resolution', 256)
        blur_config = {
            'blur_sigma0': 8.0,
            'blur_half_life': 3333
        } if self.Gaussian_blur else None
        
        self.latent_texture = LatentTexture(
            resolution=self.texture_resolution,
            latent_dim=total_latent_dim,
            predict_frame=self.predict_frame,
            init_std=0.1,
            blur_config=blur_config
        )
        
        # 2. Create BRDFDecoder
        self.decoder = BRDFDecoder(
            cfg=cfg.decoder,
            latent_dim=self.latent_dim,
            use_pos_enc=True,
            different_decoder=self.different_decoder
        )
        
        # 3. Create NeuralGeometry if enabled
        if self.neural_geometry_enabled:
            self.neural_geometry = NeuralGeometry(
                cfg=cfg.neural_geometry,
                geometry_latent_dim=self.geometry_latent_dim,
                use_local_wi_wo=self.local_wi_wo,
                use_pos_enc=self.neural_geometry_pos_enc
            )
        else:
            self.neural_geometry = None
        
        # 4. Proxy BRDF for importance sampling
        self.proxy_brdf = ProxyPBRBRDF()
    
    # ------------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------------
    def compute_uv(self, pos: torch.Tensor, width: float = 0.4, length: float = 0.4):
        """
        Compute UV coordinates from 3D positions.
        
        Args:
            pos: [B, 3] 3D positions
            width: Width of the surface
            length: Length of the surface
        
        Returns:
            uv: [B, 2] UV coordinates in [0, 1]
        """
        half_w = width * 0.5
        half_l = length * 0.5
        x, z = pos[:, 0], pos[:, 2]
        
        u = (x + half_w) / width
        v = (z + half_l) / length
        
        return torch.stack([u, v], dim=-1)
    
    def world_to_local(
        self,
        v: torch.Tensor,
        normal: torch.Tensor,
        tangent: torch.Tensor
    ):
        """
        Transform vector from world space to local tangent space.
        
        Args:
            v: [B, 3] vector in world space
            normal: [B, 3] normal in world space
            tangent: [B, 3] tangent in world space
        
        Returns:
            v_local: [B, 3] vector in local space
        """
        if tangent is None:
            # Generate arbitrary tangent perpendicular to normal
            tangent = torch.cross(
                normal,
                torch.tensor([0.0, 0.0, 1.0], device=normal.device).expand_as(normal)
            )
        
        tangent_len = tangent.norm(dim=-1, keepdim=True)
        tangent = tangent / (tangent_len + 1e-8)
        bitangent = torch.cross(normal, tangent)
        
        v_local = torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1)
        ], dim=-1)
        
        return v_local
    
    def extract_frame_from_latent(self, latent: torch.Tensor):
        """
        Extract and orthonormalize normal and tangent from latent code.
        
        Args:
            latent: [B, total_dim] latent code with last 6 dims as normal+tangent
        
        Returns:
            normal: [B, 3] normalized normal vector
            tangent: [B, 3] normalized tangent vector (orthogonal to normal)
        """
        predicted_normal = latent[..., -6:-3]
        predicted_tangent = latent[..., -3:]
        
        # Normalize
        predicted_normal = NF.normalize(predicted_normal, dim=-1)
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        
        # Gram-Schmidt orthogonalization
        predicted_tangent = predicted_tangent - \
            torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        
        return predicted_normal, predicted_tangent
    
    def _sample_from_texture(self, uv: torch.Tensor, texture: torch.Tensor):
        """
        Sample latent from a given texture at UV coordinates.
        
        Args:
            uv: [B, 2] UV coordinates in [0, 1]
            texture: [1, D, H, W] texture to sample from
        
        Returns:
            latent: [B, D] sampled latent codes
        """
        # Convert UV to grid coordinates for F.grid_sample
        grid_coords = uv * 2.0 - 1.0  # [B, 2] -> [-1, 1]
        grid_coords = grid_coords.unsqueeze(1).unsqueeze(0)  # [1, 1, B, 2]
        
        # Bilinear sampling
        latent = NF.grid_sample(
            texture,  # [1, D, H, W]
            grid_coords,  # [1, 1, B, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # [1, D, 1, B]
        
        # Reshape to [B, D]
        latent = latent.squeeze(0).squeeze(-1).transpose(0, 1)
        
        return latent
    
    # ------------------------------------------------------------------------
    # Main BRDF evaluation API
    # ------------------------------------------------------------------------
    def eval_brdf(
        self,
        gt_params,
        pos: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        normal: torch.Tensor,
        uv: torch.Tensor,
        TBN: torch.Tensor,
        latent=None,
        batch_mask=None,
        footprint_vis=None,
        dp_du=None,
        dp_dv=None
    ):
        """
        Evaluate BRDF at given geometry and directions.
        
        Args:
            gt_params: Ground truth parameters (optional)
            pos: [B, 3] 3D positions
            wi: [B, 3] incoming light directions (world space)
            wo: [B, 3] outgoing view directions (world space)
            normal: [B, 3] normals (world space)
            uv: [B, 2] UV coordinates
            TBN: [B, 3, 3] tangent-bitangent-normal frame
            latent: Ignored (latent is sampled internally)
            batch_mask: Batch mask for batched operations
            footprint_vis: Footprint for mipmap level selection
            dp_du, dp_dv: Ray differentials
        
        Returns:
            brdf: [B, 3] BRDF values
            pdf: [B, 1] probability density
            uv_offset: [B, 2] UV offset (if neural geometry enabled, else zeros)
        """
        NoL = (wi * normal).sum(-1, keepdim=True)
        NoV = (wo * normal).sum(-1, keepdim=True)
        
        # 1. Get the (optionally blurred) texture ONCE
        if self.training and self.Gaussian_blur:
            tex = self.latent_texture.apply_gaussian_blur(self.global_step)
        else:
            tex = self.latent_texture.params
        
        # 2. Sample latent from the texture
        latent = self._sample_from_texture(uv, tex)
        
        # 3. Extract frame from latent if predicting frame
        if self.predict_frame:
            predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
        else:
            # Use geometry normal and tangent
            predicted_normal = normal
            predicted_tangent = TBN[:, :, 0] if TBN is not None else None
        
        # 3. Predict UV offset if neural geometry is enabled
        uv_offset = torch.zeros_like(uv)
        if self.neural_geometry_enabled:
            # Extract geometry latent
            geometry_latent = latent[..., -6-self.geometry_latent_dim:-6] if self.predict_frame else \
                             latent[..., -self.geometry_latent_dim:]
            
            # Get directions for geometry network
            if self.local_wi_wo:
                wi_for_geo = self.world_to_local(wi, predicted_normal, predicted_tangent)
                wo_for_geo = self.world_to_local(wo, predicted_normal, predicted_tangent)
            else:
                wi_for_geo = wi
                wo_for_geo = wo
            
            # Predict UV offset
            uv_offset = self.neural_geometry(wi_for_geo, wo_for_geo, geometry_latent) * self.neural_geometry_factor
            uv = uv + uv_offset
            uv = ((uv%1)+1)%1
            # Sample from the SAME blurred texture
            latent = self._sample_from_texture(uv, tex)
            
            # Recompute frame if needed
            if self.recompute_frame and self.predict_frame:
                predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
        
        # 4. Transform directions to local space
        wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
        wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        local_normal = torch.zeros_like(wi_local)
        local_normal[..., 2] = 1.0  # (0, 0, 1) in local space
        
        # 5. Encode directions
        enc_dir = self.decoder.encode_directions(wi_local, wo_local, local_normal)
        
        # 6. Extract BRDF latent (excluding frame and geometry latents)
        if self.colorful_texture:
            if self.different_decoder:
                if self.larger_latent_dim:
                    latent_r = latent[..., :self.latent_dim]
                    latent_g = latent[..., self.latent_dim:2*self.latent_dim]
                    latent_b = latent[..., 2*self.latent_dim:3*self.latent_dim]
                else:
                    latent_r = latent[..., :self.latent_dim]
                    latent_g = latent[..., :self.latent_dim]
                    latent_b = latent[..., :self.latent_dim]
                
                brdf_r = self.decoder(enc_dir, latent_r, 'r')
                brdf_g = self.decoder(enc_dir, latent_g, 'g')
                brdf_b = self.decoder(enc_dir, latent_b, 'b')
                brdf = torch.cat([brdf_r, brdf_g, brdf_b], dim=-1)
            else:
                brdf = self.decoder(enc_dir, latent[..., :self.latent_dim], None)
        else:
            brdf = self.decoder(enc_dir, latent[..., :self.latent_dim], None)
            brdf = brdf.repeat(1, 3)  # Replicate to RGB
        
        # 7. Calculate PDF (cosine-weighted)
        pdf = NoL / math.pi
        
        return brdf, predicted_normal, pdf, uv_offset
    
    def sample_brdf(
        self,
        params,
        pos: torch.Tensor,
        sample1: torch.Tensor,
        sample2: torch.Tensor,
        wo: torch.Tensor,
        normal: torch.Tensor,
        latent=None,
        batch_mask=None
    ):
        """
        Importance sample BRDF using proxy BRDF and evaluate with neural BRDF.
        
        Args:
            params: Dictionary with 'roughness' for proxy BRDF
            pos: [B, 3] 3D positions
            sample1: [B] uniform samples [0,1] for diffuse/specular choice
            sample2: [B, 2] uniform samples for hemisphere sampling
            wo: [B, 3] outgoing view directions (world space)
            normal: [B, 3] normals (world space)
            latent: Ignored (not used)
            batch_mask: Batch mask
        
        Returns:
            wi: [B, 3] sampled incoming directions (world space)
            pdf: [B, 1] sampling probability from proxy BRDF
            brdf_weight: [B, 3] BRDF/PDF ratio
        """
        # Sample direction from proxy BRDF
        wi_proxy, pdf_proxy = self.proxy_brdf.sample_brdf(
            pos, sample1, sample2, wo, normal,
            params['roughness'], batch_mask
        )
        
        # Evaluate neural BRDF at sampled direction
        # Note: Need to compute UV and TBN here
        uv = self.compute_uv(pos)
        TBN = None  # Compute TBN if needed
        
        mlp_brdf, _, _ = self.eval_brdf(
            params, pos, wi_proxy, wo, normal, uv, TBN,
            latent, batch_mask
        )
        
        # Calculate importance weight
        stop_gradient_pdf_proxy = pdf_proxy.detach()
        brdf_weight = torch.where(
            pdf_proxy > 0,
            mlp_brdf / (stop_gradient_pdf_proxy + 1e-8),
            torch.zeros_like(mlp_brdf)
        )
        
        return wi_proxy, stop_gradient_pdf_proxy, brdf_weight
    
    # ------------------------------------------------------------------------
    # Save/Load functionality
    # ------------------------------------------------------------------------
    def save_latent(self, path: str):
        """Save latent texture to file"""
        self.latent_texture.save(path)
    
    def load_latent(self, path: str):
        """Load latent texture from file"""
        self.latent_texture.load(path)

# ============================================================================
# 7. MULTI-MATERIAL LATENT BRDF - Auto-decoder for multiple materials
# ============================================================================
class MultiMaterialLatentBRDF(LightningModule):
    """
    Multi-material BRDF model using auto-decoder architecture.
    - Per-material latent codes (M materials)
    - Per-point latent codes (sum of points across all materials)
    - Shared MLP decoder across all materials
    
    Expected folder structure:
        data_folder/
            0/                      # material_id = 0
                point_metadata.json # contains {"num_points": N, ...}
            1/                      # material_id = 1
                point_metadata.json
            ...
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Store configuration
        self.cfg = cfg
        data_folder = getattr(cfg, 'data_folder', None)
        
        # Latent dimensions
        self.latent_dim = cfg.latent_dim
        self.predict_frame = cfg.predict_frame
        self.total_latent_dim = self.latent_dim + (6 if self.predict_frame else 0)
        # BRDF decoder settings
        self.use_pos_enc = cfg.use_pos_enc
        self.different_decoder = cfg.different_decoder
        
        # Load point metadata from material subfolders
        print(f"Loading point metadata from {data_folder}...")
        self.metadata = self._load_point_metadata(data_folder)
        
        num_materials = self.metadata['num_materials']
        total_points = self.metadata['total_points']
        
        print(f"Loaded {num_materials} materials with {total_points:,} total points")

        self.point_latent_bank = nn.Embedding(
            num_embeddings=total_points,
            embedding_dim=self.total_latent_dim
        )
        nn.init.normal_(self.point_latent_bank.weight, mean=0.0, std=cfg.init_std)
        
        if self.predict_frame:
            # Last 6 dimensions: normal (0,0,1) and tangent (0,1,0)
            with torch.no_grad():
                self.point_latent_bank.weight[:, -6:-3] = torch.tensor([0.0, 0.0, 1.0])  # normal
                self.point_latent_bank.weight[:, -3:] = torch.tensor([0.0, 1.0, 0.0])    # tangent
        
        # Shared BRDF decoder
        self.decoder = BRDFDecoder(
            cfg=cfg.decoder,
            latent_dim=self.latent_dim,
            use_pos_enc=self.use_pos_enc,
            different_decoder=self.different_decoder
        )
        
        print("Initialization complete!")
    
    def _load_point_metadata(self, data_folder):
        """
        Load point metadata from material subfolders.
        
        Expected structure:
        data_folder/
            0/                      # material_id = 0
                point_metadata.json # contains {"num_points": N, "num_observations": M, ...}
            1/                      # material_id = 1
                point_metadata.json
            ...
        
        Returns:
            metadata: dict with keys:
                - num_materials: int
                - total_points: int (sum across all materials)
                - materials: List[dict] with material info including point_range
                - material_point_offsets: dict mapping material_id -> global point offset
        """
        import json
        from pathlib import Path
        
        root = Path(data_folder)
        
        # Find material folders (named by material ID: 0, 1, 2, ...)
        material_folders = sorted([
            d for d in root.iterdir() 
            if d.is_dir() and d.name.isdigit()
        ], key=lambda x: int(x.name))
        
        materials = []
        material_point_offsets = {}
        global_point_offset = 0
        
        print(f"Found {len(material_folders)} potential material folders")
        
        for mat_folder in material_folders:
            material_id = int(mat_folder.name)
            metadata_path = mat_folder / "point_metadata.json"
            
            # Skip if metadata file doesn't exist or isn't readable
            if not metadata_path.exists():
                print(f"  Warning: {metadata_path} not found, skipping material {material_id}")
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    point_meta = json.load(f)
                
                num_points = point_meta['num_points']
                num_observations = point_meta.get('num_observations', 0)
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Warning: Failed to read {metadata_path}: {e}, skipping material {material_id}")
                continue
            
            # Store material info
            materials.append({
                'material_id': material_id,
                'name': mat_folder.name,
                'num_points': num_points,
                'num_observations': num_observations,
                'point_range': (global_point_offset, global_point_offset + num_points),
                'folder': str(mat_folder)
            })
            
            material_point_offsets[material_id] = global_point_offset
            global_point_offset += num_points
            
            print(f"  Material {material_id}: {num_points:,} points, {num_observations:,} observations")
        
        if len(materials) == 0:
            raise ValueError(f"No valid material folders found in {data_folder}")
        
        metadata = {
            'num_materials': len(materials),
            'total_points': global_point_offset,
            'materials': materials,
            'material_point_offsets': material_point_offsets,
        }
        
        # Build offset tensor for efficient indexing
        # offset_tensor[material_id] = global point offset for that material
        offset_tensor = torch.zeros(len(materials), dtype=torch.long)
        for mat_info in materials:
            mat_id = mat_info['material_id']
            offset_tensor[mat_id] = material_point_offsets[mat_id]
        
        # Register as buffer so it moves with the model to GPU
        self.register_buffer('material_offset_tensor', offset_tensor)
        
        return metadata
    
    def get_global_point_id(self, material_id, local_point_id):
        """
        Convert local point ID (per-material) to global point ID.
        
        Args:
            material_id: (1, N) or int - material indices
            local_point_id: (1, N) or int - local point indices within material
        
        Returns:
            global_point_id: (1, N) or int - global point indices for latent bank lookup
        """
        # Use pre-computed offset tensor for vectorized lookup
        offsets = self.material_offset_tensor[material_id]
        return local_point_id + offsets

    def extract_frame_from_latent(self, latent: torch.Tensor):
        """
        Extract and orthonormalize normal and tangent from latent code.
        
        Args:
            latent: [B, total_dim] latent code with last 6 dims as normal+tangent
        
        Returns:
            normal: [B, 3] normalized normal vector
            tangent: [B, 3] normalized tangent vector (orthogonal to normal)
        """
        predicted_normal = latent[..., -6:-3]
        predicted_tangent = latent[..., -3:]
        
        # Normalize
        predicted_normal = NF.normalize(predicted_normal, dim=-1)
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        
        # Gram-Schmidt orthogonalization
        predicted_tangent = predicted_tangent - \
            torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        if torch.isnan(predicted_tangent).any():
            print("predicted_tangent is nan")
        
        return predicted_normal, predicted_tangent
    
    def world_to_local(self, v, normal, tangent=None):
        """
        Transform vector from world space to local tangent space.
        
        Args:
            v: [B, 3] vector in world space
            normal: [B, 3] normal in world space
            tangent: [B, 3] tangent in world space (optional)
        
        Returns:
            v_local: [B, 3] vector in local space
        """
        if tangent is None:
            # Generate arbitrary tangent perpendicular to normal
            up = torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal)
            tangent = torch.cross(up, normal)
            tangent_len = tangent.norm(dim=-1, keepdim=True)
            
            # Handle collinear case
            collinear_mask = tangent_len.squeeze(-1) < 1e-6
            if collinear_mask.any():
                right = torch.tensor([1.0, 0.0, 0.0], device=normal.device).expand_as(normal)
                tangent[collinear_mask] = torch.cross(right[collinear_mask], normal[collinear_mask])
                tangent_len = tangent.norm(dim=-1, keepdim=True)
        else:
            tangent_len = tangent.norm(dim=-1, keepdim=True)
        
        tangent = tangent / (tangent_len + 1e-8)
        bitangent = torch.cross(normal, tangent)
        
        v_local = torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1)
        ], dim=-1)
        
        return v_local
    
    def eval_brdf(
        self,
        pos,
        wi,
        wo,
        normal,
        latent=None,
        point_ids=None,
        material_ids=None,
    ):
        """
        Evaluate BRDF at given geometry and directions.
        
        Args:
            pos: [B, 3] 3D positions
            wi: [B, 3] incoming light directions (world space)
            wo: [B, 3] outgoing view directions (world space)
            normal: [B, 3] normals (world space)
            latent: Ignored (latents retrieved from banks)
            point_ids: [B] LOCAL point indices (per-material, from dataloader)
            material_ids: [B] material indices (required for global point ID computation)
        
        Returns:
            brdf: [B, 3] BRDF values
            normal: [B, 3] normals (local space)
            pdf: [B, 1] probability density
        """
        
        # point_ids and material_ids should be provided by the dataloader
        if point_ids is None or material_ids is None:
            raise ValueError("point_ids and material_ids must be provided (from dataloader)")
        
        # Convert local point IDs to global point IDs
        global_point_ids = self.get_global_point_id(material_ids, point_ids)
        
        # Retrieve latents from banks
        latent = self.point_latent_bank(global_point_ids)        # [B, latent_dim]
  
        if self.predict_frame:
            predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
                # Check valid geometry
        NoL = (wi * predicted_normal).sum(-1, keepdim=True)
        NoV = (wo * predicted_normal).sum(-1, keepdim=True)
        wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
        wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        normal_local = torch.zeros_like(wi_local)
        normal_local[..., 2] = 1.0  # Normal is always (0,0,1) in local space
        
        # Encode directions
        enc_dir = self.decoder.encode_directions(wi_local, wo_local, normal_local)
        
        # Decode BRDF
        if self.different_decoder:
            # Decode each channel separately
            brdf_r = self.decoder(enc_dir, latent[:,:self.latent_dim], channel='r')
            brdf_g = self.decoder(enc_dir, latent[:,:self.latent_dim], channel='g')
            brdf_b = self.decoder(enc_dir, latent[:,:self.latent_dim], channel='b')
            brdf = torch.cat([brdf_r, brdf_g, brdf_b], dim=-1)  # [B, 3]
        else:
            brdf = self.decoder(enc_dir, latent[:, :self.latent_dim])  # [B, 1] or [B, 3]
            if brdf.shape[-1] == 1:
                brdf = brdf.expand(-1, 3)  # Expand to RGB
        
        # Simple diffuse PDF (can be improved with importance sampling)
        pdf = NoL.clamp(min=0) / math.pi
        if torch.isnan(brdf).any():
            print("brdf is nan")
        if torch.isnan(predicted_normal).any():
            print("normal is nan")
        return brdf, predicted_normal, pdf
    
    def sample_brdf(
        self,
        params,
        pos,
        sample1,
        sample2,
        wo,
        normal,
        latent=None,
        batch_mask=None,
        point_ids=None,
        material_ids=None
    ):
        """
        Sample BRDF using importance sampling.
        
        Args:
            params: Ground truth parameters (not used)
            pos: [B, 3] 3D positions
            sample1: [B] uniform samples [0,1]
            sample2: [B, 2] uniform samples [0,1]^2
            wo: [B, 3] outgoing view directions (world space)
            normal: [B, 3] normals (world space)
            latent: Ignored
            batch_mask: Batch mask
            point_ids: [B] LOCAL point indices (per-material, from dataloader)
            material_ids: [B] material indices
        
        Returns:
            wi: [B, 3] sampled incoming light directions
            pdf: [B, 1] probability density
            brdf_weight: [B, 3] BRDF / pdf
        """
        # Simple cosine-weighted hemisphere sampling (can use proxy BRDF)
        # This is a placeholder - implement proper importance sampling if needed
        
        # Cosine-weighted sampling
        theta = torch.asin(torch.sqrt(sample2[..., 0]))
        phi = 2 * math.pi * sample2[..., 1]
        
        # Local space directions
        wi_local = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=-1)
        
        # Transform to world space
        # Build TBN frame
        up = torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal)
        tangent = torch.cross(up, normal)
        tangent_len = tangent.norm(dim=-1, keepdim=True)
        tangent = tangent / (tangent_len + 1e-8)
        bitangent = torch.cross(normal, tangent)
        
        # Local to world
        wi = (wi_local[..., 0:1] * tangent + 
              wi_local[..., 1:2] * bitangent + 
              wi_local[..., 2:3] * normal)
        
        # Evaluate BRDF at sampled direction
        brdf, pdf, _ = self.eval_brdf(
            None, pos, wi, wo, normal,
            point_ids=point_ids,
            material_ids=material_ids
        )
        
        # BRDF weight = BRDF / pdf (for Monte Carlo integration)
        brdf_weight = brdf / pdf.clamp(min=1e-6)
        
        return wi, pdf, brdf_weight