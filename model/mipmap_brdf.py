import torch
import torch.nn as nn
import torch.nn.functional as NF
import math
from pytorch_lightning import LightningModule
import sys
sys.path.append('..')

from utils.ops import *
# from noise import pnoise3
from pnoise import pnoise
from nerfstudio.field_components import encodings as encoding

def hemisphere_detection(pos):
    return (pos[:,0] + pos[:,1] + pos[:,2]) > 0
# Manual implementation of 3D Perlin noise (naive version, not gradient coherent)
def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(a, b, t):
    return a + t * (b - a)


# ──────────────────────────────────────────────────────────────────────
# 1.  UV mapping for a centred 0.4 × 0.4 m patch in the x-z plane
#    x,z ∈ [-0.2 , +0.2]  →  u,v ∈ [0 , 1]
# ──────────────────────────────────────────────────────────────────────
def compute_uv(pos, width=0.4, length=0.4):
    half_w   = width  * 0.5               # 0.2
    half_l   = length * 0.5               # 0.2
    x, z     = pos[:, 0], pos[:, 2]

    u = (x + half_w) / width              # (-0.2→0,  +0.2→1)
    v = (z + half_l) / length             # (-0.2→0,  +0.2→1)

    return torch.stack([u, v], dim=-1)     # (N,2)



# ──────────────────────────────────────────────────────────────────────
# 2.  TBN frame – constant over the whole patch
#     ∂p/∂u = (width, 0, 0)       → T  ∥  +x
#     ∂p/∂v = (0, 0, length)      → B  ∥  +z
#     N     = B × T               → +y
# ──────────────────────────────────────────────────────────────────────
def compute_tbn(pos, uv, width: float = 0.4, length: float = 0.4):
    """
    Build a fixed T-B-N frame for each point.

    Args
    ----
    pos   : (B, 3)  xyz positions (only used for device / dtype)
    uv    : (B, 2)  – not used here but kept for API compatibility
    width : float   tangent scale along +X
    length: float   bitangent scale along +Z
    """
    batch  = pos.shape[0]
    device = pos.device
    dtype  = pos.dtype

    # Constant tangent  (width, 0, 0)
    T = pos.new_tensor([width, 0.0, 0.0]).repeat(batch, 1)  # (B, 3)

    # Constant bitangent (0, 0, length)
    B = pos.new_tensor([0.0, 0.0, length]).repeat(batch, 1)  # (B, 3)

    # Normal = B × T   (right-handed)
    N = torch.cross(B, T, dim=-1)

    # Normalise
    T = NF.normalize(T, dim=-1)
    B = NF.normalize(B, dim=-1)
    N = NF.normalize(N, dim=-1)

    return T, B, N

def build_mipmap(params, mipmap_levels, use_toksvig=False):
    B, H, W, C = params.shape
    mipmaps = [params]
    roughness_channel = 7
    normal_channel = 10
    
    for level in range(1, mipmap_levels + 1):
        new_h = max(1, H // (2 ** level))
        new_w = max(1, W // (2 ** level))
        
        if use_toksvig:
            # Apply Toksvig roughness boost for roughness channel
            downsampled = torch.nn.functional.interpolate(
                params.permute(0, 3, 1, 2), size=(new_h, new_w), mode='bilinear', align_corners=True
            ).permute(0, 2, 3, 1)
            
            roughness = downsampled[..., roughness_channel:roughness_channel+1]
            # Toksvig approximation: roughness' = sqrt(roughness^2 + (1-roughness^2) * variance)
            # For simplicity, we use a fixed variance based on mip level
            variance = 0.25 * (2 ** level - 1) / (2 ** level)
            boosted_roughness = torch.sqrt(roughness**2 + (1 - roughness**2) * variance)
            downsampled[..., roughness_channel:roughness_channel+1] = boosted_roughness
            
            mipmaps.append(downsampled)
        else:
            # Standard bilinear filtering
            downsampled = torch.nn.functional.interpolate(
                params.permute(0, 3, 1, 2), size=(new_h, new_w), mode='bilinear', align_corners=True
            ).permute(0, 2, 3, 1)
            mipmaps.append(downsampled)
     
    return mipmaps

def sample_texture(params, uv):
    B, H, W, C = params.shape
    
    # UV coordinates adjustment for grid_sample
    uv_grid = uv.unsqueeze(0).unsqueeze(2) * 2 - 1  # [1,N,1,2]

    # Sample the entire texture (all channels)
    sampled_texture = torch.nn.functional.grid_sample(
        params.permute(0, 3, 1, 2), uv_grid, mode='bilinear', align_corners=True
    ).squeeze(-1).permute(0, 2, 1).squeeze(0)  # [N,C]

    return sampled_texture


def sample_texture_with_offset(params, uv, wi, normal, dp_du, dp_dv):
    B, H, W, C = params.shape
    
    # UV coordinates adjustment for grid_sample
    uv_grid = uv.unsqueeze(0).unsqueeze(2) * 2 - 1  # [1,N,1,2]
    height_map = params[:, :, :, 13:14]
    height_map_sampled = torch.nn.functional.grid_sample(
        height_map.permute(0, 3, 1, 2), uv_grid, mode='bilinear', align_corners=True
    ).squeeze(-1).squeeze(-1).squeeze(0)  # [N,1]
    
    tangent = (wi - (wi * normal).sum(-1, keepdim=True) * normal)
    tangent = NF.normalize(tangent, dim=-1)

    # 2. Offset along tangent direction
    offset = height_map_sampled * tangent

    # 3. Build Jacobian of uv mapping
    J = torch.stack([dp_du, dp_dv], dim=-1)  # [N,3,2]

    # 4. Convert offset to (du,dv)
    dudv = torch.linalg.lstsq(J, offset.unsqueeze(-1)).solution.squeeze(-1)
    uv_grid = uv_grid + dudv.unsqueeze(0).unsqueeze(2) * 2  # rescale to [-1,1]
    
    # Sample the entire texture (all channels)
    sampled_texture = torch.nn.functional.grid_sample(
        params.permute(0, 3, 1, 2), uv_grid, mode='bilinear', align_corners=True
    ).squeeze(-1).permute(0, 2, 1).squeeze(0)  # [N,C]

    return sampled_texture

def sample_texture_trilinear(params, uv, mipmap_level, wi, normal, dp_du, dp_dv, height_map=False):
    """
    params: list of length L, each tensor of shape [B, H, W, C]
    uv:     [N, 2] in [0, 1] (per-ray UVs)
    mipmap_level: [N, 1] (can be fractional; floor part selects base level)
    return: [N, C]
    """
    device = uv.device
    dtype  = uv.dtype
    L = len(params)

    # ensure usable shapes/dtypes
    mipmap_level = mipmap_level.to(device=device, dtype=dtype).squeeze(-1)  # [N]
    uv = uv.to(device=device, dtype=dtype)

    # Compute base level (integer) and fractional alpha for cross-level blend
    l0 = torch.floor(mipmap_level).clamp(0, L - 1).long()          # [N]
    alpha = (mipmap_level - l0.to(dtype)).clamp(0.0, 1.0)          # [N]
    l1 = torch.clamp(l0 + 1, max=L - 1)                            # [N]

    # Prepare output
    # Infer C from first level
    C = params[0].shape[-1]
    out = torch.empty((uv.shape[0], C), device=device, dtype=dtype)

    # grid_sample wants normalized coords in [-1, 1]
    # uv in [0,1] -> grid in [-1,1]
    def uv_to_grid(u):
        g = u * 2.0 - 1.0
        return g

    # Helper to bilinear sample a single level for a subset of rays
    def sample_level(level_idx, uv_subset, wi, normal, dp_du, dp_dv, height_map=False):
        # Take batch 0 (common in texture-parameter setups)
        if height_map:
            uv_subset_grid = uv_to_grid(uv_subset).unsqueeze(0).unsqueeze(2)  # [1, M, 1, 2]
            height_map = params[level_idx][:, :, :, 13:14]
            height_map_sampled = torch.nn.functional.grid_sample(
                height_map.permute(0, 3, 1, 2), uv_subset_grid, mode='bilinear', align_corners=True
            ).squeeze(-1).permute(0, 2, 1).squeeze(0) # [M,1]
            
            tangent = (wi - (wi * normal).sum(-1, keepdim=True) * normal)
            tangent = NF.normalize(tangent, dim=-1)
            offset = height_map_sampled * tangent
            J = torch.stack([dp_du, dp_dv], dim=-1)  # [N,3,2]
            dudv = torch.linalg.lstsq(J, offset.unsqueeze(-1)).solution.squeeze(-1)
            uv_subset_grid = uv_subset_grid + dudv.unsqueeze(0).unsqueeze(2) * 2  # rescale to [-1,1]
        else:
            uv_subset_grid = uv_to_grid(uv_subset).unsqueeze(0).unsqueeze(2)  # [1, M, 1, 2]
            
        tex = params[level_idx][0]  # [H, W, C]
        H, W, C_ = tex.shape
        assert C_ == C
        tex = tex.permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, C, H, W]
        grid = uv_subset_grid  # [1, M, 1, 2]
        sampled = NF.grid_sample(tex, grid, mode='bilinear', padding_mode='border', align_corners=True)
        # sampled: [1, C, M, 1] -> [M, C]
        return sampled.squeeze(0).squeeze(-1).permute(1, 0).contiguous()

    # Loop over levels to avoid sampling full textures for all rays unnecessarily
    for level in range(L):
        # rays whose base level == level
        mask = (l0 == level)
        if not torch.any(mask):
            continue

        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        uv_subset = uv[idx]  # [M, 2]
        wi_subset = wi[idx]
        normal_subset = normal[idx]
        dp_du_subset = dp_du[idx]
        dp_dv_subset = dp_dv[idx]
        a_subset = alpha[idx]  # [M]

        # base level sample
        s0 = sample_level(level, uv_subset, wi_subset, normal_subset, dp_du_subset, dp_dv_subset, height_map)  # [M, C]

        # upper level (could be same as base when at the top mip)
        level_up = min(level + 1, L - 1)
        if level_up != level:
            s1 = sample_level(level_up, uv_subset, wi_subset, normal_subset, dp_du_subset, dp_dv_subset, height_map)  # [M, C]
            # blend with alpha
            out[idx] = (1.0 - a_subset.unsqueeze(1)) * s0 + a_subset.unsqueeze(1) * s1
        else:
            # top level: no upper level to blend with
            out[idx] = s0

    return out  # [N, C]

def local_to_world_normal(sampled_normal, T, B, N):
    # Adjust normal map from [0,1] to [-1,1]
    sampled_normal = sampled_normal * 2 - 1
    sampled_normal = torch.nn.functional.normalize(sampled_normal, dim=-1)
    n_world = (
        sampled_normal[:, 0:1] * T +
        sampled_normal[:, 1:2] * B +
        sampled_normal[:, 2:3] * N
    )
    n_world = torch.nn.functional.normalize(n_world, dim=-1)
    return n_world

def grad(hash, x, y, z):
    h = hash & 15
    u = torch.where(h < 8, x, y)
    cond1 = h < 4
    cond2 = (h == 12) | (h == 14)
    v = torch.where(cond1, y, torch.where(cond2, x, z))
    u_term = torch.where((h & 1) == 0, u, -u)
    v_term = torch.where((h & 2) == 0, v, -v)
    return u_term + v_term

class MipmapLearnableSvPBRBRDF(nn.Module):
    """ Base BRDF class """
    def __init__(self, cfg, texture_res=512):
        super(MipmapLearnableSvPBRBRDF, self).__init__()
        self.anisotropic = cfg.anisotropic
        self.perlin_scale = cfg.perlin_scale
        self.perlin_randomness = cfg.perlin_randomness
        self.scale_factor = cfg.scale_factor
        self.normal_map = cfg.normal_map
        #self.pbr_constraint = cfg.pbr_constraint
        self.init_std = cfg.init_std
        self.soft_constraint = cfg.soft_constraint
        self.mipmap_levels = cfg.mipmap_levels
        self.prefliter = cfg.prefliter 
        self.texture_res = cfg.texture_res
        self.base_footprint = cfg.base_footprint
        self.height_map = cfg.use_height_map
        if self.prefliter:
            texture_init = torch.randn(1, self.texture_res, self.texture_res, 21) * self.init_std
            
            # Initialize normal map channels to (0, 0, 1) which represents no perturbation
            texture_init[0, :, :, 10] = 0.0  # x component
            texture_init[0, :, :, 11] = 0.0  # y component  
            texture_init[0, :, :, 12] = 1.0  # z component
            texture_init[0, :, :, 13] = 0.0  # height
            
            self.pbr_texture = nn.Parameter(texture_init)
        else:
            # Initialize mipmap levels with different resolutions
            self.mipmap_textures = nn.ParameterList()
            
            for level in range(self.mipmap_levels):
                # Calculate resolution for this mipmap level
                level_res = max(1, self.texture_res >> level)
                
                # Initialize texture for this level
                texture_init = torch.randn(1, level_res, level_res, 21) * self.init_std
                
                # Initialize normal map channels to (0, 0, 1) which represents no perturbation
                texture_init[0, :, :, 10] = 0.0  # x component
                texture_init[0, :, :, 11] = 0.0  # y component  
                texture_init[0, :, :, 12] = 1.0  # z component
                texture_init[0, :, :, 13] = 0.0  # height
                # Create parameter for this mipmap level
                mipmap_param = nn.Parameter(texture_init)
                self.mipmap_textures.append(mipmap_param)

    
    def diffuse_sampler(self, sample2, normal):
        """ sampling diffuse lobe: wi ~ NoV/math.pi 
        Args:
            sample2: Bx2 unIform samples
            normal: Bx3 normal
        Return:d
            wi: Bx3 sampled direction in world space
        """
        theta = torch.asin(sample2[...,0].sqrt())
        phi = math.pi*2*sample2[...,1]
        wi = angle2xyz(theta,phi)
        
        Nmat = get_normal_space(normal)
        wi = (wi[:,None]@Nmat.permute(0,2,1)).squeeze(1)    
        if wi.isnan().any():
            print("wi is nan")
        return wi


    def specular_sampler(self, sample2,roughness, wo, normal):
        """ sampling ggx lobe: h ~ D/(VoH*4)*NoH
        Args:
            sample2: Bx3 uniform samples
            roughness: Bx1 roughness
            wo: Bx3 viewing direction
            normal: Bx3 normal
        Return:
            wi: Bx3 sampled direction in world space
        """
        alpha = (roughness * roughness).squeeze(-1)
        
        # sample half vector
        theta = (1-sample2[...,0])/((sample2[...,0]*(alpha*alpha-1)+1))
        theta = torch.acos(theta.sqrt())

        phi = 2*math.pi*sample2[...,1]
        wh = angle2xyz(theta,phi)

        # half vector to wi
        Nmat = get_normal_space(normal)
        wh = (wh[:,None]@Nmat.permute(0,2,1)).squeeze(1)
        wi = 2*(wo*wh).sum(-1,keepdim=True)*wh-wo
        wi = NF.normalize(wi,dim=-1)
        return wi

    def compute_svbrdf_pdf(self, albedo, roughness, metallic, wi, wo, normal):
        h = NF.normalize(wi+wo,dim=-1)
        NoL = (wi*normal).sum(-1,keepdim=True).relu()
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        D = D_GGX(NoH,roughness)
        pdf_spec = D.data/(4*VoH.clamp_min(1e-4))*NoH
        pdf_diff = NoL/math.pi
        pdf = 0.5*pdf_spec + 0.5*pdf_diff

        kd = albedo*(1-metallic)
        ks = 0.04*(1-metallic) + albedo*metallic
    
        G = G_Smith(NoV,NoL,roughness)
        F = fresnelSchlick(VoH,ks)
        brdf_diff = kd/math.pi*NoL
        brdf_spec = D*G*F/4.0*NoL

        brdf = brdf_diff + brdf_spec
        
        if torch.isnan(brdf).any() or torch.isinf(brdf).any():
            print("brdf is nan or inf")
        # Debug: set all brdf to 1
        # brdf = torch.ones_like(brdf)
        return brdf, pdf

    def compute_anisotropic_svbrdf_pdf(self,
                                    diffuse, ao, ax, ay, metallic, ior,
                                    wi, wo,
                                    normal, tangent):
        """
        Inputs:
            diffuse  : [N,3] Diffuse (base-color)
            ao       : [N,1] Ambient Occlusion
            ax, ay   : [N,1] Directional roughness
            metallic : [N,1] Metallic factor
            ior      : [N,1] Specular IOR
            wi, wo   : [N,3] Incident & outgoing directions
            normal   : [N,3] Surface normal
            tangent  : [N,3] Tangent vector (bitangent computed internally)
        Returns:
            brdf [N,3] , pdf [N,1]
        """
        B = NF.normalize(torch.cross(normal, tangent, dim=-1), dim=-1)
        h = NF.normalize(wi + wo, dim=-1)

        NoL = (wi * normal).sum(-1, keepdim=True).clamp_min(0.0)
        NoV = (wo * normal).sum(-1, keepdim=True).clamp_min(0.0)
        VoH = (wo * h).sum(-1, keepdim=True).clamp_min(1e-4)
        NoH = (normal * h).sum(-1, keepdim=True).clamp_min(1e-4)

        # --- specular distribution and geometry -------------------
        D = D_GGX_aniso(h, normal, tangent, B, ax, ay)
        G = G_Smith_aniso(wi, wo, normal, tangent, B, ax, ay)

        # --- Fresnel base reflectance -----------------------------
        F0_dielectric = ((ior - 1) / (ior + 1)).pow(2)
        F0 = F0_dielectric * (1 - metallic) + diffuse * metallic
        F = fresnelSchlick(VoH, F0)

        # --- Diffuse and Specular reflectances --------------------
        kd = diffuse * (1 - metallic) * ao  # apply AO only to diffuse
        ks = 1.0                            # standard GGX specular strength (no scaling)

        # --- BRDF computation -------------------------------------
        brdf_spec = ks * (D * G * F) / (4.0 * NoL * NoV + 1e-6)
        brdf_diff = kd / math.pi
        brdf = brdf_spec + brdf_diff

        # --- PDF (half diffuse, half specular) --------------------
        pdf_spec = D * NoH / (4.0 * VoH)
        pdf_diff = NoL / math.pi
        pdf = 0.5 * (pdf_spec + pdf_diff)

        return brdf, pdf


    def eval_brdf(self, params, pos, wi, wo, normal,uv, TBN, latent=None, batch_mask=None, footprint=None, dp_du=None, dp_dv=None):
        """ wi is light direction, wo is view direction """
        TBN=TBN.permute(2,0,1)
        #print("TBN",TBN.shape)
        NoL = (wi*normal).sum(-1, keepdim=True)
        NoV = (wo*normal).sum(-1, keepdim=True)
        valid_geometry = (NoL > 0) & (NoV > 0)
        T, B, N_geo = TBN
        footprint_ratio = footprint / self.base_footprint
        mipmap_level = torch.log2(footprint_ratio).clamp(min=0, max=self.mipmap_levels - 1)
        
        if not valid_geometry.any():
            return torch.zeros_like(wi), torch.zeros(wi.shape[0], 1, device=wi.device)
        if self.prefliter:
            mipmaps = build_mipmap(self.pbr_texture, self.mipmap_levels, False)
        else:
            mipmaps = self.mipmap_textures
            
        sampled_texture = sample_texture_trilinear(mipmaps, uv, mipmap_level, wi, normal, dp_du, dp_dv, self.height_map)
            
        if self.anisotropic:
            # Extract anisotropic texture maps
            diffuse = sampled_texture[:, 0:3]                    # Color channels
            ao = sampled_texture[:, 3:4]                       # Ambient occlusion
            arm = sampled_texture[:, 6:9]                      # ARM channels
            roughness_map = arm[:, 1:2]           # Roughness map
            metallic_map = sampled_texture[:, 10:11]           # Metallic map
            ior_map = sampled_texture[:, 11:12]                # Specular IOR
            aniso_rot = sampled_texture[:, 12:15]              # Anisotropy rotation
            aniso_str = sampled_texture[:, 15:18]              # Anisotropy strength
            normal_local = sampled_texture[:, 18:21]           # Normal channels (DX)
            
            
            # Extract material properties
            roughness = roughness_map                          # Use dedicated roughness map
            metallic = metallic_map                            # Use dedicated metallic map
            ior = ior_map                                      # Use IOR map
            
            aniso_strength = aniso_str[:, 0:1]                 # Use first channel of anisotropy strength
            base_roughness = roughness       # Clamp roughness to valid range
            
            # Compute ax and ay based on anisotropy strength
            ax = base_roughness * (1.0 + aniso_strength)      # Tangent direction roughness
            ay = base_roughness * (1.0 - aniso_strength * 0.5) # Bitangent direction roughness
            # Step 3: Transform local normal to world space
            n_world = local_to_world_normal(normal_local, T, B, N_geo)
            rotation_angle = aniso_rot[:, 0:1] * 2.0 * torch.pi  # Convert [0,1] to [0, 2π]
            
            # Create rotation matrix in tangent space (rotate around normal)
            cos_theta = torch.cos(rotation_angle)
            sin_theta = torch.sin(rotation_angle)
            
            # Rotate the tangent vector in the T-B plane
            T_rot = cos_theta * T + sin_theta * B
            T_reproj = T_rot - (T_rot * n_world).sum(-1, keepdim=True) * n_world
            T_ortho  = NF.normalize(T_reproj, dim=-1)
            # The rotated tangent is already in world space since T and B are in world space
            B_ortho  = torch.cross(n_world, T_ortho, dim=-1)
            B_ortho  = NF.normalize(B_ortho, dim=-1)  
            
            # Evaluate anisotropic BRDF
            if self.soft_constraint:
                albedo = torch.sigmoid(diffuse)
                roughness = torch.sigmoid(roughness_map)
                metallic = torch.sigmoid(metallic_map)
                normal_map = NF.normalize(normal_local, dim=-1)
            else:
                albedo = torch.clamp(diffuse, 0.01, 0.99)  # [eps, inf]
                roughness = torch.clamp(roughness_map, 0.01, 0.99)  # [eps, 1-eps]
                metallic = torch.clamp(metallic_map, 0.01, 0.99)  # [eps, 1-eps]

            brdf, pdf = self.compute_anisotropic_svbrdf_pdf(
                diffuse, ao, ax, ay, metallic, ior,
                wi, wo, n_world, T_ortho
            )
            
        else:
            arm = sampled_texture[:, 6:9]      # ARM channels
            color = sampled_texture[:, 0:3]    # Color channels
            normal_local = sampled_texture[:, 10:13]  # Normal channels
            albedo, roughness, metallic = arm[:, 0:1], arm[:, 1:2], arm[:, 2:3]
            if self.soft_constraint:
                albedo = torch.sigmoid(color)
                roughness = torch.sigmoid(arm[:, 1:2])
                metallic = torch.sigmoid(arm[:, 2:3])
                normal_local = NF.normalize(normal_local, dim=-1)
            else:
                color = torch.clamp(color, 0.01, 0.99)  # [eps,1-eps]
                albedo = torch.clamp(albedo, 0.01, 0.99)  # [eps,1-eps]
                roughness = torch.clamp(arm[:, 1:2], 0.01, 0.99)  # [eps, 1-eps]
                metallic = torch.clamp(arm[:, 2:3], 0.01, 0.99)  # [eps, 1-eps]

            # Step 4: Transform local normal to world
            # n_world = local_to_world_normal(normal_local, T, B, N_geo)
            n_world = normal_local # use normal map instead of geometry normal

            # Evaluate BRDF with mapped parameters
            if self.normal_map:
                brdf, pdf = self.compute_svbrdf_pdf(albedo, roughness, metallic, wi, wo, n_world)
            else:
                brdf, pdf = self.compute_svbrdf_pdf(albedo, roughness, metallic, wi, wo, normal)
            brdf = color * brdf
            # brdf = torch.ones_like(brdf)
            
        return brdf, pdf
    
    def sample_brdf(self, params, pos, sample1, sample2, wo, normal, latent=None, batch_mask=None):
        """ TODO """
        B = sample1.shape[0]
        device = sample1.device

        # Compute blending mask
        mask = perlin_mask(pos, scale=self.perlin_scale, randomness=self.perlin_randomness).view(-1)

        # Sample diffuse directions (same across both materials)
        wi_diffuse = self.diffuse_sampler(sample2, normal)

        # Sample specular directions for both materials
        roughness0 = params['roughness'][:, 0]
        roughness1 = params['roughness'][:, 1]
        wi_spec0 = self.specular_sampler(sample2, roughness0, wo, normal)
        wi_spec1 = self.specular_sampler(sample2, roughness1, wo, normal)

        # Mix sampled directions based on mask
        wi_spec = wi_spec1

        # Choose whether to use diffuse or specular
        select_diff = sample1 > 0.5
        wi = torch.zeros(B, 3, device=device)
        wi[select_diff] = wi_diffuse[select_diff]
        wi[~select_diff] = wi_spec[~select_diff]

        # Evaluate blended BRDF
        brdf, pdf = self.eval_brdf(params, pos, wi, wo, normal, latent, batch_mask)
        brdf_weight = torch.where(pdf > 0, brdf / pdf, torch.zeros_like(brdf))
        brdf_weight[brdf_weight.isnan()] = 0

        return wi, pdf, brdf_weight
