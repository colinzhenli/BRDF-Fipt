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


def grad(hash, x, y, z):
    h = hash & 15
    u = torch.where(h < 8, x, y)
    cond1 = h < 4
    cond2 = (h == 12) | (h == 14)
    v = torch.where(cond1, y, torch.where(cond2, x, z))
    u_term = torch.where((h & 1) == 0, u, -u)
    v_term = torch.where((h & 2) == 0, v, -v)
    return u_term + v_term


def perlin_noise_3d(pos, scale=1.0):
    # torch.manual_seed(42)
    # permutation = torch.arange(256, dtype=torch.int32)
    # permutation = permutation[torch.randperm(256)]
    permutation = torch.tensor([
        151,160,137,91,90,15,
        131,13,201,95,96,53,194,233,7,225,
        140,36,103,30,69,142,8,99,37,240,21,10,23,
        190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
        35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,
        168, 68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,
        111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
        102,143,54, 65,25,63,161,1,216,80,73,209,76,132,187,208,
        89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,
        186, 3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,
        82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,
        183,170,213,119,248,152, 2,44,154,163,70,221,153,101,155,167,
        43,172,9,129,22,39,253, 19,98,108,110,79,113,224,232,178,
        185, 112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,
        179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,
        199,106,157,184,84,204,176,115,121,50,45,127, 4,150,254,138,
        236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215
        ], dtype=torch.int32).to(pos.device)
    p = torch.cat([permutation, permutation])
    pos = pos * scale
    xi = pos[:, 0].floor().long() & 255
    yi = pos[:, 1].floor().long() & 255
    zi = pos[:, 2].floor().long() & 255

    xf = pos[:, 0] - pos[:, 0].floor()
    yf = pos[:, 1] - pos[:, 1].floor()
    zf = pos[:, 2] - pos[:, 2].floor()

    u = fade(xf)
    v = fade(yf)
    w = fade(zf)

    xi1 = (xi + 1) & 255
    yi1 = (yi + 1) & 255
    zi1 = (zi + 1) & 255

    aaa = p[p[p[    xi ] +     yi ] +     zi ]
    aba = p[p[p[    xi ] +   yi1 ] +     zi ]
    aab = p[p[p[    xi ] +     yi ] +   zi1 ]
    abb = p[p[p[    xi ] +   yi1 ] +   zi1 ]
    baa = p[p[p[  xi1 ] +     yi ] +     zi ]
    bba = p[p[p[  xi1 ] +   yi1 ] +     zi ]
    bab = p[p[p[  xi1 ] +     yi ] +   zi1 ]
    bbb = p[p[p[  xi1 ] +   yi1 ] +   zi1 ]

    x1 = lerp(grad(aaa, xf    , yf    , zf    ),
              grad(baa, xf-1 , yf    , zf    ), u)
    x2 = lerp(grad(aba, xf    , yf-1 , zf    ),
              grad(bba, xf-1 , yf-1 , zf    ), u)
    y1 = lerp(x1, x2, v)

    x3 = lerp(grad(aab, xf    , yf    , zf-1 ),
              grad(bab, xf-1 , yf    , zf-1 ), u)
    x4 = lerp(grad(abb, xf    , yf-1 , zf-1 ),
              grad(bbb, xf-1 , yf-1 , zf-1 ), u)
    y2 = lerp(x3, x4, v)

    return lerp(y1, y2, w)

# Generate a blending mask based on 3D Perlin noise
# Args:
#   pos: Tensor of shape (N, 3) representing 3D positions
#   scale: Controls the frequency of the noise
#   randomness: Controls the blending strength between noise and uniform value 0.5
# Returns:
#   A tensor of shape (N, 1) containing values in [0, 1] for blending

def perlin_mask(pos, scale=1.0, randomness=0.5):
    mask = perlin_noise_3d(pos, scale)
    # Scale the mask to make it sharper at boundaries and smoother elsewhere
    # First, blend with a uniform value of 0.5 based on randomness parameter
    
    # Apply linear scaling and clipping to create transitions at boundaries
    # Scale the mask linearly and then clip to ensure values stay in [0, 1]
    scale_factor = 6  # Controls the scaling strength
    mask = mask * scale_factor
    mask = torch.clamp(mask, 0.0, 1.0)  # Ensure values stay within valid range
    return mask.view(-1, 1)

def load_pbr_texture(pbr_folder):
    import cv2
    from PIL import Image
    import torchvision.transforms as T
    import os

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

    # Detect cloth name from path
    cloth_name = "fabric_pattern_07"  # default
    if "denim" in pbr_folder.lower():
        cloth_name = "denim_fabric_03"
    
    # Collect PBR texture data
    try:
        # Read all texture maps based on the detected cloth name
        if cloth_name == "denim_fabric_03":
            diffuse = read_img(f"{cloth_name}_diff_4k.jpg")                    # [3,H,W] - Diffuse map
            ao = read_img(f"{cloth_name}_ao_4k.jpg")                         # [3,H,W] - Ambient occlusion
            arm = read_img(f"{cloth_name}_arm_4k.jpg")                       # [3,H,W] - (A, Roughness, Metalness)
            rough = read_img(f"{cloth_name}_rough_4k.exr")                   # [1,H,W] - Roughness map (EXR)
            metal = read_img(f"{cloth_name}_metal_4k.exr")                   # [1,H,W] - Metallic map (EXR)
            spec_ior = read_img(f"{cloth_name}_spec_ior_4k.exr")             # [1,H,W] - Specular IOR (EXR)
            aniso_rot = read_img(f"{cloth_name}_anisotropy_rotation_4k.jpg") # [3,H,W] - Anisotropy rotation
            aniso_str = read_img(f"{cloth_name}_anisotropy_strength_4k.jpg") # [3,H,W] - Anisotropy strength
            nor_dx = read_img(f"{cloth_name}_nor_dx_4k.exr")                 # [3,H,W] - Normal map X
            nor_gl = read_img(f"{cloth_name}_nor_gl_4k.exr")                 # [3,H,W] - Normal map GL
            
            # Combine all available channels for denim fabric
            # [Color (3) + AO (3) + ARM (3) + Roughness (1) + Metal (1) + Spec IOR (1) + Aniso Rot (3) + Aniso Str (3) + Normal DX (1) + Normal GL (1)] = 20 channels
            tex = torch.cat([
                diffuse,                # RGB color (3 channels) 0:3
                ao,                   # Ambient occlusion (3 channels) 3:6
                arm,                  # ARM texture (3 channels) 6:9
                rough,                # Roughness map (1 channel) 9:10
                metal,                # Metallic map (1 channel) 10:11
                spec_ior,             # Specular IOR (1 channel) 11:12
                aniso_rot,            # Anisotropy rotation (3 channels) 12:15
                aniso_str,            # Anisotropy strength (3 channels) 15:18
                nor_dx,               # Normal map X (3 channels) 18:21
                nor_gl                # Normal map GL (3 channels) 21:24
            ], dim=0)  # [24,H,W]
        else:
            # Default fabric pattern maps
            col_1 = read_img(f"{cloth_name}_col_1_4k.jpg")      # [3,H,W] - Color map
            ao = read_img(f"{cloth_name}_ao_4k.jpg")            # [3,H,W] - Ambient occlusion
            arm = read_img(f"{cloth_name}_arm_4k.jpg")          # [3,H,W] - (A, Roughness, Metalness)
            rough = read_img(f"{cloth_name}_rough_4k.exr")      # [1,H,W] - Roughness map (EXR)
            nor_dx = read_img(f"{cloth_name}_nor_dx_4k.exr")    # [3,H,W] - Normal map X
            nor_gl = read_img(f"{cloth_name}_nor_gl_4k.exr")    # [3,H,W] - Normal map GL
            
            # Combine all available channels for default fabric
            # [Color (3) + AO (3) + ARM (3) + Roughness (1) + Normal DX (1) + Normal GL (1)] = 12 channels
            tex = torch.cat([
                col_1,                # RGB color (3 channels) 0:3
                ao,                   # Ambient occlusion (3 channels) 3:6
                arm,                  # ARM texture (3 channels) 6:9
                rough,                # Roughness map (1 channel) 9:10
                nor_dx,               # Normal map X (3 channels) 10:13
                nor_gl                # Normal map GL (3 channels) 13:16
            ], dim=0)  # [16,H,W]
        
        return tex.permute(1, 2, 0).contiguous()  # [H,W,C] => [U,V,C]
    except Exception as e:
        print(f"Error loading PBR textures: {e}")
        # Return a default texture if loading fails
        H, W = 1024, 1024
        return torch.ones(H, W, 12)  # Default to 12 channels for compatibility

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


def sample_texture(params, uv):
    B, H, W, C = params.shape
    
    # UV coordinates adjustment for grid_sample
    uv_grid = uv.unsqueeze(0).unsqueeze(2) * 2 - 1  # [1,N,1,2]

    # Sample the entire texture (all channels)
    sampled_texture = torch.nn.functional.grid_sample(
        params.permute(0, 3, 1, 2), uv_grid, mode='bilinear', align_corners=True
    ).squeeze(-1).permute(0, 2, 1).squeeze(0)  # [N,C]

    return sampled_texture

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

class SvPBRBRDF(nn.Module):
    """ Base BRDF class """
    def __init__(self, cfg, albedo=torch.ones(1, 3), pbr_texture=None):
        super(SvPBRBRDF,self).__init__()
        # Initialize learnable material parameters
        
        self.albedo = nn.Parameter(albedo)
        self.anisotropic = cfg.anisotropic
        self.perlin_scale = cfg.perlin_scale
        self.perlin_randomness = cfg.perlin_randomness
        self.scale_factor = cfg.scale_factor
        self.normal_map = cfg.normal_map
        if pbr_texture is None:
            self.pbr_texture = load_pbr_texture(cfg.pbr_texture_path).unsqueeze(0).cuda()
        else:
            self.pbr_texture = pbr_texture.unsqueeze(0).cuda()
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
            import pdb; pdb.set_trace()

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


    def eval_brdf(self, params, pos, wi, wo, normal, uv, TBN, latent=None, batch_mask=None):
        NoL = (wi*normal).sum(-1, keepdim=True)
        NoV = (wo*normal).sum(-1, keepdim=True)
        valid_geometry = (NoL > 0) & (NoV > 0)
        
        if not valid_geometry.any():
            return torch.zeros_like(wi), torch.zeros(wi.shape[0], 1, device=wi.device)
        radius = 0.2
        factor = self.scale_factor
        params = self.pbr_texture
        H, W = params.shape[1], params.shape[2]
        crop_h = int((H - H * factor) // 2)
        crop_w = int((W - W * factor) // 2)
        new_h = int(H * factor)
        new_w = int(W * factor)
        params = params[:, crop_h:crop_h+new_h, crop_w:crop_w+new_w, :]
        # uv_fake = compute_uv(pos, 0.8, 0.8)
        # print("uv_fake",uv_fake.shape)
        # print("uv",uv.shape)
        # Step 1: TBN frame
        # T, B, N_geo = compute_tbn(pos, uv, 0.8, 0.8)
        T = TBN[:, :, 0]  # Extract tangent vectors
        B = TBN[:, :, 1]  # Extract bitangent vectors
        N_geo = TBN[:, :, 2]  # Extract normal vectors
        # Step 2: Texture sampling
        sampled_texture = sample_texture(params, uv)
        if self.anisotropic:
            # Extract anisotropic texture maps
            diffuse = sampled_texture[:, 0:3]                    # Color channels
            ao = sampled_texture[:, 3:4]                       # Ambient occlusion
            arm = sampled_texture[:, 6:9]                      # ARM channels
            roughness_map = sampled_texture[:, 9:10]           # Roughness map
            metallic_map = sampled_texture[:, 10:11]           # Metallic map
            ior_map = sampled_texture[:, 11:12]                # Specular IOR
            aniso_rot = sampled_texture[:, 12:15]              # Anisotropy rotation
            aniso_str = sampled_texture[:, 15:18]              # Anisotropy strength
            normal_local = sampled_texture[:, 18:21]           # Normal channels (DX)
            
            
            # Extract material properties
            roughness = roughness_map                          # Use dedicated roughness map
            metallic = metallic_map                            # Use dedicated metallic map
            ior = ior_map                                      # Use IOR map
            
            # Compute anisotropic roughness parameters
            # aniso_str controls the strength of anisotropy (0 = isotropic, 1 = fully anisotropic)
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
            brdf, pdf = self.compute_anisotropic_svbrdf_pdf(
                diffuse, ao, ax, ay, metallic, ior,
                wi, wo, n_world, T_ortho
            )
            
        else:
            arm = sampled_texture[:, 6:9]      # ARM channels
            color = sampled_texture[:, 0:3]    # Color channels
            normal_local = sampled_texture[:, 10:13]  # Normal channels
            albedo, roughness, metallic = arm[:, 0:1], arm[:, 1:2], arm[:, 2:3]
            # Step 4: Transform local normal to world
            n_world = local_to_world_normal(normal_local, T, B, N_geo)

            # Evaluate BRDF with mapped parameters
            if self.normal_map:
                brdf, pdf = self.compute_svbrdf_pdf(albedo, roughness, metallic, wi, wo, n_world)
            else:
                brdf, pdf = self.compute_svbrdf_pdf(albedo, roughness, metallic, wi, wo, normal)
            brdf = color * brdf
            
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

class LearnableSvPBRBRDF(nn.Module):
    """ Base BRDF class """
    def __init__(self, cfg, texture_res=512):
        super(LearnableSvPBRBRDF, self).__init__()
        self.anisotropic = cfg.anisotropic
        self.perlin_scale = cfg.perlin_scale
        self.perlin_randomness = cfg.perlin_randomness
        self.scale_factor = cfg.scale_factor
        self.normal_map = cfg.normal_map
        self.pbr_constraint = cfg.pbr_constraint
        self.init_std = cfg.init_std
        self.soft_constraint = cfg.soft_constraint

        # Randomly initialize learnable PBR texture
        # Channels: [albedo(3), roughness(1), metallic(1), normal(3)] = 8 total
        self.texture_res = texture_res
        self.pbr_texture = nn.Parameter(
            torch.randn(1, texture_res, texture_res, 21)*self.init_std  # 8 = 3+1+1+3
        )
        self.initialize_normal_channel()
        
    def initialize_normal_channel(self):
        """Initialize normal map channels to (0, 0, 1) which represents no perturbation"""
        with torch.no_grad():
            # Normal map channels are at indices 18, 19, 20 (last 3 channels)
            self.pbr_texture.data[0, :, :, 18] = 0.0  # x component
            self.pbr_texture.data[0, :, :, 19] = 0.0  # y component  
            self.pbr_texture.data[0, :, :, 20] = 1.0  # z component

    
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
            import pdb; pdb.set_trace()

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


    def eval_brdf(self, params, pos, wi, wo, normal,uv, TBN, latent=None, batch_mask=None):
        NoL = (wi*normal).sum(-1, keepdim=True)
        NoV = (wo*normal).sum(-1, keepdim=True)
        valid_geometry = (NoL > 0) & (NoV > 0)
        T = TBN[:, :, 0]  # Extract tangent vectors
        B = TBN[:, :, 1]  # Extract bitangent vectors
        N_geo = TBN[:, :, 2]  # Extract normal vectors
        
        if not valid_geometry.any():
            return torch.zeros_like(wi), torch.zeros(wi.shape[0], 1, device=wi.device)
        radius = 0.2
        factor = self.scale_factor
        params = self.pbr_texture
        H, W = params.shape[1], params.shape[2]
        crop_h = int((H - H * factor) // 2)
        crop_w = int((W - W * factor) // 2)
        new_h = int(H * factor)
        new_w = int(W * factor)
        params = params[:, crop_h:crop_h+new_h, crop_w:crop_w+new_w, :]
        # Step 2: Texture sampling
        sampled_texture = sample_texture(params, uv)
        if self.anisotropic:
            # Extract anisotropic texture maps
            diffuse = sampled_texture[:, 0:3]                    # Color channels
            ao = sampled_texture[:, 3:4]                       # Ambient occlusion
            arm = sampled_texture[:, 6:9]                      # ARM channels
            roughness_map = sampled_texture[:, 9:10]           # Roughness map
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
            n_world = local_to_world_normal(normal_local, T, B, N_geo)

            # Evaluate BRDF with mapped parameters
            if self.normal_map:
                brdf, pdf = self.compute_svbrdf_pdf(albedo, roughness, metallic, wi, wo, n_world)
            else:
                brdf, pdf = self.compute_svbrdf_pdf(albedo, roughness, metallic, wi, wo, normal)
            brdf = color * brdf
            
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
