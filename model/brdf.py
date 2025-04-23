import torch
import torch.nn as nn
import torch.nn.functional as NF
import math

import sys
sys.path.append('..')

from utils.ops import *

from nerfstudio.field_components import encodings as encoding

# class SHEncoding(nn.Module):
#     def __init__(self, degree: int):
#         """
#         Spherical Harmonics Encoding module
#         Args:
#             degree (int): SH degree level, controls the number of SH coefficients
#         """
#         super(SHEncoding, self).__init__()
#         self.degree = degree

#     def forward(self, dirs):
#         """
#         Encode directions into SH coefficients.
        
#         Args:
#             dirs (B, 3): normalized directional vectors
            
#         Returns:
#             encoding (B, (degree+1)^2): SH encoding vector
#         """
#         x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
#         phi = torch.atan2(y, x)  # azimuthal angle in [-pi, pi]
#         theta = torch.acos(torch.clamp(z, -1.0, 1.0))  # polar angle in [0, pi]

#         sh_encodings = []
#         for l in range(self.degree + 1):
#             for m in range(-l, l+1):
#                 Y_lm = sph_harm(m, l, phi, theta)  # complex SH
#                 if m < 0:
#                     # Real form (for negative m)
#                     sh_encodings.append(torch.sqrt(torch.tensor(2.0)) * (-1)**m * Y_lm.imag)
#                 elif m == 0:
#                     # Real form for m=0
#                     sh_encodings.append(Y_lm.real)
#                 else:
#                     # Real form (for positive m)
#                     sh_encodings.append(torch.sqrt(torch.tensor(2.0)) * (-1)**m * Y_lm.real)
#         encoding = torch.stack(sh_encodings, dim=-1)  # (B, (degree+1)^2)
#         return encoding


def diffuse_sampler(sample2,normal):
    """ sampling diffuse lobe: wi ~ NoV/math.pi 
    Args:
        sample2: Bx2 unIform samples
        normal: Bx3 normal
    Return:
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

def specular_sampler(sample2,roughness,wo,normal):
    """ sampling ggx lobe: h ~ D/(VoH*4)*NoH
    Args:
        sample2: Bx3 uniform samples
        roughness: Bx1 roughness
        wo: Bx3 viewing direction
        normal: Bx3 normal
    Return:
        wi: Bx3 sampled direction in world space
    """
    alpha = (roughness*roughness).squeeze(-1)
    
    # sample half vector
    theta = (1-sample2[...,0])/((sample2[...,0]*(alpha*alpha-1)+1) + 1e-8)
    theta = torch.acos(theta.sqrt())
    phi = 2*math.pi*sample2[...,1]
    wh = angle2xyz(theta,phi)

    # half vector to wi
    Nmat = get_normal_space(normal)
    wh = (wh[:,None]@Nmat.permute(0,2,1)).squeeze(1)
    wi = 2*(wo*wh).sum(-1,keepdim=True)*wh-wo
    wi = NF.normalize(wi,dim=-1)
    if wi.isnan().any():
        print("wi is nan")
    return wi

# def safe_acos(input_tensor):
#     """
#     Computes arccos safely, avoiding NaNs while maintaining smooth gradients.
#     Uses a soft projection instead of clamping.
#     """
#     safe_min = 1e-7  # Avoid zero for sqrt
#     safe_max = 1.0 - 1e-7  # Avoid precision issues at 1

#     # Soft projection: Instead of clamping, smoothly project out-of-range values
#     projected = (input_tensor - safe_min) / (safe_max - safe_min)  # Normalize to [0,1]
#     projected = projected * (safe_max - safe_min) + safe_min  # Scale back

#     return torch.acos(torch.sqrt(projected))

def acos_grad(z):
    return -1 / torch.sqrt(1 - z**2)

class safe_acos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        
        # protect ourselves from nan outputs in forward pass.
        return torch.clamp(input, min=-1 + 1e-6, max=1 - 1e-6).acos()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # protect ourselves from large gradients in backward pass.
        # outside of (-1 + epsilon, 1 - epsilon), gradient value is fixed constant to acos'(1-epsilon)
        epsilon = .05
        safe_input = torch.clamp(input, min=-1 + epsilon, max=1 - epsilon)

        return acos_grad(safe_input) * grad_input
    
class BaseBRDF(nn.Module):
    """ Base BRDF class """
    def __init__(self,):
        super(BaseBRDF,self).__init__()
        return
    
    def forward(self,):
        pass
    
    def eval_diffuse(self,wi,normal):
        """ evaluate diffuse shading 
            and pdf
        """
        pdf = (normal*wi).sum(-1,keepdim=True).relu()/math.pi
        brdf = pdf.expand(len(wi),3) 
        return brdf,pdf
    
    def sample_diffuse(self,sample2,normal):
        """ sample diffuse shading
            and get sampled weight
        """
        # get wi
        wi = diffuse_sampler(sample2,normal)
        
        # get brdf/pdf, pdf
        brdf_weight = torch.ones(normal.shape,device=normal.device)
        pdf = (normal*wi).sum(-1,keepdim=True).relu()/math.pi
        return wi,pdf,brdf_weight
    
    def eval_specular(self,wi,wo,normal,roughness):
        """" evaluate specular shadings
            and pdf
        """
        h = NF.normalize(wi+wo,dim=-1)
        NoL = (wi*normal).sum(-1,keepdim=True).relu()
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        D = D_GGX(NoH,roughness)
        pdf = D.data/(4*VoH.clamp_min(1e-4))*NoH

        G = G_Smith(NoV,NoL,roughness)
        F0,F1 = fresnelSchlick_sep(VoH)
        
        # two term corresponds to two fresnel components
        brdf_spec0 = D*G*F0/4.0*NoL
        brdf_spec1 = D*G*F1/4.0*NoL

        return brdf_spec0,brdf_spec1,pdf 

    def sample_specular(self,sample2,wo,normal,roughness):
        """ evaluate specular shadings
            and get sampled weight
        """
        # get wi
        wi = specular_sampler(sample2,roughness,wo,normal)
        
        # get brdf/pdf, pdf
        h = NF.normalize(wi+wo,dim=-1)
        NoL = (wi*normal).sum(-1,keepdim=True).relu()
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()
        
        D = D_GGX(NoH,roughness)
        pdf = D.data/(4*VoH.clamp_min(1e-4))*NoH

        G = G_Smith(NoV,NoL,roughness)
        F0,F1 = fresnelSchlick_sep(VoH)
        
        fac = G*VoH*NoL/NoH.clamp_min(1e-4)
        
        brdf_weight0 = F0*fac
        brdf_weight1 = F1*fac
        return wi,pdf,brdf_weight0,brdf_weight1

    def eval_brdf(self,wi,wo,normal,mat, half_sphere_mask):
        """ evaluate BRDF and pdf
        Args:
            wi: Bx3 light direction
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: surface BRDF dict
        Return:
            brdf: Bx3
            pdf: Bx1
        """
        # Check if both directions are on the same side
        NoL = (wi*normal).sum(-1,keepdim=True)
        NoV = (wo*normal).sum(-1,keepdim=True)
        valid_geometry = (NoL > 0) & (NoV > 0)
        
        # Early return for invalid geometry
        if not valid_geometry.any():
            return torch.zeros_like(wi), torch.zeros(wi.shape[0], 1, device=wi.device)

        albedo,roughness,metallic = mat['albedo'],mat['roughness'],mat['metallic']

        h = NF.normalize(wi+wo,dim=-1)
        NoL = NoL.relu()  # Now safe to relu after check
        NoV = NoV.relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        # get pdf
        D = D_GGX(NoH,roughness)
        pdf_spec = D.data/(4*VoH.clamp_min(1e-4))*NoH
        pdf_diff = NoL/math.pi
        pdf = 0.5*pdf_spec + 0.5*pdf_diff

        # get brdf
        kd = albedo*(1-metallic)
        ks = 0.04*(1-metallic) + albedo*metallic

        G = G_Smith(NoV,NoL,roughness)
        F = fresnelSchlick(VoH,ks)
        brdf_diff = kd/math.pi*NoL
        brdf_spec = D*G*F/4.0*NoL

        brdf = brdf_diff + brdf_spec
        
        # Zero out invalid geometry cases
        # brdf = torch.where(valid_geometry, brdf, 0.0)
        # pdf = torch.where(valid_geometry, pdf, 0.0)

        return brdf,pdf 
    
    def sample_brdf(self,sample1,sample2,wo,normal,mat, half_sphere_mask):
        """ importance sampling brdf and get brdf/pdf
        Args:
            sample1: B unifrom samples
            sample2: Bx2 uniform samples
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: material dict
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1
            brdf_weight: Bx3 brdf/pdf
        """
        B = sample1.shape[0]
        device = sample1.device

        pdf = torch.zeros(B,device=device)
        brdf = torch.zeros(B,3,device=device)
        wi = torch.zeros(B,3,device=device)


        mask = (sample1 > 0.5)
        # sample diffuse
        wi[mask] = diffuse_sampler(sample2[mask],normal[mask])
        mask = ~mask
        # sample specular
        wi[mask] = specular_sampler(sample2[mask],mat['roughness'][mask],wo[mask],normal[mask])

        # get brdf,pdf
        brdf,pdf = self.eval_brdf(wi,wo,normal,mat, half_sphere_mask)

        brdf_weight = torch.where(pdf>0,brdf/pdf,0)
        brdf_weight[brdf_weight.isnan()] = 0
        return wi,pdf,brdf_weight

class PhongBRDF(nn.Module):
    def __init__(self, diffuse_color=torch.ones(1, 3), specular_color=torch.ones(1, 3), shininess=32.0):
        super(PhongBRDF, self).__init__()
        self.diffuse_color = nn.Parameter(diffuse_color)
        self.specular_color = nn.Parameter(specular_color)
        self.shininess = nn.Parameter(torch.tensor([shininess]))

    def eval_brdf(self, wi, wo, normal):
        NoL = (normal * wi).sum(-1, keepdim=True).relu()
        NoV = (normal * wo).sum(-1, keepdim=True).relu()
        reflect_dir = NF.normalize(2 * NoV * normal - wo, dim=-1)
        RoV = (reflect_dir * wi).sum(-1, keepdim=True).relu()

        diffuse = self.diffuse_color * NoL / math.pi
        specular = self.specular_color * ((self.shininess + 2) / (2 * math.pi)) * (reflect_dir * wi).sum(-1, keepdim=True).relu().pow(self.shininess)

        brdf = self.diffuse_color * diffuse + self.specular_color * specular
        pdf_diffuse = NoL / math.pi
        pdf_specular = (self.shininess + 1) / (2 * math.pi) * RoV.pow(self.shininess)
        pdf = 0.5 * pdf_diffuse + 0.5 * pdf_specular

        return brdf, pdf

    def sample_brdf(self, sample1, sample2, wo, normal):
        mask_diffuse = (sample1 <= 0.5)
        mask_specular = ~mask_diffuse
        wi = torch.zeros_like(wo)

        if mask_diffuse.any():
            wi[mask_diffuse] = self.diffuse_sampler(sample2[mask_diffuse], normal[mask_diffuse])

        if mask_specular.any():
            reflect_dir = NF.normalize(2 * (wo * normal).sum(-1, keepdim=True) * normal - wo, dim=-1)
            wi[mask_specular] = self.specular_sampler(sample2[mask_specular], reflect_dir[mask_specular])

        brdf, pdf = self.eval_brdf(wi, wo, normal)
        pdf = pdf.clamp(min=1e-8)
        brdf_weight = torch.where(pdf > 0, brdf / pdf, torch.zeros_like(brdf))
        # brdf_weight[brdf_weight.isnan()] = 0

        return wi, pdf, brdf_weight

    def diffuse_sampler(self, sample2, normal):
        r = torch.sqrt(sample2[:, :1])
        theta = 2 * math.pi * sample2[:, 1:2]

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        z = torch.sqrt((1 - r.pow(2)).clamp(min=0))

        tangent, bitangent = self.create_tangent_frame(normal)

        wi = tangent * x + bitangent * y + normal * z
        return NF.normalize(wi, dim=-1)

    def specular_sampler(self, sample2, reflect_dir):
        phi = 2 * math.pi * sample2[:, 0:1]
        cos_theta = sample2[:, 1:2].pow(1 / (self.shininess + 1))
        sin_theta = torch.sqrt(1 - cos_theta.pow(2))

        x = sin_theta * torch.cos(phi)
        y = sin_theta * torch.sin(phi)
        z = cos_theta

        tangent, bitangent = self.create_tangent_frame(reflect_dir)

        wi = tangent * x + bitangent * y + reflect_dir * z
        return NF.normalize(wi, dim=-1)

    def create_tangent_frame(self, normal):
        up = torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal)
        tangent = torch.cross(up, normal)
        tangent = NF.normalize(tangent, dim=-1)
        bitangent = NF.normalize(torch.cross(normal, tangent), dim=-1)
        return tangent, bitangent


class PBRBRDF(nn.Module):
    """ Base BRDF class """
    def __init__(self, albedo=torch.ones(1, 3), roughness=0.2, metallic=0.5):
        super(PBRBRDF,self).__init__()
        # Initialize learnable material parameters
        self.albedo = nn.Parameter(albedo)
        self.roughness = nn.Parameter(torch.full((1, 1), roughness).cuda())  # Scalar roughness 
        self.metallic = nn.Parameter(torch.full((1, 1), metallic).cuda())  # Scalar metallic
        # Initialize parameters as dictionary
        self.mat = {
            'albedo': self.albedo,
            'roughness': self.roughness, 
            'metallic': self.metallic
        }
        return
    
    def forward(self, wi, wo, normal):
        brdf, pdf = self.eval_brdf(wi, wo, normal)
        return brdf, pdf
    
    def eval_diffuse(self,wi,normal):
        """ evaluate diffuse shading 
            and pdf
        """
        pdf = (normal*wi).sum(-1,keepdim=True).relu()/math.pi
        brdf = pdf.expand(len(wi),3) 
        return brdf,pdf
    
    def sample_diffuse(self,sample2,normal):
        """ sample diffuse shading
            and get sampled weight
        """
        # get wi
        wi = diffuse_sampler(sample2,normal)
        
        # get brdf/pdf, pdf
        brdf_weight = torch.ones(normal.shape,device=normal.device)
        pdf = (normal*wi).sum(-1,keepdim=True).relu()/math.pi
        return wi,pdf,brdf_weight
    
    def eval_specular(self,wi,wo,normal,roughness):
        """" evaluate specular shadings
            and pdf
        """
        h = NF.normalize(wi+wo,dim=-1)
        NoL = (wi*normal).sum(-1,keepdim=True).relu()
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        D = D_GGX(NoH,roughness)
        pdf = D/(4*VoH)*NoH

        G = G_Smith(NoV,NoL,roughness)
        F0,F1 = fresnelSchlick_sep(VoH)
        
        # two term corresponds to two fresnel components
        brdf_spec0 = D*G*F0/4.0*NoL
        brdf_spec1 = D*G*F1/4.0*NoL

        return brdf_spec0,brdf_spec1,pdf 

    # def sample_specular(self,sample2,wo,normal,roughness):
    #     """ evaluate specular shadings
    #         and get sampled weight
    #     """
    #     # get wi
    #     wi = specular_sampler(sample2,roughness,wo,normal)
        
    #     # get brdf/pdf, pdf
    #     h = NF.normalize(wi+wo,dim=-1)
    #     NoL = (wi*normal).sum(-1,keepdim=True).relu()
    #     NoV = (wo*normal).sum(-1,keepdim=True).relu()
    #     VoH = (wo*h).sum(-1,keepdim=True).relu()
    #     NoH = (normal*h).sum(-1,keepdim=True).relu()
        
    #     D = D_GGX(NoH,roughness)
    #     pdf = D/(4*VoH.clamp_min(1e-4))*NoH

    #     G = G_Smith(NoV,NoL,roughness)
    #     F0,F1 = fresnelSchlick_sep(VoH)
        
    #     fac = G*VoH*NoL/NoH.clamp_min(1e-4)
        
    #     brdf_weight0 = F0*fac
    #     brdf_weight1 = F1*fac
    #     return wi,pdf,brdf_weight0,brdf_weight1

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
        # # theta = safe_acos.apply(theta)
        # EPS = 0.1
        # theta = torch.acos(theta.clamp(min=0.0 + EPS, max=1.0 - EPS).sqrt())
        theta = torch.acos(theta.sqrt())

        phi = 2*math.pi*sample2[...,1]
        wh = angle2xyz(theta,phi)

        # half vector to wi
        Nmat = get_normal_space(normal)
        wh = (wh[:,None]@Nmat.permute(0,2,1)).squeeze(1)
        wi = 2*(wo*wh).sum(-1,keepdim=True)*wh-wo
        wi = NF.normalize(wi,dim=-1)
        return wi
    
    # def eval_brdf(self,wi,wo,normal):
    #     """ evaluate BRDF and pdf
    #         wi: Bx3 light direction
    #         wo: Bx3 viewing direction
    #         normal: Bx3 normal
    #         mat: surface BRDF dict
    #     Return:
    #         brdf: Bx3
    #         pdf: Bx1
    #     """
    #     # Check if both directions are on the same side
    #     NoL = (wi*normal).sum(-1,keepdim=True)
    #     NoV = (wo*normal).sum(-1,keepdim=True)
    #     valid_geometry = (NoL > 0) & (NoV > 0)
        
    #     # # Early return for invalid geometry
    #     if not valid_geometry.any():
    #         return torch.zeros_like(wi), torch.zeros(wi.shape[0], 1, device=wi.device)
    #     # Reshape albedo tensor to match expected dimensions
    #     albedo = self.mat['albedo'].view(1, 3).expand(normal.shape[0], 3)
    #     roughness = self.mat['roughness'].view(1, 1).expand(normal.shape[0], 1)
    #     metallic = self.mat['metallic'].view(1, 1).expand(normal.shape[0], 1)

    #     h = NF.normalize(wi+wo,dim=-1)
    #     NoL = NoL.relu()  # Now safe to relu after check
    #     NoV = NoV.relu()
    #     VoH = (wo*h).sum(-1,keepdim=True).relu()
    #     NoH = (normal*h).sum(-1,keepdim=True).relu()

    #     # get pdf
    #     D = D_GGX(NoH,roughness)
    #     pdf_spec = D.data/(4*VoH.clamp_min(1e-4))*NoH
    #     pdf_diff = NoL/math.pi
    #     pdf = 0.5*pdf_spec + 0.5*pdf_diff

    #     # get brdf
    #     kd = albedo*(1-metallic)
    #     ks = 0.04*(1-metallic) + albedo*metallic

    #     G = G_Smith(NoV,NoL,roughness)
    #     F = fresnelSchlick(VoH,ks)
    #     brdf_diff = kd/(math.pi*NoL + 1e-8)
    #     brdf_spec = D*G*F/(4.0*NoL + 1e-8)

    #     brdf = brdf_diff + brdf_spec


    #     return brdf,pdf 

    def eval_brdf(self,wi,wo,normal):
        """ evaluate BRDF and pdf
            wi: Bx3 light direction
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: surface BRDF dict
        Return:
            brdf: Bx3
            pdf: Bx1
        """
        # Check if both directions are on the same side
        NoL = (wi*normal).sum(-1,keepdim=True)
        NoV = (wo*normal).sum(-1,keepdim=True)
        valid_geometry = (NoL > 0) & (NoV > 0)
        
        # Early return for invalid geometry
        if not valid_geometry.any():
            return torch.zeros_like(wi), torch.zeros(wi.shape[0], 1, device=wi.device)
        # Reshape albedo tensor to match expected dimensions
        albedo = self.mat['albedo'].view(1, 3).expand(normal.shape[0], 3)
        roughness = self.mat['roughness'].view(1, 1).expand(normal.shape[0], 1)
        metallic = self.mat['metallic'].view(1, 1).expand(normal.shape[0], 1)

        h = NF.normalize(wi+wo,dim=-1)
        NoL = NoL.relu()  # Now safe to relu after check
        NoV = NoV.relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        # get pdf
        D = D_GGX(NoH,roughness)
        pdf_spec = D.data/((4*VoH.clamp_min(1e-4))*NoH + 1e-8)
        pdf_diff = NoL/math.pi
        pdf = 0.5*pdf_spec + 0.5*pdf_diff

        # get brdf
        kd = albedo*(1-metallic)
        ks = 0.04*(1-metallic) + albedo*metallic

        G = G_Smith(NoV,NoL,roughness)
        F = fresnelSchlick(VoH,ks)
        brdf_diff = kd/math.pi*NoL
        brdf_spec = D*G*F/4.0*NoL

        brdf = brdf_diff + brdf_spec


        return brdf,pdf
    
    def sample_brdf(self,sample1,sample2,wo,normal):
        """ importance sampling brdf and get brdf/pdf
        Args:
            sample1: B unifrom samples
            sample2: Bx2 uniform samples
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: material dict
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1
            brdf_weight: Bx3 brdf/pdf
        """
        B = sample1.shape[0]
        device = sample1.device

        pdf = torch.zeros(B,device=device)
        brdf = torch.zeros(B,3,device=device)
        # wi = torch.zeros(B,3,device=device)


        mask = (sample1 > 0.5)
        wi_diffuse = diffuse_sampler(sample2[mask], normal[mask])
        wi_specular = self.specular_sampler(sample2[~mask], self.mat['roughness'].expand(normal.shape[0], 1)[~mask], wo[~mask], normal[~mask])

        # Construct wi without gradient-breaking assignment
        wi = torch.zeros(B, 3, device=device)
        wi = wi.clone()  # explicitly ensures gradients
        wi[mask] = wi_diffuse
        wi[~mask] = wi_specular
        # get brdf,pdf
        brdf,pdf = self.forward(wi,wo,normal)
        brdf_weight = torch.where(pdf>0,brdf/pdf,0)
        brdf_weight[brdf_weight.isnan()] = 0
        return wi,pdf,brdf_weight

class ProxyPBRBRDF(nn.Module):
    def __init__(self, roughness=0.1):
        super(ProxyPBRBRDF, self).__init__()
        self.roughness = nn.Parameter(torch.full((1, 1), roughness).cuda())  # Scalar roughness 
        self.metallic = 0.2
        self.albedo = torch.ones(1, 3).cuda()

    def eval_brdf(self,wi,wo,normal):
        """ Used for debugging """
        """ evaluate BRDF and pdf
            wi: Bx3 light direction
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: surface BRDF dict
        Return:
            brdf: Bx3
            pdf: Bx1
        """
        # Check if both directions are on the same side
        NoL = (wi*normal).sum(-1,keepdim=True)
        NoV = (wo*normal).sum(-1,keepdim=True)
        valid_geometry = (NoL > 0) & (NoV > 0)
        
        # Early return for invalid geometry
        if not valid_geometry.any():
            return torch.zeros_like(wi), torch.zeros(wi.shape[0], 1, device=wi.device)
        # Reshape albedo tensor to match expected dimensions
        albedo = self.albedo.view(1, 3).expand(normal.shape[0], 3)
        roughness = self.roughness.view(1, 1).expand(normal.shape[0], 1)
        metallic = torch.tensor([self.metallic], device=wi.device).expand(normal.shape[0], 1)

        h = NF.normalize(wi+wo,dim=-1)
        NoL = NoL.relu()  # Now safe to relu after check
        NoV = NoV.relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        # get pdf
        D = D_GGX(NoH,roughness)
        pdf_spec = D.data/((4*VoH.clamp_min(1e-4))*NoH + 1e-8)
        pdf_diff = NoL/math.pi
        pdf = 0.5*pdf_spec + 0.5*pdf_diff

        # get brdf
        kd = albedo*(1-metallic)
        ks = 0.04*(1-metallic) + albedo*metallic

        G = G_Smith(NoV,NoL,roughness)
        F = fresnelSchlick(VoH,ks)
        brdf_diff = kd/math.pi*NoL
        brdf_spec = D*G*F/4.0*NoL

        brdf = brdf_diff + brdf_spec

        return brdf, pdf
    
    def eval_pdf(self,wi,wo,normal):
        """ evaluate BRDF and pdf
            wi: Bx3 light direction
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: surface BRDF dict
        Return:
            brdf: Bx3
            pdf: Bx1
        """
        # Check if both directions are on the same side
        NoL = (wi*normal).sum(-1,keepdim=True)
        NoV = (wo*normal).sum(-1,keepdim=True)
        valid_geometry = (NoL > 0) & (NoV > 0)
        
        # Early return for invalid geometry
        if not valid_geometry.any():
            return torch.zeros_like(wi), torch.zeros(wi.shape[0], 1, device=wi.device)

        h = NF.normalize(wi+wo,dim=-1)
        NoL = NoL.relu()  # Now safe to relu after check
        NoV = NoV.relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        # get pdf
        D = D_GGX(NoH, self.roughness.expand(normal.shape[0], 1))
        pdf_spec = D/((4*VoH.clamp_min(1e-4))*NoH.clamp_min(1e-4))
        pdf_diff = NoL/math.pi
        pdf = 0.5*pdf_spec + 0.5*pdf_diff

        return pdf

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
    
    def sample_brdf(self,sample1,sample2,wo,normal):
        """ importance sampling brdf and get brdf/pdf
        Args:
            sample1: B unifrom samples
            sample2: Bx2 uniform samples
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: material dict
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1
            brdf_weight: Bx3 brdf/pdf
        """
        B = sample1.shape[0]
        device = sample1.device

        pdf = torch.zeros(B,device=device)
        brdf = torch.zeros(B,3,device=device)
        # wi = torch.zeros(B,3,device=device)


        mask = (sample1 > 0.5)
        wi_diffuse = diffuse_sampler(sample2[mask], normal[mask])
        wi_specular = self.specular_sampler(sample2[~mask], self.roughness.expand(normal.shape[0], 1)[~mask], wo[~mask], normal[~mask])

        # Construct wi without gradient-breaking assignment
        wi = torch.zeros(B, 3, device=device)
        wi = wi.clone()  # explicitly ensures gradients
        wi[mask] = wi_diffuse
        wi[~mask] = wi_specular
        wi = wi.detach()
        # get brdf,pdf

        # brdf, pdf = self.eval_brdf(wi,wo,normal)
        pdf = self.eval_pdf(wi,wo,normal)
        # stop_gradient_pdf = pdf.detach()

        # brdf_weight = torch.where(stop_gradient_pdf>0,brdf/stop_gradient_pdf,0)
        # brdf_weight[brdf_weight.isnan()] = 0
        return wi, pdf

class MLPPBRBRDF(nn.Module):
    """ MLP-based BRDF class """
    def __init__(self, cfg, gt_roughness, gt_metallic):
        super(MLPPBRBRDF, self).__init__()

        # Add SH positional encoding module
        self.levels = 4
        self.pos_enc = False
        if self.pos_enc:
            self.sh_encoder = encoding.SHEncoding(levels=self.levels)
            
        # Calculate input dimension after SH encoding
        sh_dim = (self.levels) ** 2
        encoded_input_dim = sh_dim * 3  # wi, wo, normal each encoded by SH

        layers = []
        prev_dim = encoded_input_dim if self.pos_enc else 9
        for hidden_dim in cfg.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if cfg.activation.lower() == "relu":
                layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, cfg.output_channels))
        layers.append(nn.LeakyReLU(0.2))
        
        self.mlp = nn.Sequential(*layers)

        # self.proxy_brdf = PhongBRDF()
        self.proxy_brdf = ProxyPBRBRDF(roughness=gt_roughness)


    def forward(self, wi, wo, normal):
        """
        Evaluate BRDF using MLP
        Args:
            wi: Bx3 incoming light direction 
            wo: Bx3 outgoing view direction
        Returns:
            brdf: Bx1 BRDF values
        """
        # SH encoding
        if self.pos_enc:
            wi_enc = self.sh_encoder(wi)
            wo_enc = self.sh_encoder(wo)
            normal_enc = self.sh_encoder(normal)
        # Concatenate encoded inputs
        x = torch.cat([wi_enc, wo_enc, normal_enc], dim=-1) if self.pos_enc else torch.cat([wi, wo, normal], dim=-1)
        return self.mlp(x)

    def world_to_local(self, v, normal):
        
        # choose arbitrary tangent
        up = torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal)
        tangent = torch.cross(up, normal)
        tangent_len = tangent.norm(dim=-1, keepdim=True)
        
        # if normal is collinear with [0,1,0], choose another tangent
        collinear_mask = tangent_len.squeeze(-1) < 1e-6
        if collinear_mask.any():
            tangent[collinear_mask] = torch.cross(normal[collinear_mask], torch.tensor([1., 0., 0.].expand_as(normal[collinear_mask])), device=normal.device)
            tangent_len = tangent.norm(dim=-1, keepdim=True)

        tangent = tangent / tangent_len

        bitangent = torch.cross(normal, tangent)

        v_local = torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1)
        ], dim=-1)

        return v_local

    # def eval_brdf(self, wi, wo, normal):
    #     """
    #     Evaluate BRDF and pdf after transforming world-space vectors to local space.
    #     Args:
    #         wi: Bx3 light direction in world space
    #         wo: Bx3 viewing direction in world space
    #         normal: Bx3 normal in world space
    #     Returns:
    #         brdf: Bx3 BRDF values
    #         pdf: Bx1 probability
    #     """
    #     # Ensure normal is normalized
    #     NoL = (wi*normal).sum(-1,keepdim=True)
    #     NoV = (wo*normal).sum(-1,keepdim=True)
    #     normal = normal / normal.norm(dim=-1, keepdim=True)

    #     brdf_val = self.forward(wi, wo, normal)
    #     brdf = brdf_val.expand(-1, 3)

    #     # # Construct an orthonormal basis (T, B, N)
    #     # up = torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal)
    #     # tangent = torch.cross(up, normal)
    #     # tangent = tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-8)  # Avoid division by zero

    #     # bitangent = torch.cross(normal, tangent)  # Ensure orthogonality

    #     # # Create rotation matrix [T | B | N]
    #     # local_matrix = torch.stack([tangent, bitangent, normal], dim=-1)  # Bx3x3

    #     # # Transform wi and wo into the local frame
    #     # wi_local = torch.einsum('bij,bj->bi', local_matrix.transpose(-2, -1), wi)
    #     # wo_local = torch.einsum('bij,bj->bi', local_matrix.transpose(-2, -1), wo)

    #     # # Get BRDF from MLP using local-space vectors and normal in local space
    #     # local_normal = torch.zeros_like(wi_local)
    #     # local_normal[..., 2] = 1.0  # Normal is always (0,0,1) in local space
    #     # brdf_val = self.forward(wi_local, wo_local, local_normal)

    #     # # Expand to RGB channels
    #     # brdf = brdf_val.expand(-1, 3)

    #     pdf = NoL / math.pi

    #     return brdf, pdf
    
    def eval_brdf(self, wi, wo, normal):
        """
        Evaluate BRDF and pdf after transforming world-space vectors to local space.
        Args:
            wi: Bx3 light direction in world space
            wo: Bx3 viewing direction in world space
            normal: Bx3 normal in world space
        Returns:
            brdf: Bx3 BRDF values
            pdf: Bx1 probability
        """
        # Ensure normal is normalized
        NoL = (wi*normal).sum(-1,keepdim=True)
        NoV = (wo*normal).sum(-1,keepdim=True)
        wi_local = self.world_to_local(wi, normal)
        wo_local = self.world_to_local(wo, normal)
        local_normal = torch.zeros_like(wi_local)
        local_normal[..., 2] = 1.0  # Normal is always (0,0,1) in local space
        brdf_value = self.forward(wi_local, wo_local, local_normal)
        brdf = brdf_value.expand(-1, 3)

        pdf = NoL / math.pi

        return brdf, pdf
        
    def sample_brdf(self, sample1, sample2, wo, normal):
        """
        Importance sampling BRDF using proxy (PBRBRDF) and evaluating MLP BRDF.
        
        Args:
            sample1: B uniform samples [0,1] to select diffuse or specular sampling
            sample2: Bx2 uniform samples for hemisphere sampling
            wo: Bx3 viewing direction in world space
            normal: Bx3 normal in world space
            proxy_brdf: instance of PBRBRDF to guide sampling (importance sampling proxy)

        Returns:
            wi: Bx3 sampled incoming directions (world space)
            pdf: Bx1 sampling pdf values from proxy_brdf
            brdf_weight: Bx3 ratio (MLP evaluated BRDF / pdf)
        """
        B = sample1.shape[0]
        device = sample1.device

        # Step 1: Sample direction wi from proxy BRDF
        wi_proxy, pdf_proxy = self.proxy_brdf.sample_brdf(sample1, sample2, wo, normal)
        stop_gradient_pdf_proxy = pdf_proxy.detach()
        
        # Step 2: Evaluate the MLP-based BRDF at these sampled directions
        mlp_brdf, _ = self.eval_brdf(wi_proxy, wo, normal)

        # Step 3: Compute brdf_weight (importance sampling ratio)
        mlp_brdf = mlp_brdf * pdf_proxy / (stop_gradient_pdf_proxy + 1e-8)
        brdf_weight = torch.where(pdf_proxy > 0, mlp_brdf / (stop_gradient_pdf_proxy + 1e-8), torch.zeros_like(mlp_brdf))

        return wi_proxy, stop_gradient_pdf_proxy, brdf_weight
    
class LatentBRDF(nn.Module):
    """ MLP-based BRDF class """
    def __init__(self, cfg, gt_roughness, gt_metallic):
        super(LatentBRDF, self).__init__()
        
        layers = []
        prev_dim = cfg.input_channels
        for hidden_dim in cfg.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if cfg.activation.lower() == "relu":
                layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, cfg.output_channels))
        layers.append(nn.LeakyReLU(0.2))
        
        self.mlp = nn.Sequential(*layers)
        # self.mlp = nn.Sequential(
        #     nn.Linear(9, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1),
        #     nn.LeakyReLU(0.2)  # Ensure non-negative BRDF values
        # )

        # self.proxy_brdf = PhongBRDF()
        self.proxy_brdf = PBRBRDF(albedo=torch.ones(1, 3), roughness=gt_roughness, metallic=gt_metallic)


    def forward(self, wi, wo, normal):
        """
        Evaluate BRDF using MLP
        Args:
            wi: Bx3 incoming light direction 
            wo: Bx3 outgoing view direction
        Returns:
            brdf: Bx1 BRDF values
        """
        x = torch.cat([wi, wo, normal], dim=-1)  # Concatenate to Bx9
        return self.mlp(x)

    
    def eval_brdf(self, wi, wo, normal):
        """
        Evaluate BRDF and pdf after transforming world-space vectors to local space.
        Args:
            wi: Bx3 light direction in world space
            wo: Bx3 viewing direction in world space
            normal: Bx3 normal in world space
        Returns:
            brdf: Bx3 BRDF values
            pdf: Bx1 probability
        """
        # Ensure normal is normalized
        NoL = (wi*normal).sum(-1,keepdim=True)
        NoV = (wo*normal).sum(-1,keepdim=True)
        normal = normal / normal.norm(dim=-1, keepdim=True)

        # Construct an orthonormal basis (T, B, N)
        up = torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal)
        tangent = torch.cross(up, normal)
        tangent = tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-8)  # Avoid division by zero

        bitangent = torch.cross(normal, tangent)  # Ensure orthogonality

        # Create rotation matrix [T | B | N]
        local_matrix = torch.stack([tangent, bitangent, normal], dim=-1)  # Bx3x3

        # Transform wi and wo into the local frame
        wi_local = torch.einsum('bij,bj->bi', local_matrix.transpose(-2, -1), wi)
        wo_local = torch.einsum('bij,bj->bi', local_matrix.transpose(-2, -1), wo)

        # Get BRDF from MLP using local-space vectors and normal in local space
        local_normal = torch.zeros_like(wi_local)
        local_normal[..., 2] = 1.0  # Normal is always (0,0,1) in local space
        brdf_val = self.forward(wi_local, wo_local, local_normal)

        # Expand to RGB channels
        brdf = brdf_val.expand(-1, 3)

        # Compute PDF (cosine-weighted hemisphere sampling)
        # NoL = wi_local[:, 2:3]  # Z-component in local space

        pdf = NoL / math.pi

        return brdf, pdf
        
    def sample_brdf(self, sample1, sample2, wo, normal):
        """
        Importance sampling BRDF using proxy (PBRBRDF) and evaluating MLP BRDF.
        
        Args:
            sample1: B uniform samples [0,1] to select diffuse or specular sampling
            sample2: Bx2 uniform samples for hemisphere sampling
            wo: Bx3 viewing direction in world space
            normal: Bx3 normal in world space
            proxy_brdf: instance of PBRBRDF to guide sampling (importance sampling proxy)

        Returns:
            wi: Bx3 sampled incoming directions (world space)
            pdf: Bx1 sampling pdf values from proxy_brdf
            brdf_weight: Bx3 ratio (MLP evaluated BRDF / pdf)
        """
        B = sample1.shape[0]
        device = sample1.device

        # Step 1: Sample direction wi from proxy BRDF
        wi_proxy, pdf_proxy, _ = self.proxy_brdf.sample_brdf(sample1, sample2, wo, normal)
        pdf_proxy = pdf_proxy.detach() # Stop gradient
        wi_proxy = wi_proxy.detach()
        
        # Step 2: Evaluate the MLP-based BRDF at these sampled directions
        mlp_brdf, _ = self.eval_brdf(wi_proxy, wo, normal)

        # Step 3: Compute brdf_weight (importance sampling ratio)
        eps = 1e-8  # Small epsilon to avoid division by zero
        pdf_proxy = pdf_proxy.clamp(min=eps)
        brdf_weight = torch.where(pdf_proxy > eps, mlp_brdf / pdf_proxy, torch.zeros_like(mlp_brdf))

        return wi_proxy, pdf_proxy, brdf_weight