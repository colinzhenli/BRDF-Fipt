import torch
import torch.nn as nn
import torch.nn.functional as NF
import math

import sys
sys.path.append('..')

from utils.ops import *

from nerfstudio.field_components import encodings as encoding

def hemisphere_detection(pos):
    return (pos[:,0] + pos[:,1] + pos[:,2]) > 0

class PBRBRDF(nn.Module):
    """ Base BRDF class """
    def __init__(self, albedo=torch.ones(1, 3)):
        super(PBRBRDF,self).__init__()
        # Initialize learnable material parameters
        self.albedo = nn.Parameter(albedo)

    
    def diffuse_sampler(self, sample2, normal):
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
    
    def eval_brdf(self, params, pos, wi, wo, normal, latent=None, batch_mask=None):
        """ evaluate BRDF and pdf
            pos: Bx3 position
            wi: Bx3 light direction
            wo: Bx3 viewing direction
            normal: Bx3 normal
            params: surface BRDF dict with batched parameters
            latent: optional latent code for material
            batch_mask: B indices indicating which batch each ray belongs to
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
        albedo = self.albedo.expand(normal.shape[0], 3)
        
        # Use batch_mask to index into the batched parameters
        hemisphere_mask = hemisphere_detection(pos).view(-1, 1)
        roughness = torch.where(
            hemisphere_mask,
            params['roughness'][:, 0].expand_as(hemisphere_mask),
            params['roughness'][:, 1].expand_as(hemisphere_mask)
        )
        metallic = torch.where(
            hemisphere_mask,
            params['metallic'][:, 0].expand_as(hemisphere_mask),
            params['metallic'][:, 1].expand_as(hemisphere_mask)
        )
        # roughness = roughness[batch_mask]
        # metallic = metallic[batch_mask]

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

        return brdf, pdf
    
    def sample_brdf(self, params, pos, sample1, sample2, wo, normal, latent=None, batch_mask=None):
        """ importance sampling brdf and get brdf/pdf
        Args:
            params: Bx2 material parameters
            pos: Bx3 position
            sample1: B unifrom samples
            sample2: Bx2 uniform samples
            wo: Bx3 viewing direction
            normal: Bx3 normal
            batch_mask: B indices indicating which batch each ray belongs to
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1
            brdf_weight: Bx3 brdf/pdf
        """
        B = sample1.shape[0]
        device = sample1.device

        pdf = torch.zeros(B,device=device)
        brdf = torch.zeros(B,3,device=device)


        mask = (sample1 > 0.5)
        wi_diffuse = self.diffuse_sampler(sample2[mask], normal[mask])
        hemisphere_mask = hemisphere_detection(pos).view(-1, 1)
        roughness = torch.where(
            hemisphere_mask,
            params['roughness'][:, 0].expand_as(hemisphere_mask),
            params['roughness'][:, 1].expand_as(hemisphere_mask)
        )
        # roughness = roughness[batch_mask]
        wi_specular = self.specular_sampler(sample2[~mask], roughness[~mask], wo[~mask], normal[~mask])

        # Construct wi without gradient-breaking assignment
        wi = torch.zeros(B, 3, device=device)
        wi = wi.clone()  # explicitly ensures gradients
        wi[mask] = wi_diffuse
        wi[~mask] = wi_specular
        # get brdf,pdf
        brdf,pdf = self.eval_brdf(params, pos,wi, wo, normal, latent, batch_mask)
        brdf_weight = torch.where(pdf>0,brdf/pdf,0)
        brdf_weight[brdf_weight.isnan()] = 0
        return wi,pdf,brdf_weight
    
class ProxyPBRBRDF(nn.Module):
    def __init__(self, roughness=0.1):
        super(ProxyPBRBRDF, self).__init__()
        self.roughness = nn.Parameter(torch.full((1, 1), roughness).cuda())  # Default roughness parameter
        self.albedo = torch.ones(1, 3).cuda()
    
    def eval_pdf(self, wi, wo, normal, roughness=None):
        """ evaluate BRDF and pdf
            wi: Bx3 light direction
            wo: Bx3 viewing direction
            normal: Bx3 normal
            roughness: Bx1 roughness parameter (optional)
        Return:
            brdf: Bx3
            pdf: Bx1
        """
        # Use provided roughness or default class parameter
        if roughness is None:
            roughness = self.roughness.expand(normal.shape[0], 1)
            
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
        D = D_GGX(NoH, roughness)
        pdf_spec = D/(4*VoH.clamp_min(1e-4))*NoH
        pdf_diff = NoL/math.pi
        pdf = 0.5*pdf_spec + 0.5*pdf_diff

        return pdf

    def diffuse_sampler(self, sample2, normal):
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
    
    def specular_sampler(self, sample2, roughness, wo, normal):
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
    
    def sample_brdf(self, pos, sample1, sample2, wo, normal, roughness=None, batch_mask=None):
        """ importance sampling brdf and get brdf/pdf
        Args:
            pos: Bx3 position
            sample1: B unifrom samples
            sample2: Bx2 uniform samples
            wo: Bx3 viewing direction
            normal: Bx3 normal
            roughness: Bx1 roughness parameter (optional)
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1
        """
        B = sample1.shape[0]
        device = sample1.device
        hemisphere_mask = hemisphere_detection(pos).view(-1, 1)
        roughness = torch.where(
            hemisphere_mask,
            roughness[:, 0].expand_as(hemisphere_mask),
            roughness[:, 1].expand_as(hemisphere_mask)
        )
        # roughness = roughness[batch_mask]

        pdf = torch.zeros(B, device=device)

        mask = (sample1 > 0.5)
        wi_diffuse = self.diffuse_sampler(sample2[mask], normal[mask])
        wi_specular = self.specular_sampler(sample2[~mask], roughness[~mask], wo[~mask], normal[~mask])

        # Construct wi without gradient-breaking assignment
        wi = torch.zeros(B, 3, device=device)
        wi = wi.clone()  # explicitly ensures gradients
        wi[mask] = wi_diffuse
        wi[~mask] = wi_specular
        wi = wi.detach()

        pdf = self.eval_pdf(wi, wo, normal, roughness)
        return wi, pdf
    
class LatentModel(nn.Module):
    """ MLP-based BRDF class with latent conditioning """
    def __init__(self, cfg):
        super(LatentModel, self).__init__()

        # Latent dimension from config
        self.latent_dim = cfg.latent_dim
        
        # Add SH positional encoding module
        self.degree = 3
        self.pos_enc = True
        if self.pos_enc:
            self.sh_encoder = lambda x: components_from_spherical_harmonics(self.degree, x)
            
        # Calculate input dimension after SH encoding
        sh_dim = num_sh_bases(self.degree)
        encoded_input_dim = sh_dim * 3  # wi, wo, normal each encoded by SH
        
        # Add latent dimension to input
        input_dim = encoded_input_dim + self.latent_dim if self.pos_enc else cfg.input_channels + self.latent_dim
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in cfg.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if cfg.activation.lower() == "relu":
                layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, cfg.output_channels))
        layers.append(nn.LeakyReLU(0.2))
        
        self.mlp = nn.Sequential(*layers)

        # Initialize proxy BRDF for importance sampling
        self.proxy_brdf = ProxyPBRBRDF()  # Default roughness

    def forward(self, pos, wi, wo, normal, latent=None, batch_mask=None):
        """
        Evaluate BRDF using MLP with latent conditioning
        Args:
            wi: Bx3 incoming light direction 
            wo: Bx3 outgoing view direction
            normal: Bx3 normal
            latent: Bx{latent_dim} latent code for material
        Returns:
            brdf: Bx1 BRDF values
        """
        hemisphere_mask = hemisphere_detection(pos).view(-1, 1)
        # Determine which latent code to use based on hemisphere
        latent = latent.reshape(latent.shape[0], 2, self.latent_dim)
        latent = torch.where(
            hemisphere_mask,
            latent[:, 0].expand(hemisphere_mask.shape[0], self.latent_dim),
            latent[:, 1].expand(hemisphere_mask.shape[0], self.latent_dim)
        )

        # latent = latent[batch_mask]
        if self.pos_enc:
            wi_enc = self.sh_encoder(wi)
            wo_enc = self.sh_encoder(wo)
            normal_enc = self.sh_encoder(normal)
            x = torch.cat([wi_enc, wo_enc, normal_enc, latent], dim=-1)
            # x = torch.cat([wi_enc, wo_enc, normal_enc], dim=-1)
        else:
            x = torch.cat([wi, wo, normal, latent], dim=-1)
            
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
    
    def eval_brdf(self, gt_params, pos, wi, wo, normal, latent=None, batch_mask=None):
        """
        Evaluate BRDF and pdf after transforming world-space vectors to local space.
        Args:
            pos: Bx3 position
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
        brdf_value = self.forward(pos, wi_local, wo_local, local_normal, latent, batch_mask)
        brdf = brdf_value.expand(-1, 3)

        pdf = NoL / math.pi

        return brdf, pdf
    
    def sample_brdf(self, params, pos, sample1, sample2, wo, normal, latent=None, batch_mask=None):
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

        wi_proxy, pdf_proxy = self.proxy_brdf.sample_brdf(pos, sample1, sample2, wo, normal, params['roughness'], batch_mask)
        stop_gradient_pdf_proxy = pdf_proxy.detach()
        mlp_brdf, _ = self.eval_brdf(params, pos, wi_proxy, wo, normal, latent, batch_mask)
        mlp_brdf = mlp_brdf * pdf_proxy / (stop_gradient_pdf_proxy + 1e-8)
        brdf_weight = torch.where(pdf_proxy > 0, mlp_brdf / (stop_gradient_pdf_proxy + 1e-8), torch.zeros_like(mlp_brdf))

        return wi_proxy, stop_gradient_pdf_proxy, brdf_weight
    