import torch
import torch.nn.functional as NF

import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

from .ops import *

def ray_intersect(scene,xs,ds):
    """ warpper of mitsuba ray-mesh intersection 
    Args:
        xs: Bx3 pytorch ray origin
        ds: Bx3 pytorch ray direction
    Return:
        positions: Bx3 intersection location
        normals: Bx3 normals
        uvs: Bx2 uv coordinates
        idx: B triangle indices, -1 indicates no intersection
        valid: B whether a valid intersection
    """

    # convert pytorch tensor to mitsuba
    xs_mi = mitsuba.Point3f(xs[...,0],xs[...,1],xs[...,2])
    ds_mi = mitsuba.Vector3f(ds[...,0],ds[...,1],ds[...,2])
    rays_mi = mitsuba.Ray3f(xs_mi,ds_mi)
    
    ret = scene.ray_intersect_preliminary(rays_mi)
    idx = mitsuba.Int(ret.prim_index).torch().long()
    ret = ret.compute_surface_interaction(rays_mi)
    
    positions = ret.p.torch()
    normals = ret.sh_frame.n.torch()
    normals = NF.normalize(normals,dim=-1)
    
    # check if invalid intersection
    ts  = ret.t.torch()
    valid = (~ts.isinf())
    
    idx[~valid] = -1
    normals = double_sided(-ds,normals)
    return positions,normals,ret.uv.torch(),idx,valid

# def ray_sphere_intersect(self, ray_o, ray_d):
#     """ Ray-sphere intersection test
#     Args:
#         ray_o: Bx3 ray origins
#         ray_d: Bx3 ray directions (normalized)
#     Returns:
#         hit_pos: Bx3 intersection points
#         normals: Bx3 surface normals
#         valid: B whether ray hits sphere
#     """
#     # Solve quadratic equation for ray-sphere intersection
#     oc = ray_o - self.position
#     a = (ray_d * ray_d).sum(-1)
#     b = 2.0 * (oc * ray_d).sum(-1)
#     c = (oc * oc).sum(-1) - self.radius * self.radius
#     disc = b * b - 4 * a * c
    
#     valid = disc > 0
#     t = torch.zeros_like(disc)
#     t[valid] = (-b[valid] - torch.sqrt(disc[valid])) / (2.0 * a[valid])
#     valid = valid & (t > 0)

#     # Compute intersection points and normals
#     hit_pos = ray_o + ray_d * t.unsqueeze(-1)
#     normals = NF.normalize(hit_pos - self.position, dim=-1)
    
#     return hit_pos, normals, valid

def ray_rectangle_intersect_TBN(xs, ds, center=[0, 0, 0], width=0.4, length=0.4):
    """
    Ray-rectangle intersection using mathematical computation.
    Rectangle is defined by center, width (x), length (y), with normal along z-axis.
    
    Args:
        xs: (N, 3) ray origins
        ds: (N, 3) ray directions (normalized)
        center: [x, y, z] center of rectangle
        width: width along x-axis
        length: length along y-axis
    
    Returns:
        positions: (N, 3) intersection points
        normals: (N, 3) surface normals (along z-axis)
        uv: (N, 2) texture coordinates [0,1]
        dp_du: (N, 3) surface partial derivative wrt u
        dp_dv: (N, 3) surface partial derivative wrt v
        idx: (N,) primitive index (-1 for invalid)
        valid: (N,) boolean mask for valid intersections
        TBN: (N, 3, 3) tangent-bitangent-normal frame
    """
    device = xs.device
    N = xs.shape[0]
    
    # Rectangle plane: normal is [0, 0, 1] in local space
    center_pt = torch.tensor(center, dtype=xs.dtype, device=device)
    plane_normal = torch.tensor([0.0, 0.0, 1.0], dtype=xs.dtype, device=device)
    
    # Ray-plane intersection: t = (center - xs) · n / (ds · n)
    numerator = ((center_pt - xs) * plane_normal).sum(-1)
    denominator = (ds * plane_normal).sum(-1)
    
    # Check if ray is parallel to plane
    valid = denominator.abs() > 1e-8
    t = torch.zeros(N, dtype=xs.dtype, device=device)
    t[valid] = numerator[valid] / denominator[valid]
    
    # Check if intersection is in front of ray
    valid = valid & (t > 1e-6)
    
    # Compute intersection points
    positions = xs + ds * t.unsqueeze(-1)
    
    # Check if intersection is within rectangle bounds
    local_pos = positions - center_pt
    half_width = width / 2.0
    half_length = length / 2.0
    
    in_bounds = (local_pos[..., 0].abs() <= half_width) & \
                (local_pos[..., 1].abs() <= half_length)
    valid = valid & in_bounds
    
    # Compute UV coordinates [0, 1]
    uv = torch.zeros(N, 2, dtype=xs.dtype, device=device)
    uv[:, 0] = (local_pos[:, 0] / width) + 0.5  # u: [0, 1]
    uv[:, 1] = (local_pos[:, 1] / length) + 0.5  # v: [0, 1]
    
    # Surface partials: dp/du and dp/dv
    # u maps to x-axis, v maps to y-axis
    dp_du = torch.zeros(N, 3, dtype=xs.dtype, device=device)
    dp_dv = torch.zeros(N, 3, dtype=xs.dtype, device=device)
    dp_du[:, 0] = width   # ∂p/∂u = width along x
    dp_dv[:, 1] = length  # ∂p/∂v = length along y
    
    # Normals (all pointing along z-axis)
    normals = plane_normal.unsqueeze(0).expand(N, 3).clone()
    normals = double_sided(-ds, normals)
    
    # TBN frame: tangent (u-direction), bitangent (v-direction), normal (z)
    tangent = torch.zeros(N, 3, dtype=xs.dtype, device=device)
    tangent[:, 0] = 1.0  # tangent along x (u-direction)
    
    bitangent = torch.zeros(N, 3, dtype=xs.dtype, device=device)
    bitangent[:, 1] = 1.0  # bitangent along y (v-direction)
    
    TBN = torch.stack([tangent, bitangent, normals], dim=-1)  # [N, 3, 3]
    
    # Primitive index (0 for valid hits, -1 for invalid)
    idx = torch.zeros(N, dtype=torch.long, device=device)
    idx[~valid] = -1
    
    return positions, normals, uv, dp_du, dp_dv, idx, valid, TBN


def ray_intersect_with_tbn(scene, xs, ds):
    xs_mi = mitsuba.Point3f(xs[...,0], xs[...,1], xs[...,2])
    ds_mi = mitsuba.Vector3f(ds[...,0], ds[...,1], ds[...,2])
    rays_mi = mitsuba.Ray3f(xs_mi, ds_mi)

    ret = scene.ray_intersect_preliminary(rays_mi)
    idx = mitsuba.Int(ret.prim_index).torch().long()
    ret = ret.compute_surface_interaction(rays_mi)

    positions = ret.p.torch()
    normals   = NF.normalize(ret.n.torch(), dim=-1)

    tangent   = NF.normalize(ret.sh_frame.s.torch(), dim=-1)
    bitangent = NF.normalize(ret.sh_frame.t.torch(), dim=-1)
    TBN = torch.stack([tangent, bitangent, normals], dim=-1)  # [B, 3, 3]

    ts  = ret.t.torch()
    valid = (~ts.isinf())
    idx[~valid] = -1
    normals = double_sided(-ds, normals)

    return positions, normals, ret.uv.torch(), ret.dp_du.torch(), ret.dp_dv.torch(), idx, valid, TBN

def uv_footprint_simple(ro, wi, p, n, dp_du, dp_dv, dx_du, dy_dv, eps=1e-8):
    t = ((p - ro) * wi).sum(-1, keepdim=True).clamp_min(0.0)
    cosi = (n * wi).sum(-1, keepdim=True).abs().clamp_min(eps)
    theta = 0.5 * torch.sqrt((dx_du**2).sum(-1, keepdim=True) + (dy_dv**2).sum(-1, keepdim=True))
    r_surf = (t * theta) / cosi
    ru = r_surf / dp_du.norm(dim=-1, keepdim=True).clamp_min(eps)
    rv = r_surf / dp_dv.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.sqrt(ru * rv)  # isotropic radius in UV

def compute_footprint(rays_o, rays_d, p, dp_du, dp_dv, dx_du, dy_dv, eps=1e-8):
    """
    Compute the surface footprint (per-pixel projected area) for each ray hit.

    Args:
        rays_o, rays_d: (N,3) ray origins and directions
        p:              (N,3) hit positions
        dp_du, dp_dv:   (N,3) surface partials at hit (∂p/∂u, ∂p/∂v)
        dx_du, dy_dv:   (N,3) direction differentials wrt screen x and y (from get_rays)
        eps:            small epsilon to avoid division by zero

    Returns:
        dp_dx, dp_dy:   (N,3) 3D intersection differentials on the surface for screen x,y
        area:           (N,)  footprint area on the surface (per pixel)
        n:              (N,3) shading normal at the hit
        t:              (N,)  parameter distance along each ray to the hit
    """
    # Distance t to hit; works whether rays_d is normalized or not
    d = rays_d
    o = rays_o
    t = ((p - o) * d).sum(-1) / (d * d).sum(-1).clamp_min(eps)  # (N,)

    # Surface normal
    n = torch.cross(dp_du, dp_dv, dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + eps)  # (N,3)

    ndotd = (n * d).sum(-1).clamp(min=-1.0, max=1.0)  # (N,)
    # Avoid near-grazing division
    ndotd = torch.where(ndotd.abs() < eps, ndotd.sign() * eps, ndotd)

    def one_axis(dd):  # dd = d_x or d_y
        ndotdd = (n * dd).sum(-1)                # (N,)
        dt = - t * ndotdd / ndotd                # (N,)
        dp = t[..., None] * dd + dt[..., None] * d  # (N,3)

        # optional: remove any tiny normal component (numerical stability)
        dp = dp - (dp * n).sum(-1, keepdim=True) * n
        return dp

    dp_dx = one_axis(dx_du)
    dp_dy = one_axis(dy_dv)

    # Footprint area = area of parallelogram spanned by dp_dx and dp_dy
    area = torch.linalg.norm(torch.cross(dp_dx, dp_dy, dim=-1), dim=-1)  # (N,)

    return area
    
def batched_path_tracing_dynamic_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv,spp, brdf_sampling, emitter_sampling, gt_params=None, latent=None):
    """ Path trace current scene
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: BxNx3 ray origin
        rays_d: BxNx3 ray direction
        dx_du,dy_dv: BxNx3 ray differential
        spp: samples per pixel
        brdf_sampling: boolean flag for BRDF importance sampling
        emitter_sampling: boolean flag for emitter importance sampling
        gt_params: optional ground truth material parameters
        latent: optional batched latent code for material network
    Return:
        L: (B*N)x3 traced results unbatched
    """
    # flatten the rays
    # Create batch mask where each row contains the same batch index
    # For rays with shape B, N, 3, create mask with shape B, N
    batch_mask = torch.arange(len(rays_o), device=rays_o.device).view(rays_o.shape[0], 1).expand(rays_o.shape[0], rays_o.shape[1])
    # batch_mask = torch.zeros(len(rays_o), device=rays_o.device)
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    dx_du = dx_du.reshape(-1,3)
    dy_dv = dy_dv.reshape(-1,3)
    batch_mask = batch_mask.reshape(-1)
    N = len(rays_o)
    N_lights = emitter_net.num_lights
    device = rays_o.device
    
    # sample camera ray
    # du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    # wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    wi = rays_d
    # Add mask for wi z component
    position = rays_o.repeat_interleave(spp,0)
    
    # compute first intersection
    position,normal,_, _,vis = ray_intersect(scene,position,wi)
    # position, normal, vis = ray_sphere_intersect(scene,position,wi)
    L = torch.zeros(vis.shape[0],3,device=device)
    if not vis.any():
        print("No valid intersection")
        return L.reshape(N,spp,3).mean(1), None, None
    position = position[vis]
    normal = normal[vis]
    batch_mask = batch_mask[vis]
    wo = -wi[vis]
    
    # deterministic sampling
    wi,emit_pdf, emit_position, idx = emitter_net.sample_emitter(position)
    normal = normal.repeat_interleave(emitter_net.num_lights,0)
    position = position.repeat_interleave(emitter_net.num_lights,0)
    wo = wo.repeat_interleave(emitter_net.num_lights,0)
    batch_mask = batch_mask.repeat_interleave(emitter_net.num_lights,0)
    # visibility test
    emit_weight,_,_ = emitter_net.eval_emitter(emit_position, idx)
    emit_vis = (wi*normal).sum(-1,keepdim=True) > 0 # B, 1
    
    # goemetry term (assume double sided area light)
    G = 1 / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
    emit_weight = emit_weight*emit_vis*G[...,None]/emit_pdf.clamp_min(1e-6)
    
    # Now, reshape and average over light dimension
    emit_brdf,_ = material_net.eval_brdf(gt_params,position, wi,wo,normal, latent, batch_mask)
    L[vis] += (emit_brdf*emit_weight).reshape(-1, N_lights,3).mean(1)
    L = L.reshape(N,spp,3).mean(1)
    ray_params = torch.cat([position, wi, wo], dim=-1)
    return L, vis, ray_params

def batched_path_tracing_preset_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv, light_id, spp,  brdf_sampling, emitter_sampling, gt_params=None, latent=None):
    """ Path trace aligned with real capture
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: BxNx3 ray origin
        rays_d: BxNx3 ray direction
        dx_du,dy_dv: BxNx3 ray differential
        spp: samples per pixel
        brdf_sampling: boolean flag for BRDF importance sampling
        emitter_sampling: boolean flag for emitter importance sampling
        gt_params: optional ground truth material parameters
        latent: optional batched latent code for material network
    Return:
        L: (B*N)x3 traced results unbatched
    """
    # flatten the rays
    # Create batch mask where each row contains the same batch index
    # For rays with shape B, N, 3, create mask with shape B, N
    batch_mask = torch.arange(len(rays_o), device=rays_o.device).view(rays_o.shape[0], 1).expand(rays_o.shape[0], rays_o.shape[1])
    # batch_mask = torch.zeros(len(rays_o), device=rays_o.device)
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    light_id = light_id.reshape(-1)
    dx_du = dx_du.reshape(-1,3)
    dy_dv = dy_dv.reshape(-1,3)
    batch_mask = batch_mask.reshape(-1)
    N = len(rays_o)
    device = rays_o.device
    
    # sample camera ray
    du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    # wi = rays_d
    # Add mask for wi z component
    position = rays_o.repeat_interleave(spp,0)
    
    # compute first intersection
    position,normal,uv, _,vis = ray_intersect(scene,position,wi)
    # position, normal, vis = ray_sphere_intersect(scene,position,wi)
    L = torch.zeros(vis.shape[0],3,device=device)
    if not vis.any():
        print("No valid intersection")
        return L.reshape(N,spp,3).mean(1), None, None
    position = position[vis]
    normal_raw=normal
    normal = normal[vis]
    uv=uv[vis]
    batch_mask = batch_mask[vis]
    wo = -wi[vis]
    light_id = light_id[vis]
    
    # deterministic sampling
    wi,emit_pdf, emit_position, idx = emitter_net.sample_emitter(position, light_id)
    # visibility test
    emit_weight,_,_ = emitter_net.eval_emitter(emit_position, idx)
    emit_vis = (wi*normal).sum(-1,keepdim=True) > 0 # B, 1
    
    # goemetry term (assume double sided area light)
    G = 1 / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
    emit_weight = emit_weight*emit_vis*G[...,None]/emit_pdf.clamp_min(1e-6)
    
    # Now, reshape and average over light dimension
    emit_brdf,_ = material_net.eval_brdf(None, position, wi,wo,normal,uv, latent, batch_mask)
    debug_mode = False
    if not debug_mode:
        L[vis] += (emit_brdf*emit_weight)
    else:
        L[vis] += torch.tensor([1.0, 0.0, 0.0], device=L.device).expand_as(L[vis])
    ray_params = torch.cat([position, wi, wo], dim=-1)

    emit_vis_raw = torch.zeros((normal_raw.shape[0],1),dtype=emit_vis.dtype, device=emit_vis.device) 
    emit_vis_raw[vis] = emit_vis
    #return emit_vis_raw, vis, ray_params
    #return (normal_raw+1)/2, vis, ray_params
    return L, vis, ray_params

def batched_path_tracing_dynamic_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv,spp, brdf_sampling, emitter_sampling, gt_params=None, latent=None):
    """ Path trace current scene
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: BxNx3 ray origin
        rays_d: BxNx3 ray direction
        dx_du,dy_dv: BxNx3 ray differential
        spp: samples per pixel
        brdf_sampling: boolean flag for BRDF importance sampling
        emitter_sampling: boolean flag for emitter importance sampling
        gt_params: optional ground truth material parameters
        latent: optional batched latent code for material network
    Return:
        L: (B*N)x3 traced results unbatched
    """
    # flatten the rays
    # Create batch mask where each row contains the same batch index
    # For rays with shape B, N, 3, create mask with shape B, N
    batch_mask = torch.arange(len(rays_o), device=rays_o.device).view(rays_o.shape[0], 1).expand(rays_o.shape[0], rays_o.shape[1])
    # batch_mask = torch.zeros(len(rays_o), device=rays_o.device)
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    dx_du = dx_du.reshape(-1,3)
    dy_dv = dy_dv.reshape(-1,3)
    batch_mask = batch_mask.reshape(-1)
    N = len(rays_o)
    N_lights = emitter_net.num_lights
    device = rays_o.device
    
    # sample camera ray
    # du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    # wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    wi = rays_d
    # Add mask for wi z component
    position = rays_o.repeat_interleave(spp,0)
    
    # compute first intersection
    position,normal,_, _,vis = ray_intersect(scene,position,wi)
    # position, normal, vis = ray_sphere_intersect(scene,position,wi)
    L = torch.zeros(vis.shape[0],3,device=device)
    if not vis.any():
        print("No valid intersection")
        return L.reshape(N,spp,3).mean(1), None, None
    position = position[vis]
    normal = normal[vis]
    batch_mask = batch_mask[vis]
    wo = -wi[vis]
    
    # deterministic sampling
    wi,emit_pdf, emit_position, idx = emitter_net.sample_emitter(position)
    normal = normal.repeat_interleave(emitter_net.num_lights,0)
    position = position.repeat_interleave(emitter_net.num_lights,0)
    wo = wo.repeat_interleave(emitter_net.num_lights,0)
    batch_mask = batch_mask.repeat_interleave(emitter_net.num_lights,0)
    # visibility test
    emit_weight,_,_ = emitter_net.eval_emitter(emit_position, idx)
    emit_vis = (wi*normal).sum(-1,keepdim=True) > 0 # B, 1
    
    # goemetry term (assume double sided area light)
    G = 1 / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
    emit_weight = emit_weight*emit_vis*G[...,None]/emit_pdf.clamp_min(1e-6)
    
    # Now, reshape and average over light dimension
    emit_brdf,_ = material_net.eval_brdf(gt_params,position, wi,wo,normal, latent, batch_mask)
    L[vis] += (emit_brdf*emit_weight).reshape(-1, N_lights,3).mean(1)
    L = L.reshape(N,spp,3).mean(1)
    ray_params = torch.cat([position, wi, wo], dim=-1)
    return L, vis, ray_params

def batched_path_tracing_tbn_preset_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv, light_id, spp,  brdf_sampling, emitter_sampling, gt_params=None, latent=None):
    """ Path trace aligned with real capture
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: BxNx3 ray origin
        rays_d: BxNx3 ray direction
        dx_du,dy_dv: BxNx3 ray differential
        spp: samples per pixel
        brdf_sampling: boolean flag for BRDF importance sampling
        emitter_sampling: boolean flag for emitter importance sampling
        gt_params: optional ground truth material parameters
        latent: optional batched latent code for material network
    Return:
        L: (B*N)x3 traced results unbatched
    """
    # flatten the rays
    # Create batch mask where each row contains the same batch index
    # For rays with shape B, N, 3, create mask with shape B, N
    batch_mask = torch.arange(len(rays_o), device=rays_o.device).view(rays_o.shape[0], 1).expand(rays_o.shape[0], rays_o.shape[1])
    # batch_mask = torch.zeros(len(rays_o), device=rays_o.device)
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    light_id = light_id.reshape(-1)
    dx_du = dx_du.reshape(-1,3)
    dy_dv = dy_dv.reshape(-1,3)
    batch_mask = batch_mask.reshape(-1)
    N = len(rays_o)
    device = rays_o.device
    
    # sample camera ray
    du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    # wi = rays_d
    # Add mask for wi z component
    position = rays_o.repeat_interleave(spp,0)
    
    # compute first intersection
    position,normal,uv, _,vis, TBN = ray_intersect_with_tbn(scene,position,wi)
    # position, normal, vis = ray_sphere_intersect(scene,position,wi)
    L = torch.zeros(vis.shape[0],3,device=device)
    if not vis.any():
        print("No valid intersection")
        return L.reshape(N,spp,3).mean(1), None, None
    position = position[vis]
    normal_raw=normal
    normal = normal[vis]
    uv=uv[vis]
    batch_mask = batch_mask[vis]
    wo = -wi[vis]
    light_id = light_id[vis]
    TBN = TBN[vis]
    
    # deterministic sampling
    wi,emit_pdf, emit_position, idx = emitter_net.sample_emitter(position, light_id)
    # visibility test
    emit_weight,_,_ = emitter_net.eval_emitter(emit_position, idx)
    emit_vis = (wi*normal).sum(-1,keepdim=True) > 0 # B, 1
    
    # goemetry term (assume double sided area light)
    G = 1 / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
    emit_weight = emit_weight*emit_vis*G[...,None]/emit_pdf.clamp_min(1e-6)
    
    # Now, reshape and average over light dimension
    emit_brdf,_ = material_net.eval_brdf(None, position, wi,wo,normal,uv, TBN, latent, batch_mask)
    debug_mode = False
    if not debug_mode:
        L[vis] += (emit_brdf*emit_weight)
    else:
        L[vis] += torch.tensor([1.0, 0.0, 0.0], device=L.device).expand_as(L[vis])
    ray_params = torch.cat([position, wi, wo], dim=-1)

    emit_vis_raw = torch.zeros((normal_raw.shape[0],1),dtype=emit_vis.dtype, device=emit_vis.device) 
    emit_vis_raw[vis] = emit_vis
    #return emit_vis_raw, vis, ray_params
    #return (normal_raw+1)/2, vis, ray_params
    return L, vis, ray_params

def path_tracing_envmap_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv,spp, brdf_sampling, emitter_sampling, gt_params=None, latent=None):
    """ Path trace current scene
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: Bx3 ray origin
        rays_d: Bx3 ray direction
        dx_du,dy_dv: Bx3 ray differential
        spp: sampler per pixel
        indir_depth: indirect illumination depth
    Return:
        L: Bx3 traced results
    """
    B = len(rays_o)
    device = rays_o.device
    
    # sample camera ray
    # Set fixed random seed for reproducibility
    # torch.manual_seed(42)    
    # Generate random offsets for ray sampling
    du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    
    # Add mask for wi z component
    position = rays_o.repeat_interleave(spp,0)
    
    # compute first intersection
    position,normal,_, _,vis = ray_intersect(scene,position,wi)
    L,_,valid_next = emitter_net.eval_emitter(position,wi)
    L[vis] = 0 # set the radiance to 0 for the valid intersection
    valid_next = vis
    # drop invalid intersection
    if not valid_next.any():
        return L.reshape(B,spp,3).mean(1)
    position = position[valid_next]
    normal = normal[valid_next]
    wo = -wi[valid_next]
    active_next = valid_next.clone()
    # Create batch mask for all samples (all batch 0s since this is not batched)
    batch_size = position.shape[0]
    batch_mask = torch.zeros(batch_size, dtype=torch.long, device=position.device)

    # Sample the environment map instead of a point emitter
    if emitter_sampling:
        wi, emit_pdf, _ = emitter_net.sample_emitter(torch.rand_like(position[..., :2]), position)

        # Evaluate the environment map along sampled directions
        emit_weight, emit_pdf, _ = emitter_net.eval_emitter(position, wi)
        emit_weight = emit_weight / emit_pdf.clamp_min(1e-6)
        # emit brdf
        emit_brdf,brdf_pdf = material_net.eval_brdf(gt_params,position, wi,wo,normal,latent, batch_mask) # gt_params will not be used in neural brdf model
        w_mis = torch.where((emit_pdf>0)&(~brdf_pdf.isinf()),emit_pdf*emit_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
        w_mis[emit_pdf.isinf()|(brdf_pdf==0)] = 1
        L[active_next] += emit_brdf*emit_weight

    # sample brdf
    if brdf_sampling:
        wi,brdf_pdf,brdf_weight = material_net.sample_brdf(
            gt_params,
            position,
            torch.rand(len(normal),device=device),
            torch.rand(len(normal),2,device=device),
            wo,normal,
            latent,
            batch_mask
        ) # ground truth roughness will be used in brdf sampling
    
        # Evaluate Le from environment map
        Le, emit_pdf, valid_next = emitter_net.eval_emitter(position, wi)

        # Update BRDF PDF
        brdf_pdf = brdf_pdf 
        
        w_mis = torch.where((brdf_pdf>0)&(~emit_pdf.isinf()),brdf_pdf*brdf_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
        w_mis[brdf_pdf.isinf()|(emit_pdf==0)] = 1
        w_mis[w_mis.isnan()] = 0
        L[active_next] += brdf_weight*Le * w_mis

    L = L.reshape(B,spp,3).mean(1)
    return L

# def batched_path_tracing_tbn_real_area_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv, light_id, spp, brdf_sampling, emitter_sampling, gt_params=None, latent=None):
#     """ Path trace with real capture
#     Args:
#         scene: mitsuba scene
#         emitter_net: emitter object
#         material_net: material object
#         rays_o: BxNx3 ray origin
#         rays_d: BxNx3 ray direction
#         dx_du,dy_dv: BxNx3 ray differential
#         spp: samples per pixel
#         brdf_sampling: boolean flag for BRDF importance sampling
#         emitter_sampling: boolean flag for emitter importance sampling
#         gt_params: optional ground truth material parameters
#         latent: optional batched latent code for material network
#     Return:
#         L: (B*N)x3 traced results unbatched
#     """
#     # flatten the rays
#     # Create batch mask where each row contains the same batch index
#     # For rays with shape B, N, 3, create mask with shape B, N
#     batch_mask = torch.arange(len(rays_o), device=rays_o.device).view(rays_o.shape[0], 1).expand(rays_o.shape[0], rays_o.shape[1])
#     # batch_mask = torch.zeros(len(rays_o), device=rays_o.device)
#     rays_o = rays_o.reshape(-1,3)
#     rays_d = rays_d.reshape(-1,3)
#     dx_du = dx_du.reshape(-1,3)
#     dy_dv = dy_dv.reshape(-1,3)
#     light_id = light_id.reshape(-1)
#     batch_mask = batch_mask.reshape(-1)
#     N = len(rays_o)
#     device = rays_o.device
    
#     # sample camera ray
#     du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
#     wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
#     # wi = rays_d
#     # Add mask for wi z component
#     position = rays_o.repeat_interleave(spp,0)
    
#     # compute first intersection
#     position,normal,uv, _,vis, TBN = ray_intersect_with_tbn(scene,position,wi)
#     # position, normal, vis = ray_sphere_intersect(scene,position,wi)
#     L = torch.zeros(vis.shape[0],3,device=device)
#     if not vis.any():
#         print("No valid intersection")
#         return L.reshape(N,spp,3).mean(1), None, None
#     position = position[vis]
#     normal = normal[vis]
#     uv=uv[vis]
#     batch_mask = batch_mask[vis]
#     wo = -wi[vis]
#     light_id = light_id[vis]
#     TBN = TBN[vis]
    
#     # deterministic sampling
#     if emitter_sampling:
#         wi, emit_pdf, emit_position, emitter_normal= emitter_net.sample_emitter(torch.rand_like(position[..., :2]), position, light_id)
#         # visibility test
#         emit_weight,emit_pdf, _ = emitter_net.eval_emitter(position, wi, light_id)
#         G = (-wi*emitter_normal).sum(-1).abs() / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
#         emit_weight = emit_weight*G[...,None]/emit_pdf.clamp_min(1e-6)
#         # emit brdf
#         emit_brdf,brdf_pdf = material_net.eval_brdf(None, position, wi,wo,normal,uv, TBN, latent, batch_mask) # gt_params will not be used in neural brdf model
#         w_mis = torch.where((emit_pdf>0)&(~brdf_pdf.isinf()),emit_pdf*emit_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
#         w_mis[emit_pdf.isinf()|(brdf_pdf==0)] = 1
#         L[vis] += emit_brdf*emit_weight
#     # sample brdf
#     if brdf_sampling:
#         wi,brdf_pdf,brdf_weight = material_net.sample_brdf(
#             gt_params,
#             position,
#             torch.rand(len(normal),device=device),
#             torch.rand(len(normal),2,device=device),
#             wo,normal,
#             latent,
#             batch_mask
#         ) # ground truth roughness will be used in brdf sampling
    
#         # Evaluate Le
#         Le, emit_pdf, _ = emitter_net.eval_emitter(position, wi, light_id)
#         G = (-wi*normal).sum(-1).abs() / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
#         Le = Le*G[...,None]/emit_pdf.clamp_min(1e-6)
        
#         w_mis = torch.where((brdf_pdf>0)&(~emit_pdf.isinf()),brdf_pdf*brdf_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
#         w_mis[brdf_pdf.isinf()|(emit_pdf==0)] = 1
#         w_mis[w_mis.isnan()] = 0
#         L[vis] += brdf_weight*Le * w_mis
#     ray_params = torch.cat([position, wi, wo], dim=-1)

#     return L, vis, ray_params

def batched_path_tracing_tbn_real_area_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv, light_id, spp, brdf_sampling, emitter_sampling, gt_params=None, latent=None):
    """ Path trace with real capture
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: BxNx3 ray origin
        rays_d: BxNx3 ray direction
        dx_du,dy_dv: BxNx3 ray differential
        spp: samples per pixel
        brdf_sampling: boolean flag for BRDF importance sampling
        emitter_sampling: boolean flag for emitter importance sampling
        gt_params: optional ground truth material parameters
        latent: optional batched latent code for material network
    Return:
        L: (B*N)x3 traced results unbatched
    """
    # flatten the rays
    # Create batch mask where each row contains the same batch index
    # For rays with shape B, N, 3, create mask with shape B, N
    batch_mask = torch.arange(len(rays_o), device=rays_o.device).view(rays_o.shape[0], 1).expand(rays_o.shape[0], rays_o.shape[1])
    # batch_mask = torch.zeros(len(rays_o), device=rays_o.device)
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    dx_du = dx_du.reshape(-1,3)
    dy_dv = dy_dv.reshape(-1,3)
    light_id = light_id.reshape(-1)
    batch_mask = batch_mask.reshape(-1)
    N = len(rays_o)
    device = rays_o.device
    
    # sample camera ray
    du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    
    # wi = rays_d.repeat_interleave(spp, 0)
    # Add mask for wi z component
    position = rays_o.repeat_interleave(spp,0)
    light_id = light_id.repeat_interleave(spp,0)
    
    # compute first intersection
    # Check if scene is a dictionary (scene parameters) or a Mitsuba scene object
    if isinstance(scene, dict):
        # Use mathematical ray-rectangle intersection
        position, normal, uv, dp_du, dp_dv, _, vis, TBN = ray_rectangle_intersect_TBN(
            position, wi, 
            center=scene.get('center', [0, 0, 0]),
            width=scene.get('width', 0.4),
            length=scene.get('length', 0.4)
        )
    else:
        # Use Mitsuba scene intersection
        position, normal, uv, dp_du, dp_dv, _, vis, TBN = ray_intersect_with_tbn(scene, position, wi)
    footprint_vis = uv_footprint_simple(rays_o.repeat_interleave(spp,0)[vis], wi[vis], position[vis], normal[vis], dp_du[vis], dp_dv[vis], dx_du.repeat_interleave(spp,0)[vis], dy_dv.repeat_interleave(spp,0)[vis])
    
    # position, normal, vis = ray_sphere_intersect(scene,position,wi)
    L = torch.zeros(vis.shape[0],3,device=device)
    if not vis.any():
        print("No valid intersection")
        return L.reshape(N,spp,3).mean(1), None, None
    batch_size = position.shape[0]
    batch_mask = torch.zeros(batch_size, dtype=torch.long, device=position.device)
    batch_mask = batch_mask[vis]
    position = position[vis]

    normal = normal[vis]
    uv=uv[vis]
    dp_du = dp_du[vis]
    dp_dv = dp_dv[vis]

    wo = -wi[vis]
    light_id = light_id[vis]
    TBN = TBN[vis]
    
    angle_ok_all = torch.zeros(N*spp, dtype=torch.bool, device=device)
    gray_patch_idx = torch.zeros(N*spp, dtype=torch.long, device=device)
    
    # deterministic sampling
    if emitter_sampling:
        wi, emit_pdf, emit_position, emitter_normal= emitter_net.sample_emitter(torch.rand_like(position[..., :2]), position, light_id)
        # visibility test
        emit_weight,emit_pdf, _ = emitter_net.eval_emitter(position, wi, light_id)
        G = (wi*normal).sum(-1).abs() * (-wi*emitter_normal).sum(-1).abs() / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
        emit_weight = emit_weight*G[...,None]/emit_pdf.clamp_min(1e-6)
        # emit brdf
        emit_brdf,brdf_pdf,angle_ok, patch_index = material_net.eval_brdf(None, position, wi,wo,normal,uv, TBN, latent, batch_mask, footprint_vis, dp_du, dp_dv) # gt_params will not be used in neural brdf model
        """ mask out the brdf for the invalid angle, used for radiometric calibration """
        angle_ok_all[vis] = angle_ok
        gray_patch_idx[vis] = patch_index
        # pixel_all_ok = angle_ok_full.reshape(N, spp).all(dim=1)
        
        # emit_brdf = emit_brdf * pixel_all_ok.repeat_interleave(spp)[vis].unsqueeze(-1)
        w_mis = torch.where((emit_pdf>0)&(~brdf_pdf.isinf()),emit_pdf*emit_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
        w_mis[emit_pdf.isinf()|(brdf_pdf==0)] = 1
        # Avoid in-place indexed operation for cleaner autograd graph
        contribution = emit_brdf * emit_weight
        L_update = torch.zeros_like(L)
        L_update[vis] = contribution
        L = L + L_update
        # L[vis] += emit_weight * emit_brdf
    # sample brdf
    if brdf_sampling:
        wi,brdf_pdf,brdf_weight = material_net.sample_brdf(
            gt_params,
            position,
            torch.rand(len(normal),device=device),
            torch.rand(len(normal),2,device=device),
            wo,normal,
            latent,
            batch_mask,
            dp_du,
            dp_dv
        ) # ground truth roughness will be used in brdf sampling
    
        # Evaluate Le
        Le, emit_pdf, _ = emitter_net.eval_emitter(position, wi, light_id)
        G = (wi*normal).sum(-1).abs() * (-wi*emitter_normal).sum(-1).abs() / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
        Le = Le*G[...,None]/emit_pdf.clamp_min(1e-6)
        
        w_mis = torch.where((brdf_pdf>0)&(~emit_pdf.isinf()),brdf_pdf*brdf_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
        w_mis[brdf_pdf.isinf()|(emit_pdf==0)] = 1
        w_mis[w_mis.isnan()] = 0
        # Avoid in-place indexed operation for cleaner autograd graph
        contribution = brdf_weight * Le * w_mis
        L_update = torch.zeros_like(L)
        L_update[vis] = contribution
        L = L + L_update
    ray_params = torch.cat([position, wi, wo], dim=-1)
    L = L.reshape(N,spp,3).mean(1)
    # Merge visibility across multiple spp - if any ray is visible, vis is true
    vis_reshaped = vis.reshape(N, spp)
    gray_patch_idx = gray_patch_idx.reshape(N, spp)
    angle_ok = angle_ok_all.reshape(N, spp)
    
    # Check if patch indices are consistent across spp for each pixel (treating -1 same as others)
    # For each pixel, all patch indices (including -1) should be the same
    patch_min = gray_patch_idx.min(dim=1)[0]  # [N]
    patch_max = gray_patch_idx.max(dim=1)[0]  # [N]
    patch_inconsistent = patch_min != patch_max  # [N]
    
    # pixel_all_ok is True only if all angles are ok AND patch indices are consistent
    pixel_all_ok = angle_ok.all(dim=1) & (~patch_inconsistent)
    gray_patch_idx = gray_patch_idx[:, 0]
    
    vis = vis_reshaped.any(dim=1)
    return L, vis, ray_params, gray_patch_idx, pixel_all_ok 