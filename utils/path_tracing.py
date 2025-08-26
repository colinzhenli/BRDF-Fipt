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
    #normals[:,0]=0
    #normals[:,1]=1
    #normals[:,2]=0
    normals = double_sided(-ds,normals)
    return positions,normals,ret.uv.torch(),idx,valid

def ray_intersect_with_tbn(scene, xs, ds):
    xs_mi = mitsuba.Point3f(xs[...,0], xs[...,1], xs[...,2])
    ds_mi = mitsuba.Vector3f(ds[...,0], ds[...,1], ds[...,2])
    rays_mi = mitsuba.Ray3f(xs_mi, ds_mi)

    ret = scene.ray_intersect_preliminary(rays_mi)
    idx = mitsuba.Int(ret.prim_index).torch().long()
    ret = ret.compute_surface_interaction(rays_mi)

    positions = ret.p.torch()
    normals   = NF.normalize(ret.sh_frame.n.torch(), dim=-1)

    tangent   = NF.normalize(ret.dp_du.torch(), dim=-1)
    bitangent = NF.normalize(torch.cross(normals, tangent), dim=-1)
    TBN = torch.stack([tangent, bitangent, normals], dim=-1)  # [B, 3, 3]

    ts  = ret.t.torch()
    valid = (~ts.isinf())
    idx[~valid] = -1
    normals = double_sided(-ds, normals)

    return positions, normals, ret.uv.torch(), idx, valid, TBN

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
        L[active_next] += emit_brdf*emit_weight * w_mis

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
    normal = normal[vis]
    uv=uv[vis]
    batch_mask = batch_mask[vis]
    wo = -wi[vis]
    light_id = light_id[vis]
    TBN = TBN[vis]
    
    # deterministic sampling
    if emitter_sampling:
        wi, emit_pdf, emit_position, emitter_normal= emitter_net.sample_emitter(torch.rand_like(position[..., :2]), position)
        # visibility test
        emit_weight,emit_pdf, _ = emitter_net.eval_emitter(emit_position, wi)
        G = (-wi*emitter_normal).sum(-1).abs() / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
        emit_weight = emit_weight*G[...,None]/emit_pdf.clamp_min(1e-6)
        # emit brdf
        emit_brdf,brdf_pdf = material_net.eval_brdf(None, position, wi,wo,normal,uv, TBN, latent, batch_mask) # gt_params will not be used in neural brdf model
        w_mis = torch.where((emit_pdf>0)&(~brdf_pdf.isinf()),emit_pdf*emit_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
        w_mis[emit_pdf.isinf()|(brdf_pdf==0)] = 1
        L[vis] += emit_brdf*emit_weight * w_mis
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
    
        # Evaluate Le
        Le, emit_pdf, _ = emitter_net.eval_emitter(position, wi)
        G = (-wi*normal).sum(-1).abs() / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
        Le = Le*G[...,None]/emit_pdf.clamp_min(1e-6)
        
        w_mis = torch.where((brdf_pdf>0)&(~emit_pdf.isinf()),brdf_pdf*brdf_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
        w_mis[brdf_pdf.isinf()|(emit_pdf==0)] = 1
        w_mis[w_mis.isnan()] = 0
        L[vis] += brdf_weight*Le * w_mis
    ray_params = torch.cat([position, wi, wo], dim=-1)

    return L, vis, ray_params