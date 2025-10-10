"""
Profiling script to identify bottlenecks in batched_path_tracing_tbn_real_area_emitter
"""
import torch
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    """Simple timing context manager"""
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"{name}: {elapsed*1000:.2f}ms")

def profile_path_tracing_detailed(scene, emitter_net, material_net, rays_o, rays_d, dx_du, dy_dv, light_id, spp, brdf_sampling, emitter_sampling, gt_params=None, latent=None):
    """
    Instrumented version of batched_path_tracing_tbn_real_area_emitter with detailed timing
    """
    import torch.nn.functional as NF
    from utils.path_tracing import ray_intersect_with_tbn, uv_footprint_simple
    
    print("\n=== Profiling batched_path_tracing_tbn_real_area_emitter ===")
    
    with timer("1. Setup and reshape"):
        batch_mask = torch.arange(len(rays_o), device=rays_o.device).view(rays_o.shape[0], 1).expand(rays_o.shape[0], rays_o.shape[1])
        rays_o_flat = rays_o.reshape(-1,3)
        rays_d_flat = rays_d.reshape(-1,3)
        dx_du_flat = dx_du.reshape(-1,3)
        dy_dv_flat = dy_dv.reshape(-1,3)
        light_id_flat = light_id.reshape(-1)
        batch_mask_flat = batch_mask.reshape(-1)
        N = len(rays_o_flat)
        device = rays_o_flat.device
    
    with timer("2. Sample camera rays"):
        du, dv = torch.rand(2, len(rays_o_flat), spp, 1, device=device) - 0.5
        wi = NF.normalize(rays_d_flat[:,None] + dx_du_flat[:,None]*du + dy_dv_flat[:,None]*dv, dim=-1).reshape(-1,3)
        position = rays_o_flat.repeat_interleave(spp, 0)
        light_id_spp = light_id_flat.repeat_interleave(spp, 0)
    
    with timer("3. Ray intersection with TBN (MITSUBA)"):
        position, normal, uv, dp_du, dp_dv, _, vis, TBN = ray_intersect_with_tbn(scene, position, wi)
    
    with timer("4. UV footprint computation"):
        footprint_vis = uv_footprint_simple(
            rays_o_flat.repeat_interleave(spp,0)[vis], 
            wi[vis], 
            position[vis], 
            normal[vis], 
            dp_du[vis], 
            dp_dv[vis], 
            dx_du_flat.repeat_interleave(spp,0)[vis], 
            dy_dv_flat.repeat_interleave(spp,0)[vis]
        )
    
    L = torch.zeros(vis.shape[0], 3, device=device)
    if not vis.any():
        print("No valid intersection")
        return L.reshape(N, spp, 3).mean(1), None, None
    
    with timer("5. Filter valid intersections"):
        batch_size = position.shape[0]
        batch_mask_new = torch.zeros(batch_size, dtype=torch.long, device=position.device)
        batch_mask_new = batch_mask_new[vis]
        position_vis = position[vis]
        normal_vis = normal[vis]
        uv_vis = uv[vis]
        dp_du_vis = dp_du[vis]
        dp_dv_vis = dp_dv[vis]
        wo = -wi[vis]
        light_id_vis = light_id_spp[vis]
        TBN_vis = TBN[vis]
    
    if emitter_sampling:
        with timer("6a. Emitter sampling"):
            wi_emit, emit_pdf, emit_position, emitter_normal = emitter_net.sample_emitter(
                torch.rand_like(position_vis[..., :2]), 
                position_vis, 
                light_id_vis
            )
        
        with timer("6b. Emitter evaluation"):
            emit_weight, emit_pdf, _ = emitter_net.eval_emitter(position_vis, wi_emit, light_id_vis)
        
        with timer("6c. Geometry term computation"):
            G = (-wi_emit*emitter_normal).sum(-1).abs() / (emit_position-position_vis).pow(2).sum(-1).clamp_min(1e-6)
            emit_weight = emit_weight * G[...,None] / emit_pdf.clamp_min(1e-6)
        
        with timer("6d. Material BRDF evaluation (MAIN BOTTLENECK?)"):
            emit_brdf, brdf_pdf = material_net.eval_brdf(
                None, position_vis, wi_emit, wo, normal_vis, uv_vis, 
                TBN_vis, latent, batch_mask_new, footprint_vis, dp_du_vis, dp_dv_vis
            )
        
        with timer("6e. MIS weight and accumulation"):
            w_mis = torch.where(
                (emit_pdf>0) & (~brdf_pdf.isinf()),
                emit_pdf*emit_pdf / (emit_pdf*emit_pdf + brdf_pdf*brdf_pdf),
                0
            )
            w_mis[emit_pdf.isinf() | (brdf_pdf==0)] = 1
            L[vis] += emit_brdf * emit_weight
    
    if brdf_sampling:
        with timer("7a. BRDF sampling"):
            wi_brdf, brdf_pdf_samp, brdf_weight = material_net.sample_brdf(
                gt_params, position_vis,
                torch.rand(len(normal_vis), device=device),
                torch.rand(len(normal_vis), 2, device=device),
                wo, normal_vis, latent, batch_mask_new, dp_du_vis, dp_dv_vis
            )
        
        with timer("7b. Emitter evaluation for BRDF samples"):
            Le, emit_pdf_brdf, _ = emitter_net.eval_emitter(position_vis, wi_brdf, light_id_vis)
        
        with timer("7c. Geometry and MIS"):
            G_brdf = (-wi_brdf*normal_vis).sum(-1).abs() / (emit_position-position_vis).pow(2).sum(-1).clamp_min(1e-6)
            Le = Le * G_brdf[...,None] / emit_pdf_brdf.clamp_min(1e-6)
            w_mis_brdf = torch.where(
                (brdf_pdf_samp>0) & (~emit_pdf_brdf.isinf()),
                brdf_pdf_samp*brdf_pdf_samp / (emit_pdf_brdf*emit_pdf_brdf + brdf_pdf_samp*brdf_pdf_samp),
                0
            )
            w_mis_brdf[brdf_pdf_samp.isinf() | (emit_pdf_brdf==0)] = 1
            w_mis_brdf[w_mis_brdf.isnan()] = 0
            L[vis] += brdf_weight * Le * w_mis_brdf
    
    with timer("8. Final reshape and reduction"):
        ray_params = torch.cat([position_vis, wi_emit if emitter_sampling else wi_brdf, wo], dim=-1)
        L = L.reshape(N, spp, 3).mean(1)
        vis_reshaped = vis.reshape(N, spp)
        vis_final = vis_reshaped.any(dim=1)
    
    print("=== End Profiling ===\n")
    return L, vis_final, ray_params


def profile_material_eval_brdf(material_net, position, wi, wo, normal, uv, TBN, latent, batch_mask, footprint_vis, dp_du, dp_dv):
    """
    Detailed profiling of material_net.eval_brdf
    """
    print("\n=== Profiling material_net.eval_brdf ===")
    
    with timer("BRDF 1. Input validation"):
        NoL = (wi*normal).sum(-1, keepdim=True)
        NoV = (wo*normal).sum(-1, keepdim=True)
    
    with timer("BRDF 2. Texture sampling (latent grid)"):
        if material_net.training and material_net.Gaussian_blur:
            tex = material_net._blur_latent(material_net.global_step)
        else:
            tex = material_net.latent_texture
        latent_sampled = material_net.sample_latent_from_texture(uv, tex)
    
    with timer("BRDF 3. World to local transformation"):
        if material_net.predict_normal:
            normal_pred = torch.nn.functional.normalize(latent_sampled[..., -3:], dim=-1)
        else:
            normal_pred = normal
        wi_local = material_net.world_to_local(wi, normal_pred)
        wo_local = material_net.world_to_local(wo, normal_pred)
        local_normal = torch.zeros_like(wi_local)
        local_normal[..., 2] = 1.0
    
    if material_net.colorful_texture and material_net.larger_latent_dim:
        with timer("BRDF 4. Split latent for RGB"):
            latent_r = latent_sampled[..., :material_net.latent_dim]
            latent_g = latent_sampled[..., material_net.latent_dim:2*material_net.latent_dim]
            latent_b = latent_sampled[..., 2*material_net.latent_dim:3*material_net.latent_dim]
        
        with timer("BRDF 5a. Forward pass (R channel)"):
            brdf_r = material_net.forward(position, wi_local, wo_local, local_normal, latent_r, batch_mask, 'r')
        
        with timer("BRDF 5b. Forward pass (G channel)"):
            brdf_g = material_net.forward(position, wi_local, wo_local, local_normal, latent_g, batch_mask, 'g')
        
        with timer("BRDF 5c. Forward pass (B channel)"):
            brdf_b = material_net.forward(position, wi_local, wo_local, local_normal, latent_b, batch_mask, 'b')
        
        with timer("BRDF 6. Concatenate channels"):
            brdf = torch.cat([brdf_r, brdf_g, brdf_b], dim=-1)
    else:
        with timer("BRDF 5. Forward pass (single)"):
            brdf = material_net.forward(position, wi_local, wo_local, local_normal, latent_sampled, batch_mask)
            if not material_net.colorful_texture:
                brdf = brdf.repeat(1, 3)
    
    with timer("BRDF 7. Compute PDF"):
        pdf = NoL / 3.14159265359
    
    print("=== End BRDF Profiling ===\n")
    return brdf, pdf


# Usage example:
if __name__ == "__main__":
    print("To use this profiler, import and replace the path tracing function in your training code:")
    print("  from profile_path_tracing import profile_path_tracing_detailed")
    print("  # Then use profile_path_tracing_detailed instead of batched_path_tracing_tbn_real_area_emitter")

