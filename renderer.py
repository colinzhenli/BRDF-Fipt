import torch
from utils.path_tracing import path_tracing_envmap_emitter, batched_path_tracing_dynamic_emitter, batched_path_tracing_preset_emitter, batched_path_tracing_tbn_preset_emitter, batched_path_tracing_tbn_real_area_emitter
from utils.scene_loader import load_uv_obj_to_mitsuba_scene, create_rectangle_scene
from mitsuba import load_dict
import mitsuba as mi
from model.emitter import EnvMapEmitter, DynamicPointEmitter
class ForwardRenderer:
    def __init__(self, cfg, material):
        self.cfg = cfg
        self.device = 'cuda'
        '''
        self.scene = load_dict({
            "type": "scene",
            "shape_id": {
                "type": "rectangle",
                "to_world": (
                    mi.ScalarTransform4f.rotate([1, 0, 0], -90) @   # xy → xz
                    mi.ScalarTransform4f.scale(0.4)                 # 1 m → 0.4 m
                ),
                "flip_normals": False
            }
        })
        '''
        if cfg.renderer.mesh.path is not None:
            self.scene=load_uv_obj_to_mitsuba_scene(cfg.renderer.mesh.path)
        else:
            self.scene = create_rectangle_scene(
                center=cfg.renderer.mesh.rectangle.center,
                width=cfg.renderer.mesh.rectangle.width,
                length=cfg.renderer.mesh.rectangle.length
            )
        self.material = material.to(self.device)

        print("type",cfg.renderer.emitter.type)
        if cfg.renderer.emitter.type == 'envmap':
            self.ray_tracer = path_tracing_envmap_emitter
        elif cfg.renderer.emitter.type == 'presetpoint':
            self.ray_tracer = batched_path_tracing_tbn_preset_emitter
        elif cfg.renderer.emitter.type == 'realarea':
            self.ray_tracer = batched_path_tracing_tbn_real_area_emitter
        # elif cfg.renderer.emitter.type == 'tbnpresetpoint':
        #     self.ray_tracer = batched_path_tracing_tbn_preset_emitter
        emitter_cfg = cfg.renderer.emitter
        if cfg.renderer.emitter.type == 'envmap':
            self.emitter = EnvMapEmitter(emitter_cfg.envmap_path).to(self.device)

        self.SPP_chunk = cfg.renderer.SPP_chunk
    
    def render(self, emitter, rays, light_idx, spp, gt_params=None, latent=None):
        rays_x, rays_d, dxdu, dydv = rays[..., :3], rays[..., 3:6], rays[..., 6:9], rays[..., 9:12]
        L = torch.zeros_like(rays_x)
        ray_params = torch.zeros_like(rays)
        if spp < self.SPP_chunk:
            self.SPP_chunk = spp
        if emitter is None:
            for _ in range(spp // self.SPP_chunk):
                L0, vis, ray_params = self.ray_tracer(
                    self.scene, self.emitter, self.material,
                    rays_x, rays_d, dxdu, dydv, 
                    light_idx, self.SPP_chunk, brdf_sampling=self.cfg.renderer.brdf_sampling, emitter_sampling=self.cfg.renderer.emitter_sampling, gt_params=gt_params, latent=latent
                )
                L += L0
        else:
            for _ in range(spp // self.SPP_chunk):
                L0, vis, ray_params = self.ray_tracer(
                    self.scene, emitter, self.material,
                    rays_x, rays_d, dxdu, dydv, 
                    light_idx, self.SPP_chunk, brdf_sampling=self.cfg.renderer.brdf_sampling, emitter_sampling=self.cfg.renderer.emitter_sampling, gt_params=gt_params, latent=latent
                )
                L += L0
        rgbs = L / (spp // self.SPP_chunk)
        rgbs = rgbs.squeeze(0) # squeeze the batch dimension
        return rgbs, vis, ray_params
