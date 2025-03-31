import torch
from utils.path_tracing import path_tracing_fix_emitter, path_tracing_envmap_emitter
from mitsuba import load_dict
from model.emitter import EnvMapEmitter, PointEmitter
class ForwardRenderer:
    def __init__(self, cfg, material):
        self.cfg = cfg
        self.device = 'cuda'
        self.scene = load_dict({
            "type": "scene",
            "shape_id": {
                "type": "sphere",
                "center": [0, 0, 0], 
                "radius": 0.2,
                "flip_normals": False
            }
        })
        self.material = material.to(self.device)

        if cfg.renderer.emitter.type == 'point':
            self.ray_tracer = path_tracing_fix_emitter
        else:
            self.ray_tracer = path_tracing_envmap_emitter
        emitter_cfg = cfg.renderer.emitter
        if cfg.renderer.emitter.type == 'point':
            self.emitter = PointEmitter(
                torch.tensor(emitter_cfg.position),
                torch.tensor(emitter_cfg.intensity),
                emitter_cfg.radius
            ).to(self.device)
        else:
            self.emitter = EnvMapEmitter(emitter_cfg.envmap_path).to(self.device)

        self.SPP_chunk = cfg.renderer.SPP_chunk
    
    def render(self, rays_x, rays_d, dxdu, dydv, img_hw, spp):
        L = torch.zeros_like(rays_x)
        for _ in range(spp // self.SPP_chunk):
            L += self.ray_tracer(
                self.scene, self.emitter, self.material,
                rays_x, rays_d, dxdu, dydv, 
                self.SPP_chunk, indir_depth=0, brdf_sampling=self.cfg.renderer.brdf_sampling, emitter_sampling=self.cfg.renderer.emitter_sampling
            )
        rgbs = L.reshape(*img_hw, -1) / (spp // self.SPP_chunk)
        return rgbs
