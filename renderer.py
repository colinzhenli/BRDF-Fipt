import torch
from utils.path_tracing import path_tracing_fix_emitter, path_tracing_envmap_emitter, path_tracing_multipoint_emitter, path_tracing_dynamic_emitter
from mitsuba import load_dict
from model.emitter import EnvMapEmitter, PointEmitter, MultiPointsEmitter
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
        elif cfg.renderer.emitter.type == 'multipoint':
            self.ray_tracer = path_tracing_multipoint_emitter
        elif cfg.renderer.emitter.type == 'dynamicpoint':
            self.ray_tracer = path_tracing_dynamic_emitter
        else:
            self.ray_tracer = path_tracing_envmap_emitter
        emitter_cfg = cfg.renderer.emitter
        self.emitter = None
        if cfg.renderer.emitter.type == 'point':
            self.emitter = PointEmitter(
                torch.tensor(emitter_cfg.position),
                torch.tensor(emitter_cfg.intensity),
                emitter_cfg.radius
            ).to(self.device)

        elif cfg.renderer.emitter.type == 'multipoint':
            self.emitter = MultiPointsEmitter(
                dist=emitter_cfg.dist,
                n_theta=emitter_cfg.n_theta,
                n_phi=emitter_cfg.n_phi,
                num_lights=emitter_cfg.num_lights
            ).to(self.device)
        elif cfg.renderer.emitter.type == 'envmap':
            self.emitter = EnvMapEmitter(emitter_cfg.envmap_path).to(self.device)

        self.SPP_chunk = cfg.renderer.SPP_chunk
    
    def render(self, emitter, rays_x, rays_d, dxdu, dydv, img_hw, spp, light_indices=None):
        L = torch.zeros_like(rays_x)
        if spp < self.SPP_chunk:
            self.SPP_chunk = spp
        if emitter is None:
            for _ in range(spp // self.SPP_chunk):
                L += self.ray_tracer(
                    self.scene, self.emitter, self.material,
                    rays_x, rays_d, dxdu, dydv, 
                    self.SPP_chunk, indir_depth=0, brdf_sampling=self.cfg.renderer.brdf_sampling, emitter_sampling=self.cfg.renderer.emitter_sampling, light_indices=light_indices
                )
        else:
            L = self.ray_tracer(
                self.scene, emitter, self.material,
                rays_x, rays_d, dxdu, dydv, 
                self.SPP_chunk, indir_depth=0, brdf_sampling=self.cfg.renderer.brdf_sampling, emitter_sampling=self.cfg.renderer.emitter_sampling, light_indices=light_indices
            )
        rgbs = L / (spp // self.SPP_chunk)
        return rgbs
