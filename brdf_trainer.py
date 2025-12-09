import torch
import torch.nn.functional as NF
import pytorch_lightning as pl
import pl_bolts
from renderer import ForwardRenderer
import torchvision
from torchviz import make_dot
import json
from viztracer import VizTracer
import cv2
import numpy as np
from tqdm import tqdm
import math
from model.emitter import DynamicPointEmitter, PresetPointEmitter, RealAreaEmitter
import os
from utils.pose_refiner import GlobalHandEyeRefiner

class BRDFTrainer(pl.LightningModule):
    def __init__(self, cfg, material, gt_material, roughness, metallic):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.material = material
        self.gt_material = gt_material
        self.gt_folder = cfg.gt_folder
        self.camera_factor = cfg.renderer.camera.linear_factor
        
        #self.latent_dim = cfg.material.latent_dim
        # Create a mapping from roughness-metallic pairs to train latent indices

        self.latent_reg_weight = cfg.model.latent_reg_weight if hasattr(cfg.model, 'latent_reg_weight') else 1e-4
        self.visualize_uv = cfg.data.visualize_uv
        self.inference_lr = cfg.model.optimizer.inference_lr
        self.inference_steps = cfg.model.optimizer.inference_steps
        print("after latent reg weight")
        self.renderer = ForwardRenderer(cfg, self.material)
        self.gt_renderer = ForwardRenderer(cfg, self.gt_material)
        if cfg.data.handeye_refiner:
            self.handeye_refiner = GlobalHandEyeRefiner(sigma_t_mm=0.5, sigma_r_deg=0.1, json_path=cfg.data.metadata_path)
        else:
            self.handeye_refiner = None
        # self.emitter = PresetPointEmitter(
        #     read_from_metadata=True,
        #     metadata_path=os.path.join(self.cfg.metadata_path, 'emitter_metadata.json'),
        #     positions=None, 
        #     intensities=None
        # )
        self.emitter = RealAreaEmitter(
            cfg = cfg.renderer.emitter,
            json_path = cfg.data.metadata_path
        )
        self.img_hw = (cfg.renderer.camera.intrinsics.height, cfg.renderer.camera.intrinsics.width)

    # def gamma(self, x):
    #     mask = x <= 0.0031308
    #     ret = torch.empty_like(x)
    #     ret[mask] = 12.92 * x[mask]
    #     ret[~mask] = 1.055 * x[~mask].pow(1/2.4) - 0.055
    #     return ret
    def gamma(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor of linear-light RGB values to sRGB.
        Matches Blender's built-in OCIO conversion (Standard view-transform).

        Parameters
        ----------
        x : torch.Tensor
            *Linear* RGB values in **[0 … ∞)**. Negative values are clamped to 0.

        Returns
        -------
        torch.Tensor
            sRGB-encoded values in the display range **[0 … 1]**.
        """
        # --- constants taken from the official sRGB transfer function ---
        _A   = 0.055           # 1.055 - 1
        _K0  = 0.0031308       # linear-to-sRGB break-point
        _PHI = 1.0 / 2.4       # 0.416̅  = 1/γ

        x_lin = x.clamp(min=0.0)               # Blender never shows negative light
        low   = 12.92 * x_lin                  # linear segment
        high  = 1.055 * torch.pow(x_lin, _PHI) - _A

        return torch.where(x_lin <= _K0, low, high).clamp(0.0, 1.0)

    def tone_mapping(self, x):
        """
        Apply tone mapping to convert HDR image to LDR.
        
        Args:
            x (torch.Tensor): HDR image with values in range [0, 1]
            
        Returns:
            torch.Tensor: LDR image with tone mapping applied
        """
        # Simple Reinhard tone mapping: x / (1 + x)
        return x / (1 + x)
    
    def configure_optimizers(self):  
        params_to_optimize = self.parameters()
        
        if self.hparams.model.optimizer.name == "SGD":
            optimizer = torch.optim.SGD(
                params_to_optimize,
                lr=self.hparams.model.optimizer.lr,
                momentum=0.9,
                weight_decay=1e-4,
            )
            scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=int(self.hparams.model.optimizer.warmup_steps_ratio * self.hparams.model.trainer.max_steps),
                max_epochs=self.hparams.model.trainer.max_steps,
                eta_min=0,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }

        elif self.hparams.model.optimizer.name == 'Adam':
            optimizer = torch.optim.Adam(
                params_to_optimize,
                lr=self.hparams.model.optimizer.lr,
                betas=(0.9, 0.999),
                weight_decay=self.hparams.model.optimizer.weight_decay,
            )
            return optimizer

        else:
            logging.error('Optimizer type not supported')

    def load_pbr_texture(self, pbr_texture_path):
        pbr_folder = '/localhome/zla247/theia2_data/theia2_data/BRDF-Fipt/fabric_pattern_07_4k/textures'
        import cv2
        from PIL import Image
        import torchvision.transforms as T

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

        # Collect PBR texture data
        try:
            # Read all texture maps based on the image
            col_1 = read_img("fabric_pattern_07_col_1_4k.jpg")  # [3,H,W] - Color map
            ao = read_img("fabric_pattern_07_ao_4k.jpg")        # [3,H,W] - Ambient occlusion
            arm = read_img("fabric_pattern_07_arm_4k.jpg")      # [3,H,W] - (A, Roughness, Metalness)
            rough = read_img("fabric_pattern_07_rough_4k.exr")  # [1,H,W] - Roughness map (EXR)
            nor_dx = read_img("fabric_pattern_07_nor_dx_4k.exr") # [1,H,W] - Normal map X
            nor_gl = read_img("fabric_pattern_07_nor_gl_4k.exr") # [1,H,W] - Normal map GL
            
            
            # Combine all available channels into a texture
            # [Color (3) + AO (3) + ARM (3) + Roughness (1) + Normal DX (1) + Normal GL (1)] = 12 channels
            tex = torch.cat([
                col_1,                # RGB color (3 channels) 0:3
                ao,                   # Ambient occlusion (3 channels) 3:6
                arm,                  # ARM texture (3 channels) 6:9
                rough,                # Roughness map (1 channel) 9:10
                nor_dx,               # Normal map X (3 channel) 10:13
                nor_gl                # Normal map GL (3 channel) 13:16
            ], dim=0)  # [16,H,W]
            
            return tex.permute(1, 2, 0).contiguous().to(self.device)  # [H,W,6] => [U,V,6]
        except Exception as e:
            print(f"Error loading PBR textures: {e}")
            # Return a default texture if loading fails
            H, W = 1024, 1024
            return torch.ones(H, W, 6)
    
    def save_pbr_texture(self, output_dir, batch_idx, b):
        if self.hparams.material.type == "AnisotropicLatentTexturedModel" or self.hparams.material.type == "MipmapAniLatentTexturedModel": # Save normal and tangent map
            # Extract normal and tangent from latent texture (last 6 dimensions)
            # latent_texture = self.material.latent_texture.data[0]  # [latent_dim, H, W]
            
            # # Last 6 dimensions: normal (3) and tangent (3)
            # normal_map = latent_texture[-6:-3, :, :].permute(1, 2, 0)  # [H, W, 3]
            # tangent_map = latent_texture[-3:, :, :].permute(1, 2, 0)   # [H, W, 3]
            
            # torchvision.utils.save_image(
            #     normal_map.permute(2, 0, 1),
            #     os.path.join(output_dir, f'normal_map_{batch_idx}_{b}.png')
            # )
            # torchvision.utils.save_image(
            #     tangent_map.permute(2, 0, 1),
            #     os.path.join(output_dir, f'tangent_map_{batch_idx}_{b}.png')
            # )
            return
        else:
            # Check if material has prefilter option
            if self.hparams.material.prefliter:
                # Save the finest level PBR texture when using prefilter
                pbr_texture_data = self.material.pbr_texture.data[0]
                
                # Save albedo map (channels 0-3)
                pbr_albedo_map = pbr_texture_data[:, :, 0:3]
                torchvision.utils.save_image(
                    pbr_albedo_map.permute(2, 0, 1),
                    os.path.join(output_dir, f'pbr_albedo_map_{batch_idx}_{b}.png')
                )
                
                # Save normal map (channels 10-13)
                pbr_normal_map = pbr_texture_data[:, :, 10:13]
                torchvision.utils.save_image(
                    pbr_normal_map.permute(2, 0, 1),
                    os.path.join(output_dir, f'pbr_normal_map_{batch_idx}_{b}.png')
                )
                
                # save height map
                pbr_height_map = pbr_texture_data[:, :, 13:14]
                torchvision.utils.save_image(
                    pbr_height_map.permute(2, 0, 1),
                    os.path.join(output_dir, f'pbr_height_map_{batch_idx}_{b}.png')
                )
                
                # Save roughness map (channel 7)
                pbr_roughness_map = pbr_texture_data[:, :, 7:8]
                torchvision.utils.save_image(
                    pbr_roughness_map.permute(2, 0, 1),
                    os.path.join(output_dir, f'pbr_roughness_map_{batch_idx}_{b}.png')
                )
                
                # Save metallic map (channel 8)
                pbr_metallic_map = pbr_texture_data[:, :, 8:9]
                torchvision.utils.save_image(
                    pbr_metallic_map.permute(2, 0, 1),
                    os.path.join(output_dir, f'pbr_metallic_map_{batch_idx}_{b}.png')
                )
            else:
                # Save individual mipmap textures when not using prefilter
                if hasattr(self.material, 'mipmap_textures'):
                    for level, mipmap_texture in enumerate(self.material.mipmap_textures):
                        texture_data = mipmap_texture.data[0]
                        
                        # Save albedo mipmap
                        mipmap_albedo = texture_data[:, :, 0:3]
                        torchvision.utils.save_image(
                            mipmap_albedo.permute(2, 0, 1),
                            os.path.join(output_dir, f'mipmap_albedo_level_{level}_{batch_idx}_{b}.png')
                        )
                        
                        # Save normal mipmap
                        mipmap_normal = texture_data[:, :, 10:13]
                        torchvision.utils.save_image(
                            mipmap_normal.permute(2, 0, 1),
                            os.path.join(output_dir, f'mipmap_normal_level_{level}_{batch_idx}_{b}.png')
                        )
                        
                        # Save height mipmap
                        mipmap_height = texture_data[:, :, 13:14]
                        torchvision.utils.save_image(
                            mipmap_height.permute(2, 0, 1),
                            os.path.join(output_dir, f'mipmap_height_map_level_{level}_{batch_idx}_{b}.png')
                        )
                        
                        # Save roughness mipmap
                        mipmap_roughness = texture_data[:, :, 7:8]
                        torchvision.utils.save_image(
                            mipmap_roughness.permute(2, 0, 1),
                            os.path.join(output_dir, f'mipmap_roughness_level_{level}_{batch_idx}_{b}.png')
                        )
                        
                        # Save metallic mipmap
                        mipmap_metallic = texture_data[:, :, 8:9]
                        torchvision.utils.save_image(
                            mipmap_metallic.permute(2, 0, 1),
                            os.path.join(output_dir, f'mipmap_metallic_level_{level}_{batch_idx}_{b}.png')
                        )

    def loss_function(self, rgbs, rgbs_gt, vis, weighted_pdf=None):
        # Calculate per-pixel loss
        if self.hparams.model.loss.recon_loss.name == "l1":
            per_pix = torch.abs(rgbs[vis] - rgbs_gt.squeeze(0)[vis]).mean(dim=-1)
        elif self.hparams.model.loss.recon_loss.name == "l2":  # "l2"
            per_pix = torch.pow(rgbs[vis] - rgbs_gt.squeeze(0)[vis], 2).mean(dim=-1)
        else:  # "logrel" from paper
            rho_ref = getattr(
                self.hparams.model.loss.recon_loss.log_space, "logrel_ref", 0.5
            )  # reference reflectance/radiance
            eps = getattr(
                self.hparams.model.loss.recon_loss.log_space, "logrel_eps", 1e-3
            )  # fixed small constant

            ref = torch.as_tensor(rho_ref, dtype=rgbs.dtype, device=rgbs.device)

            def log_mapping(x):
                return torch.log((x + eps) / (ref + eps) + 1.0)

            per_pix_log = (log_mapping(rgbs) - log_mapping(rgbs_gt.squeeze(0))).abs().mean(dim=-1)  # [N]
            per_pix = per_pix_log

        
        # Calculate importance weights
        if self.hparams.data.importance_sampling and weighted_pdf is not None:
            weights = 1.0 / weighted_pdf.clamp(min=1e-5)
        else:
            weights = 1.0
        
        # Calculate reconstruction loss
        recon_loss = (per_pix * weights).mean()
        
        # Add regularizer
        loss = recon_loss
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """
        with importance sampling
        """
        # ------------------------------------------------------------------
        # 1. Un-pack inputs
        # ------------------------------------------------------------------
        rays, rgbs_gt, emitter_ids, weighted_pdf, camera_ids = batch['rays'], batch['rgbs'], batch['emitter_ids'], batch['pdf'], batch['camera_ids']
        prior = 0.0
        if self.handeye_refiner:
            rays, prior = self.handeye_refiner.apply_handeye_delta_to_rays(rays, camera_ids)
        # forward renders
        rgbs, vis, ray_params, _ = self.renderer.render(self.emitter, rays, emitter_ids, self.cfg.renderer.spp.train,
                                        None, None)                       # f(r)
        rgbs = rgbs * self.camera_factor
        loss = self.loss_function(rgbs, rgbs_gt, vis, weighted_pdf)

        psnr_loss  = torch.nn.functional.mse_loss(rgbs[vis], rgbs_gt.squeeze(0)[vis], reduction='mean')
        max_val = torch.max(torch.stack([rgbs.max(), rgbs_gt.squeeze(0).max()])).clamp_min(1e-8)
        psnr       = 10.0 * torch.log10((max_val ** 2) / psnr_loss.clamp_min(1e-5))

        # ------------------------------------------------------------------
        # 7.  Logging  (now includes diagnostics)
        # ------------------------------------------------------------------
        self.log_dict({
            'train/recon_loss':   loss,
            'train/total_loss':   loss,
            'train/psnr':         psnr,
        }, prog_bar=True, batch_size=rays.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        """ batch pbr texture: [B, H, W, 16] """
        rays, rgbs_gt, emitter_ids = batch['rays'], batch['rgbs'], batch['emitter_ids']
        prior = 0.0
        if self.handeye_refiner:
            rays, prior = self.handeye_refiner.apply_handeye_delta_to_rays(rays, batch['camera_ids'])

        # forward renders
        rgbs, vis, ray_params, uv_offset = self.renderer.render(self.emitter, rays, emitter_ids, self.cfg.renderer.spp.train, None, None)            
        rgbs = rgbs * self.camera_factor
        loss = self.loss_function(rgbs, rgbs_gt, vis)

        psnr_loss = torch.nn.functional.mse_loss(rgbs[vis], rgbs_gt.squeeze(0)[vis], reduction='mean')
        max_val = torch.max(torch.stack([rgbs.max(), rgbs_gt.squeeze(0).max()])).clamp_min(1e-8)
        psnr = 10.0 * torch.log10((max_val ** 2) / psnr_loss.clamp_min(1e-5))
        emitter_radiance = self.emitter.light_radiance.detach().cpu().numpy()
        
        loss = self.loss_function(rgbs, rgbs_gt, vis)
        
        # Handle batch of images
        batch_size = 1
        batched_rgbs = rgbs.reshape(batch_size, *self.img_hw, -1)
        batched_rgbs_gt = rgbs_gt.reshape(batch_size, *self.img_hw, -1)
        
        # Reshape uv_offset for visualization if it exists
        if uv_offset is not None:
            batched_uv_offset = uv_offset.reshape(batch_size, *self.img_hw, 2)
        
        for b in range(batch_size):
            # Reshape individual sample in batch
            sample_rgbs = batched_rgbs[b]
            sample_rgbs_gt = batched_rgbs_gt[b]
            
            # Create output directory for each sample
            output_dir = os.path.join(
                self.cfg.exp_output_root_path,
                f'images'
            )
            os.makedirs(output_dir, exist_ok=True)

            # Convert float32 (0-65535) to uint16 (0-65535)
            sample_rgbs_gt_16bit = np.clip(sample_rgbs_gt.cpu().numpy(), 0, 65535).astype(np.uint16)
            sample_rgbs_16bit = np.clip(sample_rgbs.cpu().numpy(), 0, 65535).astype(np.uint16)

            # Save as 16-bit PNG (OpenCV expects BGR)
            cv2.imwrite(
                os.path.join(output_dir, f'gt_view_{batch_idx}_{b}.png'),
                cv2.cvtColor(sample_rgbs_gt_16bit, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                os.path.join(output_dir, f'result_view_{batch_idx}_{b}.png'),
                cv2.cvtColor(sample_rgbs_16bit, cv2.COLOR_RGB2BGR)
            )
            
            # Visualize UV offsets as grayscale images
            if self.visualize_uv and (uv_offset is not None):
                sample_uv_offset = batched_uv_offset[b].cpu().numpy()  # Shape: [H, W, 2]
                u_offset = sample_uv_offset[:, :, 0]  # U offset
                v_offset = sample_uv_offset[:, :, 1]  # V offset
                
                # Normalize offsets to [0, 255] for visualization
                # Map the range of values to 0-255, with 127 representing zero offset
                u_min, u_max = u_offset.min(), u_offset.max()
                v_min, v_max = v_offset.min(), v_offset.max()
                
                # Normalize to [0, 255] with proper handling of positive and negative values
                u_range = max(abs(u_min), abs(u_max))
                v_range = max(abs(v_min), abs(v_max))
                
                if u_range > 0:
                    u_offset_vis = ((u_offset / u_range) * 127 + 127).astype(np.uint8)
                else:
                    u_offset_vis = np.full_like(u_offset, 127, dtype=np.uint8)
                    
                if v_range > 0:
                    v_offset_vis = ((v_offset / v_range) * 127 + 127).astype(np.uint8)
                else:
                    v_offset_vis = np.full_like(v_offset, 127, dtype=np.uint8)
                
                # Save grayscale offset images
                cv2.imwrite(
                    os.path.join(output_dir, f'u_offset_{batch_idx}_{b}.png'),
                    u_offset_vis
                )
                cv2.imwrite(
                    os.path.join(output_dir, f'v_offset_{batch_idx}_{b}.png'),
                    v_offset_vis
                ) 
                
                # Create colorful merged visualization using Red and Green channels
                # R=U offset, G=V offset, B=neutral (127)
                blue_channel = np.full_like(u_offset_vis, 127, dtype=np.uint8)
                uv_color = np.stack([u_offset_vis, v_offset_vis, blue_channel], axis=2)
                cv2.imwrite(
                    os.path.join(output_dir, f'uv_offset_color_{batch_idx}_{b}.png'),
                    cv2.cvtColor(uv_color, cv2.COLOR_RGB2BGR)
                )          
            # # Save the tone-mapped ground truth and result images
            # torchvision.utils.save_image(
            #     sample_rgbs_gt.permute(2, 0, 1),
            #     os.path.join(output_dir, f'gt_view_{batch_idx}_{b}.png')
            # )
            # torchvision.utils.save_image(
            #     sample_rgbs.permute(2, 0, 1),
            #     os.path.join(output_dir, f'result_view_{batch_idx}_{b}.png')
            # )
            # # Save non-gamma-corrected result image
            # torchvision.utils.save_image(
            #     sample_rgbs_gt.permute(2, 0, 1),
            #     os.path.join(output_dir, f'gt_view_linear_{batch_idx}_{b}.png')
            # )
            # torchvision.utils.save_image(
            #     # self.gamma(sample_rgbs.permute(2, 0, 1)),
            #     sample_rgbs.permute(2, 0, 1),
            #     os.path.join(output_dir, f'result_view_{batch_idx}_{b}.png')
            # )
            
        os.makedirs(os.path.join(self.cfg.exp_output_root_path, f'pbr_map_images'), exist_ok=True)
        self.save_pbr_texture(os.path.join(self.cfg.exp_output_root_path, f'pbr_map_images'), batch_idx, b)
            
        self.log('val/loss', loss)
        self.log('val/emitter_radiance', emitter_radiance.mean())
        self.log('val/psnr', psnr)        
        return

    def on_train_batch_start(self, batch, batch_idx):
        step = self.global_step
        self.trainer.train_dataloader.dataset.datasets.set_step(step)