import torch
import torch.nn.functional as NF
import pytorch_lightning as pl
import pl_bolts
from renderer import ForwardRenderer
import torchvision
from torchviz import make_dot
from viztracer import VizTracer
from tqdm import tqdm
import math
from model.emitter import DynamicPointEmitter
import os

class BRDFTrainer(pl.LightningModule):
    def __init__(self, cfg, material, gt_material, roughness, metallic):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.material = material
        self.gt_material = gt_material
        self.gt_folder = cfg.gt_folder
        
        self.latent_dim = cfg.material.latent_dim
        # Create a mapping from roughness-metallic pairs to train latent indices

        
        self.latent_reg_weight = cfg.model.latent_reg_weight if hasattr(cfg.model, 'latent_reg_weight') else 1e-4
        self.inference_lr = cfg.model.optimizer.inference_lr
        self.inference_steps = cfg.model.optimizer.inference_steps
        
        self.renderer = ForwardRenderer(cfg, self.material)
        self.gt_renderer = ForwardRenderer(cfg, self.gt_material)
        self.img_hw = cfg.renderer.resolution

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
    # def training_step(self, batch, batch_idx):
    #     # Handle batch of roughness and metallic values
    #     batch_indices = []
    #     gt_params = batch['gt_params']
        
    #     # randomly initialize emitter
    #     emitter = DynamicPointEmitter(
    #         dist=self.cfg.renderer.emitter.dist,
    #         num_lights=self.cfg.renderer.emitter.num_lights,
    #         fix_seed = False
    #     )
        
    #     # Render step
    #     rays, gt_params = batch['rays'], batch['gt_params']
    #     rgbs = self.renderer.render(emitter, rays, self.cfg.renderer.spp.train, None, None)

    #     if self.gt_folder is None:
    #         with torch.no_grad():
    #             rgbs_gt = self.gt_renderer.render(emitter, rays, self.cfg.renderer.spp.train, gt_params, None)
    #     else:
    #         rgbs_gt = batch['rgbs']
        
    #     # Reconstruction loss
    #     recon_loss = NF.l1_loss(rgbs, rgbs_gt)
    #     latent_reg = 0
    #     loss = recon_loss + self.hparams.model.loss.latent_reg_loss.weight * latent_reg
        
    #     psnr_loss = NF.l1_loss(self.gamma(rgbs), self.gamma(rgbs_gt))
    #     psnr = -10.0 * torch.log10(psnr_loss.clamp_min(1e-5))
        
    #     self.log('train/recon_loss', recon_loss)
    #     self.log('train/latent_reg', latent_reg)
    #     self.log('train/total_loss', loss)
    #     self.log('train/psnr', psnr)
        
    #     return loss
    def training_step(self, batch, batch_idx):
        """
        with importance sampling
        """
        # ------------------------------------------------------------------
        # 1. Un-pack inputs
        # ------------------------------------------------------------------
        rays, gt_params = batch['rays'], batch['gt_params']
        # gt_params = self.load_pbr_texture(gt_params).unsqueeze(0)
        # rays: [B, N, 3]
        # gt_params: [B, N, 16]
        emitter = DynamicPointEmitter(
            dist=self.cfg.renderer.emitter.dist,
            num_lights=self.cfg.renderer.emitter.num_lights,
            fix_seed=False
        )

        # forward renders
        rgbs, vis, ray_params = self.renderer.render(emitter, rays, self.cfg.renderer.spp.train,
                                        None, None)                       # f(r)
        if self.gt_folder is None:                                        # f̂ target
            with torch.no_grad():
                rgbs_gt, *_ = self.gt_renderer.render(
                    emitter, rays, self.cfg.renderer.spp.train, gt_params, None)
        else:
            rgbs_gt = batch['rgbs']
        if self.cfg.data.uniform_sampling:
            x, wi, wo = ray_params.split([3, 3, 3], dim=-1)
            assert x.shape == wi.shape == wo.shape
            L = self.cfg.renderer.emitter.num_lights
            x, wo = x.view(-1, L, 3)[:, 0], wo.view(-1, L, 3)[:, 0]  # → (N, 3)
            wi = wi.view(-1, L, 3)                          # (N,M,3)

            # ------------ scene constants ----------------------------
            r        = 0.20
            R_cam    = 2.00
            R_lgt    = 4.00
            theta_fov= 0.5 * 0.5                              # radians
            Omega    = 2.0 * math.pi * (1.0-math.cos(theta_fov))
            eps      = 1e-6

            # ------------ unit normal --------------------------------
            nx = x / r                                         # (N,3)

            # ------------ cosines wrt camera (shared for all m) -------
            cos_o_pos = torch.clamp((nx * wo).sum(-1), min=eps)        # (N,)

            # ------------ helper: endpoint on given radius ------------
            def endpoint_along(dir_vec, R_target):
                # dir_vec: (N,M,3) or (N,1,3) broadcasting ok
                dot = (x.unsqueeze(1) * dir_vec).sum(-1, keepdim=True)  # (N,M,1)
                s   = -dot + torch.sqrt(dot**2 + R_target*R_target - r*r)
                return x.unsqueeze(1) + s * dir_vec                     # (N,M,3)

            # ------------ camera & light centres ----------------------
            c = endpoint_along(-wo.unsqueeze(1), R_cam)[:,0]  # (N,3) (same for all m)
            l = endpoint_along( wi, R_lgt)                    # (N,M,3)

            # ------------ geometric terms -----------------------------
            dist2_cam = ((x - c) ** 2).sum(-1, keepdim=True)           # (N,1)
            dist2_lgt = ((x.unsqueeze(1) - l) ** 2).sum(-1)            # (N,M)

            cos_i_pos = torch.clamp((nx.unsqueeze(1) * wi).sum(-1), min=eps)  # (N,M)

            # ------------ per-sample weights, then average ------------
            w_each = (dist2_cam / cos_o_pos.unsqueeze(1)) * (cos_i_pos / dist2_lgt) * Omega  # (N,M)
            w_px   = w_each.mean(dim=1)                               # (N,)   equation (★)
            w_px   = w_px / w_px.mean()                               # optional normalise

            # ------------ pixel-wise L1 loss --------------------------
            per_pix_l1 = torch.abs(rgbs - rgbs_gt).mean(-1)             # (N,)

            recon_loss = (w_px * per_pix_l1[vis]).mean()

        elif self.cfg.data.importance_sampling:
            luminance = (0.2126 * rgbs_gt[..., 0] +
                        0.7152 * rgbs_gt[..., 1] +
                        0.0722 * rgbs_gt[..., 2]).clamp(min=1e-6)    # (N,)

            pdf = luminance.detach() / luminance.sum()               # f̂(r),  stops grad

            # When every pixel is black the above becomes NaN; fall back to uniform.
            if not torch.isfinite(pdf).all():
                pdf = torch.full_like(pdf, 1.0 / pdf.numel())
            N_tot        = rays.shape[1]
            n_samples    = getattr(self.cfg.data, "importance_sampling_num", N_tot)   # use all by default
            sample_idx   = torch.multinomial(pdf, n_samples, replacement=True)   # (S,)

            # gather the sampled quantities
            rgbs_s       = rgbs[sample_idx]
            rgbs_gt_s    = rgbs_gt[sample_idx]
            pdf_s        = pdf[sample_idx]                                # f̂(r_i)
            if self.hparams.model.loss.recon_loss.name == "l1":
                per_pix_l1 = torch.abs(rgbs_s - rgbs_gt_s).mean(dim=-1) 
                weighted   = per_pix_l1 / pdf_s                               # multiply by Q=1
            elif self.hparams.model.loss.recon_loss.name == "l2":
                per_pix_l2 = torch.pow(rgbs_s - rgbs_gt_s, 2).mean(dim=-1)       # (S,)
                weighted   = per_pix_l2 / pdf_s                               # multiply by Q=1
            recon_loss = weighted.mean()

        else:
            if self.cfg.loss.recon_loss.name == "l1":
                recon_loss = NF.l1_loss(rgbs, rgbs_gt)
            elif self.cfg.loss.recon_loss.name == "l2":
                recon_loss = NF.mse_loss(rgbs, rgbs_gt)

        latent_reg = 0.0
        loss       = recon_loss + self.hparams.model.loss.latent_reg_loss.weight * latent_reg

        psnr_loss = NF.l1_loss(self.gamma(rgbs), self.gamma(rgbs_gt))
        psnr      = -10.0 * torch.log10(psnr_loss.clamp_min(1e-5))

        # ------------------------------------------------------------------
        # 6. Logging
        # ------------------------------------------------------------------
        self.log_dict({
            'train/recon_loss': recon_loss,
            'train/latent_reg': latent_reg,
            'train/total_loss': loss,
            'train/psnr':       psnr
        }, prog_bar=True, batch_size=rays.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        """ batch pbr texture: [B, H, W, 16] """
        gt_params = batch['gt_params']
        # Get latents for the batch using the indices

        emitter = DynamicPointEmitter(
            dist=self.cfg.renderer.emitter.dist,
            num_lights=self.cfg.renderer.emitter.num_lights,
            fix_seed = True
        )
        
        # Render step logic
        rays, gt_params = batch['rays'], batch['gt_params']
        rgbs, *_ = self.renderer.render(emitter, rays, self.cfg.renderer.spp.train, None, None)

        if self.gt_folder is None:
            with torch.no_grad():
                rgbs_gt, *_ = self.gt_renderer.render(emitter, rays, self.cfg.renderer.spp.train, gt_params, None)
        else:
            rgbs_gt = batch['rgbs']

        psnr_loss = NF.l1_loss(self.gamma(rgbs), self.gamma(rgbs_gt))
        psnr = -10.0 * torch.log10(psnr_loss.clamp_min(1e-5))
        
        if self.hparams.model.loss.recon_loss.name == "l1":
            recon_loss = NF.l1_loss(rgbs, rgbs_gt)
        elif self.hparams.model.loss.recon_loss.name == "l2":
            recon_loss = NF.mse_loss(rgbs, rgbs_gt)
        latent_reg = 0
        loss = recon_loss + self.hparams.model.loss.latent_reg_loss.weight * latent_reg
        
        # Handle batch of images
        batch_size = 1
        batched_rgbs = rgbs.reshape(batch_size, *self.img_hw, -1)
        batched_rgbs_gt = rgbs_gt.reshape(batch_size, *self.img_hw, -1)
        for b in range(batch_size):
            # Reshape individual sample in batch
            sample_rgbs = batched_rgbs[b]
            sample_rgbs_gt = batched_rgbs_gt[b]
            
            # Create output directory for each sample
            output_dir = os.path.join(
                self.cfg.exp_output_root_path,
                f'fabric_pattern_07_4k'
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Save ground truth and result images
            torchvision.utils.save_image(
                self.gamma(sample_rgbs_gt.permute(2, 0, 1)),
                os.path.join(output_dir, f'gt_view_{batch_idx}_{b}.png')
            )
            # Save non-gamma-corrected result image
            torchvision.utils.save_image(
                sample_rgbs_gt.permute(2, 0, 1),
                os.path.join(output_dir, f'gt_view_linear_{batch_idx}_{b}.png')
            )
            torchvision.utils.save_image(
                self.gamma(sample_rgbs.permute(2, 0, 1)),
                os.path.join(output_dir, f'result_view_{batch_idx}_{b}.png')
            )
        
        self.log('val/loss', loss)
        self.log('val/psnr', psnr)        
        return
