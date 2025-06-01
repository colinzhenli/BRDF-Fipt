import torch
import torch.nn.functional as NF
import pytorch_lightning as pl
import pl_bolts
from renderer import ForwardRenderer
import torchvision
from torchviz import make_dot
from viztracer import VizTracer
from tqdm import tqdm
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
        rays,  gt_params = batch['rays'], batch['gt_params']
        # rays: [B, N, 3]
        # gt_params: [B, N, 16]
        emitter = DynamicPointEmitter(
            dist=self.cfg.renderer.emitter.dist,
            num_lights=self.cfg.renderer.emitter.num_lights,
            fix_seed=False
        )

        # forward renders
        rgbs = self.renderer.render(emitter, rays, self.cfg.renderer.spp.train,
                                        None, None)                       # f(r)
        if self.gt_folder is None:                                        # f̂ target
            with torch.no_grad():
                rgbs_gt = self.gt_renderer.render(
                    emitter, rays, self.cfg.renderer.spp.train, gt_params, None)
        else:
            rgbs_gt = batch['rgbs']

        if self.cfg.data.importance_sampling:
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
        rgbs = self.renderer.render(emitter, rays, self.cfg.renderer.spp.train, None, None)

        if self.gt_folder is None:
            with torch.no_grad():
                rgbs_gt = self.gt_renderer.render(emitter, rays, self.cfg.renderer.spp.train, gt_params, None)
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
