"""Stage-2 trainer for UBO2014 BTF dataset (single-material overfitting).

Forward model (direct BRDF supervision):
  predicted = BRDF(wi, wo)   where wi, wo are local-frame unit vectors

The BTF data provides direct per-texel BRDF values for known
(light, view) directions on a flat sample, so there is no inverse
rendering pipeline, no emitter calibration, and no xyz positions.

Compared to Stage2Trainer_Bonn:
  - No material_ids (single material only)
  - No xyz positions / world-to-local transform (directions already local)
  - No LLS / panchromatic branches (only polychromatic RGB)
  - LDR data: simple gamma for PNG display, no HDR tone mapping
"""

import torch
import torch.nn.functional as NF
import pytorch_lightning as pl
import pl_bolts
import cv2
import numpy as np
import os
import math


class Stage2Trainer_UBO(pl.LightningModule):
    """Stage-2 trainer for UBO2014 BTF single-material overfitting."""

    def __init__(self, cfg, material, gt_material=None, roughness=None, metallic=None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.material = material
        self.freeze_decoder = cfg.model.freeze_decoder
        self.more_visualizations = True

        # Loss
        self.latent_reg_weight = getattr(cfg.model, 'latent_reg_weight', 1e-4)
        self.smooth_reg_weight = getattr(cfg.model, 'smooth_reg_weight', 1e-3)

    # ------------------------------------------------------------------
    # Optimizer  (dense Adam, same pattern as other stage-2 trainers)
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        if self.freeze_decoder:
            print("Decoder frozen! Only optimizing latent bank.")
            decoder_params = set(self.material.decoder.parameters()) if hasattr(self.material, 'decoder') else set()
            params_to_optimize = [p for p in self.parameters() if p not in decoder_params]
        else:
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

        elif self.hparams.model.optimizer.name == 'Adam8bit':
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(
                params_to_optimize,
                lr=self.hparams.model.optimizer.lr,
                betas=(0.9, 0.999),
                weight_decay=self.hparams.model.optimizer.weight_decay,
            )
            print(f"Using Adam8bit, lr={self.hparams.model.optimizer.lr}")
            return optimizer

        else:
            raise ValueError(f"Optimizer type '{self.hparams.model.optimizer.name}' not supported")

    # ------------------------------------------------------------------
    # BRDF evaluation
    # ------------------------------------------------------------------
    def _eval_brdf(self, wi, wo, point_ids):
        """Thin wrapper around material.eval_brdf.

        Returns (brdf [B,3], smooth_loss scalar).
        """
        brdf, smooth_loss = self.material.eval_brdf(
            wi, wo, point_ids=point_ids)
        return brdf, smooth_loss

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _compute_loss(self, pred, gt):
        loss_cfg = self.hparams.model.loss.recon_loss
        name = loss_cfg.name

        if name == 'l1':
            per_pix = (pred - gt).abs().mean(dim=-1)
        elif name == 'l2':
            per_pix = (pred - gt).pow(2).mean(dim=-1)
        else:  # logrel
            rho_ref = getattr(loss_cfg.log_space, 'logrel_ref', 0.5)
            eps     = getattr(loss_cfg.log_space, 'logrel_eps', 1e-3)
            ref = torch.as_tensor(rho_ref, dtype=pred.dtype, device=pred.device)
            def lm(x):
                return torch.log((x + eps) / (ref + eps) + 1.0)
            per_pix = (lm(pred) - lm(gt)).abs().mean(dim=-1)

        return per_pix.mean()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        wi        = batch['wi'].squeeze(0)
        wo        = batch['wo'].squeeze(0)
        rgbs_gt   = batch['rgbs'].squeeze(0)
        point_ids = batch['point_ids'].squeeze(0)

        brdf, smooth_loss = self._eval_brdf(wi, wo, point_ids)

        recon_loss = self._compute_loss(brdf, rgbs_gt)
        total_loss = recon_loss + self.smooth_reg_weight * smooth_loss

        # PSNR
        mse = NF.mse_loss(brdf, rgbs_gt)
        max_val = rgbs_gt.max().clamp_min(1e-8)
        psnr = 10.0 * torch.log10(max_val ** 2 / mse.clamp_min(1e-10))

        log_dict = {
            'train/total_loss': total_loss,
            'train/recon_loss': recon_loss,
            'train/psnr':       psnr,
        }
        if hasattr(self.material, 'learnable_factor') and self.material.learnable_factor:
            factor_val = self.material.factor.detach()
            log_dict['train/learnable_factor_r'] = factor_val[0]
            log_dict['train/learnable_factor_g'] = factor_val[1]
            log_dict['train/learnable_factor_b'] = factor_val[2]
        self.log_dict(log_dict, prog_bar=True, batch_size=wi.shape[0])

        return total_loss

    # ------------------------------------------------------------------
    # Validation  (full-image from UBOBTFValDataset)
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        wi        = batch['wi'].squeeze(0)
        wo        = batch['wo'].squeeze(0)
        rgbs_gt   = batch['rgbs'].squeeze(0)
        point_ids = batch['point_ids'].squeeze(0)
        img_hw    = batch['img_hw'].squeeze(0)

        brdf, _ = self._eval_brdf(wi, wo, point_ids)

        loss = self._compute_loss(brdf, rgbs_gt)
        mse  = NF.mse_loss(brdf, rgbs_gt)
        max_val = rgbs_gt.max().clamp_min(1e-8)
        psnr = 10.0 * torch.log10(max_val ** 2 / mse.clamp_min(1e-10))

        log_dict = {'val/loss': loss, 'val/psnr': psnr}
        if hasattr(self.material, 'learnable_factor') and self.material.learnable_factor:
            factor_val = self.material.factor.detach()
            log_dict['val/learnable_factor_r'] = factor_val[0]
            log_dict['val/learnable_factor_g'] = factor_val[1]
            log_dict['val/learnable_factor_b'] = factor_val[2]
        self.log_dict(log_dict, prog_bar=True, batch_size=wi.shape[0])

        # ---- reconstruct 2-D images and save ----------------------------
        H, W = img_hw[0].item(), img_hw[1].item()

        gt_img   = rgbs_gt.reshape(H, W, 3)
        pred_img = brdf.reshape(H, W, 3)

        output_dir = os.path.join(self.cfg.exp_output_root_path, 'images')
        os.makedirs(output_dir, exist_ok=True)

        psnr_str = f'{psnr.item():.2f}'

        gt_png   = (gt_img.clamp(0.0, 1.0) * 255).byte().cpu().numpy()
        pred_png = (pred_img.clamp(0.0, 1.0) * 255).byte().cpu().numpy()
        cv2.imwrite(
            os.path.join(output_dir, f'gt_view{batch_idx}.png'),
            cv2.cvtColor(gt_png, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            os.path.join(output_dir,
                         f'pred_view{batch_idx}_psnr{psnr_str}.png'),
            cv2.cvtColor(pred_png, cv2.COLOR_RGB2BGR))

        # ---- save normal / tangent maps and BRDF lobes (first val step) ----
        if batch_idx == 0:
            if self.material.predict_frame:
                with torch.no_grad():
                    latent = self.material.point_latent_bank(point_ids)
                    pred_normal, pred_tangent = self.material.extract_frame_from_latent(latent)

                normal_img  = pred_normal.reshape(H, W, 3)
                tangent_img = pred_tangent.reshape(H, W, 3)
                normal_png  = ((normal_img.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255).byte().cpu().numpy()
                tangent_png = ((tangent_img.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255).byte().cpu().numpy()
                cv2.imwrite(
                    os.path.join(output_dir, 'normal.png'),
                    cv2.cvtColor(normal_png, cv2.COLOR_RGB2BGR))
                cv2.imwrite(
                    os.path.join(output_dir, 'tangent.png'),
                    cv2.cvtColor(tangent_png, cv2.COLOR_RGB2BGR))

            if self.more_visualizations:
                self.visualize_brdf_lobe(output_dir=output_dir, num_latents=10, resolution=64)

        return loss

    # ------------------------------------------------------------------
    # BRDF lobe visualisation
    # ------------------------------------------------------------------
    def visualize_brdf_lobe(self, output_dir, num_latents=10, resolution=64):
        if not hasattr(self.material, 'decoder') or not hasattr(self.material, 'point_latent_bank'):
            return
        if not hasattr(self.material.decoder, 'encode_directions'):
            return  # PBRDecoder uses a different API

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        device = next(self.material.parameters()).device
        latent_dim   = self.material.latent_dim
        total_points = self.material.point_latent_bank.num_embeddings

        torch.manual_seed(42)
        point_indices = torch.randint(0, total_points, (num_latents,), device=device)

        with torch.no_grad():
            all_latents  = self.material.point_latent_bank(point_indices)
        brdf_latents = all_latents[:, :latent_dim]

        local_normal  = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        brdf_lobe_dir = os.path.join(output_dir, 'brdf_lobes')
        os.makedirs(brdf_lobe_dir, exist_ok=True)

        # ---- Fix wi, vary wo ------------------------------------------------
        theta_i_values = [15.0, 30.0, 45.0, 60.0, 75.0]

        for latent_idx in range(num_latents):
            latent = brdf_latents[latent_idx:latent_idx + 1]
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

            for theta_i_deg in theta_i_values:
                theta_i  = np.radians(theta_i_deg)
                wi = torch.tensor([[np.sin(theta_i), 0.0, np.cos(theta_i)]],
                                  device=device, dtype=torch.float32)

                theta_o_range = np.linspace(-np.pi / 2, np.pi / 2, resolution * 2)
                wo_batch = torch.zeros(len(theta_o_range), 3, device=device)
                for j, theta_o in enumerate(theta_o_range):
                    if theta_o >= 0:
                        wo_batch[j, 0] = -np.sin(theta_o)
                        wo_batch[j, 2] =  np.cos(theta_o)
                    else:
                        wo_batch[j, 0] =  np.sin(-theta_o)
                        wo_batch[j, 2] =  np.cos(-theta_o)

                wi_batch     = wi.expand(len(theta_o_range), -1)
                normal_batch = local_normal.expand(len(theta_o_range), -1)
                latent_batch = latent.expand(len(theta_o_range), -1)

                enc_dir = self.material.decoder.encode_directions(wi_batch, wo_batch, normal_batch)
                with torch.no_grad():
                    brdf = self.material.decoder(enc_dir, latent_batch)

                ax.plot(theta_o_range, brdf.mean(dim=-1).cpu().numpy(), label=f'θ_i={theta_i_deg}°')
                ax.axvline(x=np.radians(theta_i_deg),
                           color=ax.lines[-1].get_color(), linestyle='--', alpha=0.5)

            ax.set_theta_zero_location('N')
            ax.set_theta_direction(1)
            ax.set_thetamin(-90)
            ax.set_thetamax(90)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.set_title(f'BRDF Polar Plot (vary wo) - Point {point_indices[latent_idx].item()}\n'
                         f'(Fixed wi, vary wo; 0°=normal, dashed=specular direction)')
            plt.tight_layout()
            plt.savefig(os.path.join(brdf_lobe_dir,
                                     f'polar_brdf_vary_wo_pt_{point_indices[latent_idx].item()}.png'), dpi=150)
            plt.close()

        # ---- Fix wo, vary wi  (BRDF × cos_theta_i) --------------------------
        theta_o_values = [15.0, 30.0, 45.0, 60.0]

        for latent_idx in range(num_latents):
            latent = brdf_latents[latent_idx:latent_idx + 1]
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

            for theta_o_deg in theta_o_values:
                theta_o = np.radians(theta_o_deg)
                wo = torch.tensor([[np.sin(theta_o), 0.0, np.cos(theta_o)]],
                                  device=device, dtype=torch.float32)

                theta_i_range = np.linspace(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, resolution * 2)
                wi_batch = torch.zeros(len(theta_i_range), 3, device=device)
                for j, theta_i in enumerate(theta_i_range):
                    if theta_i >= 0:
                        wi_batch[j, 0] = -np.sin(theta_i)
                        wi_batch[j, 2] =  np.cos(theta_i)
                    else:
                        wi_batch[j, 0] =  np.sin(-theta_i)
                        wi_batch[j, 2] =  np.cos(-theta_i)

                wo_batch     = wo.expand(len(theta_i_range), -1)
                normal_batch = local_normal.expand(len(theta_i_range), -1)
                latent_batch = latent.expand(len(theta_i_range), -1)

                enc_dir = self.material.decoder.encode_directions(wi_batch, wo_batch, normal_batch)
                with torch.no_grad():
                    brdf = self.material.decoder(enc_dir, latent_batch)

                cos_theta_i = wi_batch[:, 2].clamp(min=0).cpu().numpy()
                ax.plot(theta_i_range, brdf.mean(dim=-1).cpu().numpy() * cos_theta_i,
                        label=f'θ_o={theta_o_deg}°')
                ax.axvline(x=np.radians(theta_o_deg),
                           color=ax.lines[-1].get_color(), linestyle='--', alpha=0.5)

            ax.set_theta_zero_location('N')
            ax.set_theta_direction(1)
            ax.set_thetamin(-90)
            ax.set_thetamax(90)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.set_title(f'BRDF × cos(θ_i) Polar Plot - Point {point_indices[latent_idx].item()}\n'
                         f'(Fixed wo, vary wi; 0°=normal, dashed=specular direction)')
            plt.tight_layout()
            plt.savefig(os.path.join(brdf_lobe_dir,
                                     f'polar_brdf_vary_wi_pt_{point_indices[latent_idx].item()}.png'), dpi=150)
            plt.close()

        print(f"[BRDF Lobe Visualization] Saved {num_latents * 2} figures to {brdf_lobe_dir}")

    def on_train_batch_start(self, batch, batch_idx):
        step = self.global_step
        self.trainer.train_dataloader.dataset.datasets.set_step(step)
