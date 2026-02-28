import torch
import torch.nn.functional as NF
import pytorch_lightning as pl
import pl_bolts
import cv2
import numpy as np
import os
import math
from utils.dataset.bonn import DTYPE_POLY, DTYPE_PAN, DTYPE_LLS


# ---------------------------------------------------------------------------
# LLS Monte-Carlo utility
# ---------------------------------------------------------------------------

def sample_quad_uniform(corners, spp):
    """Sample points uniformly on a quadrilateral via bilinear interpolation.

    Args:
        corners: (N, 4, 3)  four corner positions
        spp:     int         number of samples per pixel

    Returns:
        sample_pos: (N, spp, 3)
    """
    N = corners.shape[0]
    device = corners.device
    u = torch.rand(N, spp, 1, device=device)
    v = torch.rand(N, spp, 1, device=device)
    c0 = corners[:, 0:1, :]
    c1 = corners[:, 1:2, :]
    c2 = corners[:, 2:3, :]
    c3 = corners[:, 3:4, :]
    return (1 - u) * (1 - v) * c0 + u * (1 - v) * c1 + u * v * c2 + (1 - u) * v * c3


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Stage2Trainer_Bonn(pl.LightningModule):
    """Stage-2 trainer for Bonn SVBRDF dataset (single-material overfitting).

    Forward model (after white-frame calibration):
      - Point-lit (poly / pan):  predicted = BRDF(wi, wo)
      - LLS:  predicted = pi * sum(BRDF(wi_k, wo) * w_k) / sum(w_k)
              where w_k = cos_theta_i_k / dist_k^2

    Compared to Stage1Trainer_Bonn:
      - Uses dense Adam optimizer (same as Stage2Trainer for real data)
      - Supports freezing the decoder to optimise latent bank only
      - Uses automatic optimisation (no manual backward)
    """

    def __init__(self, cfg, material, gt_material=None, roughness=None, metallic=None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.material = material
        self.freeze_decoder = cfg.model.freeze_decoder

        # Loss weights
        self.pan_loss_weight = getattr(cfg.model.loss, 'pan_weight', 0.5)
        self.lls_loss_weight = getattr(cfg.model.loss, 'lls_weight', 0.5)
        self.lls_spp = getattr(cfg.model, 'lls_spp', 16)
        self.latent_reg_weight = getattr(cfg.model, 'latent_reg_weight', 1e-4)
        self.smooth_reg_weight = getattr(cfg.model, 'smooth_reg_weight', 1e-3)

        # Approximate RGB→gray weights for panchromatic supervision
        self.register_buffer('pan_weights', torch.tensor([0.34, 0.36, 0.28]))

    # ------------------------------------------------------------------
    # Optimiser  (dense Adam, same pattern as Stage2Trainer for real data)
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

        else:
            raise ValueError(f"Optimizer type '{self.hparams.model.optimizer.name}' not supported")

    # ------------------------------------------------------------------
    # BRDF helpers
    # ------------------------------------------------------------------
    def _eval_brdf(self, xyz, wi, wo, point_ids, material_ids):
        """Thin wrapper around material.eval_brdf.

        Returns (brdf [B,3], predicted_normal [B,3], smooth_loss scalar).
        """
        dummy_normal = torch.zeros_like(wi)
        dummy_normal[..., 2] = 1.0
        brdf, pred_normal, _pdf, smooth_loss = self.material.eval_brdf(
            xyz, wi, wo, dummy_normal,
            point_ids=point_ids, material_ids=material_ids)
        return brdf, pred_normal, smooth_loss

    def _lls_monte_carlo(self, xyz, wo, lls_corners, point_ids, material_ids, spp):
        """Monte-Carlo integration over LLS quad (white-frame calibrated).

        predicted = pi * sum_k(BRDF(wi_k, wo) * w_k) / sum_k(w_k)
        w_k = max(0, cos_theta_i_k) / dist_k^2

        Args:
            xyz:          (N, 3)
            wo:           (N, 3)
            lls_corners:  (N, 4, 3)
            point_ids:    (N,)
            material_ids: (N,)
            spp:          int

        Returns:
            pred: (N, 3)  predicted calibrated measurement (RGB)
        """
        N = xyz.shape[0]
        device = xyz.device

        # Predicted normal for geometric weights
        global_pids = self.material.get_global_point_id(material_ids, point_ids)
        latent = self.material.point_latent_bank(global_pids)
        pred_normal, _ = self.material.extract_frame_from_latent(latent)  # (N, 3)

        # Sample K points on the quad
        sample_pos = sample_quad_uniform(lls_corners, spp)           # (N, spp, 3)

        xyz_exp = xyz.unsqueeze(1).expand(-1, spp, -1)               # (N, spp, 3)
        diff = sample_pos - xyz_exp                                  # (N, spp, 3)
        dist_sq = (diff * diff).sum(-1, keepdim=True).clamp(min=1e-8)
        wi_k = diff / dist_sq.sqrt()                                 # (N, spp, 3)

        normal_exp = pred_normal.unsqueeze(1).expand(-1, spp, -1)
        cos_theta_i = (wi_k * normal_exp).sum(-1, keepdim=True).clamp(min=0)
        w_k = cos_theta_i / dist_sq                                  # (N, spp, 1)

        # Flatten for batch BRDF evaluation
        wi_flat  = wi_k.reshape(N * spp, 3)
        wo_flat  = wo.unsqueeze(1).expand(-1, spp, -1).reshape(N * spp, 3)
        xyz_flat = xyz_exp.reshape(N * spp, 3)
        pid_flat = point_ids.unsqueeze(1).expand(-1, spp).reshape(N * spp)
        mid_flat = material_ids.unsqueeze(1).expand(-1, spp).reshape(N * spp)

        brdf_flat, _, _ = self._eval_brdf(xyz_flat, wi_flat, wo_flat, pid_flat, mid_flat)
        brdf_k = brdf_flat.reshape(N, spp, 3)                       # (N, spp, 3)

        numerator   = (brdf_k * w_k).sum(dim=1)                     # (N, 3)
        denominator = w_k.sum(dim=1).clamp(min=1e-8)                # (N, 1)
        return math.pi * numerator / denominator

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _compute_loss(self, pred, gt, confidence=None):
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

        if confidence is not None:
            per_pix = per_pix * confidence
        return per_pix.mean()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        xyz          = batch['xyz'].squeeze(0)
        wi           = batch['wi'].squeeze(0)
        wo           = batch['wo'].squeeze(0)
        rgbs_gt      = batch['rgbs'].squeeze(0)
        point_ids    = batch['point_ids'].squeeze(0)
        material_ids = batch['material_ids'].squeeze(0)
        data_type    = batch['data_type'].squeeze(0)
        lls_corners  = batch['lls_corners'].squeeze(0)
        confidence   = batch['confidence'].squeeze(0)

        poly_mask = data_type == DTYPE_POLY
        pan_mask  = data_type == DTYPE_PAN
        lls_mask  = data_type == DTYPE_LLS

        total_loss   = torch.tensor(0.0, device=xyz.device)
        smooth_total = torch.tensor(0.0, device=xyz.device)
        poly_pred    = None

        # --- polychromatic (RGB) loss ---
        if poly_mask.any():
            brdf, _, sm = self._eval_brdf(
                xyz[poly_mask], wi[poly_mask], wo[poly_mask],
                point_ids[poly_mask], material_ids[poly_mask])
            total_loss = total_loss + self._compute_loss(brdf, rgbs_gt[poly_mask],
                                                         confidence[poly_mask])
            smooth_total = smooth_total + sm
            poly_pred = brdf

        # --- panchromatic (grayscale) loss ---
        if pan_mask.any():
            brdf_pan, _, sm = self._eval_brdf(
                xyz[pan_mask], wi[pan_mask], wo[pan_mask],
                point_ids[pan_mask], material_ids[pan_mask])
            pred_gray = (brdf_pan * self.pan_weights).sum(-1, keepdim=True)
            gt_gray   = rgbs_gt[pan_mask][:, :1]
            total_loss = total_loss + self.pan_loss_weight * self._compute_loss(
                pred_gray, gt_gray, confidence[pan_mask])
            smooth_total = smooth_total + sm

        # --- LLS (Monte-Carlo) loss ---
        if lls_mask.any():
            lls_pred = self._lls_monte_carlo(
                xyz[lls_mask], wo[lls_mask], lls_corners[lls_mask],
                point_ids[lls_mask], material_ids[lls_mask], self.lls_spp)
            pred_gray = (lls_pred * self.pan_weights).sum(-1, keepdim=True)
            gt_gray   = rgbs_gt[lls_mask][:, :1]
            total_loss = total_loss + self.lls_loss_weight * self._compute_loss(
                pred_gray, gt_gray, confidence[lls_mask])

        # Smoothness regularisation (from poly branch only to avoid double-counting)
        total_loss = total_loss + self.smooth_reg_weight * smooth_total

        # PSNR — computed on poly RGB only
        psnr = torch.tensor(0.0, device=xyz.device)
        if poly_pred is not None and poly_pred.numel() > 0:
            gt_poly = rgbs_gt[poly_mask]
            mse = NF.mse_loss(poly_pred, gt_poly)
            max_val = gt_poly.max().clamp_min(1e-8)
            psnr = 10.0 * torch.log10(max_val ** 2 / mse.clamp_min(1e-8))

        self.log_dict({
            'train/total_loss': total_loss,
            'train/psnr':       psnr,
        }, prog_bar=True, batch_size=xyz.shape[0])

        return total_loss

    # ------------------------------------------------------------------
    # Validation  (full-image from BonnValDataset)
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        xyz          = batch['xyz'].squeeze(0)
        wi           = batch['wi'].squeeze(0)
        wo           = batch['wo'].squeeze(0)
        rgbs_gt      = batch['rgbs'].squeeze(0)
        point_ids    = batch['point_ids'].squeeze(0)
        material_ids = batch['material_ids'].squeeze(0)
        img_hw       = batch['img_hw'].squeeze(0)              # (2,)

        brdf, _, _ = self._eval_brdf(xyz, wi, wo, point_ids, material_ids)

        loss = self._compute_loss(brdf, rgbs_gt)
        mse  = NF.mse_loss(brdf, rgbs_gt)
        max_val = rgbs_gt.max().clamp_min(1e-8)
        psnr = 10.0 * torch.log10(max_val ** 2 / mse.clamp_min(1e-8))

        log_dict = {'val/loss': loss, 'val/psnr': psnr}
        if hasattr(self.material, 'factor'):
            log_dict['val/factor'] = self.material.factor
        self.log_dict(log_dict, prog_bar=True, batch_size=xyz.shape[0])

        # ---- reconstruct 2-D images and save ----------------------------
        H, W = img_hw[0].item(), img_hw[1].item()

        gt_img   = rgbs_gt.reshape(H, W, 3)
        pred_img = brdf.reshape(H, W, 3)

        output_dir = os.path.join(self.cfg.exp_output_root_path, 'images')
        os.makedirs(output_dir, exist_ok=True)

        psnr_str = f'{psnr.item():.2f}'
        mat_id   = material_ids[0].item()

        # # Save EXR (full HDR precision)
        # gt_exr   = gt_img.cpu().numpy().astype(np.float32)
        # pred_exr = pred_img.cpu().numpy().astype(np.float32)
        # cv2.imwrite(
        #     os.path.join(output_dir, f'gt_mat{mat_id:04d}_view{batch_idx}.exr'),
        #     cv2.cvtColor(gt_exr, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(
        #     os.path.join(output_dir,
        #                  f'pred_mat{mat_id:04d}_view{batch_idx}_psnr{psnr_str}.exr'),
        #     cv2.cvtColor(pred_exr, cv2.COLOR_RGB2BGR))

        # Save 8-bit PNG (tone-mapped + gamma for quick inspection)
        gt_png   = self._tonemap_for_display(gt_img)
        pred_png = self._tonemap_for_display(pred_img)
        cv2.imwrite(
            os.path.join(output_dir, f'gt_mat{mat_id:04d}_view{batch_idx}.png'),
            cv2.cvtColor(gt_png, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            os.path.join(output_dir,
                         f'pred_mat{mat_id:04d}_view{batch_idx}.png'),
            cv2.cvtColor(pred_png, cv2.COLOR_RGB2BGR))

        return loss

    # ------------------------------------------------------------------
    # Tone mapping for display
    # ------------------------------------------------------------------
    @staticmethod
    def _tonemap_for_display(img_hwc):
        """Reinhard tone-map + sRGB gamma → uint8 numpy (H, W, 3)."""
        x = img_hwc.clamp(min=0.0)
        x = x / (1.0 + x)                       # Reinhard
        low  = 12.92 * x
        high = 1.055 * x.pow(1.0 / 2.4) - 0.055
        x = torch.where(x <= 0.0031308, low, high).clamp(0.0, 1.0)
        return (x * 255).byte().cpu().numpy()

    def on_train_batch_start(self, batch, batch_idx):
        step = self.global_step
        self.trainer.train_dataloader.dataset.datasets.set_step(step)
