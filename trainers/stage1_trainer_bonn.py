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

class Stage1Trainer_Bonn(pl.LightningModule):
    """Trainer for Bonn SVBRDF dataset.

    Forward model (after white-frame calibration):
      - Point-lit (poly / pan):  predicted = BRDF(wi, wo)
      - LLS:  predicted = pi * sum(BRDF(wi_k, wo) * w_k) / sum(w_k)
              where w_k = cos_theta_i_k / dist_k^2
    """

    def __init__(self, cfg, material, gt_material=None, roughness=None, metallic=None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.automatic_optimization = False
        self.more_visualizations = True

        self.material = material
        self.freeze_decoder = cfg.model.freeze_decoder

        # Loss weights
        self.pan_loss_weight = getattr(cfg.model.loss, 'pan_weight', 0.5)
        self.lls_loss_weight = getattr(cfg.model.loss, 'lls_weight', 0.5)
        self.lls_spp = getattr(cfg.model, 'lls_spp', 16)
        self.latent_reg_weight = getattr(cfg.model, 'latent_reg_weight', 1e-4)
        self.smooth_reg_weight = getattr(cfg.model, 'smooth_reg_weight', 1e-3)
        self.reset_latent_momentum = getattr(cfg.model.optimizer, 'reset_latent_momentum_on_chunk_switch', False)
        self._opt_name = getattr(cfg.model.optimizer, 'name', 'SparseAdam')

        # Approximate RGB→gray weights for panchromatic supervision
        self.register_buffer('pan_weights', torch.tensor([0.34, 0.36, 0.28]))

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        lr = self.hparams.model.optimizer.lr
        decoder_lr = getattr(self.hparams.model.optimizer, 'decoder_lr', lr)
        wd = self.hparams.model.optimizer.weight_decay
        opt_name = getattr(self.hparams.model.optimizer, 'name', 'SparseAdam')

        embedding_params = list(self.material.point_latent_bank.parameters())
        factor_params = [self.material.factor] if getattr(self.material, 'learnable_factor', False) else []
        
        decoder_params = [p for p in self.parameters()
                          if p not in set(embedding_params) and p not in set(factor_params)]

        if self.freeze_decoder:
            for p in decoder_params:
                p.requires_grad = False
            print("Decoder frozen – optimising latent bank and learnable factor (if present).")

        if opt_name == 'SparseAdam':
            latent_opt = torch.optim.SparseAdam(embedding_params, lr=lr)
            
            dense_params = decoder_params + factor_params if not self.freeze_decoder else factor_params
            if len(dense_params) > 0:
                dense_opt = torch.optim.Adam(
                    dense_params, lr=decoder_lr, betas=(0.9, 0.999), weight_decay=wd,
                )
                print(f"Using SparseAdam (embedding lr={lr}) + Adam (dense lr={decoder_lr})")
                return [latent_opt, dense_opt]
            else:
                print(f"Using SparseAdam (embedding only), lr={lr}")
                return latent_opt

        elif opt_name == 'Adam':
            # Better to explicitly create two groups
            opt_groups = [{'params': embedding_params, 'lr': lr}]
            
            dense_params = decoder_params + factor_params if not self.freeze_decoder else factor_params
            if len(dense_params) > 0:
                opt_groups.append({'params': dense_params, 'lr': decoder_lr})
                
            opt = torch.optim.Adam(opt_groups, betas=(0.9, 0.999), weight_decay=wd)

            print(f"Using Dense Adam (embedding lr={lr}, dense lr={decoder_lr})")
            return opt

        elif opt_name == 'SGD':
            opt_groups = [{'params': embedding_params, 'lr': lr}]
            
            dense_params = decoder_params + factor_params if not self.freeze_decoder else factor_params
            if len(dense_params) > 0:
                opt_groups.append({'params': dense_params, 'lr': decoder_lr})
                
            opt = torch.optim.SGD(opt_groups, momentum=0.0, weight_decay=wd)
            print(f"Using SGD (embedding lr={lr}, dense lr={decoder_lr})")
            return opt
        
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    # ------------------------------------------------------------------
    # BRDF helpers
    # ------------------------------------------------------------------
    def _eval_brdf(self, xyz, wi, wo, point_ids, material_ids, normals=None):
        """Thin wrapper around material.eval_brdf.

        Returns (brdf [B,3], predicted_normal [B,3], smooth_loss scalar).
        """
        if normals is None:
            normals = torch.zeros_like(wi)
            normals[..., 2] = 1.0
        brdf, pred_normal, _pdf, smooth_loss = self.material.eval_brdf(
            xyz, wi, wo, normals,
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
        confidence   = batch['confidence'].squeeze(0)

        gt_normals = batch.get('gt_normals')
        if gt_normals is not None:
            gt_normals = gt_normals.squeeze(0)

        # All rays are polychromatic (RGB) — no pan/lls branching
        brdf, _, smooth_loss = self._eval_brdf(
            xyz, wi, wo, point_ids, material_ids, normals=gt_normals)
        recon_loss = self._compute_loss(brdf, rgbs_gt, confidence)
        total_loss = recon_loss + self.smooth_reg_weight * smooth_loss

        # PSNR over valid (non-occluded) pixels only
        valid = confidence > 0
        if valid.any():
            mse = ((brdf[valid] - rgbs_gt[valid]) ** 2).mean()
            max_val = rgbs_gt[valid].max().clamp_min(1e-8)
        else:
            mse = torch.tensor(1.0, device=brdf.device)
            max_val = torch.tensor(1.0, device=brdf.device)
        psnr = 10.0 * torch.log10(max_val ** 2 / mse.clamp_min(1e-8))

        self.log_dict({
            'train/total_loss': total_loss,
            'train/psnr':       psnr,
            'train/poly_pred_mean': brdf.mean(),
            'train/poly_gt_mean':   rgbs_gt.mean(),
        }, prog_bar=True, batch_size=xyz.shape[0])

        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        for opt in opts:
            opt.zero_grad()
        self.manual_backward(total_loss)

        # ----- Gradient & weight diagnostics (manual, since PL's
        #       track_grad_norm is broken with automatic_optimization=False) ----
        bs = xyz.shape[0]
        total_grad_norm_sq = torch.zeros(1, device=xyz.device)

        # Per-layer decoder gradient norms & weight norms
        for name, param in self.material.decoder.named_parameters():
            w_norm = param.detach().norm(2)
            self.log(f'weight_norm/decoder.{name}', w_norm, batch_size=bs)
            if param.grad is not None:
                g_norm = param.grad.detach().norm(2)
                self.log(f'grad_norm/decoder.{name}', g_norm, batch_size=bs)
                total_grad_norm_sq += g_norm.pow(2)

        # Latent bank: only active (non-zero grad) latents
        lat_w = self.material.point_latent_bank.weight
        lat_w_norm = lat_w.detach().norm(2)
        self.log('weight_norm/latent_bank_total', lat_w_norm, batch_size=bs)
        self.log('weight_norm/latent_bank_mean', lat_w.detach().norm(dim=1).mean(), batch_size=bs)
        self.log('weight_norm/latent_bank_std', lat_w.detach().norm(dim=1).std(), batch_size=bs)
        if lat_w.grad is not None:
            lg = lat_w.grad.detach()
            # For sparse grads, count how many latents were actually touched
            if lg.is_sparse:
                lg_dense = lg.to_dense()
            else:
                lg_dense = lg
            active_mask = lg_dense.norm(dim=1) > 0
            n_active = active_mask.sum()
            self.log('grad_norm/latent_bank_total', lg_dense.norm(2), batch_size=bs)
            self.log('grad_norm/latent_bank_n_active', n_active.float(), batch_size=bs)
            if n_active > 0:
                self.log('grad_norm/latent_bank_active_mean',
                         lg_dense[active_mask].norm(dim=1).mean(), batch_size=bs)
            total_grad_norm_sq += lg_dense.norm(2).pow(2)

        # Learnable BRDF scale factor
        if hasattr(self.material, 'learnable_factor') and self.material.learnable_factor:
            factor_val = self.material.factor.detach()
            self.log('train/learnable_factor_r', factor_val[0], batch_size=bs)
            self.log('train/learnable_factor_g', factor_val[1], batch_size=bs)
            self.log('train/learnable_factor_b', factor_val[2], batch_size=bs)
            if self.material.factor.grad is not None:
                self.log('grad_norm/learnable_factor', self.material.factor.grad.detach().norm(2), batch_size=bs)

        self.log('train/grad_norm_2', total_grad_norm_sq.sqrt(),
                 prog_bar=False, batch_size=bs)

        for opt in opts:
            opt.step()

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
        confidence   = batch['confidence'].squeeze(0)          # (N,)
        img_hw       = batch['img_hw'].squeeze(0)              # (2,)

        gt_normals = batch.get('gt_normals')
        if gt_normals is not None:
            gt_normals = gt_normals.squeeze(0)

        brdf, _, _ = self._eval_brdf(xyz, wi, wo, point_ids, material_ids, normals=gt_normals)

        # Zero out brdf at occluded pixels
        brdf = brdf * confidence.unsqueeze(-1)

        loss = self._compute_loss(brdf, rgbs_gt, confidence)

        # PSNR over valid (non-occluded) pixels only
        valid = confidence > 0
        if valid.any():
            mse = ((brdf[valid] - rgbs_gt[valid]) ** 2).mean()
            max_val = rgbs_gt[valid].max().clamp_min(1e-8)
        else:
            mse = torch.tensor(1.0, device=brdf.device)
            max_val = torch.tensor(1.0, device=brdf.device)
        psnr = 10.0 * torch.log10(max_val ** 2 / mse.clamp_min(1e-8))

        self.log_dict({
            'val/loss': loss,
            'val/psnr': psnr,
        }, prog_bar=True, batch_size=xyz.shape[0])

        if hasattr(self.material, 'learnable_factor') and self.material.learnable_factor:
            factor_val = self.material.factor.detach()
            self.log_dict({
                'val/learnable_factor_r': factor_val[0],
                'val/learnable_factor_g': factor_val[1],
                'val/learnable_factor_b': factor_val[2],
            }, batch_size=xyz.shape[0])

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

        # Save 8-bit PNG (with proper tone mapping for HDR to LDR display)
        gt_png   = self._tonemap_for_display(gt_img.detach())
        pred_png = self._tonemap_for_display(pred_img.detach())
        
        cv2.imwrite(
            os.path.join(output_dir, f'gt_mat{mat_id:04d}_view{batch_idx}.png'),
            cv2.cvtColor(gt_png, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            os.path.join(output_dir,
                         f'pred_mat{mat_id:04d}_view{batch_idx}_psnr{psnr_str}.png'),
            cv2.cvtColor(pred_png, cv2.COLOR_RGB2BGR))

        # ---- save normal and tangent maps once (view-independent) ----------
        if batch_idx == 0:
            with torch.no_grad():
                global_pids = self.material.get_global_point_id(material_ids, point_ids)
                latent      = self.material.point_latent_bank(global_pids)
                if self.material.predict_frame or gt_normals is None:
                    pred_normal, pred_tangent = self.material.extract_frame_from_latent(latent)
                else:
                    pred_normal, pred_tangent = self.material.extract_frame_from_latent(latent, gt_normals)

            normal_img  = pred_normal.reshape(H, W, 3)
            tangent_img = pred_tangent.reshape(H, W, 3)
            # map [-1, 1] → [0, 255]
            normal_png  = ((normal_img.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255).byte().cpu().numpy()
            tangent_png = ((tangent_img.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255).byte().cpu().numpy()
            cv2.imwrite(
                os.path.join(output_dir, f'normal_mat{mat_id:04d}.png'),
                cv2.cvtColor(normal_png, cv2.COLOR_RGB2BGR))
            cv2.imwrite(
                os.path.join(output_dir, f'tangent_mat{mat_id:04d}.png'),
                cv2.cvtColor(tangent_png, cv2.COLOR_RGB2BGR))

        if self.more_visualizations:
            if batch_idx == 0:
                self.visualize_brdf_lobe(output_dir=output_dir, num_latents=10, resolution=64)

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

    def visualize_brdf_lobe(self, output_dir, num_latents=10, resolution=64):
        if not hasattr(self.material, 'decoder') or not hasattr(self.material, 'point_latent_bank'):
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        device = next(self.material.parameters()).device
        latent_dim   = self.material.latent_dim
        total_points = self.material.point_latent_bank.num_embeddings

        torch.manual_seed(42)
        point_indices = torch.randint(0, total_points, (num_latents,), device=device)

        with torch.no_grad():
            all_latents  = self.material.point_latent_bank(point_indices)  # [num_latents, total_latent_dim]
        brdf_latents = all_latents[:, :latent_dim]  # [num_latents, latent_dim]

        local_normal  = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        brdf_lobe_dir = os.path.join(output_dir, 'brdf_lobes')
        os.makedirs(brdf_lobe_dir, exist_ok=True)

        # ---- Visualization 1: Fix wi, vary wo --------------------------------
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

        # ---- Visualization 2: Fix wo, vary wi  (BRDF × cos_theta_i) ----------
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
        dataset = self.trainer.train_dataloader.dataset.datasets
        dataset.set_step(step)
