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
from model.emitter import DynamicPointEmitter, PresetPointEmitter, RealAreaEmitter
import os

class BRDFTrainer(pl.LightningModule):
    def __init__(self, cfg, material, gt_material, roughness, metallic):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.material = material
        self.gt_material = gt_material
        self.gt_folder = cfg.gt_folder
        
        #self.latent_dim = cfg.material.latent_dim
        # Create a mapping from roughness-metallic pairs to train latent indices

        self.latent_reg_weight = cfg.model.latent_reg_weight if hasattr(cfg.model, 'latent_reg_weight') else 1e-4
        self.inference_lr = cfg.model.optimizer.inference_lr
        self.inference_steps = cfg.model.optimizer.inference_steps
        print("after latent reg weight")
        self.renderer = ForwardRenderer(cfg, self.material)
        self.gt_renderer = ForwardRenderer(cfg, self.gt_material)
        # self.emitter = PresetPointEmitter(
        #     read_from_metadata=True,
        #     metadata_path=os.path.join(self.cfg.metadata_path, 'emitter_metadata.json'),
        #     positions=None, 
        #     intensities=None
        # )
        self.emitter = RealAreaEmitter(
            radius=cfg.renderer.emitter.radius,
            positions=cfg.renderer.emitter.positions,
            intensities=cfg.renderer.emitter.intensities
        )
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
        
    def training_step(self, batch, batch_idx):
        """
        with importance sampling
        """
        # ------------------------------------------------------------------
        # 1. Un-pack inputs
        # ------------------------------------------------------------------
        rays, rgbs_gt, emitter_ids, weighted_pdf = batch['rays'], batch['rgbs'], batch['emitter_ids'], batch['pdf']
        # forward renders
        rgbs, vis, ray_params = self.renderer.render(self.emitter, rays, emitter_ids, self.cfg.renderer.spp.train,
                                        None, None)                       # f(r)
        if self.hparams.model.loss.recon_loss.name == "l1":
            per_pix = torch.abs(rgbs - rgbs_gt).mean(dim=-1)        # (S,)
        else:  # "l2"
            per_pix = torch.pow(rgbs - rgbs_gt, 2).mean(dim=-1)     # (S,)

        if self.hparams.data.importance_sampling:
            weights    = 1.0 / weighted_pdf.clamp(min=1e-5)                              # importance weights
        else:
            weights    = 1.0               # importance weights
        recon_loss = (per_pix * weights).mean()  

        # ------------------------------------------------------------------
        # 6.  Add regulariser + compute PSNR
        # ------------------------------------------------------------------
        latent_reg = 0.0
        loss       = recon_loss + self.hparams.model.loss.latent_reg_loss.weight * latent_reg

        psnr_loss  = torch.nn.functional.mse_loss(self.gamma(rgbs),
                                                self.gamma(rgbs_gt),
                                                reduction='mean')
        psnr       = 10.0 * torch.log10(1.0 / psnr_loss.clamp_min(1e-5))

        # ------------------------------------------------------------------
        # 7.  Logging  (now includes diagnostics)
        # ------------------------------------------------------------------
        self.log_dict({
            'train/recon_loss':   recon_loss,
            'train/latent_reg':   latent_reg,
            'train/total_loss':   loss,
            'train/psnr':         psnr,
        }, prog_bar=True, batch_size=rays.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        """ batch pbr texture: [B, H, W, 16] """
        rays, rgbs_gt, emitter_ids = batch['rays'], batch['rgbs'], batch['emitter_ids']

        # forward renders
        rgbs, vis, ray_params = self.renderer.render(self.emitter, rays, emitter_ids, self.cfg.renderer.spp.train, None, None)    

        psnr_loss = torch.nn.functional.mse_loss(self.gamma(rgbs), self.gamma(rgbs_gt), reduction='mean')
        psnr = 10.0 * torch.log10((1.0 ** 2) / psnr_loss.clamp_min(1e-5))
        
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
                f'demin_fabric_03_4k'
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
            
        # save the learned pbr normal map
        pbr_normal_map = self.material.pbr_texture.data[0, :, :, 10:13]
        torchvision.utils.save_image(
            pbr_normal_map.permute(2, 0, 1),
            os.path.join(output_dir, f'pbr_normal_map_{batch_idx}_{b}.png')
        )
        
        self.log('val/loss', loss)
        self.log('val/psnr', psnr)        
        return

    def on_train_batch_start(self, batch, batch_idx):
        step = self.global_step
        self.trainer.train_dataloader.dataset.datasets.set_step(step)
