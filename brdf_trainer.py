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
        self.train_latents = torch.nn.Embedding(int(cfg.data.train_num), self.latent_dim)
        self.test_latents = torch.nn.Embedding(int(cfg.data.test_num), self.latent_dim)
        # Create a mapping from roughness-metallic pairs to train latent indices
        self.train_roughness_metallic_to_index = {}
        self.test_roughness_metallic_to_index = {}
        
        # Store roughness and metallic values for test rendering
        self.roughness = roughness
        self.metallic = metallic

        
        self.latent_reg_weight = cfg.model.latent_reg_weight if hasattr(cfg.model, 'latent_reg_weight') else 1e-4
        self.inference_lr = cfg.model.optimizer.inference_lr
        self.inference_steps = cfg.model.optimizer.inference_steps
        
        self.renderer = ForwardRenderer(cfg, self.material)
        self.gt_renderer = ForwardRenderer(cfg, self.gt_material)
        self.img_hw = cfg.renderer.resolution

    def gamma(self, x):
        mask = x <= 0.0031308
        ret = torch.empty_like(x)
        ret[mask] = 12.92 * x[mask]
        ret[~mask] = 1.055 * x[~mask].pow(1/2.4) - 0.055
        return ret

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

    def training_step(self, batch, batch_idx):
        # Handle batch of roughness and metallic values
        batch_indices = []
        roughness = batch['gt_params']['roughness']
        metallic = batch['gt_params']['metallic']
        keys = [f"{r:.2f}_{m:.2f}" for r, m in zip(roughness, metallic)]
        for key in keys:
            if key not in self.train_roughness_metallic_to_index:
                self.train_roughness_metallic_to_index[key] = len(self.train_roughness_metallic_to_index)
        batch_indices = [self.train_roughness_metallic_to_index[key] for key in keys]
        # Get latents for the batch using the indices
        latent = self.train_latents(torch.tensor(batch_indices, device=self.device))
        
        # randomly initialize emitter
        emitter = DynamicPointEmitter(
            dist=self.cfg.renderer.emitter.dist,
            num_lights=self.cfg.renderer.emitter.num_lights
        )
        
        # Render step
        rays, gt_params = batch['rays'], batch['gt_params']
        rgbs = self.renderer.render(emitter, rays, self.cfg.renderer.spp.train, None, latent)

        if self.gt_folder is None:
            with torch.no_grad():
                rgbs_gt = self.gt_renderer.render(emitter, rays, self.cfg.renderer.spp.train, gt_params, None)
        else:
            rgbs_gt = batch['rgbs']
        
        # Reconstruction loss
        recon_loss = NF.l1_loss(rgbs, rgbs_gt)
        latent_reg = torch.norm(latent, p=2)
        loss = recon_loss + self.hparams.model.loss.latent_reg_loss.weight * latent_reg
        
        psnr_loss = NF.mse_loss(self.gamma(rgbs), self.gamma(rgbs_gt))
        psnr = -10.0 * torch.log10(psnr_loss.clamp_min(1e-5))
        
        self.log('train/recon_loss', recon_loss)
        self.log('train/latent_reg', latent_reg)
        self.log('train/total_loss', loss)
        self.log('train/psnr', psnr)
        
        return loss

    def validation_step(self, batch, batch_idx):
        batch_indices = []
        roughness = batch['gt_params']['roughness']
        metallic = batch['gt_params']['metallic']
        keys = [f"{r:.2f}_{m:.2f}" for r, m in zip(roughness, metallic)]
        for key in keys:
            if key not in self.train_roughness_metallic_to_index:
                self.train_roughness_metallic_to_index[key] = len(self.train_roughness_metallic_to_index)
        batch_indices = [self.train_roughness_metallic_to_index[key] for key in keys]
        # Get latents for the batch using the indices
        latent = self.train_latents(torch.tensor(batch_indices, device=self.device))

        emitter = DynamicPointEmitter(
            dist=self.cfg.renderer.emitter.dist,
            num_lights=self.cfg.renderer.emitter.num_lights
        )
        
        # Render step logic
        rays, gt_params = batch['rays'], batch['gt_params']
        rgbs = self.renderer.render(emitter, rays, self.cfg.renderer.spp.train, None, latent)

        if self.gt_folder is None:
            with torch.no_grad():
                rgbs_gt = self.gt_renderer.render(emitter, rays, self.cfg.renderer.spp.train, gt_params, None)
        else:
            rgbs_gt = batch['rgbs']

        psnr_loss = NF.mse_loss(self.gamma(rgbs), self.gamma(rgbs_gt))
        psnr = -10.0 * torch.log10(psnr_loss.clamp_min(1e-5))
        
        recon_loss = NF.l1_loss(rgbs, rgbs_gt)
        latent_reg = torch.norm(latent, p=2)
        loss = recon_loss + self.hparams.model.loss.latent_reg_loss.weight * latent_reg
        
        # Handle batch of images
        batch_size = roughness.shape[0]
        batched_rgbs = rgbs.reshape(batch_size, *self.img_hw, -1)
        batched_rgbs_gt = rgbs_gt.reshape(batch_size, *self.img_hw, -1)
        for b in range(batch_size):
            # Reshape individual sample in batch
            sample_rgbs = batched_rgbs[b]
            sample_rgbs_gt = batched_rgbs_gt[b]
            
            # Create output directory for each sample
            output_dir = os.path.join(
                self.cfg.exp_output_root_path,
                f'val_roughness_{roughness[b].item():.2f}_metallic_{metallic[b].item():.2f}'
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Save ground truth and result images
            torchvision.utils.save_image(
                self.gamma(sample_rgbs_gt.permute(2, 0, 1)),
                os.path.join(output_dir, f'gt_view_{batch_idx}_{b}.png')
            )
            torchvision.utils.save_image(
                self.gamma(sample_rgbs.permute(2, 0, 1)),
                os.path.join(output_dir, f'result_view_{batch_idx}_{b}.png')
            )
        
        self.log('val/loss', loss)
        self.log('val/psnr', psnr)        
        return

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        latent = torch.randn(1, self.hparams.material.latent_dim, device=self.device) * 0.01
        latent_optimizer = torch.optim.Adam([latent], lr=self.inference_lr)
        rays, rgbs_gt, gt_params = batch['rays'], batch['rgbs'], batch['gt_params']
        
        if self.cfg.renderer.emitter.type == 'envmap':
            rgbs = self.renderer.render(None, rays, self.cfg.renderer.spp.test, gt_params, latent)
            if rgbs_gt is None:
                with torch.no_grad():
                    rgbs_gt = self.gt_renderer.render(None, rays, self.cfg.renderer.spp.test, gt_params, latent)
        else:
            for i in range(self.inference_steps):
                latent_optimizer.zero_grad()
                emitter = DynamicPointEmitter(
                    dist=self.cfg.renderer.emitter.dist,
                    num_lights=self.cfg.renderer.emitter.num_lights
                )
                rgbs = self.renderer.render(emitter, rays, self.cfg.renderer.spp.test, None, latent)
                if rgbs_gt is None:
                    with torch.no_grad():
                        rgbs_gt = self.gt_renderer.render(emitter, rays, self.cfg.renderer.spp.test, gt_params, None)

                recon_loss = NF.l1_loss(rgbs, rgbs_gt)
                latent_reg = torch.norm(latent, p=2)
                loss = recon_loss + self.hparams.model.loss.latent_reg_loss.weight * latent_reg
                loss.backward()
                latent_optimizer.step()
        torch.set_grad_enabled(False)
        psnr_loss = NF.mse_loss(self.gamma(rgbs), self.gamma(rgbs_gt))
        psnr = -10.0 * torch.log10(psnr_loss.clamp_min(1e-5))
        
        rgbs = rgbs.reshape(*self.img_hw, -1)
        rgbs_gt = rgbs_gt.reshape(*self.img_hw, -1)
        os.makedirs(os.path.join(self.cfg.exp_output_root_path,f'roughness_{self.roughness:.2f}_metallic_{self.metallic:.2f}'), exist_ok=True)

        torchvision.utils.save_image(
            self.gamma(rgbs_gt.permute(2, 0, 1)),
            os.path.join(self.cfg.exp_output_root_path,f'roughness_{self.roughness:.2f}_metallic_{self.metallic:.2f}', f'gt_view_{batch_idx}.png')
            )
        torchvision.utils.save_image(
            self.gamma(rgbs.permute(2, 0, 1)),
            os.path.join(self.cfg.exp_output_root_path,f'roughness_{self.roughness:.2f}_metallic_{self.metallic:.2f}', f'result_view_{batch_idx}.png')
        )
        self.log('test/psnr', psnr)
        return 
