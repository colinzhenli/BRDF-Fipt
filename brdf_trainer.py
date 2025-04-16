import torch
import torch.nn.functional as NF
import pytorch_lightning as pl
import pl_bolts
from renderer import ForwardRenderer
import torchvision
from torchviz import make_dot
import os

class BRDFTrainer(pl.LightningModule):
    def __init__(self, cfg, material, roughness, metallic):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.roughness = roughness
        self.metallic = metallic

        self.material = material
        self.material_latents = {}
        self.renderer = ForwardRenderer(cfg, self.material)
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

    def render_step(self, batch, spp):
        rays, rgbs_gt, light_indices = batch['rays'], batch['rgbs'], batch['light_indices']
        rays_x, rays_d = rays[..., :3], rays[..., 3:6]
        dxdu, dydv = rays[..., 6:9], rays[..., 9:12]

        rgbs = self.renderer.render(rays_x, rays_d, dxdu, dydv, self.img_hw, spp, light_indices)
        loss = NF.l1_loss(rgbs, rgbs_gt)
        # loss = NF.mse_loss(self.gamma(rgbs), self.gamma(rgbs_gt))
        psnr = -10.0 * torch.log10(loss.clamp_min(1e-5))
        return rgbs, loss, psnr

    def training_step(self, batch, batch_idx):
        _, loss, psnr = self.render_step(batch, self.cfg.renderer.spp.train)
        self.log('train/loss', loss)
        self.log('train/psnr', psnr)
        # self.log('train/roughness', self.material.proxy_brdf.roughness)
        # loss = loss.sum()
        # loss.backward(retain_graph=True)

        # # Print gradients explicitly
        # print(f"Gradient w.r.t roughness: {self.material.proxy_brdf.roughness.grad}")
        # # Gradient Visualization
        # dot = make_dot(loss, params={
        #     'roughness': self.material.proxy_brdf.roughness,
        # })
        # dot.render(f"New_gradient_flow_batch_{batch_idx}", format="png")
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss, psnr = self.render_step(batch, self.cfg.renderer.spp.val)
        self.log('val/loss', loss)
        self.log('val/psnr', psnr)
        return

    def test_step(self, batch, batch_idx):
        camera_idx = batch_idx // self.cfg.renderer.emitter.num_lights
        light_idx = batch_idx % self.cfg.renderer.emitter.num_lights
        rgbs, loss, psnr = self.render_step(batch, self.cfg.renderer.spp.test)
        rgbs = rgbs.reshape(*self.img_hw, -1)
        os.makedirs(os.path.join(self.cfg.exp_output_root_path,f'roughness_{self.roughness:.2f}_metallic_{self.metallic:.2f}'), exist_ok=True)
        torchvision.utils.save_image(
            self.gamma(rgbs.permute(2, 0, 1)),
            os.path.join(self.cfg.exp_output_root_path,f'roughness_{self.roughness:.2f}_metallic_{self.metallic:.2f}', f'result_view_{camera_idx}_light_{light_idx}.png')
        )
        self.log('test/psnr', psnr)
        # self.log('test/roughness', self.material.proxy_brdf.roughness)
        return 
