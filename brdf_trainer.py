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
        rays, rgbs_gt = batch['rays'], batch['rgbs']
        rays_x, rays_d = rays[..., :3], rays[..., 3:6]
        dxdu, dydv = rays[..., 6:9], rays[..., 9:12]

        rgbs = self.renderer.render(rays_x, rays_d, dxdu, dydv, self.img_hw, spp)
        loss = NF.mse_loss(self.gamma(rgbs), self.gamma(rgbs_gt))
        psnr = -10.0 * torch.log10(loss.clamp_min(1e-5))
        return rgbs, loss, psnr

    def training_step(self, batch, batch_idx):
        _, loss, psnr = self.render_step(batch, self.cfg.renderer.spp.train)
        # self.log('train/loss', loss)
        # self.log('train/psnr', psnr)
        loss = loss.sum()
        loss.backward(retain_graph=True)

        # Print gradients explicitly
        print(f"Gradient w.r.t roughness: {self.material.proxy_brdf.roughness.grad}")
        # # Gradient Visualization
        # dot = make_dot(loss, params={
        #     'roughness': self.material.proxy_brdf.roughness,
        # })
        # dot.render(f"New_gradient_flow_batch_{batch_idx}", format="png")
        return loss
    # def training_step(self, batch, batch_idx):
    #     rays, rgbs_gt = batch['rays'], batch['rgbs']
    #     rays_x, rays_d = rays[..., :3], rays[..., 3:6]
    #     dxdu, dydv = rays[..., 6:9], rays[..., 9:12]
        
    #     B = rays_x.shape[0]
    #     device = rays_x.device

    #     sample1 = torch.rand(B, device=device)
    #     sample2 = torch.rand(B, 2, device=device)
    #     wo = torch.rand(B, 3, device=device).requires_grad_()
    #     normal = torch.nn.functional.normalize(torch.rand(B, 3, device=device), dim=-1)

    #     wi_proxy, pdf_proxy, brdf_weight = self.material.sample_brdf(sample1, sample2, wo, normal)

    #     loss = pdf_proxy.sum()
    #     loss.backward(retain_graph=True)

    #     # Print gradients explicitly
    #     print(f"Gradient w.r.t roughness: {self.material.proxy_brdf.roughness.grad}")
    #     # print(f"Gradient w.r.t debug_params: {self.material.proxy_brdf.debug_params.grad}")
    #     # Gradient Visualization
    #     dot = make_dot(loss, params={
    #         'roughness': self.material.proxy_brdf.roughness,
    #         'debug_params': self.material.proxy_brdf.debug_params
    #     })
    #     # dot.render(f"gradient_flow_batch_{batch_idx}", format="png")

    #     self.log('train/loss', loss.item())

    #     return loss

    def validation_step(self, batch, batch_idx):
        _, loss, psnr = self.render_step(batch, self.cfg.renderer.spp.val)
        self.log('val/loss', loss)
        self.log('val/psnr', psnr)
        return

    def test_step(self, batch, batch_idx):
        rgbs, loss, psnr = self.render_step(batch, self.cfg.renderer.spp.test)
        torchvision.utils.save_image(
            self.gamma(rgbs.permute(2, 0, 1)),
            os.path.join(self.cfg.exp_output_root_path, f'test_{batch_idx}_{self.roughness:.2f}_{self.metallic:.2f}.png')
        )
        self.log('test/psnr', psnr)
        return
