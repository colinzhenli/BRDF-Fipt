import torch
import torchvision
import json
import os
from tqdm import tqdm
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from renderer import ForwardRenderer
from brdf_trainer import BRDFTrainer
from model.brdf import MLPPBRBRDF, PBRBRDF, PhongBRDF
from torch.utils.data import DataLoader
from utils.dataset import SphereDataset
import hydra
from omegaconf import DictConfig
from pytorch_lightning.strategies import DDPStrategy
import importlib
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def gamma(x):
    mask = x <= 0.0031308
    ret = torch.empty_like(x)
    ret[mask] = 12.92 * x[mask]
    ret[~mask] = 1.055 * x[~mask].pow(1/2.4) - 0.055
    return ret


def get_dataset(cfg, split, gt_path=None):
    return SphereDataset(
        cfg,
        gt_path,
        split
    )

def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.model.checkpoint_monitor)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_monitor, lr_monitor]

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.global_train_seed, workers=True)
    rendered_image_paths = {}
    os.makedirs(cfg.exp_output_root_path, exist_ok=True)
    checkpoint_output_path = os.path.join(cfg.exp_output_root_path, "training")
    os.makedirs(checkpoint_output_path, exist_ok=True)

    roughness_vals = torch.linspace(cfg.model.roughness_range[0], cfg.model.roughness_range[1], cfg.model.parameter_num)
    metallic_vals = torch.linspace(cfg.model.metallic_range[0], cfg.model.metallic_range[1], cfg.model.parameter_num)

    for roughness in tqdm(roughness_vals, desc="[render ground truth] Roughness"):
        for metallic in tqdm(metallic_vals, desc="[render ground truth] Metallic", leave=False):
            material = PBRBRDF(albedo=torch.tensor([[1.0, 1.0, 1.0]]), roughness=roughness.item(), metallic=metallic.item())
            renderer = ForwardRenderer(cfg, material)
            dataset = get_dataset(cfg, 'test')

            sample = dataset[0]
            rays = sample['rays'].to(renderer.device)
            rays_x, rays_d = rays[..., :3], rays[..., 3:6]
            dxdu, dydv = rays[..., 6:9], rays[..., 9:12]
            with torch.no_grad():
                img = renderer.render(rays_x, rays_d, dxdu, dydv, cfg.renderer.resolution, cfg.renderer.spp.test)

            filename = f'output_{roughness:.2f}_{metallic:.2f}.png'
            output_path = os.path.join(cfg.exp_output_root_path, filename)
            torchvision.utils.save_image(img.permute(2, 0, 1), output_path)

            vis_filename = f'output_gamma_{roughness:.2f}_{metallic:.2f}.png'
            vis_path = os.path.join(cfg.exp_output_root_path, vis_filename)
            torchvision.utils.save_image(gamma(img.permute(2, 0, 1)), vis_path)

            rendered_image_paths[(roughness.item(), metallic.item())] = output_path

    # Training
    psnr_results = {}
    psnr_file = os.path.join(cfg.exp_output_root_path, 'psnr_results.json')

    if os.path.exists(psnr_file):
        with open(psnr_file, 'r') as f:
            psnr_results = json.load(f)

    for (roughness, metallic), gt_path in tqdm(rendered_image_paths.items(), desc="Training models"):
        material_module = importlib.import_module('model.brdf')
        material = getattr(material_module, cfg.material.type)(cfg.material, roughness, metallic)
        model = BRDFTrainer(cfg, material, roughness, metallic)

        print("==> initializing data ...")          
        train_loader = DataLoader(get_dataset(cfg, 'train', gt_path), batch_size=None, num_workers=cfg.data.num_workers)
        val_loader = DataLoader(get_dataset(cfg, 'val', gt_path), batch_size=None, num_workers=cfg.data.num_workers)
        test_loader = DataLoader(get_dataset(cfg, 'test', gt_path), batch_size=None, num_workers=cfg.data.num_workers)

        print("==> initializing logger ...")
        logger = hydra.utils.instantiate(cfg.model.logger, save_dir=cfg.exp_output_root_path)

        print("==> initializing monitor ...")
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(cfg.model.checkpoint_monitor.dirpath, f'model_{roughness:.2f}_{metallic:.2f}'),
            filename=cfg.model.checkpoint_monitor.filename,
            save_top_k=cfg.model.checkpoint_monitor.save_top_k, 
            every_n_epochs=cfg.model.checkpoint_monitor.every_n_epochs,
            monitor='val/loss'
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')

        print("==> initializing trainer ...")

        trainer = pl.Trainer(
            gradient_clip_val=1.0, # Gradient clipping (prevents gradient explosion)
            gradient_clip_algorithm='norm', 
            callbacks=[checkpoint_callback, lr_monitor], logger=logger, **cfg.model.trainer, strategy=DDPStrategy(find_unused_parameters=True)
        )

        trainer.fit(model, train_loader, val_loader)
        test_results = trainer.test(model, dataloaders=test_loader)

        psnr = test_results[0]['test/psnr']
        psnr_results[f"{roughness:.2f}_{metallic:.2f}"] = psnr

        with open(psnr_file, 'w') as f:
            json.dump(psnr_results, f, indent=4)

        torch.cuda.empty_cache()

    print('Training and Testing Complete!')


if __name__ == "__main__":
    main()