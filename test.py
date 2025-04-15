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
from model.brdf import MLPPBRBRDF, PBRBRDF, PhongBRDF, ProxyPBRBRDF
from torch.utils.data import DataLoader
from utils.dataset import SphereDataset
import hydra

from pytorch_lightning.strategies import DDPStrategy
import importlib
import warnings
import logging
import cv2
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
            # Create parameter-specific output folder
            if cfg.gt_folder is not None: # if gt folder is provided, use it
                output_folder = cfg.gt_folder
            else: # if gt fold er is not provided, render and save the image
                output_folder = os.path.join(cfg.exp_output_root_path, f'roughness_{roughness:.2f}_metallic_{metallic:.2f}')
                os.makedirs(output_folder, exist_ok=True)
                for idx, batch in tqdm(enumerate(dataset)):
                    rays = batch['rays'].to(renderer.device)
                    rays_x,rays_d = rays[...,:3],rays[...,3:6]
                    dxdu, dydv = rays[..., 6:9], rays[..., 9:12]
                    with torch.no_grad():
                        img = renderer.render(rays_x, rays_d, dxdu, dydv, cfg.renderer.resolution, cfg.renderer.spp.test)
                        img = img.reshape(*cfg.renderer.resolution, -1)

                    filename = f'output_view_{idx}.exr'
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, img[...,[2,1,0]].cpu().numpy())

                    vis_filename = f'output_gamma_view_{idx}.png'
                    vis_path = os.path.join(output_folder, vis_filename)
                    torchvision.utils.save_image(gamma(img.permute(2, 0, 1)), vis_path)

            rendered_image_paths[(roughness.item(), metallic.item())] = output_folder
            


    # Test
    psnr_results = {}
    psnr_file = os.path.join(cfg.exp_output_root_path, 'psnr_results.json')

    if os.path.exists(psnr_file):
        with open(psnr_file, 'r') as f:
            psnr_results = json.load(f)

    for (roughness, metallic), gt_folder in tqdm(rendered_image_paths.items(), desc="Training models"):
        material = MLPPBRBRDF(cfg.material, roughness, metallic)
        model = BRDFTrainer(cfg, material, roughness, metallic)
        # model = BRDFTrainer.load_from_checkpoint(cfg.model.ckpt_path,  weights_only=False)
        # Load checkpoint
        if os.path.isfile(cfg.model.ckpt_path):
            print(f"=> loading model checkpoint '{cfg.model.ckpt_path}'")
            checkpoint = torch.load(cfg.model.ckpt_path, map_location=model.device, weights_only=False)

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint successfully.")
        else:
            raise FileNotFoundError(f"No checkpoint found at '{cfg.model.ckpt_path}'. Please ensure the path is correct.")
        
        print("==> initializing data ...")          
        test_loader = DataLoader(get_dataset(cfg, 'test', gt_folder), batch_size=None, num_workers=cfg.data.num_workers)

        print("==> initializing trainer ...")

        # trainer = pl.Trainer(
        #     callbacks=None, logger=None, max_epochs=1, inference_mode=True
        # )
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            inference_mode=True
        )


        test_results = trainer.test(model, dataloaders=test_loader)

        test_psnr = sum(result['test/psnr'] for result in test_results) / len(test_results)
        psnr_results[f"{roughness:.2f}_{metallic:.2f}"] = test_psnr 
        print(f"PSNR for roughness {roughness:.2f}, metallic {metallic:.2f}: {test_psnr:.2f}")

        with open(psnr_file, 'w') as f:
            json.dump(psnr_results, f, indent=4)

        torch.cuda.empty_cache()

    print('Training and Testing Complete!')


if __name__ == "__main__":
    main()