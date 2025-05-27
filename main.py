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
from model.brdf import SvLatentModel, SvPBRBRDF, LatentTexturedModel
from torch.utils.data import DataLoader
from utils.dataset import SphereIterableDataset, SphereValDataset, UniformSphereIterableDataset
import hydra
from omegaconf import DictConfig
from pytorch_lightning.strategies import DDPStrategy
import importlib
import warnings
import logging
from viztracer import VizTracer
import cv2
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.model.checkpoint_monitor)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_monitor, lr_monitor]

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.global_train_seed, workers=True)
    os.makedirs(cfg.exp_output_root_path, exist_ok=True)
    checkpoint_output_path = os.path.join(cfg.exp_output_root_path, "training")
    os.makedirs(checkpoint_output_path, exist_ok=True)

    # Load ground truth material parameters from pbr config
    gt_material_cfg = hydra.compose(config_name="config", overrides=["material=pbr"]).material
    
    # Use ground truth parameters from pbr.yaml
    albedo = gt_material_cfg.albedo
    roughness = gt_material_cfg.roughness
    metallic = gt_material_cfg.metallic
    
    output_folder = os.path.join(cfg.exp_output_root_path, f'fabric_pattern_07_4k')
    os.makedirs(output_folder, exist_ok=True)

    # Initialize materials using different configs
    # material_module = importlib.import_module('model.brdf')
    # material = getattr(material_module, cfg.material.type)(cfg)
    if cfg.material.type == "SvLatentModel":
        material = SvLatentModel(cfg.material)  # MLP model uses mlp_pbr config
    elif cfg.material.type == "LatentTexturedModel":
        material = LatentTexturedModel(cfg.material)  # MLP model uses mlp_pbr config
    gt_material = SvPBRBRDF(
        albedo=torch.tensor(albedo)
    )  # Ground truth uses pbr config

    model = BRDFTrainer(cfg, material, gt_material, roughness, metallic)

    print("==> initializing data ...")          
    train_dataset = UniformSphereIterableDataset(cfg, gt_folder=None, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    val_dataset = SphereValDataset(cfg, gt_folder=None)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=cfg.data.num_workers,
    )

    print("==> initializing logger ...")
    logger = hydra.utils.instantiate(cfg.model.logger, save_dir=cfg.exp_output_root_path)

    print("==> initializing monitor ...")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.model.checkpoint_monitor.dirpath, f'model_{roughness:.2f}_{metallic:.2f}'),
        filename=cfg.model.checkpoint_monitor.filename,
        save_top_k=cfg.model.checkpoint_monitor.save_top_k, 
        every_n_epochs=cfg.model.checkpoint_monitor.every_n_epochs,
        monitor='val/loss',
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    print("==> initializing trainer ...")

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor], logger=logger, **cfg.model.trainer
    )

    trainer.fit(model, train_loader, val_loader)
    """  Skipping testing for now """
    # test_results = trainer.test(model, dataloaders=test_loader)

    # test_psnr = sum(result['test/psnr'] for result in test_results) / len(test_results)
    # print(f"PSNR for roughness {roughness:.2f}, metallic {metallic:.2f}: {test_psnr:.2f}")

    torch.cuda.empty_cache()

    print('Training and Testing Complete!')


if __name__ == "__main__":
    main()