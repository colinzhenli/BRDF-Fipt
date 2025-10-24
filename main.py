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
from model.brdf import SvPBRBRDF
from model.neural_brdf import SvLatentModel, LatentTexturedModel, AnisotropicLatentTexturedModel, LearnableSvPBRBRDF
from model.mipmap_brdf import MipmapLearnableSvPBRBRDF
from torch.utils.data import DataLoader
from utils.dataset import RealImageDataset, RealValDataset
from model.brdf import GreyPatchBRDF
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
    gt_material_cfg = hydra.compose(config_name="config", overrides=["material=svpbr"]).material
    
    # Use ground truth parameters from pbr.yaml
    albedo = gt_material_cfg.albedo
    roughness = gt_material_cfg.roughness
    metallic = gt_material_cfg.metallic
    
    output_folder = os.path.join(cfg.exp_output_root_path, f'fabric_pattern_07_4k')
    os.makedirs(output_folder, exist_ok=True)

    # Initialize materials using different configs
    # material_module = importlib.import_module('model.brdf')
    # material = getattr(material_module, cfg.material.type)(cfg)
    if cfg.material.type == "LatentTexturedModel":
        material = LatentTexturedModel(cfg.material)  # MLP model uses mlp_pbr config
    elif cfg.material.type == "SvLatentModel":
        material = SvLatentModel(cfg.material)  # MLP model uses mlp_pbr config
    elif cfg.material.type == "AnisotropicLatentTexturedModel":
        material = AnisotropicLatentTexturedModel(cfg.material)  # MLP model uses mlp_pbr config
    elif cfg.material.type == "LearnableSvPBRBRDF":
        material = LearnableSvPBRBRDF(cfg.material)  # MLP model uses mlp_pbr config
    elif cfg.material.type == "MipmapLearnableSvPBRBRDF":
        material = MipmapLearnableSvPBRBRDF(cfg.material)  # MLP model uses mlp_pbr config
    elif cfg.material.type == "GreyPatchBRDF":
        material = GreyPatchBRDF(cfg.material)  # MLP model uses mlp_pbr config
    else:
        raise ValueError(f"Invalid material type: {cfg.material.type}")
    gt_material = SvPBRBRDF(
        cfg=gt_material_cfg,
        albedo=torch.tensor(albedo)
    )  # Ground truth uses pbr config
    print("before trainer init")
    model = BRDFTrainer(cfg, material, gt_material, roughness, metallic)
    if cfg.model.ckpt_path is not None and os.path.isfile(cfg.model.ckpt_path):
        print(f"=> loading model checkpoint '{cfg.model.ckpt_path}'")
        checkpoint = torch.load(cfg.model.ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
        # Load parameters that exist in the checkpoint, keep new parameters as initialized
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"=> loaded checkpoint successfully. {len(pretrained_dict)}/{len(model_dict)} parameters loaded.")
    print("after trainer init")
    print("==> initializing data ...")          
    train_dataset = RealImageDataset(cfg, gt_folder=cfg.gt_folder, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    val_dataset = RealValDataset(cfg, gt_folder=cfg.gt_folder)
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
    # tracer = VizTracer()
    # tracer.start()
    # trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)
    # tracer.stop()
    # tracer.save(f"Ray-rect-intersection_tracer.json")
    """  Skipping testing for now """
    # test_results = trainer.test(model, dataloaders=val_loader)

    # test_psnr = sum(result['test/psnr'] for result in test_results) / len(test_results)
    # print(f"PSNR for roughness {roughness:.2f}, metallic {metallic:.2f}: {test_psnr:.2f}")

    torch.cuda.empty_cache()

    print('Training and Testing Complete!')


if __name__ == "__main__":
    main()