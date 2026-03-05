import torch
import torch.nn as nn
import torchvision
import json
import os
from tqdm import tqdm
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from renderer import ForwardRenderer
from trainers import get_trainer_class
from model.brdf import SvPBRBRDF
from torch.utils.data import DataLoader
from utils.dataset import RealImageDataset, RealValDataset, MultiMaterialPointDataset, MERLBRDFIterableDataset,MERLBRDFIterableDataset_hd,MERLBRDFFixedDataset_hd,MERLBRDFFixedDataset, RealNovelViewDataset, BonnDataset, BonnValDataset, BonnSingleMaterialDataset, BonnSingleMaterialValDataset
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

    # Initialize material dynamically from module.type config
    material_module = importlib.import_module(cfg.material.module)
    material_class = getattr(material_module, cfg.material.type)
    material = material_class(cfg.material)
    gt_material = SvPBRBRDF(
        cfg=gt_material_cfg,
        albedo=torch.tensor(albedo)
    )  # Ground truth uses pbr config
    print("before trainer init")
    # Get the appropriate trainer class based on stage and data type
    stage = cfg.model.get('stage', 1)  # Default to stage 1
    data_type = cfg.data.get('dataset_name', 'default')
    TrainerClass = get_trainer_class(stage, data_type)
    
    print(f"Using trainer for stage {stage}: {TrainerClass.__name__}")
    model = TrainerClass(cfg, material, gt_material, roughness, metallic)
    
    # Track whether to use Lightning's resume functionality
    resume_ckpt_path = None
    
    if cfg.model.ckpt_path is not None and os.path.isfile(cfg.model.ckpt_path):
        print(f"=> loading model checkpoint '{cfg.model.ckpt_path}'")
        
        # If continue_training is enabled, use PyTorch Lightning's native resume
        # This will restore model weights, optimizer state, scheduler state, epoch, etc.
        continue_training = getattr(cfg.model, 'continue_training', False)
        if continue_training:
            print(f"=> continue_training=True: will resume full training state from checkpoint")
            resume_ckpt_path = cfg.model.ckpt_path
        else:
            # Manual weight loading for transfer learning / partial loading
            checkpoint = torch.load(cfg.model.ckpt_path, map_location='cpu', weights_only=False)
            
            if stage == 2:
                if cfg.model.test:
                    # Filter out emitter parameters from checkpoint
                    model_dict = model.state_dict()
                    filtered_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                                     if 'emitter' not in k and k in model_dict}
                    model_dict.update(filtered_dict)
                    model.load_state_dict(model_dict)
                    print(f"=> loaded model checkpoint successfully (excluding emitter). {len(filtered_dict)}/{len(checkpoint['state_dict'])} parameters loaded.")
                else:
                    # Stage 2: Only load the decoder weights from checkpoint
                    # Load material.decoder.* weights only (not latent codes)
                    model_dict = model.state_dict()
                    decoder_dict = {}
                    for k, v in checkpoint['state_dict'].items():
                        if 'material.decoder.' in k:
                            if k in model_dict:
                                decoder_dict[k] = v
                    
                    model_dict.update(decoder_dict)
                    model.load_state_dict(model_dict)
                    print(f"=> Stage 2: loaded decoder checkpoint successfully. {len(decoder_dict)}/{len([k for k in model_dict if 'material.decoder.' in k])} decoder parameters loaded.")
                    
                    # If use_latent_bank is enabled, also load the latent bank from checkpoint
                    use_latent_bank = getattr(cfg.material, 'use_latent_bank', False)
                    if use_latent_bank:
                        latent_bank_key = 'material.point_latent_bank.weight'
                        if latent_bank_key in checkpoint['state_dict']:
                            latent_weights = checkpoint['state_dict'][latent_bank_key]
                            num_points, latent_dim = latent_weights.shape
                            # Create embedding from checkpoint weights directly
                            model.material.point_latent_bank = nn.Embedding(num_points, latent_dim)
                            model.material.point_latent_bank.weight.data = latent_weights
                            print(f"=> Stage 2: loaded latent bank from checkpoint: {num_points} x {latent_dim}")
                        else:
                            print(f"=> Stage 2: use_latent_bank=True but no latent bank weights found in checkpoint.")
                    
                    # If initialize_from_std is enabled, reinitialize latent texture using std from checkpoint's latent bank
                    initialize_from_std = getattr(cfg.material, 'initialize_from_std', False)
                    if initialize_from_std:
                        latent_bank_key = 'material.point_latent_bank.weight'
                        if latent_bank_key in checkpoint['state_dict']:
                            latent_weights = checkpoint['state_dict'][latent_bank_key]
                            # latent_weights: [num_points, latent_dim]
                            # Last 6 dimensions have special meaning (normal + tangent), exclude them
                            brdf_latent_weights = latent_weights[:, :-6]
                            
                            # Compute std from the brdf latent dimensions
                            computed_std = brdf_latent_weights.std().item()
                            
                            # Reinitialize only the first N-6 dimensions, keep last 6 unchanged
                            latent_texture = model.material.latent_texture
                            resolution = latent_texture.resolution
                            num_brdf_dims = brdf_latent_weights.shape[1]
                            
                            # Reinitialize first N-6 dimensions with computed std
                            latent_texture.params.data[:, :num_brdf_dims, :, :] = torch.randn(
                                1, num_brdf_dims, resolution, resolution,
                                device=latent_texture.params.device,
                                dtype=latent_texture.params.dtype
                            ) * computed_std
                            
                            print(f"=> Stage 2: initialized latent texture (first {num_brdf_dims} dims) from checkpoint std={computed_std:.6f}")
                        else:
                            print(f"=> Stage 2: initialize_from_std=True but no latent bank weights found in checkpoint.")
            else:
                # Stage 1: Only load material parameters
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict and k.startswith('material.')}

                # Debug: inject latents for one hardcoded material only (offset-aware)
                debug_load_material_id = 1
                latent_bank_key = 'material.point_latent_bank.weight'
                if cfg.data.debug and latent_bank_key in pretrained_dict:
                    ckpt_latents = pretrained_dict.pop(latent_bank_key)   # [N_ckpt, D]
                    if hasattr(model.material, 'material_offset_tensor'):
                        offset = model.material.material_offset_tensor[debug_load_material_id].item()
                    else:
                        offset = 0  # single-material mode: no global offset
                    n = ckpt_latents.shape[0]
                    model_dict[latent_bank_key][offset:offset + n] = ckpt_latents
                    print(f"[Debug] Injected latents for material {debug_load_material_id}: "
                          f"ckpt rows 0:{n} → bank rows {offset}:{offset + n}")

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print(f"=> loaded material checkpoint successfully. {len(pretrained_dict)}/{len([k for k in model_dict if k.startswith('material.')])} material parameters loaded.")
    print("after trainer init")
    print("==> initializing data ...")   
    if cfg.data.dataset_name == "real":
        if not cfg.model.test:
            train_dataset = RealImageDataset(cfg, gt_folder=cfg.gt_folder, split="train")
            val_dataset = RealValDataset(cfg, gt_folder=cfg.gt_folder)
        else:
            if cfg.model.test_novel_view:
                val_dataset = RealNovelViewDataset(cfg, gt_folder=cfg.gt_folder)
            else:
                val_dataset = RealValDataset(cfg, gt_folder=cfg.gt_folder)
    elif cfg.data.dataset_name == "merl":
        if cfg.model.stage == 1:
            train_dataset = MERLBRDFIterableDataset(cfg,data_folder=cfg.dataset_folder,batch_size=1048576,split="train")
            if cfg.data.debug & cfg.data.valid_on_train_set:
                val_dataset = MERLBRDFIterableDataset(cfg,data_folder=cfg.dataset_folder,batch_size=1048576,split="val")
            else:
                val_dataset = MERLBRDFIterableDataset(cfg,data_folder=cfg.dataset_folder,batch_size=1048576,split="val")
        else:
            train_dataset = MERLBRDFFixedDataset(cfg,data_folder=cfg.dataset_folder,batch_size=100,split="train")
            if cfg.data.debug & cfg.data.valid_on_train_set:
                val_dataset = MERLBRDFFixedDataset(cfg,data_folder=cfg.dataset_folder,batch_size=1048576,split="val")
            else:
                val_dataset = MERLBRDFFixedDataset(cfg,data_folder=cfg.dataset_folder,batch_size=1048576,split="val")
    elif cfg.data.dataset_name == "points":
        train_dataset = MultiMaterialPointDataset(cfg, root_folder=cfg.dataset_folder, split="train")
        if cfg.data.debug & cfg.data.valid_on_train_set:
            val_dataset = MultiMaterialPointDataset(cfg, root_folder=cfg.dataset_folder, split="train")
        else:
            val_dataset = MultiMaterialPointDataset(cfg, root_folder=cfg.dataset_folder, split="val")
    elif cfg.data.dataset_name == "bonn":
        if cfg.model.stage == 1:
            train_dataset = BonnDataset(cfg, root_folder=cfg.dataset_folder, split="train")
            if cfg.data.debug & cfg.data.valid_on_train_set:
                val_dataset = BonnValDataset(cfg, root_folder=cfg.dataset_folder)
            else:
                val_dataset = BonnValDataset(cfg, root_folder=cfg.dataset_folder)
        else:
            train_dataset = BonnSingleMaterialDataset(cfg, root_folder=cfg.dataset_folder, split="train")
            if cfg.data.debug & cfg.data.valid_on_train_set:
                val_dataset = BonnSingleMaterialValDataset(cfg, root_folder=cfg.dataset_folder)
            else:
                val_dataset = BonnSingleMaterialValDataset(cfg, root_folder=cfg.dataset_folder)
        
    else:
        raise ValueError(f"Invalid dataset name: {cfg.data.dataset_name}")
    if not cfg.model.test:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=False,
        )
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
        callbacks=[checkpoint_callback, lr_monitor], logger=logger, 
        track_grad_norm=2,  # Log L2 norm of gradients
        # gradient_clip_val=1.0,  # Optional: clip gradients
        **cfg.model.trainer
    )
    # tracer = VizTracer()
    # tracer.start()
    if cfg.model.test:
        trainer.validate(model, val_loader)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt_path)
    # tracer.stop()
    # tracer.save(f"Ray-rect-intersection_tracer.json")
    """  Skipping testing for now """
    # test_results = trainer.test(model, dataloaders=test_loader)

    # test_psnr = sum(result['test/psnr'] for result in test_results) / len(test_results)
    # print(f"PSNR for roughness {roughness:.2f}, metallic {metallic:.2f}: {test_psnr:.2f}")

    torch.cuda.empty_cache()

    print('Training and Testing Complete!')


if __name__ == "__main__":
    main()
