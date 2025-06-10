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
from model.brdf import SvPBRBRDF, SvLatentModel
from model.emitter import DynamicPointEmitter
from torch.utils.data import DataLoader
from itertools import islice
from utils.dataset import SphereTestDataset, SphereValDataset
import hydra

from pytorch_lightning.strategies import DDPStrategy
import importlib
import warnings
import logging
import cv2
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def generate_video_from_results(output_folder, video_name="results_video.mp4", path_string="result_view_*.png", fps=10):
    import cv2
    import os
    from glob import glob
    img_paths = sorted(glob(os.path.join(output_folder, path_string)))
    if len(img_paths) == 0:
        print(f"No PNG images found in {output_folder}")
        return
    frame = cv2.imread(img_paths[0])
    height, width, _ = frame.shape
    out_path = os.path.join(output_folder, video_name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for img_path in img_paths:
        frame = cv2.imread(img_path)
        writer.write(frame)

    writer.release()
    print(f"Saved video to {out_path}")

def gamma(x):
    mask = x <= 0.0031308
    ret = torch.empty_like(x)
    ret[mask] = 12.92 * x[mask]
    ret[~mask] = 1.055 * x[~mask].pow(1/2.4) - 0.055
    return ret

def latent_train_collate_fn(batch_list, device=None):
    rays = torch.stack([b['rays'] for b in batch_list])        # [B, N, 12]
    rgbs = torch.stack([b['rgbs'] for b in batch_list])        # [B, N, 3]
    gt_params = {
        'roughness': torch.tensor([b['gt_params']['roughness'] for b in batch_list], device=device),
        'metallic': torch.tensor([b['gt_params']['metallic'] for b in batch_list], device=device)
    }
    rays = rays.to(device) if device else rays
    rgbs = rgbs.to(device) if device else rgbs
    return {'rays': rays, 'rgbs': rgbs, 'gt_params': gt_params}


def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.model.checkpoint_monitor)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_monitor, lr_monitor]

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    pl.seed_everything(cfg.global_train_seed, workers=True)
    os.makedirs(cfg.exp_output_root_path, exist_ok=True)
    checkpoint_output_path = os.path.join(cfg.exp_output_root_path, "training")
    os.makedirs(checkpoint_output_path, exist_ok=True)

    gt_material_cfg = hydra.compose(config_name="config", overrides=["material=svpbr"]).material  
    point_emitter_cfg = hydra.compose(config_name="config", overrides=["renderer=dynamicpoint_emitter"])
    albedo = gt_material_cfg.albedo
    roughness = gt_material_cfg.roughness
    metallic = gt_material_cfg.metallic

    material = SvLatentModel(cfg.material)
    gt_material = SvPBRBRDF(
        cfg=gt_material_cfg,
        albedo=torch.tensor(albedo)
    )  # Ground truth uses pbr config

    model = BRDFTrainer(point_emitter_cfg, material, gt_material, roughness, metallic)
    model.cuda()

    if os.path.isfile(cfg.model.ckpt_path):
        print(f"=> loading model checkpoint '{cfg.model.ckpt_path}'")
        checkpoint = torch.load(cfg.model.ckpt_path, map_location=model.device, weights_only=False)
        # Load parameters that exist in the checkpoint, keep new parameters as initialized
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"=> loaded checkpoint successfully. {len(pretrained_dict)}/{len(model_dict)} parameters loaded.")
    else:
        raise FileNotFoundError(f"No checkpoint found at '{cfg.model.ckpt_path}'.")


    
    # Initialize the dataset for batch processing
    dataset = SphereValDataset(point_emitter_cfg, gt_folder=None)
    dataloader = DataLoader(dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    print("==> rendering ground truth and predictions using environmental map...")
    resolution = cfg.renderer.resolution

    model.eval()
    gt_renderer = ForwardRenderer(cfg, gt_material)
    renderer = ForwardRenderer(cfg, material)
    emitter = DynamicPointEmitter(
        dist=cfg.renderer.emitter.dist,
        num_lights=cfg.renderer.emitter.num_lights,
        fix_seed = True
    )
    psnr_list = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataset), total=len(dataset), desc="Processing materials"):
            # Render step logic
            rays, gt_params = batch['rays'].to(model.device).unsqueeze(0), batch['gt_params'].to(model.device).unsqueeze(0)
            
            rgbs_pred, *_ = renderer.render(emitter, rays, cfg.renderer.spp.test, gt_params, None)
            if cfg.gt_folder is None:
                with torch.no_grad():
                    rgbs_gt, *_ = gt_renderer.render(emitter, rays, cfg.renderer.spp.test, gt_params, None)
            else:
                rgbs_gt = batch['rgbs'].to(model.device)

            psnr_loss = torch.nn.functional.l1_loss(model.gamma(rgbs_pred), model.gamma(rgbs_gt))
            psnr = -10.0 * torch.log10(psnr_loss.clamp_min(1e-5))
            psnr_list.append(psnr.item())

            # Reshape for visualization
            img_pred = rgbs_pred.reshape(*resolution, -1)
            img_gt = rgbs_gt.reshape(*resolution, -1)
            
            # Compute error map between prediction and ground truth with sign
            error_map = img_pred - img_gt  # shape: [H, W, 3]
            error_scalar = error_map.sum(dim=-1, keepdim=True)  # shape: [H, W, 1]
            
            # Calculate brightness of ground truth for normalization
            brightness = img_gt.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # Avoid division by zero
            
            # Normalize error by the brightness (relative error)
            relative_error = error_scalar / brightness
            error_magnitude = relative_error.abs()
            
            # Create red-blue error visualization
            error_map_r = torch.zeros_like(relative_error)
            error_map_b = torch.zeros_like(relative_error)
            error_map_r[relative_error > 0] = error_magnitude[relative_error > 0]
            error_map_b[relative_error < 0] = error_magnitude[relative_error < 0]
            error_map_display = torch.cat([error_map_r, torch.zeros_like(error_map_r), error_map_b], dim=-1)
            
            # Save error map and images
            output_dir = os.path.join(
                cfg.exp_output_root_path,
                f'fabric_pattern_07_4k'
            )
            os.makedirs(output_dir, exist_ok=True)
            torchvision.utils.save_image(
                error_map_display.permute(2, 0, 1), 
                os.path.join(output_dir, f'error_map_view_{idx}.png')
            )
            torchvision.utils.save_image(model.gamma(img_gt.permute(2, 0, 1)), os.path.join(output_dir, f'gt_view_{idx}.png'))
            torchvision.utils.save_image(model.gamma(img_pred.permute(2, 0, 1)), os.path.join(output_dir, f'result_view_{idx}.png'))

    avg_psnr = sum(psnr_list) / len(psnr_list)
    print(f"Final average PSNR across all test materials: {avg_psnr:.2f}")

if __name__ == "__main__":
    main()
