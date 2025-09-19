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
from model.neural_brdf import SvLatentModel, AnisotropicLatentTexturedModel, LearnableSvPBRBRDF
from model.emitter import DynamicPointEmitter
from torch.utils.data import DataLoader
from itertools import islice
from utils.dataset import SphereTestDataset, SphereValDataset
import hydra
from pytorch_lightning.strategies import DDPStrategy
import importlib
import warnings
import logging
from utils.dataset import RealValDataset
from model.emitter import RealAreaEmitter
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

# Compute Delta E (CIE76) color difference
# Convert RGB to LAB color space for perceptual color difference
def compute_delta_e(img_pred, img_gt, gamma_fn):
    """Compute Delta E (CIE76) color difference between two images."""
    def rgb_to_xyz(rgb):
        # sRGB to XYZ conversion matrix (D65 illuminant)
        rgb = rgb.clamp(0, 1)
        # Apply inverse gamma correction
        rgb_linear = torch.where(rgb <= 0.04045, rgb / 12.92, torch.pow((rgb + 0.055) / 1.055, 2.4))
        
        # sRGB to XYZ matrix
        M = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], device=rgb.device, dtype=rgb.dtype)
        
        xyz = torch.matmul(rgb_linear, M.T)
        return xyz
    
    def xyz_to_lab(xyz):
        # D65 white point
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        
        x = xyz[..., 0] / Xn
        y = xyz[..., 1] / Yn
        z = xyz[..., 2] / Zn
        
        # Apply the cube root function with linear segment for small values
        def f(t):
            delta = 6.0 / 29.0
            return torch.where(t > delta**3, torch.pow(t, 1/3), t / (3 * delta**2) + 4/29)
        
        fx = f(x)
        fy = f(y)
        fz = f(z)
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return torch.stack([L, a, b], dim=-1)
    
    def delta_e_cie76(lab1, lab2):
        diff = lab1 - lab2
        return torch.sqrt(torch.sum(diff**2, dim=-1))
    
    # Convert gamma-corrected images to LAB
    img_pred_gamma = gamma_fn(img_pred)
    img_gt_gamma = gamma_fn(img_gt)
    
    xyz_pred = rgb_to_xyz(img_pred_gamma)
    xyz_gt = rgb_to_xyz(img_gt_gamma)
    
    lab_pred = xyz_to_lab(xyz_pred)
    lab_gt = xyz_to_lab(xyz_gt)
    
    delta_e = delta_e_cie76(lab_pred, lab_gt)
    return delta_e
    
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

    area_emitter_cfg = hydra.compose(config_name="config", overrides=["renderer=realcapture_area_emitter"])

    if cfg.material.type == "LatentTexturedModel":
        material = LatentTexturedModel(cfg.material)  # MLP model uses mlp_pbr config
    elif cfg.material.type == "SvLatentModel":
        material = SvLatentModel(cfg.material)  # MLP model uses mlp_pbr config
    elif cfg.material.type == "AnisotropicLatentTexturedModel":
        material = AnisotropicLatentTexturedModel(cfg.material)  # MLP model uses mlp_pbr config
    elif cfg.material.type == "LearnableSvPBRBRDF":
        material = LearnableSvPBRBRDF(cfg.material)  # MLP model uses mlp_pbr config
    else:
        raise ValueError(f"Invalid material type: {cfg.material.type}")

    model = BRDFTrainer(area_emitter_cfg, material, material, None, None)
    model.cuda()

    if os.path.isfile(cfg.model.ckpt_path):
        print(f"=> loading model checkpoint '{cfg.model.ckpt_path}'")
        ckpt = torch.load(cfg.model.ckpt_path, map_location=model.device, weights_only=False)
        state = ckpt["state_dict"].copy()
        model.load_state_dict(state, strict=False)

    else:
        raise FileNotFoundError(f"No checkpoint found at '{cfg.model.ckpt_path}'.")

    # Initialize the dataset for batch processing
    dataset = RealValDataset(cfg, gt_folder=cfg.gt_folder)
    output_dir = os.path.join(cfg.exp_output_root_path, "multiple-spp_rotated_view_images")
    os.makedirs(output_dir, exist_ok=True)

    print("==> rendering ground truth and predictions using environmental map...")
    resolution = cfg.renderer.camera.intrinsics.height, cfg.renderer.camera.intrinsics.width

    model.eval()
    renderer = ForwardRenderer(cfg, material)
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataset), total=len(dataset), desc="Processing materials"):
            turntable_center = torch.tensor(cfg.renderer.turntable_center, device=model.device)
            model.emitter._update_poses_for_vis(turntable_center, 100)
            # Fix camera id to 0 and use emitter index same as idx
            """ TODO: change the dataset to get rays only from the first camera for testing """
            batch['camera_ids'] = torch.zeros_like(batch['camera_ids'])
            batch['emitter_ids'] = torch.full_like(batch['emitter_ids'], idx)
            rays, emitter_ids = batch['rays'].to(model.device).unsqueeze(0), batch['emitter_ids'].to(model.device).unsqueeze(0)

            rgbs_pred, *_ = renderer.render(model.emitter, rays, emitter_ids, cfg.renderer.spp.test, None, None)    
            
            img_pred = rgbs_pred.reshape(*resolution, -1)
            torchvision.utils.save_image((img_pred.permute(2, 0, 1)), os.path.join(output_dir, f'result_view_{idx}.png'))

if __name__ == "__main__":
    main()
