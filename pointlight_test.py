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
from model.brdf import LatentTexturedModel
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

    gt_material_cfg = hydra.compose(config_name="config", overrides=["material=svpbr"]).material  
    point_emitter_cfg = hydra.compose(config_name="config", overrides=["renderer=dynamicpoint_emitter"])
    albedo = gt_material_cfg.albedo
    roughness = gt_material_cfg.roughness
    metallic = gt_material_cfg.metallic

    if cfg.material.type == "LatentTexturedModel":
        material = LatentTexturedModel(cfg.material)  # MLP model uses mlp_pbr config
    elif cfg.material.type == "SvLatentModel":
        material = SvLatentModel(cfg.material)  # MLP model uses mlp_pbr config
    else:
        raise ValueError(f"Invalid material type: {cfg.material.type}")
    
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
    psnr_list = []
    delta_e_list = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataset), total=len(dataset), desc="Processing materials"):
            # Render step logic
            rays, gt_params = batch['rays'].to(model.device).unsqueeze(0), batch['gt_params'].to(model.device).unsqueeze(0)
            emitter = DynamicPointEmitter(
                dist=cfg.renderer.emitter.dist,
                num_lights=cfg.renderer.emitter.num_lights,
                camera_phi = batch['phi'],
                theta_angle = cfg.renderer.emitter.theta_angle,
                random_positions = False,
                random_intensities = False
            )
            
            rgbs_pred, *_ = renderer.render(emitter, rays, cfg.renderer.spp.test, gt_params, None)
            if cfg.gt_folder is None:
                with torch.no_grad():
                    rgbs_gt, *_ = gt_renderer.render(emitter, rays, cfg.renderer.spp.test, gt_params, None)
            else:
                rgbs_gt = batch['rgbs'].to(model.device)

            
            delta_e = compute_delta_e(img_pred, img_gt, model.gamma)
            avg_delta_e = delta_e.mean().item()
            
            print(f"View {idx}: PSNR = {psnr.item():.2f}, Delta E = {avg_delta_e:.2f}")
            psnr_loss = torch.nn.functional.mse_loss(model.gamma(rgbs_pred), model.gamma(rgbs_gt), reduction='mean')
            psnr = 10.0 * torch.log10((1.0 ** 2) / psnr_loss.clamp_min(1e-5))
            psnr_list.append(psnr.item())
            delta_e_list.append(avg_delta_e)
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
    avg_delta_e = sum(delta_e_list) / len(delta_e_list)
    print(f"Final average Delta E across all test materials: {avg_delta_e:.2f}")

if __name__ == "__main__":
    main()
