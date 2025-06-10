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
from model.brdf import PBRBRDF, SvLatentModel
from model.emitter import DynamicPointEmitter, EnvMapEmitter
from torch.utils.data import DataLoader
from itertools import islice
from utils.dataset import SphereTestDataset, SphereIterableDataset
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

    gt_material_cfg = hydra.compose(config_name="config", overrides=["material=pbr"]).material
    point_emitter_cfg = hydra.compose(config_name="config", overrides=["renderer=dynamicpoint_emitter"])
    albedo = gt_material_cfg.albedo
    roughness = gt_material_cfg.roughness
    metallic = gt_material_cfg.metallic

    material = SvLatentModel(cfg.material)
    gt_material = PBRBRDF(
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

    print("==> optimizing latents for test materials...")

    # # Create a separate tensor for dual-latent codes instead of using model.test_latents
    # test_latents = torch.nn.Embedding(cfg.data.test_num, 2 * cfg.material.latent_dim, device=model.device)
    # torch.nn.init.normal_(test_latents.weight, mean=0.0, std=0.01)
    # # When used, we'll reshape to [B, 2, N] where N is the latent dimension
    
    # Optimize the spatial latent encoder and proxy BRDF parameters instead of test latents
    optimizer = torch.optim.Adam([
        {'params': material.spatial_encoder.parameters()},
        {'params': material.proxy_brdf.parameters()}
    ], lr=cfg.model.optimizer.inference_lr, betas=(0.9, 0.999), weight_decay=cfg.model.optimizer.weight_decay)
    
    # Setup cosine learning rate decay
    warmup_steps = int(0.5 * cfg.model.optimizer.inference_steps)  # 50% warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=cfg.model.optimizer.inference_steps - warmup_steps,
        eta_min=cfg.model.optimizer.inference_lr * 0.1  # Minimum LR is 10% of initial LR
    )
    
    # Create a separate dictionary to map roughness-metallic combinations to indices
    roughness_metallic_to_index = {}
    
    # Initialize the dataset for batch processing
    dataset = SphereIterableDataset(point_emitter_cfg, gt_folder=None, split="test")
    dataloader = DataLoader(dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    data_iter = iter(dataloader)
    if not cfg.test_on_train:
        for step in tqdm(range(cfg.model.optimizer.inference_steps), desc="Optimizing latents"):
            optimizer.zero_grad()
            
            # iterator = iter(dataloader)
            # batch = latent_train_collate_fn(
            #     list(islice(iterator, cfg.data.batch_size)),
            #     device=model.device
            # )

            batch = next(data_iter)  

            rays = batch['rays'].to(model.device)
            gt_params = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch['gt_params'].items()}
            
            # Create dynamic lighting for this step
            emitter = DynamicPointEmitter(
                dist=point_emitter_cfg.renderer.emitter.dist,
                num_lights=point_emitter_cfg.renderer.emitter.num_lights
            )
            
            # Extract roughness and metallic values from gt_params
            roughness = gt_params['roughness']
            metallic = gt_params['metallic']
            
            # # Create unique keys for each roughness-metallic combination
            # keys = [f"{r.item():.2f}_{m.item():.2f}" for r, m in zip(roughness, metallic)]
                
            # # Map each unique key to an index
            # for key in keys:
            #     if key not in roughness_metallic_to_index:
            #         roughness_metallic_to_index[key] = len(roughness_metallic_to_index)
                    
            # Get indices for the current batch
            # batch_indices = torch.tensor([roughness_metallic_to_index[key] for key in keys], device=model.device)
            batch_indices = torch.tensor([0], device=model.device)
            
            # # Get latents for the batch using the indices
            # latents = test_latents(batch_indices)
            latents = None
            
            # Render all samples in batch
            rgbs = model.renderer.render(emitter, rays, point_emitter_cfg.renderer.spp.test, None, latents)
            
            if cfg.gt_folder is None:
                with torch.no_grad():
                    rgbs_gt = model.gt_renderer.render(emitter, rays, point_emitter_cfg.renderer.spp.train, gt_params, None)
            else:
                rgbs_gt = batch['rgbs'].to(model.device)
            
            # Compute loss for the entire batch
            recon_loss = torch.nn.functional.l1_loss(rgbs, rgbs_gt)
            # latent_reg = torch.norm(latents, p=2, dim=1).mean()
            # total_loss = recon_loss + cfg.model.loss.latent_reg_loss.weight * latent_reg
            total_loss = recon_loss
            # Backward and optimize
            total_loss.backward()
            optimizer.step()
            
            # Apply learning rate scheduler after optimizer step
            if step >= warmup_steps:
                scheduler.step()
            
            if step % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                tqdm.write(f"Step {step}/{cfg.model.optimizer.inference_steps}, Loss: {total_loss.item():.6f}, Recon: {recon_loss.item():.6f}, LR: {current_lr:.6f}")
    else:
        print("==> skipping latent optimization...")

    print("==> rendering ground truth and predictions using environmental map...")
    psnr_record = {}
    resolution = cfg.renderer.resolution
    if cfg.test_on_train:
        dataset = SphereTestDataset(cfg, None, split="train")
    else:
        dataset = SphereTestDataset(cfg, None, split="test")

    model.eval()
    env_emitter = EnvMapEmitter(cfg.renderer.emitter.envmap_path).to(model.device)
    gt_renderer = ForwardRenderer(cfg, gt_material)
    renderer = ForwardRenderer(cfg, material)
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataset), total=len(dataset), desc="Processing materials"):
            gt_params = batch[0]['gt_params'] # different batch share the same gt_params
            # Move all parameters to the device
            for key in gt_params:
                gt_params[key] = gt_params[key].unsqueeze(0).to(model.device)
            r1, r2 = gt_params['roughness'][0][0].item(), gt_params['roughness'][0][1].item()
            m1, m2 = gt_params['metallic'][0][0].item(), gt_params['metallic'][0][1].item()
            param_key = f"{r1:.2f}_{m1:.2f}_{r2:.2f}_{m2:.2f}"

            output_folder = os.path.join(cfg.exp_output_root_path, param_key)
            os.makedirs(output_folder, exist_ok=True)

            if cfg.test_on_train:
                if param_key not in model.roughness_metallic_to_index:
                    model.roughness_metallic_to_index[param_key] = len(model.roughness_metallic_to_index)
                latent = model.train_latents(torch.tensor([model.roughness_metallic_to_index[param_key]], device=model.device)) # use the learned latents in the training set
            else:
                # latent = test_latents(torch.tensor([0], device=model.device))
                latent = None
            psnr_list = []

            for view_idx in tqdm(range(dataset.number_of_views), desc=f"Rendering views for {param_key}", leave=False):
                batch_view = batch[view_idx]  # reload view each loop
                rays = batch_view['rays'].to(model.device)
                
                rgbs_pred = renderer.render(None, rays, cfg.renderer.spp.test, gt_params, latent)
                if cfg.gt_folder is None:
                    rgbs_gt = gt_renderer.render(None, rays, cfg.renderer.spp.test, gt_params, None)
                else:
                    rgbs_gt = batch_view['rgbs'].to(model.device)

                psnr_loss = torch.nn.functional.mse_loss(gamma(rgbs_pred), gamma(rgbs_gt))
                psnr = -10.0 * torch.log10(psnr_loss.clamp_min(1e-5))
                psnr_list.append(psnr.item())

                img_pred = rgbs_pred.reshape(*resolution, -1)
                img_gt = rgbs_gt.reshape(*resolution, -1)
                # Compute error map between prediction and ground truth with sign
                # Signed error as sum over RGB channels
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
                
                # Save error map
                torchvision.utils.save_image(
                    error_map_display.permute(2, 0, 1), 
                    os.path.join(output_folder, f'error_map_view_{view_idx}.png')
                )

                torchvision.utils.save_image(gamma(img_gt.permute(2, 0, 1)), os.path.join(output_folder, f'gt_view_{view_idx}.png'))
                torchvision.utils.save_image(gamma(img_pred.permute(2, 0, 1)), os.path.join(output_folder, f'result_view_{view_idx}.png'))

            avg_psnr = sum(psnr_list) / len(psnr_list)
            psnr_record[param_key] = avg_psnr

            # generate_video_from_results(output_folder, video_name="results_video.mp4", path_string="result_view_*.png", fps=10)
            # generate_video_from_results(output_folder, video_name="gt_video.mp4", path_string="gt_view_*.png", fps=10)

        with open(os.path.join(cfg.exp_output_root_path, "psnr_results.json"), "w") as f:
            json.dump(psnr_record, f, indent=2)

    final_psnr = sum(psnr_record.values()) / len(psnr_record)
    print(f"Final average PSNR across all test materials: {final_psnr:.2f}")


if __name__ == "__main__":
    main()
