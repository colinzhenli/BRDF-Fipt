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
from model.brdf import MLPPBRBRDF, PBRBRDF, ProxyPBRBRDF
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
    pl.seed_everything(cfg.global_train_seed, workers=True)
    os.makedirs(cfg.exp_output_root_path, exist_ok=True)
    checkpoint_output_path = os.path.join(cfg.exp_output_root_path, "training")
    os.makedirs(checkpoint_output_path, exist_ok=True)

    gt_material_cfg = hydra.compose(config_name="config", overrides=["material=pbr"]).material
    albedo = gt_material_cfg.albedo
    roughness = gt_material_cfg.roughness
    metallic = gt_material_cfg.metallic

    material = LatentModel(cfg.material)
    gt_material = PBRBRDF(
        albedo=torch.tensor(albedo)
    )  # Ground truth uses pbr config

    model = BRDFTrainer(cfg, material, gt_material, roughness, metallic)

    if os.path.isfile(cfg.model.ckpt_path):
        print(f"=> loading model checkpoint '{cfg.model.ckpt_path}'")
        checkpoint = torch.load(cfg.model.ckpt_path, map_location=model.device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint successfully.")
    else:
        raise FileNotFoundError(f"No checkpoint found at '{cfg.model.ckpt_path}'.")

    print("==> optimizing latents for test materials...")
    test_dataset = SphereTestIterableDataset(cfg, gt_folder=None)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.data.num_workers)

    optimized_latents = []
    # Process all test samples in a batch
    all_gt_params = []
    for batch in test_loader:
        all_gt_params.append({k: v.to(model.device) for k, v in batch['gt_params'].items()})
    
    batch_size = len(all_gt_params)
    # Use model.test_latents for latent codes
    optimizer = torch.optim.Adam([model.test_latents], lr=cfg.model.optimizer.inference_lr)
    
    # Initialize the dataset for batch processing
    dataset = SphereTestIterableDataset(cfg, gt_folder=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    
    for step in tqdm(range(cfg.model.optimizer.inference_steps), desc="Optimizing latents"):
        optimizer.zero_grad()
        
        # Get a batch of data
        batch = next(iter(dataloader))
        rays, gt_params = batch['rays'].to(model.device), batch['gt_params'].to(model.device)
        
        # Create dynamic lighting for this step
        emitter = DynamicPointEmitter(
            dist=cfg.renderer.emitter.dist,
            num_lights=cfg.renderer.emitter.num_lights
        )
        
        # Extract roughness and metallic values from gt_params
        roughness = torch.cat([params['roughness'] for params in all_gt_params])
        metallic = torch.cat([params['metallic'] for params in all_gt_params])
        
        # Create unique keys for each roughness-metallic combination
        keys = [f"{r.item():.2f}_{m.item():.2f}" for r, m in zip(roughness, metallic)]
            
        # Map each unique key to an index
        for key in keys:
            if key not in model.test_roughness_metallic_to_index:
                model.test_roughness_metallic_to_index[key] = len(model.test_roughness_metallic_to_index)
                
        # Get indices for the current batch
        batch_indices = torch.tensor([model.test_roughness_metallic_to_index[key] for key in keys], device=model.device)
        
        # Get latents for the batch using the indices
        latents = model.test_latents[batch_indices]
        
        # Render all samples in batch
        rgbs = model.renderer.render(emitter, rays, cfg.renderer.spp.test, None, latents)
        
        if cfg.gt_folder is None:
            with torch.no_grad():
                rgbs_gt = model.gt_renderer.render(emitter, rays, cfg.renderer.spp.train, gt_params, None)
        else:
            rgbs_gt = batch['rgbs'].to(model.device)
        
        # Compute loss for the entire batch
        recon_loss = torch.nn.functional.l1_loss(rgbs, rgbs_gt)
        latent_reg = torch.norm(latents, p=2, dim=1).mean()
        total_loss = recon_loss + cfg.model.loss.latent_reg_loss.weight * latent_reg
        
        # Backward and optimize
        total_loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            tqdm.write(f"Step {step}/{cfg.model.optimizer.inference_steps}, Loss: {total_loss.item():.6f}, Recon: {recon_loss.item():.6f}, Reg: {latent_reg.item():.6f}")
    # Store optimized latents

    optimized_latents = [latent.detach().cpu() for latent in latents]

    torch.save(optimized_latents, os.path.join(cfg.exp_output_root_path, "optimized_latents.pt"))

    print("==> rendering ground truth images using environmental map...")
    renderer = ForwardRenderer(cfg, gt_material)
    dataset = SphereDataset(cfg, None, split='test')
    output_folder = os.path.join(cfg.exp_output_root_path, f'roughness_{roughness:.2f}_metallic_{metallic:.2f}')
    os.makedirs(output_folder, exist_ok=True)
    for idx, batch in tqdm(enumerate(dataset)):
        rays = batch['rays'].to(renderer.device)
        gt_params = batch['gt_params'].to(renderer.device)
        with torch.no_grad():
            img = renderer.render(None, rays, cfg.renderer.spp.test, gt_params, None)
            img = img.reshape(*cfg.renderer.resolution, -1)
        exr_path = os.path.join(output_folder, f'output_view_{idx}.exr')
        png_path = os.path.join(output_folder, f'output_gamma_view_{idx}.png')
        cv2.imwrite(exr_path, img[...,[2,1,0]].cpu().numpy())
        torchvision.utils.save_image(gamma(img.permute(2, 0, 1)), png_path)

    print("==> testing using optimized latents and environment map lighting...")
    test_loader = DataLoader(SphereDataset(cfg, output_folder, split='test'), batch_size=None, num_workers=cfg.data.num_workers)

    trainer = pl.Trainer(accelerator='gpu', devices=1, inference_mode=True)
    test_results = trainer.test(model, dataloaders=test_loader)
    avg_psnr = sum(res['test/psnr'] for res in test_results) / len(test_results)
    print(f"Final PSNR with EnvMap: {avg_psnr:.2f}")

    generate_video_from_results(output_folder, video_name="results_video.mp4", path_string="result_view_*.png", fps=10)
    generate_video_from_results(output_folder, video_name="gt_video.mp4", path_string="output_gamma_view_*.png", fps=10)

if __name__ == "__main__":
    main()
