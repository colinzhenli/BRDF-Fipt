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
from model.neural_brdf import LearnableSvPBRBRDF, SvLatentModel, AnisotropicLatentTexturedModel, LatentTexturedModel
from model.emitter import DynamicPointEmitter, PresetPointEmitter
from torch.utils.data import DataLoader
from itertools import islice
from utils.dataset import SphereTestDataset, SphereValDataset
import hydra
import numpy as np
from pytorch_lightning.strategies import DDPStrategy
import importlib
import warnings
import logging
import cv2
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def gamma(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor of linear-light RGB values to sRGB.
    Matches Blender's built-in OCIO conversion (Standard view-transform).

    Parameters
    ----------
    x : torch.Tensor
        *Linear* RGB values in **[0 … ∞)**. Negative values are clamped to 0.

    Returns
    -------
    torch.Tensor
        sRGB-encoded values in the display range **[0 … 1]**.
    """
    # --- constants taken from the official sRGB transfer function ---
    _A   = 0.055           # 1.055 - 1
    _K0  = 0.0031308       # linear-to-sRGB break-point
    _PHI = 1.0 / 2.4       # 0.416̅  = 1/γ

    x_lin = x.clamp(min=0.0)               # Blender never shows negative light
    low   = 12.92 * x_lin                  # linear segment
    high  = 1.055 * torch.pow(x_lin, _PHI) - _A

    return torch.where(x_lin <= _K0, low, high).clamp(0.0, 1.0)

# Generate camera positions uniformly distributed on hemisphere
def get_camera_dicts(number_of_views, distance, look_at, up, debug_mode=False):
    """Generate camera positions on the +Y hemisphere (y > 0)."""
    camera_dict = []
    if debug_mode:
        # Debug mode: 5 specific camera views
        # 1. Top down view (theta = 0, phi = 0)
        camera_dict.append({
            "position": [0.0, distance, 0.0],
            "look_at":  look_at,
            "up":       up,
            "theta":    0.0,
            "phi":      0.0
        })
        
        # 2-5. Four views at theta = 45 degrees with phi = 0, 90, 180, 270 degrees
        theta_45 = np.pi / 4.0  # 45 degrees in radians
        phi_angles = [0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0]  # 0, 90, 180, 270 degrees
        
        for phi in phi_angles:
            x = distance * np.sin(theta_45) * np.cos(phi)
            z = distance * np.sin(theta_45) * np.sin(phi)
            y = distance * np.cos(theta_45)
            
            camera_dict.append({
                "position": [x, y, z],
                "look_at":  look_at,
                "up":       up,
                "theta":    theta_45,
                "phi":      phi
            })
        
        return camera_dict
    
    # Original hemisphere distribution
    n_phi   = max(4, int(np.sqrt(max(number_of_views - 1, 1))))
    n_theta = max(2, int(np.ceil((number_of_views - 1) / n_phi)) + 1)
    
    # θ in (0, π/2)  ⇒  y > 0, exclude equator (θ = π/2) and pole (θ = 0)
    thetas = np.linspace(0.0, np.pi / 2.0, n_theta, endpoint=True)[1:-1]
    phis   = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing="ij")
    for theta, phi in zip(theta_grid.ravel(), phi_grid.ravel()):
        x = distance * np.sin(theta) * np.cos(phi)
        z = distance * np.sin(theta) * np.sin(phi)
        y = distance * np.cos(theta)             # guaranteed y > 0
        
        camera_dict.append({
            "position": [x, y, z],
            "look_at":  look_at,
            "up":       up,
            "theta":    theta,
            "phi":      phi
        })
    
    # Add the single north-pole view (θ = 0, y = +distance)
    camera_dict.append({
        "position": [0.0, distance, 0.0],
        "look_at":  look_at,
        "up":       up,
        "theta":    0.0,
        "phi":      0.0
    })
    
    return camera_dict

# Generate light positions uniformly distributed on hemisphere
def get_light_positions(num_lights, distance, debug_mode=False):
    """Generate light positions uniformly distributed on hemisphere."""
    light_positions = []
    
    # Debug mode: only one top-down light
    if debug_mode:
        light_positions.append({
            "position": [0.0, distance, 0.0],
            "intensity": 50.0,
            "theta": 0.0,
            "phi": 0.0
        })
        return light_positions
    
    n_phi   = max(4, int(np.sqrt(max(num_lights - 1, 1))))
    n_theta = max(2, int(np.ceil((num_lights - 1) / n_phi)) + 1)
    
    # θ in (0, π/2)  ⇒  y > 0, exclude equator (θ = π/2) and pole (θ = 0)
    thetas = np.linspace(0.0, np.pi / 2.0, n_theta, endpoint=True)[1:-1]
    phis   = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing="ij")
    for theta, phi in zip(theta_grid.ravel(), phi_grid.ravel()):
        x = distance * np.sin(theta) * np.cos(phi)
        z = distance * np.sin(theta) * np.sin(phi)
        y = distance * np.cos(theta)             # guaranteed y > 0
        
        light_positions.append({
            "position": [x, y, z],
            "intensity": 50.0,  # Fixed intensity
            "theta": theta,
            "phi": phi
        })
    
    # Add the single north-pole light (θ = 0, y = +distance)
    light_positions.append({
        "position": [0.0, distance, 0.0],
        "intensity": 50.0,
        "theta": 0.0,
        "phi": 0.0
    })
    
    return light_positions

def get_ray_directions(H, W, focal):
    """ get camera ray direction
    Args:
        H,W: height and width
        focal: focal length
    """
    x_coords = torch.linspace(0.5, W - 0.5, W)
    y_coords = torch.linspace(0.5, H - 0.5, H)
    j, i = torch.meshgrid([y_coords, x_coords])
    directions = \
        torch.stack([-(i-W/2)/focal, -(j-H/2)/focal, torch.ones_like(i)], -1) 

    return directions

def get_rays(directions, c2w, focal=None):
    """ world space camera ray
    Args:
        directions: camera ray direction (local)
        c2w: 3x4 camera to world matrix
        focal: if not None, return ray differentials as well
    """
    R = c2w[:,:3]
    rays_d = directions @ R.T
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    if focal is not None:
        dxdu = torch.tensor([1.0/focal,0,0])[None,None].expand_as(directions)@R.T
        dydv = torch.tensor([0,1.0/focal,0])[None,None].expand_as(directions)@R.T
        dxdu = dxdu.view(-1,3)
        dydv = dydv.view(-1,3)
        return rays_o, rays_d, dxdu, dydv
    else:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        return rays_o, rays_d
    
def get_c2w(camera):
    position = torch.tensor(camera['position'], dtype=torch.float32)
    target = torch.tensor(camera['look_at'], dtype=torch.float32)
    up = torch.tensor(camera.get('up', [0,1,0]), dtype=torch.float32)
    
    forward = target - position
    forward = forward / torch.norm(forward)
    # Ensure `up` is not parallel to `forward`
    if torch.abs(torch.dot(forward, up)) > 0.99:  # Too parallel, adjust up
        up = torch.tensor([1.0, 0.0, 0.0]) if torch.abs(forward[0]) < 0.99 else torch.tensor([0.0, 1.0, 0.0])
    """ right hand coordinate system """
    right = torch.cross(up, forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)
    
    c2w = torch.eye(4)
    c2w[:3,:3] = torch.stack([right, up, forward], dim=1)
    c2w[:3,3] = position
    c2w = c2w[:3,:4]
    return c2w

def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.model.checkpoint_monitor)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_monitor, lr_monitor]

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    pl.seed_everything(cfg.global_train_seed, workers=True)
    if not os.path.exists(cfg.exp_output_root_path):
        os.makedirs(cfg.exp_output_root_path)
    
    # Setup ground truth material
    gt_material_cfg = hydra.compose(config_name="config", overrides=["material=svpbr"]).material  
    albedo = gt_material_cfg.albedo
    
    gt_material = SvPBRBRDF(
        cfg=gt_material_cfg,
        albedo=torch.tensor(albedo)
    )
    
    # Setup renderer
    gt_renderer = ForwardRenderer(cfg, gt_material)
    
    # Setup camera and light parameters
    number_of_views = cfg.renderer.camera.number_of_views
    number_of_lights = cfg.renderer.emitter.num_lights
    camera_distance = cfg.renderer.camera.distance
    light_distance = cfg.renderer.emitter.dist
    look_at = cfg.renderer.camera.look_at
    up = cfg.renderer.camera.up
    resolution = cfg.renderer.resolution
    
    # Generate camera and light positions
    debug_mode = False
    camera_dicts = get_camera_dicts(number_of_views, camera_distance, look_at, up, debug_mode)
    light_positions = get_light_positions(number_of_lights, light_distance, debug_mode)
    
    # Setup ray generation
    h, w = resolution
    camera_angle_x = cfg.renderer.camera.camera_angle_x
    focal = (0.5 * w / np.tan(0.5 * camera_angle_x)).item()
    directions = get_ray_directions(h, w, focal)
    
    emitter = PresetPointEmitter(
        positions=torch.tensor([pos["position"] for pos in light_positions], device="cuda"),
        intensities=torch.tensor([[50.0, 50.0, 50.0] for _ in range(number_of_lights)], device="cuda")
    )
    
    print(f"==> Generating {len(camera_dicts)} x {len(light_positions)} = {len(camera_dicts) * len(light_positions)} reference images...")
    
    # Create output directory
    output_dir = cfg.dataset_folder
    os.makedirs(output_dir, exist_ok=True)
    
    # Metadata for all images
    metadata = []
    image_idx = 0

    # Store camera metadata
    camera_metadata = []
    for cam_idx, camera_dict in enumerate(camera_dicts):
        c2w = get_c2w(camera_dict)
        camera_metadata.append({
            "camera_id": cam_idx,
            "position": camera_dict["position"],
            "look_at": look_at,
            "up": up,
            "distance": camera_distance,
            "camera_angle_x": camera_angle_x,
            "focal": focal,
            "c2w_matrix": c2w.tolist()
        })
    
    # Store emitter metadata
    emitter_metadata = []
    for light_idx, light_pos in enumerate(light_positions):
        emitter_metadata.append({
            "emitter_id": light_idx,
            "position": light_pos["position"],
            "intensity": [50.0, 50.0, 50.0],
            "distance": light_distance
        })
    with torch.no_grad():
        for cam_idx, camera_dict in tqdm(enumerate(camera_dicts), total=len(camera_dicts), desc="Processing cameras"):
            # Generate camera-to-world matrix
            c2w = get_c2w(camera_dict)
            rays_o, rays_d, dxdu, dydv = get_rays(directions, c2w, focal=focal)
            rays = torch.cat([rays_o, rays_d, dxdu, dydv], dim=-1).unsqueeze(0).cuda()
            
            for light_idx, light_pos in enumerate(light_positions):
                # Create emitter with single light at specific position

                light_idx_tensor = torch.tensor([light_idx], device="cuda").expand(rays.shape[0], rays.shape[1])
                
                # Render image
                gt_params = torch.zeros(1).cuda()
                rgbs_gt, *_ = gt_renderer.render(emitter, rays, light_idx_tensor, cfg.renderer.spp.test, gt_params, None)
                
                # Reshape and save before gamma correction
                img_gt = rgbs_gt.reshape(*resolution, -1)
                
                # Save linear image (before gamma)
                linear_filename = f"image_{image_idx:06d}_original.png"
                torchvision.utils.save_image(
                    img_gt.permute(2, 0, 1), 
                    os.path.join(output_dir, linear_filename)
                )
                
                # Apply gamma correction
                img_gt_gamma = gamma(img_gt)
                
                # Save gamma-corrected image
                image_filename = f"image_{image_idx:06d}.png"
                torchvision.utils.save_image(
                    img_gt_gamma.permute(2, 0, 1), 
                    os.path.join(output_dir, image_filename)
                )
                # Store metadata
                metadata.append({
                    "image_id": image_idx,
                    "filename": image_filename,
                    "camera_id": cam_idx,
                    "emitter_id": light_idx
                })
                
                image_idx += 1
    
    # Save metadata to JSON file
    metadata_filename = os.path.join(output_dir, "metadata.json")
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2, default=lambda x: list(x) if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)) else str(x))
        
    # Save camera metadata to separate JSON file
    camera_metadata_filename = os.path.join(output_dir, "camera_metadata.json")
    with open(camera_metadata_filename, 'w') as f:
        json.dump(camera_metadata, f, indent=2, default=lambda x: list(x) if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)) else str(x))
    
    # Save emitter metadata to separate JSON file
    emitter_metadata_filename = os.path.join(output_dir, "emitter_metadata.json")
    with open(emitter_metadata_filename, 'w') as f:
        json.dump(emitter_metadata, f, indent=2, default=lambda x: list(x) if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)) else str(x))
    
    print(f"==> Generated {len(metadata)} reference images")
    print(f"==> Images saved to: {output_dir}")
    print(f"==> Metadata saved to: {metadata_filename}")

if __name__ == "__main__":
    main()
