
import torch
torch.set_float32_matmul_precision('high')
import torch.nn.functional as NF
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_scatter
import time
import matplotlib.pyplot as plt

# import wandb

import json
import torch
import time
from pathlib import Path
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import mitsuba
import torchvision
mitsuba.set_variant('cuda_ad_rgb')

import math
# import tqdm
from tqdm import tqdm
import os
from pathlib import Path
from argparse import Namespace, ArgumentParser


from configs.config import default_options
from utils.dataset import InvRealDataset,RealDataset,InvSyntheticDataset,SyntheticDataset, SphereDataset
from utils.ops import *
from utils.path_tracing import ray_intersect, path_tracing_fix_emitter, path_tracing_envmap_emitter
from model.mlps import ImplicitMLP      
from model.brdf import NGPBRDF, PBRBRDF, MLPPBRBRDF
from model.emitter import SLFEmitter, PointEmitter, EnvMapEmitter

def gamma(x):
    """ tone mapping function """
    mask = x <= 0.0031308
    ret = torch.empty_like(x)
    ret[mask] = 12.92*x[mask]
    mask = ~mask
    ret[mask] = 1.055*x[mask].pow(1/2.4) - 0.055
    return ret

class ForwardRenderer():
    """ BRDF-emission mask training code """
    def __init__(self, hparams: Namespace, args, **kwargs):
        super(ForwardRenderer, self).__init__()
        self.hparams = hparams
        self.spp = 2048
        self.SPP = 16  # batch size 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # load scene geometry
        self.scene = mitsuba.load_dict({
            "type": "scene",
            "shape_id": {
                "type": "sphere",
                "center": [0.0, 0.0, 0.0], 
                "radius": 0.2,
                "flip_normals": False,
            }
        })

        # initiallize BRDF
        self.material = PBRBRDF(albedo=torch.tensor([[1.0, 1.0, 1.0]]), roughness=args['roughness'], metallic=args['metallic']).to(self.device)
        if args['emitter'] == 'point':
            self.emitter = PointEmitter(position=torch.tensor([-2, 2.0, 0.0]),
                        intensity=torch.tensor([50.0, 50.0, 50.0]),
                        radius=0.1).to(self.device)
        elif args['emitter'] == 'envmap':
            self.emitter = EnvMapEmitter('envmap.exr').to(self.device)
            
        # Initialize dataset for getting rays
        self.dataset = SphereDataset(
            root_dir=None,
            gt_image = None,
            gt_path=None,  # No ground truth needed
            split='train',
            pixel=False,
            ray_diff=True
        )

    @ torch.no_grad()
    def render(self):
        # Get rays from dataset
        sample = self.dataset[0]
        rays = sample['rays'].to(self.device)
        rays_x = rays[...,:3]
        rays_d = rays[...,3:6]
        dxdu, dydv = rays[...,6:9], rays[...,9:12]
        
        L = torch.zeros_like(rays_x).to(rays_x.device)
        for _ in tqdm(range(self.spp//self.SPP)):
            L += path_tracing_fix_emitter(self.scene, self.emitter, self.material, rays_x, rays_d, dxdu, dydv, self.SPP, indir_depth=0)
            torch.cuda.empty_cache()
        rgbs = L.reshape(*self.dataset.img_hw,-1)/(self.spp//self.SPP)

        return rgbs

class ModelTrainer(pl.LightningModule):
    """ BRDF-emission mask training code """
    def __init__(self, hparams: Namespace, **kwargs):
        super(ModelTrainer, self).__init__()
        self.save_hyperparameters(hparams)
        
        # Load scene geometry
        self.scene = mitsuba.load_dict({
            "type": "scene",
            "shape_id": {
                "type": "sphere",
                "center": [0.0, 0.0, 0.0], 
                "radius": 0.2,
                "flip_normals": False,
            }
        })

        # Initialize BRDF
        self.material = MLPPBRBRDF()
        # self.material = PBRBRDF(albe do=torch.ones(1, 3), roughness=0.5, metallic=0.5).to(self.device)
        
        # Set emitter type
        if self.hparams.emitter == 'point':
            self.emitter = PointEmitter(position=torch.tensor([-2, 2.0, 0.0]),
                        intensity=torch.tensor([50.0, 50.0, 50.0]),
                        radius=0.1)
        elif self.hparams.emitter == 'envmap':
            self.emitter = EnvMapEmitter('envmap.exr')

    def __repr__(self):
        return repr(self.hparams)
    
    def gamma(self,x):
        """ tone mapping function """
        mask = x <= 0.0031308
        ret = torch.empty_like(x)
        ret[mask] = 12.92*x[mask]
        mask = ~mask
        ret[mask] = 1.055*x[mask].pow(1/2.4) - 0.055
        return ret
    
    def configure_optimizers(self):
        if self.hparams.optimizer == 'SGD':
            opt = optim.SGD
        elif self.hparams.optimizer == 'Adam':
            opt = optim.Adam
        
        optimizer = opt(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)    
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=self.hparams.scheduler_rate)
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        dataset_name, dataset_path, cache_path, _ = self.hparams.dataset
        
        if dataset_name == 'synthetic':
            dataset = InvSyntheticDataset(dataset_path, cache_path, pixel=True, split='train',
                                          batch_size=self.hparams.batch_size, has_part=self.hparams.has_part)
        elif dataset_name == 'real':
            dataset = InvRealDataset(dataset_path, cache_path, pixel=True, split='train',
                                      batch_size=self.hparams.batch_size)
        elif dataset_name == 'sphere':
            dataset = SphereDataset(dataset_path, None, self.hparams.gt_path, split='train', 
                                    pixel=True, ray_diff=True)
       
        return DataLoader(dataset, batch_size=None, num_workers=self.hparams.num_workers)

    def on_train_epoch_start(self,):
        """ resample training batch """
        # self.train_dataloader().dataset
        return
    
    def val_dataloader(self,):
        dataset_name, dataset_path, cache_path, _ = self.hparams.dataset
        self.dataset_name = dataset_name

        if dataset_name == 'synthetic':
            dataset = SyntheticDataset(dataset_path,pixel=False,split='val')
        elif dataset_name == 'real':
            dataset = RealDataset(dataset_path,pixel=False,split='val')
        elif dataset_name == 'sphere':
            dataset = SphereDataset(dataset_path, None, self.hparams.gt_path, pixel=False,split='val')
        
        self.img_hw = dataset.img_hw
        return DataLoader(dataset, shuffle=False, batch_size=None, num_workers=self.hparams.num_workers)
    
    def test_dataloader(self,):
        dataset_name, dataset_path, cache_path, _ = self.hparams.dataset
        self.dataset_name = dataset_name

        if dataset_name == 'synthetic':
            dataset = SyntheticDataset(dataset_path,pixel=False,split='test')       
        elif dataset_name == 'real':
            dataset = RealDataset(dataset_path,pixel=False,split='test')
        elif dataset_name == 'sphere':
            dataset = SphereDataset(dataset_path, None, self.hparams.gt_path, pixel=False,split='test')
        
        self.img_hw = dataset.img_hw
        return DataLoader(dataset, shuffle=False, batch_size=None, num_workers=self.hparams.num_workers)

    def forward(self, points, view):
        return
    
    def training_step(self, batch, batch_idx):
        spp = 64
        SPP = 16
        rays, rgbs_gt = batch['rays'], batch['rgbs']
        rays_x, rays_d = rays[...,:3], rays[...,3:6]
        dxdu, dydv = rays[...,6:9], rays[...,9:12]
        
        L = torch.zeros_like(rays_x).to(rays_x.device)
        for _ in range(spp // SPP):
            L += path_tracing_fix_emitter(self.scene, self.emitter, self.material, rays_x, rays_d, dxdu, dydv, SPP, indir_depth=0)
        rgbs = L.reshape(*self.img_hw, -1) / (spp // SPP)
        
        loss_c = NF.mse_loss(self.gamma(rgbs), self.gamma(rgbs_gt))
        psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))
        
        self.log('train/loss', loss_c)
        self.log('train/psnr', psnr)
        return loss_c
    
    def validation_step(self, batch, batch_idx):
        spp = 1024
        SPP = 16
        rays, rgbs_gt = batch['rays'], batch['rgbs']
        rays_x, rays_d = rays[...,:3], rays[...,3:6]
        dxdu, dydv = rays[...,6:9], rays[...,9:12]
        
        L = torch.zeros_like(rays_x).to(rays_x.device)
        for _ in range(spp // SPP):
            L += path_tracing_fix_emitter(self.scene, self.emitter, self.material, rays_x, rays_d, dxdu, dydv, SPP, indir_depth=0)
        rgbs = L.reshape(*self.img_hw, -1) / (spp // SPP)
        
        loss_c = NF.mse_loss(self.gamma(rgbs), self.gamma(rgbs_gt))
        psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))
        
        self.log('val/loss', loss_c)
        self.log('val/psnr', psnr)
        return loss_c
    
    def test_step(self, batch, batch_idx):
        spp = self.hparams.test_spp
        SPP = 16
        rays, rgbs_gt = batch['rays'], batch['rgbs']
        rays_x, rays_d = rays[...,:3], rays[...,3:6]
        dxdu, dydv = rays[...,6:9], rays[...,9:12]
        
        L = torch.zeros_like(rays_x).to(rays_x.device)
        for _ in range(spp // SPP):
            L += path_tracing_fix_emitter(self.scene, self.emitter, self.material, rays_x, rays_d, dxdu, dydv, SPP, indir_depth=0)
        rgbs = L.reshape(*self.img_hw, -1) / (spp // SPP)
        
        loss_c = NF.mse_loss(self.gamma(rgbs), self.gamma(rgbs_gt))
        psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))
        
        self.log('test/psnr', psnr)
        torchvision.utils.save_image(
            gamma(rgbs.permute(2, 0, 1)),
            os.path.join('./Over-fit_experiments/Mar12_results_new', f'results_no-normal_output_{self.hparams.roughness:.2f}_{self.hparams.metallic:.2f}.png'),
        )   
        
        return psnr


            
def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            if(args['type'] == bool):
                parser.add_argument('--{}'.format(name), type=eval, choices=[True, False], default=str(args.get('default')))
            else:
                parser.add_argument('--{}'.format(name), **args)
        return parser
        
if __name__ == '__main__':
    torch.manual_seed(9)
    torch.cuda.manual_seed(9)

    parser = ArgumentParser()
    parser = add_model_specific_args(parser)
    hparams, _ = parser.parse_known_args()

    # Add program-level args
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--device', type=int, required=False, default=None)
    parser.add_argument('--logger', type=str, choices=['tensorboard', 'wandb'], default='wandb')
    parser.add_argument('--wandb_project', type=str, default='brdf-capture')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--test_spp', type=int, default=4096)
    parser.add_argument('--emitter', type=str, choices=['point', 'envmap'], default='point')
    
    parser.set_defaults(resume=False)
    args = parser.parse_args()
    args.gpus = [args.device]
    args.emitter = 'point'
    args.parameter_num = 1
    experiment_name = args.experiment_name

    checkpoint_path = Path(args.checkpoint_path) / experiment_name
    log_path = Path(args.log_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val/loss', save_top_k=1, save_last=True)
    lr_logger = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(log_path, name=experiment_name) if args.logger == 'tensorboard' else WandbLogger(
        project=args.wandb_project, name=experiment_name, entity=args.wandb_entity, save_dir=str(log_path)
    )

    last_ckpt = str(checkpoint_path / 'last.ckpt') if args.resume and (checkpoint_path / 'last.ckpt').exists() else None
    
    # Generate Rendered Images for Different Roughness & Metallic Values
    rendered_image_paths = {}
    output_dir = "./Over-fit_experiments/Mar12_results_new"
    os.makedirs(output_dir, exist_ok=True)
    
    for roughness in torch.linspace(0.20, 0.20, args.parameter_num):
        for metallic in torch.linspace(0.20, 0.20, args.parameter_num):
            renderer_args = {'roughness': roughness.item(), 'metallic': metallic.item(), 'emitter': args.emitter}
            renderer = ForwardRenderer(hparams, renderer_args)
            img = renderer.render()
            
            # Save original HDR image
            filename = f'output_{roughness:.2f}_{metallic:.2f}.png'
            output_path = os.path.join(output_dir, filename)
            torchvision.utils.save_image(
                img.permute(2, 0, 1),
                output_path,
            )
            rendered_image_paths[(roughness.item(), metallic.item())] = output_path
            
            # Save gamma-corrected PNG for visualization only
            vis_filename = f'output_No-normal_gamma_{roughness:.2f}_{metallic:.2f}.png'
            vis_path = os.path.join(output_dir, vis_filename)
            torchvision.utils.save_image(
                gamma(img.permute(2, 0, 1)),
                vis_path,
            )
    # Train Model for Each Rendered Image
    psnr_results = {}
    psnr_file = 'psnr_results.json'
    
    # Load existing results if file exists
    if os.path.exists(psnr_file):
        with open(psnr_file, 'r') as f:
            psnr_results = json.load(f)
    
    for (roughness, metallic), gt_path in tqdm(rendered_image_paths.items(), desc="Training models"):
        hparams.gt_path = gt_path  # Set path to ground truth image
        hparams.emitter = args.emitter
        hparams.roughness = roughness
        hparams.metallic = metallic
        hparams.test_spp = 2048
        model = ModelTrainer(hparams)
        trainer = pl.Trainer(
            accelerator='gpu', devices=[0], gpus=None, 
            logger=logger,
            callbacks=[checkpoint_callback, lr_logger],
            log_every_n_steps=50,
            max_epochs=args.max_epochs, 
            check_val_every_n_epoch=25
        )
        trainer.fit(model, ckpt_path=last_ckpt)
        
        # Test Step
        test_results = trainer.test(model)
        psnr = test_results[0]['test/psnr']
        psnr_results[f"{roughness:.2f}_{metallic:.2f}"] = psnr

        # Save all PSNR results after each iteration
        with open(psnr_file, 'w') as f:
            json.dump(psnr_results, f, indent=4)
    
    print('Training and Testing Complete!')
