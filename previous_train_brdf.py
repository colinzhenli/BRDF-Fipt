class ModelTrainer(pl.LightningModule):
    """ BRDF-emission mask training code """
    def __init__(self, hparams: Namespace, *args, **kwargs):
        super(ModelTrainer, self).__init__()
        self.save_hyperparameters(hparams)
        
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
        # self.material = PBRBRDF()
        self.material = MLPPBRBRDF()
        # self.emitter = EnvMapEmitter('envmap.exr')
        if args['emitter'] == 'point':
            self.emitter = PointEmitter(position=torch.tensor([-2, 2.0, 0.0]),
                        intensity=torch.tensor([50.0, 50.0, 50.0]),
                        radius=0.1)
        elif args['emitter'] == 'envmap':
            self.emitter = EnvMapEmitter('envmap.exr')
       
        
    def __repr__(self):
        return repr(self.hparams)

    def configure_optimizers(self):
        if(self.hparams.optimizer == 'SGD'):
            opt = optim.SGD
        if(self.hparams.optimizer == 'Adam'):
            opt = optim.Adam
        
        optimizer = opt(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)    
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.hparams.milestones,gamma=self.hparams.scheduler_rate)
        return [optimizer], [scheduler]
    
    def train_dataloader(self,):
        dataset_name,dataset_path,cache_path,gt_img_path = self.hparams.dataset
        
        if dataset_name == 'synthetic':
            dataset = InvSyntheticDataset(dataset_path,cache_path,pixel=True,split='train',
                                       batch_size=self.hparams.batch_size,has_part=self.hparams.has_part)
        elif dataset_name == 'real':
            dataset = InvRealDataset(dataset_path,cache_path,pixel=True,split='train',
                                       batch_size=self.hparams.batch_size)
        elif dataset_name == 'sphere':
            dataset = SphereDataset(dataset_path, gt_img_path, split='train', 
                                  pixel=True, ray_diff=True)
       
        return DataLoader(dataset, batch_size=None, num_workers=self.hparams.num_workers)
       
    def on_train_epoch_start(self,):
        """ resample training batch """
        # self.train_dataloader().dataset
        return
    
    def val_dataloader(self,):
        dataset_name,dataset_path,cache_path, gt_img_path = self.hparams.dataset
        self.dataset_name = dataset_name

        if dataset_name == 'synthetic':
            dataset = SyntheticDataset(dataset_path,pixel=False,split='val')
        elif dataset_name == 'real':
            dataset = RealDataset(dataset_path,pixel=False,split='val')
        elif dataset_name == 'sphere':
            dataset = SphereDataset(dataset_path, gt_img_path, pixel=False,split='val')
        
        self.img_hw = dataset.img_hw
        return DataLoader(dataset, shuffle=False, batch_size=None, num_workers=self.hparams.num_workers)

    def forward(self, points, view):
        return

    def gamma(self,x):
        """ tone mapping function """
        mask = x <= 0.0031308
        ret = torch.empty_like(x)
        ret[mask] = 12.92*x[mask]
        mask = ~mask
        ret[mask] = 1.055*x[mask].pow(1/2.4) - 0.055
        return ret
    
    def training_step(self, batch, batch_idx):
        """ one training step """
        spp = 64
        SPP = 16
        rays,rgbs_gt = batch['rays'], batch['rgbs']
        rays_x = rays[...,:3]
        rays_d = rays[...,3:6]
        dxdu, dydv = rays[...,6:9], rays[...,9:12]
        
        L = torch.zeros_like(rays_x).to(rays_x.device)
        for _ in range(spp//SPP):
            L += path_tracing_envmap_emitter(self.scene, self.emitter, self.material, rays_x, rays_d, dxdu, dydv, SPP, indir_depth=0)
        rgbs = L.reshape(*self.img_hw,-1)/(spp//SPP)
        # tonemapped mse loss
        loss_c = NF.mse_loss(self.gamma(rgbs),self.gamma(rgbs_gt))
        # vsualize rendering brdf
        psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))
        loss = loss_c
        self.log('train/loss', loss)
        self.log('train/psnr', psnr)
        # loss.backward()

        # # **Gradient Checking**
        # print("=== Gradient Checking ===")
        # for name, param in self.material.named_parameters():
        #     if param.grad is None:
        #         print(f"{name}: No gradient!")
        #     else:
        #         print(f"{name}: grad mean={param.grad.mean().item():.6f}, max={param.grad.abs().max().item():.6f}")
                
        #         # Check for NaNs or Infs
        #         if torch.isnan(param.grad).any():
        #             print(f"🚨 NaN detected in {name} gradient!")
        #         if torch.isinf(param.grad).any():
        #             print(f"🚨 Inf detected in {name} gradient!")

        return loss
    
    def validation_step(self, batch, batch_idx):
        """ one validation step """
        spp = 1024
        SPP = 16
        rays,rgbs_gt = batch['rays'], batch['rgbs']
        rays_x = rays[...,:3]
        rays_d = rays[...,3:6]
        dxdu, dydv = rays[...,6:9], rays[...,9:12]
        
        L = torch.zeros_like(rays_x).to(rays_x.device)
        for _ in range(spp//SPP):
            L += path_tracing_envmap_emitter(self.scene, self.emitter, self.material, rays_x, rays_d, dxdu, dydv, SPP, indir_depth=0)
        rgbs = L.reshape(*self.img_hw,-1)/(spp//SPP)
        # tonemapped mse loss
        loss_c = NF.mse_loss(self.gamma(rgbs),self.gamma(rgbs_gt))
        # vsualize rendering brdf
        psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))
        loss = loss_c
        self.log('val/loss', loss)
        self.log('val/psnr', psnr)
        # Save gamma-corrected rendered and ground truth images
        gamma_rgbs = self.gamma(rgbs).reshape(*self.img_hw,3).permute(2, 0, 1).clamp(0,1)
        gamma_rgbs_gt = self.gamma(rgbs_gt).reshape(*self.img_hw,3).permute(2, 0, 1).clamp(0,1)
        
        # # Save rendered and ground truth images to disk
        # os.makedirs('images', exist_ok=True)
        # plt.imsave(f'images_2/rendered_epoch{self.current_epoch}_batch{batch_idx}.png', gamma_rgbs.permute(1,2,0).cpu().numpy())
        # plt.imsave(f'images_2/gt_epoch{self.current_epoch}_batch{batch_idx}.png', gamma_rgbs_gt.permute(1,2,0).cpu().numpy())
        # """ visualize diffuse reflectance kd
        # """
        # rays,rgb_gt = batch['rays'], batch['rgbs']
        # if self.dataset_name == 'synthetic':
        #     emission_mask_gt = batch['emission'].mean(-1,keepdim=True) == 0
        # else:
        #     emission_mask_gt = torch.ones_like(rays[...,:1])
        # rays_x = rays[:,:3]
        # rays_d = NF.normalize(rays[:,3:6],dim=-1)

        # positions,normals,_,_,valid = ray_intersect(self.scene,rays_x,rays_d)
        # position = positions[valid]

        # # batched rendering diffuse reflectance
        # B = valid.sum()
        # batch_size = 10240
        # albedo_ = []
        # for b in range(math.ceil(B*1.0/batch_size)):
        #     b0 = b*batch_size
        #     b1 = min(b0+batch_size,B)
        #     mat = self.material(position[b0:b1])
        #     albedo_.append(mat['albedo']*(1-mat['metallic']))
        # albedo_ = torch.cat(albedo_)
        # albedo = torch.zeros(len(valid),3,device=valid.device)
        # albedo[valid] = albedo_
        
        # if self.dataset_name == 'synthetic':
        #     albedo_gt = batch['albedo']
        # else: # show rgb is no ground truth kd
        #     albedo_gt = rgb_gt.pow(1/2.2).clamp(0,1)

        # # mask out emissive regions
        # albedo = albedo*emission_mask_gt
        # albedo_gt = albedo_gt * emission_mask_gt
        # loss_c = NF.mse_loss(albedo_gt,albedo)
        
        # loss = loss_c
        # psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))
        
        
        # self.log('val/loss', loss)
        # self.log('val/psnr', psnr)

        # self.logger.experiment.add_image('val/gt_image', albedo_gt.reshape(*self.img_hw,3).permute(2, 0, 1).clamp(0,1), batch_idx)
        # self.logger.experiment.add_image('val/inf_image', albedo.reshape(*self.img_hw,3).permute(2, 0, 1).clamp(0,1), batch_idx)
        return

            
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

    # add PROGRAM level args
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--ft', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--device', type=int, required=False,default=None)
    parser.add_argument('--logger', type=str, choices=['tensorboard', 'wandb'], default='wandb')
    parser.add_argument('--wandb_project', type=str, default='brdf-capture')
    parser.add_argument('--wandb_entity', type=str, default=None)

    parser.set_defaults(resume=False)
    args = parser.parse_args()
    args.gpus = [args.device]
    experiment_name = args.experiment_name

    # setup checkpoint loading
    checkpoint_path = Path(args.checkpoint_path) / experiment_name
    log_path = Path(args.log_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val/loss', save_top_k=1, save_last=True)
    
    # Setup logger
    if args.logger == 'tensorboard':
        logger = TensorBoardLogger(log_path, name=experiment_name)
    else:  # wandb
        logger = WandbLogger(
            project=args.wandb_project,
            name=experiment_name,
            entity=args.wandb_entity,
            save_dir=str(log_path)
        )

    last_ckpt = checkpoint_path / 'last.ckpt' if args.resume else None
    if (last_ckpt is None) or (not (last_ckpt.exists())):
        last_ckpt = None
    else:
        last_ckpt = str(last_ckpt)
    
    # setup model trainer
    model = ModelTrainer(hparams)
    
    # trainer = Trainer.from_argparse_args(
    #     args,
    #     resume_from_checkpoint=last_ckpt,
    #     logger=logger,
    #     checkpoint_callback=checkpoint_callback,
    #     flush_logs_every_n_steps=1,
    #     log_every_n_steps=1,
    #     max_epochs=args.max_epochs
    # )

    # trainer.fit(model)
    
    # Update to lightning 1.9
    trainer = Trainer.from_argparse_args(
        args,
        accelerator='gpu', devices=[0], gpus=None, 
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50,
        max_epochs=args.max_epochs, 
        check_val_every_n_epoch=1
    )

    start_time = time.time()
    
    trainer.fit(
        model, 
        ckpt_path=last_ckpt, 
        )
    
    print('[train - BRDF-emission] time (s): ', time.time()-start_time)
output the modified MolelTrainer first