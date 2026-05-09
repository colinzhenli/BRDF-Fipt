"""Smoke + equivalence test for DDP on Stage-1 Bonn training.

What it does:
  1. Builds the same model + trainer + dataset stack main.py uses, but minimal.
  2. Runs trainer.fit for a few steps with debug materials and a tiny rays_num.
  3. Each rank dumps its final state_dict; rank 0 verifies all ranks match.

Why a separate driver: avoids hydra+wandb setup in main.py and lets us inject
state-dump verification without touching production code.

Usage:
  # single-GPU baseline (writes /tmp/ddp_bonn_test/state_devices1_rank0.pt)
  CUDA_VISIBLE_DEVICES=1 python scripts/test_ddp_bonn.py devices=1

  # DDP-2 (writes ..._devices2_rank{0,1}.pt; rank 0 prints PASS/FAIL)
  CUDA_VISIBLE_DEVICES=1,2 python scripts/test_ddp_bonn.py devices=2
"""
import importlib
import os
import sys
import time
import warnings
import logging
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from utils.dataset import BonnDataset
from trainers import get_trainer_class
from model.brdf import SvPBRBRDF


def _parse_args():
    args = {
        'devices': 1,
        'rays_num': 512,
        'debug_num': 2,
        'max_steps': 5,
        'dump_dir': '/tmp/ddp_bonn_test',
        'dataset_folder': '/media/raid/cloth/Bonn_train',
        'use_pan': True,
        'use_lls': True,
    }
    for a in sys.argv[1:]:
        if '=' not in a:
            continue
        k, v = a.split('=', 1)
        if k in args:
            cur = args[k]
            if isinstance(cur, bool):
                args[k] = v.lower() in ('1', 'true', 'yes')
            elif isinstance(cur, int):
                args[k] = int(v)
            else:
                args[k] = v
    return args


def _compose_cfg(devices, rays_num, debug_num, max_steps, use_pan, use_lls,
                 dataset_folder, exp_root):
    """Compose the same config main.py would build for stage-1 bonn, with test
    overrides: very small batch, few materials, no checkpoint/wandb."""
    config_dir = str(REPO / 'config')
    overrides = [
        'data=bonn',
        f'dataset_folder={dataset_folder}',
        'data.debug=True',
        f'data.debug_num={debug_num}',
        f'data.use_pan={use_pan}',
        f'data.use_lls={use_lls}',
        f'data.rays_num={rays_num}',
        'data.num_load_workers=4',
        'data.filter_observations=False',
        'renderer=multiarea_emitter',
        'material=bonn_latent',
        'material.decoder.use_skip_connection=True',
        'material.decoder.use_film=False',
        'material.decoder.use_color_decomp=False',
        'material.latent_dim=24',
        'material.decoder.degree=3',
        'material.decoder.smooth_reg=False',
        'material.different_decoder=False',
        'model.stage=1',
        'model.test=False',
        'model.continue_training=False',
        'model.optimizer.name=Adam',
        'model.optimizer.decoder_lr=1e-4',
        'model.loss.recon_loss.name=logrel',
        # 'model.loss.recon_loss.log_space.logrel_ref=0.05',
        'model.loss.recon_loss.log_space.logrel_ref=0.02',
        'model.loss.reg_loss.weight=0.0',
        'model.loss.pan_weight=0.5',
        'model.loss.lls_weight=0.5',
        'model.lls_spp=4',
        f'model.trainer.devices={devices}',
        'model.trainer.max_epochs=1',
        f'model.trainer.limit_train_batches={max_steps}',
        'model.trainer.check_val_every_n_epoch=999',
        'model.trainer.enable_checkpointing=False',
        'model.trainer.num_sanity_val_steps=0',
        'experiment_name=ddp_bonn_test',
        f'exp_output_root_path={exp_root}',
    ]
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = hydra.compose(config_name='config', overrides=overrides)
        gt_material_cfg = hydra.compose(
            config_name='config', overrides=['material=svpbr']).material
    return cfg, gt_material_cfg


def main():
    args = _parse_args()
    devices = int(args['devices'])
    rays_num = int(args['rays_num'])
    debug_num = int(args['debug_num'])
    max_steps = int(args['max_steps'])
    dump_dir = Path(args['dump_dir'])
    dump_dir.mkdir(parents=True, exist_ok=True)
    exp_root = str(dump_dir / 'exp')

    pl.seed_everything(42, workers=True)

    cfg, gt_material_cfg = _compose_cfg(
        devices, rays_num, debug_num, max_steps,
        args['use_pan'], args['use_lls'], args['dataset_folder'], exp_root,
    )

    # Build material + trainer module the way main.py does.
    material_module = importlib.import_module(cfg.material.module)
    material_class = getattr(material_module, cfg.material.type)
    material = material_class(cfg.material)
    gt_material = SvPBRBRDF(
        cfg=gt_material_cfg,
        albedo=torch.tensor(gt_material_cfg.albedo),
    )
    TrainerClass = get_trainer_class(cfg.model.stage, cfg.data.dataset_name)
    model = TrainerClass(cfg, material, gt_material,
                         gt_material_cfg.roughness, gt_material_cfg.metallic)

    train_dataset = BonnDataset(cfg, root_folder=cfg.dataset_folder, split='train')
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0,
                              pin_memory=False)

    # Strategy: 'auto' = single device, DDP for >1.
    # find_unused_parameters=True is conservative because this trainer logs
    # learnable_factor/factor params only when present, and DDP otherwise
    # complains when a param has no grad on the backward pass.
    if devices > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = 'auto'

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=devices,
        strategy=strategy,
        max_epochs=1,
        limit_train_batches=max_steps,
        check_val_every_n_epoch=999,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        logger=False,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    print(f"\n=== DDP test: devices={devices}, strategy={strategy} ===")
    t0 = time.time()
    trainer.fit(model, train_loader)
    elapsed = time.time() - t0

    rank = trainer.global_rank
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    out = dump_dir / f'state_devices{devices}_rank{rank}.pt'
    torch.save({'state_dict': state, 'elapsed': elapsed}, out)
    print(f"[rank {rank}] elapsed={elapsed:.1f}s, dumped {len(state)} tensors → {out}")

    # Cross-rank consistency check (only meaningful when devices > 1).
    if devices > 1:
        # Make sure every rank has finished writing before rank 0 reads.
        torch.distributed.barrier()
        if rank == 0:
            ranks_states = []
            for r in range(devices):
                p = dump_dir / f'state_devices{devices}_rank{r}.pt'
                ranks_states.append(torch.load(p, map_location='cpu')['state_dict'])
            ref = ranks_states[0]
            max_diff = 0.0
            worst_key = None
            mismatched_keys = 0
            for r in range(1, devices):
                sd = ranks_states[r]
                missing = set(ref) - set(sd)
                if missing:
                    print(f"[FAIL] keys missing in rank {r}: {sorted(missing)[:5]}")
                    mismatched_keys += len(missing)
                    continue
                for k in ref:
                    a, b = ref[k], sd[k]
                    if a.shape != b.shape:
                        print(f"[FAIL] shape mismatch {k}: {a.shape} vs {b.shape}")
                        mismatched_keys += 1
                        continue
                    d = (a.float() - b.float()).abs().max().item()
                    if d > max_diff:
                        max_diff = d
                        worst_key = k
            verdict = "PASS" if max_diff < 1e-5 and mismatched_keys == 0 else "FAIL"
            print(f"\n=== CROSS-RANK WEIGHT CHECK [{verdict}]  "
                  f"max_diff={max_diff:.2e}  worst_key={worst_key}  "
                  f"mismatched_keys={mismatched_keys} ===")


if __name__ == '__main__':
    main()
