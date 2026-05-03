"""Stage-2 test entry point.

Loads a trained stage-2 checkpoint, builds the matching trainer + val dataset,
and runs Lightning's validation epoch on the held-out views. Metrics
(val/loss, val/psnr) are aggregated over the FULL validation set; per-view
images and JSON for the first ``data.valid_num`` views are written by the
trainer's validation_step.

Currently supports the UBO BTF dataset only (data.dataset_name=ubo).
"""

import os
import json
import importlib
import warnings
import logging

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import hydra

from trainers import get_trainer_class
from model.brdf import SvPBRBRDF
from utils.dataset import UBOBTFValDataset

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def _build_val_dataset(cfg):
    if cfg.data.dataset_name == 'ubo':
        btf_path = os.path.join(cfg.dataset_folder, cfg.data.btf_filename)
        return UBOBTFValDataset(cfg, btf_path=btf_path)
    raise ValueError(
        f"test.py currently supports data.dataset_name=ubo only "
        f"(got '{cfg.data.dataset_name}')")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    pl.seed_everything(cfg.global_train_seed, workers=True)
    os.makedirs(cfg.exp_output_root_path, exist_ok=True)

    stage = cfg.model.get('stage', 2)
    if stage != 2:
        raise ValueError(f"test.py supports stage 2 only (got stage={stage})")

    gt_material_cfg = hydra.compose(
        config_name="config", overrides=["material=svpbr"]).material
    albedo    = gt_material_cfg.albedo
    roughness = gt_material_cfg.roughness
    metallic  = gt_material_cfg.metallic

    material_module = importlib.import_module(cfg.material.module)
    material_class  = getattr(material_module, cfg.material.type)
    material        = material_class(cfg.material)
    gt_material     = SvPBRBRDF(cfg=gt_material_cfg, albedo=torch.tensor(albedo))

    TrainerClass = get_trainer_class(stage, cfg.data.get('dataset_name', 'default'))
    print(f"==> trainer: {TrainerClass.__name__}")
    model = TrainerClass(cfg, material, gt_material, roughness, metallic)

    ckpt_path = cfg.model.ckpt_path
    if not (ckpt_path and os.path.isfile(ckpt_path)):
        raise FileNotFoundError(f"model.ckpt_path missing or not a file: {ckpt_path}")
    print(f"==> loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_dict = model.state_dict()
    filtered = {
        k: v for k, v in checkpoint['state_dict'].items()
        if 'emitter' not in k and k in model_dict
    }
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    print(f"==> loaded {len(filtered)}/{len(checkpoint['state_dict'])} parameters "
          f"(emitter keys excluded)")

    val_dataset = _build_val_dataset(cfg)
    val_loader  = DataLoader(
        val_dataset, batch_size=1, num_workers=cfg.data.num_workers)

    trainer_kwargs = dict(cfg.model.trainer)
    for k in ('max_epochs', 'max_steps', 'check_val_every_n_epoch',
              'limit_train_batches', 'num_sanity_val_steps', 'log_every_n_steps'):
        trainer_kwargs.pop(k, None)
    trainer_kwargs['enable_checkpointing'] = False

    print(f"==> running validation on {len(val_dataset)} held-out views ...")
    trainer = pl.Trainer(logger=False, **trainer_kwargs)
    results = trainer.validate(model, dataloaders=val_loader, verbose=False)

    print("\n" + "=" * 60)
    print(f"Test results — full validation set ({len(val_dataset)} views)")
    print("=" * 60)
    if results:
        for k, v in results[0].items():
            print(f"  {k}: {v:.6f}")
        out_json = os.path.join(cfg.exp_output_root_path, 'test_results.json')
        with open(out_json, 'w') as f:
            json.dump({
                'ckpt_path':   ckpt_path,
                'num_views':   len(val_dataset),
                'metrics':     results[0],
            }, f, indent=2)
        print(f"\nAggregated metrics → {out_json}")
    print(f"Per-view images / JSON (first {cfg.data.valid_num}): "
          f"{os.path.join(cfg.exp_output_root_path, 'images')}")
    print("=" * 60)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
