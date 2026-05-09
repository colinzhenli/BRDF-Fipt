#!/bin/bash
# Debug script: test the zero-grazing-angle auxiliary supervision on the
# `ours` (Dataset_Nov11) trainer. Uses 3-material debug list, very few
# batches/epochs so the run finishes in seconds. Sets
# `model.grazing_ratio=0.05`; the helper draws latents uniformly across the
# whole bank and recovers material_id via searchsorted so the per-material
# camera_factor still applies.
#
# Verify in stdout:
#   - `train/recon_loss` and `train/grazing_loss` both finite
#   - run completes without exception
#
# Run: bash scripts/jobs/stacked/run_stage1_ours_debug_grazing.sh

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11 \
    data.training_list_path=/media/raid/cloth/capture_data/Dataset_Nov11/training_list_debug3.txt \
    output_folder=/tmp/brdf_debug_output \
    data=points_dense \
    data.legacy_swap_indexing=False \
    data.rays_num=50000 \
    data.point_subsample_ratio=0.1 \
    renderer=multiarea_emitter \
    material=multi_material_latent \
    experiment_name=DEBUG_grazing_ours \
    model.optimizer.name=Adam8bit \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.lr=2e-3 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.continue_training=False \
    model.trainer.enable_checkpointing=False \
    model.trainer.max_epochs=2 \
    model.trainer.check_val_every_n_epoch=1 \
    model.trainer.limit_train_batches=10 \
    model.trainer.log_every_n_steps=1 \
    model.grazing_ratio=0.05 \
    material.decoder.use_skip_connection=True \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False \
    data.switch_iters=100 \
    data.chunk_size=2 \
    renderer.spp.train=4
