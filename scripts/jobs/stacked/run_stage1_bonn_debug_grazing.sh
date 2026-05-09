#!/bin/bash
# Debug script: test the zero-grazing-angle auxiliary supervision on the Bonn
# trainer. Uses local paths, 1 material, very few batches/epochs so the run
# finishes in seconds. Sets `model.grazing_ratio=0.05` so each step adds
# int(0.05 * batch_size) synthetic samples drawn uniformly from the entire
# latent bank, with BRDF=0 supervision at cos(wi, N)=0.
#
# Verify in stdout:
#   - `grazing_ratio` echo when constructing the trainer
#   - `train/recon_loss` and `train/grazing_loss` both finite
#   - run completes without exception
#
# Run: bash scripts/jobs/stacked/run_stage1_bonn_debug_grazing.sh

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

python main.py \
    dataset_folder=/media/raid/cloth/Bonn_train \
    output_folder=/tmp/brdf_debug_output \
    data=bonn \
    data.num_load_workers=2 \
    data.use_pan=True \
    data.use_lls=True \
    data.debug=True \
    data.debug_num=1 \
    data.rays_num=8192 \
    data.point_subsample_ratio=0.1 \
    data.valid_num=2 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name=DEBUG_grazing_bonn \
    model.optimizer.name=Adam8bit \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.lr=2e-3 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.loss.pan_weight=1.0 \
    model.loss.lls_weight=1.0 \
    model.lls_spp=4 \
    model.stage=1 \
    model.apply_cosine_weight=False \
    model.test=False \
    model.continue_training=False \
    model.trainer.enable_checkpointing=False \
    model.trainer.max_epochs=2 \
    model.trainer.limit_train_batches=10 \
    model.trainer.check_val_every_n_epoch=1 \
    model.trainer.log_every_n_steps=1 \
    model.grazing_ratio=0.05 \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False
