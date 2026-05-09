#!/bin/bash
#
# Debug script: test poly + pan training on 1 material from scratch.
# Verifies the pan data loading and panchromatic loss pipeline.
#
# Expected output:
#   - Data loading should report poly=100, pan=288 images for mat0001 (lls disabled)
#   - Training logs should show train/total_loss decreasing
#   - No crashes from pan data_type handling
#

export CUDA_VISIBLE_DEVICES=3

python main.py \
    dataset_folder=/media/raid/cloth/Bonn_train \
    data=bonn \
    data.use_pan=True \
    data.use_lls=False \
    data.debug=True \
    data.debug_num=1 \
    data.rays_num=8192 \
    data.valid_num=15 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name=DEBUG_stage1_pan-only_test \
    model.optimizer.name=Adam8bit \
    model.loss.recon_loss.name=logrel \
    # model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.recon_loss.log_space.logrel_ref=0.02 \
    model.loss.reg_loss.weight=0.0 \
    model.loss.pan_weight=0.5 \
    model.loss.lls_weight=0.0 \
    model.lls_spp=16 \
    model.stage=1 \
    model.apply_cosine_weight=False \
    model.test=False \
    model.continue_training=False \
    model.trainer.enable_checkpointing=False \
    model.trainer.max_epochs=400 \
    model.trainer.limit_train_batches=512 \
    model.trainer.check_val_every_n_epoch=10 \
    model.optimizer.decoder_lr=1e-4 \
    model.optimizer.lr=1e-3 \
    material.latent_dim=16 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.different_decoder=False \
    data.filter_observations=False
