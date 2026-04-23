#!/bin/bash
# Stage-1 Bonn training, DDP on 2 GPUs.
#
# Mirrors run_stage1_bonn.sh exactly except for:
#   - CUDA_VISIBLE_DEVICES=1,2  (use idle GPUs; GPU 0 is shared)
#   - model.trainer.devices=2
#   - model.trainer.strategy=ddp_find_unused_parameters_true
#   - data.num_load_workers=16  (parallel cold load; ~3-4x speedup at full scale)
#
# Notes:
#   - Effective batch size = devices * data.rays_num. Consider scaling decoder_lr
#     by sqrt(devices) when going to >2 GPUs.
#   - limit_train_batches is per-GPU, so 2 GPUs means 2x the actual samples per
#     epoch. Reduce it (or max_epochs) if you want to keep total work constant.

export CUDA_VISIBLE_DEVICES=0,1

python main.py \
    dataset_folder=/media/raid/cloth/Bonn_train \
    data=bonn \
    data.use_pan=True \
    data.use_lls=True \
    data.debug=False \
    data.debug_num=10 \
    data.rays_num=65535 \
    data.num_load_workers=32 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name=Stage-1_Bonn_DDP2_debug_run_2 \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.name=Adam8bit \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.loss.pan_weight=0.5 \
    model.loss.lls_weight=0.5 \
    model.lls_spp=4 \
    model.stage=1 \
    model.test=False \
    model.continue_training=False \
    model.trainer.max_epochs=3000 \
    model.trainer.devices=2 \
    model.trainer.strategy=ddp_find_unused_parameters_true \
    model.optimizer.decoder_lr=1e-4 \
    model.trainer.limit_train_batches=512 \
    model.trainer.check_val_every_n_epoch=20 \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False

