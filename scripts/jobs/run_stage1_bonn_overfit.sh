#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python main.py \
    dataset_folder=/media/raid/cloth/Bonn_train \
    data=bonn \
    data.rays_num=65536 \
    data.random_observations=True \
    data.subsample_ratio=1.0 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name=Mean-value_track_lr-1e-4-1e-3_LeakyReLU_l2_run_1 \
    model.optimizer.name=SparseAdam \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.stage=1 \
    model.test=False \
    model.loss.recon_loss.name=l2 \
    model.trainer.limit_train_batches=512 \
    model.loss.reg_loss.weight=0.0 \
    material.decoder.smooth_reg=False \
    material.decoder.smooth_reg_eps=0.01 \
    material.decoder.use_skip_connection=True \
    data.chunk_size=1 \
    data.switch_iters=20000 \
    data.debug_num=1 \
    data.filter_observations=False \
    data.debug=True
