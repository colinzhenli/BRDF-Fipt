#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python main.py \
    dataset_folder=/media/raid/cloth/Bonn_train \
    data=bonn \
    data.use_pan=False \
    data.use_lls=False \
    data.debug=True \
    data.debug_num=10 \
    data.rays_num=131072 \
    data.subsample_ratio=1.0 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name=Stage-1_SparseAdam_Bonn_Overfit-10-materials-Decoder-lr-0.01-0.001-All-pixels-no-switch-RGB-only_run_1 \
    model.optimizer.decoder_lr=0.001 \
    model.optimizer.lr=0.01 \
    model.optimizer.name=SparseAdam \
    model.loss.recon_loss.name=l2 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=3000 \
    model.trainer.limit_train_batches=512 \
    material.decoder.use_skip_connection=True \
    material.latent_dim=16 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=True \
    data.switch_iters=400000 \
    data.chunk_size=10 \
    data.filter_observations=False \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/Stage-1_ReLU_Overfit-Material-0_run_1/training/model_0.20_0.20/last.ckpt

