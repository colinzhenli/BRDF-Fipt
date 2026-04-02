#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

python main.py \
    dataset_folder=/home/zla247/scratch/data/Bonn/train \
    data=bonn \
    data.use_pan=False \
    data.use_lls=False \
    data.debug=False \
    data.debug_num=10 \
    data.subsample_ratio=0.02 \
    data.random_observations=True \
    data.rays_num=131072 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name=Stage-1_Fir_Softplus_Bonn_Observations_SparseAdam-Optimizer-Same-chunk-as-ours_RGB-only_run_1 \
    output_folder=/home/zla247/scratch/output/BRDF \
    model.optimizer.name=SparseAdam \
    model.loss.recon_loss.name=l2 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=3000 \
    model.optimizer.decoder_lr=0.001 \
    model.trainer.limit_train_batches=512 \
    material.decoder.use_skip_connection=True \
    material.latent_dim=16 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.switch_iters=2000 \
    data.chunk_size=20 \
    data.filter_observations=False \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/Stage-1_ReLU_Overfit-Material-0_run_1/training/model_0.20_0.20/last.ckpt

