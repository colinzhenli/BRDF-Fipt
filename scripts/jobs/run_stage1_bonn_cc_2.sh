#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

python main.py \
    dataset_folder=/home/zla247/scratch/data/Bonn/train \
    data=bonn \
    data.use_pan=True \
    data.use_lls=True \
    data.debug=False \
    data.debug_num=10 \
    data.subsample_ratio=0.05 \
    data.random_observations=True \
    data.rays_num=131072 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name=Stage-1_Fir_Bonn_Observations-Subsample-0.2-Chunk-100-Same-Decoder-lr-0.0005-Switch-10K-All-data_run_1 \
    model.loss.recon_loss.name=l2 \
    output_folder=/home/zla247/scratch/output/BRDF \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=3000 \
    model.optimizer.decoder_lr=0.0005 \
    model.trainer.limit_train_batches=512 \
    material.decoder.use_skip_connection=True \
    material.latent_dim=16 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.switch_iters=10000 \
    data.chunk_size=100 \
    data.filter_observations=False \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/Stage-1_ReLU_Overfit-Material-0_run_1/training/model_0.20_0.20/last.ckpt
