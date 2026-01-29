#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Debug_Dataset_Jan20/105 \
    data=real \
    renderer=multiarea_emitter \
    material=ani_latent_texture_model \
    material.different_decoder=False \
    material.neural_geometry.factor=0.02 \
    material.decoder.use_skip_connection=True \
    experiment_name=Stage-2_Latent-dim-16_Degree-5_Single-Head-decoder_Neural-geometry-factor-0.02_Initialize-from-std_Leaky-ReLU_Material-105_run_1 \
    model.stage=2 \
    model.test=False \
    material.use_latent_bank=False \
    material.latent_dim=16 \
    material.decoder.degree=5 \
    model.freeze_decoder=True \
    data.switch_iters=3000 \
    data.debug=False \
    model.ckpt_path=/media/raid/cloth/output/BRDF/points/L2_All-materials_Latent-dim-16_Large-batch_training_Single-color-head_degree-5_run_2/training/model_0.20_0.20/last.ckpt

