#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Debug_Feb10/227 \
    data=real \
    renderer=multiarea_emitter \
    material=ani_latent_texture_model \
    material.learnable_factor=True \
    material.mono_brdf=False \
    material.different_decoder=True \
    material.neural_geometry.factor=0.04 \
    material.decoder.use_skip_connection=True \
    experiment_name=Theia-1_Stage-2_Train-on-all-data_Correct-4000K-light_Optimize-different-decoder_From-Real_Latent-dim-16_Degree-5_Fixed-Val-227_run_1 \
    model.stage=2 \
    model.test=False \
    model.continue_training=True \
    material.use_latent_bank=False \
    material.latent_dim=16 \
    material.decoder.degree=5 \
    model.freeze_decoder=False \
    data.switch_iters=3000 \
    data.chunk_size=200 \
    data.debug=False \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/L2_All-materials_Latent-dim-16_Large-batch_training_Single-color-head_degree-5_run_2/training/model_0.20_0.20/last.ckpt