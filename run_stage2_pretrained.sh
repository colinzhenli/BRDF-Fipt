#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11/227 \
    data=real \
    renderer=multiarea_emitter \
    material=learnable_pbr_texture_model \
    material.learnable_factor=True \
    material.mono_brdf=False \
    material.different_decoder=False \
    material.neural_geometry.factor=0.02 \
    material.decoder.use_skip_connection=True \
    experiment_name=Stage-2_Isotropic-PBR_Material-227_run_1 \
    model.stage=2 \
    model.test=False \
    material.use_latent_bank=False \
    material.latent_dim=16 \
    material.decoder.degree=3 \
    model.freeze_decoder=False \
    data.switch_iters=2000 \
    data.chunk_size=100 \
    data.debug=False \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/L2_All-materials_training_Single-color-head_run_1/training/model_0.20_0.20/last.ckpt