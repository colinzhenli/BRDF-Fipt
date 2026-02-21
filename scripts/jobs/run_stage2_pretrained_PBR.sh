#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11/242 \
    data=real \
    renderer=multiarea_emitter \
    material=learnable_pbr_texture_model \
    material.learnable_factor=True \
    material.mono_brdf=False \
    material.disney=True \
    material.different_decoder=False \
    material.neural_geometry.factor=0.04 \
    material.decoder.use_skip_connection=True \
    experiment_name=Theia-2_Stage-2_Disney_242_run_1 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    material.use_latent_bank=False \
    material.latent_dim=16 \
    material.decoder.degree=3 \
    model.freeze_decoder=False \
    data.switch_iters=3000 \
    data.chunk_size=200 \
    data.debug=False \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/real/Visualizations/Real/real/Stage-2_Anisotropic_PBR_Material-227_run_1/training/model_0.20_0.20/last-v1.ckpt