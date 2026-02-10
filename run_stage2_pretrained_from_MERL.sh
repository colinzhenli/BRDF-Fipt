#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11/227 \
    data=real \
    renderer=multiarea_emitter \
    material=ani_latent_texture_model \
    material.learnable_factor=True \
    material.mono_brdf=False \
    material.different_decoder=False \
    material.neural_geometry.factor=0.04 \
    material.decoder.use_skip_connection=True \
    experiment_name=Stage-2_Continue-training_From-Real_Latent-dim-16_Degree-5_Material-227_run_1 \
    model.stage=2 \
    model.test=False \
    model.continue_training=True \
    material.use_latent_bank=False \
    material.latent_dim=16 \
    material.decoder.degree=5 \
    model.freeze_decoder=True \
    data.switch_iters=2000 \
    data.chunk_size=100 \
    data.debug=False \
    model.ckpt_path=/media/raid/cloth/output/BRDF/real/Stage-2_From-Real_Latent-dim-16_Degree-5_Material-227_run_1/training/model_0.20_0.20/last.ckpt