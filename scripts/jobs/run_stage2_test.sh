#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11/227 \
    data=real \
    renderer=multiarea_emitter \
    material=learnable_pbr_texture_model \
    material.different_decoder=False \
    material.decoder.use_skip_connection=True \
    material.learnable_factor=True \
    experiment_name=Test_Ani-PBR_Material-119_run_1 \
    model.stage=2 \
    model.test=True \
    material.use_latent_bank=False \
    material.neural_geometry.factor=0.02 \
    model.freeze_decoder=True \
    data.switch_iters=3000 \
    data.debug=False \
    data.valid_on_train_set=False \
    model.ckpt_path=/media/raid/cloth/output/BRDF/real/Stage-2_Anisotropic_Baseline-PBR_Material-227_run_1/training/model_0.20_0.20/last.ckpt