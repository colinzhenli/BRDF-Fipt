#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python main.py \
    dataset_folder=/media/raid/cloth/Bonn_train \
    data=bonn \
    renderer=multiarea_emitter \
    material=bonn_latent \
    material.different_decoder=False \
    material.decoder.use_skip_connection=True \
    material.learnable_factor=True \
    experiment_name=Stage-2_Bonn-3_From-Real_run_1 \
    model.stage=2 \
    model.test=False \
    material.latent_dim=16 \
    material.decoder.degree=3 \
    material.learnable_factor=True \
    model.freeze_decoder=True \
    data.debug=False \
    data.valid_on_train_set=False \
    model.ckpt_path=/media/raid/cloth/output/BRDF/points/All-materials_training_Single-color-head_run_1/training/model_0.20_0.20/last.ckpt

