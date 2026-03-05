#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Debug_Feb10/227 \
    data=real \
    data.use_fixed_val=True \
    data.valid_num=100 \
    data.hold_out_val_num=240 \
    renderer=multiarea_emitter \
    material=ani_latent_texture_model \
    material.learnable_factor=True \
    material.mono_brdf=False \
    material.different_decoder=True \
    material.neural_geometry.factor=0.04 \
    material.decoder.use_skip_connection=True \
    experiment_name=Stage-2_From-Bonn_Material_Fixed-Val-227_run_1 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    material.use_latent_bank=False \
    material.latent_dim=16 \
    material.decoder.degree=3 \
    model.freeze_decoder=True \
    data.switch_iters=3000 \
    data.chunk_size=200 \
    data.debug=False \
    model.ckpt_path=/media/raid/cloth/output/BRDF/bonn_Theia-2/Stage-1_Bonn_Observations-Subsample-0.05-Chunk-20-Decoder-lr-0.0003-Switch-2K-RGB-only_run_1/training/model_0.20_0.20/last.ckpt
