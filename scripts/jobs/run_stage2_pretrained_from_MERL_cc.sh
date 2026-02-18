#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1

python main.py \
    dataset_folder=/home/zla247/scratch/data/debug/242 \
    data=real \
    renderer=multiarea_emitter \
    material=ani_latent_texture_model \
    material.learnable_factor=True \
    material.mono_brdf=False \
    material.different_decoder=False \
    material.neural_geometry.factor=0.04 \
    material.decoder.use_skip_connection=True \
    experiment_name=Stage-2_Fir_From-MERL_Material_Fixed-Val-242_run_1 \
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
    model.ckpt_path=/home/zla247/scratch/checkpoints/Stage1_from_merl.ckpt