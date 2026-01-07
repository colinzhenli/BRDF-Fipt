#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11/119 \
    data=real \
    renderer=multiarea_emitter \
    material=ani_latent_texture_model \
    material.different_decoder=True \
    material.decoder.use_skip_connection=True \
    experiment_name=Multi-Head-decoder_Add-boundary_Leaky-ReLU_Stage-2-Skip-connection-from-pretrained-decoder_Material-119_run_1 \
    model.stage=2 \
    model.test=False \
    material.use_latent_bank=False \
    model.freeze_decoder=True \
    data.switch_iters=3000 \
    data.debug=False \
    model.ckpt_path=/media/raid/cloth/output/BRDF/points/Multi-Head-decoder_Add-boundary_Leaky-ReLU_Stage-1_Material-100-40_Skip-connection_run_1/training/model_0.20_0.20/last.ckpt

