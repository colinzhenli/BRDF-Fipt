#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11/119 \
    data=real \
    renderer=multiarea_emitter \
    material=ani_latent_texture_model \
    experiment_name=Stage-2_Material-119_ReLU_Optimize_from-scratch-decoder_run_2 \
    model.stage=2 \
    model.test=False \
    material.use_latent_bank=False \
    model.freeze_decoder=False \
    data.switch_iters=3000 \
    data.debug=False \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/10-Materials_Points-training_run_1/training/model_0.20_0.20/last.ckpt

