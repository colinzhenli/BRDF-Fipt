#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/New_center_Nov25/0 \
    data=real \
    renderer=multiarea_emitter \
    material=ani_latent_texture_model \
    experiment_name=Leaky-ReLU_Colmap-camera_Correct-Spp_Stage-2_Optimize_from-scratch-decoder_run_2 \
    model.stage=2 \
    model.test=False \
    model.freeze_decoder=False \
    data.switch_iters=3000 \
    data.debug=False \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/10-Materials_Points-training_run_1/training/model_0.20_0.20/last.ckpt

