#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11/2 \
    data=points \
    renderer=multiarea_emitter \
    material=ani_latent_texture_model \
    experiment_name=Stage-1_test \
    model.stage=1 \
    model.test=False \
    data.switch_iters=1000 \
    model.ckpt_path=/media/raid/cloth/output/BRDF/points/10-Materials_Points-training_run_1/training/model_0.20_0.20/last.ckpt

