#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11/2 \
    data=real \
    renderer=multiarea_emitter \
    material=ani_latent_texture_model \
    experiment_name=Stage-2_ReLU_Overfit-New-center-material_stage1_from-Overfit-material-0_pretrained-decoder_run_1 \
    model.stage=2 \
    model.test=False \
    model.freeze_decoder=True \
    data.switch_iters=3000 \
    data.debug=False \
    model.ckpt_path=/media/raid/cloth/output/BRDF/points/Stage-1_Overfit-material-0_run_1/training/model_0.20_0.20/last.ckpt

