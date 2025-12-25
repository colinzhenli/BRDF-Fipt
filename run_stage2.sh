#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11/0 \
    data=real \
    renderer=multiarea_emitter \
    material=ani_latent_texture_model \
    experiment_name=Stage-2-Latent-bank_Material-0_Correct-chunk-switching_run_1 \
    model.stage=2 \
    model.test=True \
    material.use_latent_bank=True \
    model.freeze_decoder=True \
    data.switch_iters=3000 \
    data.debug=False \
    model.ckpt_path=/media/raid/cloth/output/BRDF/points/Correct-chunk-switching_No-filter-observations_Stage-1_Dataset-Nov11_20-materials_run_1/training/model_0.20_0.20/last.ckpt

