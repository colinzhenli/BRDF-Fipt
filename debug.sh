#!/bin/bash

python main.py \
    dataset_folder=/home/zla247/scratch/data/105 \
    data=real \
    renderer=multiarea_emitter \
    material=ani_latent_texture_model \
    experiment_name=CC_Debug_run_1 \
    model.stage=2 \
    model.test=False \
    material.use_latent_bank=False \
    model.freeze_decoder=False \
    data.switch_iters=3000 \
    data.debug=False \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/10-Materials_Points-training_run_1/training/model_0.20_0.20/last.ckpt

