#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11 \
    data=points \
    renderer=multiarea_emitter \
    material=multi_material_latent \
    experiment_name=Stage-1_ReLU_Correct-observation_10-materials_run_1 \
    model.stage=1 \
    model.test=False \
    data.switch_iters=1000 \
    data.debug=True \
    data.val_ratio=0.2 \
    data.debug_num_materials=10 \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/10-Materials_Points-training_run_1/training/model_0.20_0.20/last.ckpt

