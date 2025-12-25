#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11 \
    data=points \
    renderer=multiarea_emitter \
    material=multi_material_latent \
    experiment_name=Normalized-l1_Clip-gradients_Stage-1_Material-2_No-skip-connection_debug \
    model.loss.recon_loss.name=normalized_l1 \
    model.stage=1 \
    model.test=False \
    model.trainer.limit_train_batches=512 \
    material.decoder.use_skip_connection=False \
    data.switch_iters=2000 \
    data.filter_observations=False \
    data.debug=True \
    data.val_ratio=0.2 \
    data.num_materials=1 \
    data.start_material_id=2 \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/Stage-1_ReLU_Overfit-Material-0_run_1/training/model_0.20_0.20/last.ckpt

