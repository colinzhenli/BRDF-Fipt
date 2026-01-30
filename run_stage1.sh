#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python main.py \
    dataset_folder=/media/raid/cloth/BRDFDatabase/brdfs \
    data=points \
    renderer=realcapture_area_emitter \
    material=merl_brdf_model \
    experiment_name=MERL_run_1 \
    model.loss.recon_loss.name=l1 \
    model.stage=1 \
    model.test=False \
    model.trainer.limit_train_batches=512 \
    material.decoder.use_skip_connection=True \
    data.switch_iters=2000 \
    data.filter_observations=False \
    data.debug=True \
    data.val_ratio=0.2 \
    data.num_materials=20 \
    data.start_material_id=100 \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/Stage-1_ReLU_Overfit-Material-0_run_1/training/model_0.20_0.20/last.ckpt

