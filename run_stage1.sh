#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python main.py \
    dataset_folder=/home/featurize/data \
    data=points \
    renderer=realcapture_area_emitter \
    material=merl_brdf_model \
    experiment_name=Leaky-ReLU_Stage-1_Material-100-40_Skip-connection_run_1 \
    output_folder=/home/featurize/data/output \
    model.loss.recon_loss.name=l1 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=3000 \
    model.trainer.limit_train_batches=512 \
    material.decoder.use_skip_connection=True \
    material.latent_dim=16 \
    material.decoder.degree=5 \
    material.different_decoder=False \
    data.switch_iters=2000 \
    data.chunk_size=20 \
    data.filter_observations=False \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/Stage-1_ReLU_Overfit-Material-0_run_1/training/model_0.20_0.20/last.ckpt

