#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DATASET_FOLDER=/media/raid/cloth/Bonn_val
MAT_ID=318

python main.py \
    dataset_folder=${DATASET_FOLDER} \
    data=bonn \
    data.overfit_mat_id=${MAT_ID} \
    data.rays_num=500000 \
    renderer=multiarea_emitter \
    material=bonn_pbr_latent \
    material.learnable_factor=True \
    material.predict_frame=True \
    material.disney=True \
    material.anisotropic=True \
    material.soft_constraint=True \
    experiment_name=Stage2_Adam8bit_Bonn${MAT_ID}_PBR_run_1 \
    model.stage=2 \
    model.test=False \
    model.apply_cosine_weight=True \
    model.continue_training=False \
    model.freeze_decoder=False \
    model.optimizer.name=Adam8bit \
    model.optimizer.lr=0.002 \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.trainer.max_epochs=100 \
    model.trainer.check_val_every_n_epoch=4 \
    model.trainer.limit_train_batches=8000
