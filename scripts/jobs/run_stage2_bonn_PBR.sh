#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python main.py \
    dataset_folder=/media/raid/cloth/Bonn_val \
    data=bonn \
    data.overfit_mat_id=3 \
    renderer=multiarea_emitter \
    material=bonn_pbr_latent \
    material.learnable_factor=True \
    material.disney=True \
    material.anisotropic=True \
    material.soft_constraint=True \
    material.predict_frame=True \
    experiment_name=Stage-2_PBR-Disney_Bonn-3_Lr-1e-3_run_1 \
    model.optimizer.name=Adam \
    model.optimizer.lr=0.001 \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    model.trainer.max_epochs=4000 \
    model.trainer.check_val_every_n_epoch=200 \
    model.freeze_decoder=False \
    model.trainer.limit_train_batches=512
