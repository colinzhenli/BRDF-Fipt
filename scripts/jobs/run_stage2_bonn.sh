#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=offline

python main.py \
    dataset_folder=/home/haoran/axf/AxF-Decoding-SDK/sample/material_out \
    output_folder=/home/haoran/BRDF-Fipt/output/BRDF \
    data=bonn \
    renderer=multiarea_emitter \
    material=axfbrdf \
    data.overfit_mat_id=1 \
    material.axf_path=/home/haoran/axf/mat0001.axf \
    material.learnable_factor=False \
    experiment_name=Stage-2_Bonn-1_AxFBRDF_run_1 \
    model.optimizer.name=Adam \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    model.trainer.accelerator=cpu \
    model.trainer.devices=1 \
    model.trainer.max_epochs=4000 \
    model.trainer.check_val_every_n_epoch=1 \
    model.optimizer.lr=0.01 \
    model.trainer.limit_train_batches=10 \
    model.trainer.num_sanity_val_steps=0 \
    model.freeze_decoder=False \
    model.logger=false \
    model.ckpt_path=""
