#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

DATASET_FOLDER=/media/raid/cloth/Bonn_val
MAT_ID=318

python main.py \
    dataset_folder=${DATASET_FOLDER} \
    output_folder=/media/raid/cloth/output/BRDF/Bonn \
    data=bonn \
    data.overfit_mat_id=${MAT_ID} \
    data.rays_num=500000 \
    data.valid_num=20 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    material.learnable_factor=True \
    material.different_decoder=False \
    material.latent_dim=24 \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    experiment_name=Stage2_Adam8bit_Bonn${MAT_ID}_from_MERL_run_1 \
    model.stage=2 \
    model.test=False \
    model.apply_cosine_weight=True \
    model.continue_training=False \
    model.freeze_decoder=True \
    model.optimizer.name=Adam8bit \
    model.optimizer.lr=0.002 \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.02 \
    model.loss.reg_loss.weight=0.0 \
    model.trainer.max_epochs=40 \
    model.trainer.check_val_every_n_epoch=2 \
    model.trainer.limit_train_batches=1000 \
    model.ckpt_path=/media/raid/cloth/output/BRDF/Stage-1-Finals/MERL_480K.ckpt
