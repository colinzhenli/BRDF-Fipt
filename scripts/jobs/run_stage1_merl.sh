#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python main.py \
    dataset_folder=/media/raid/cloth/BRDFDatabase/brdfs \
    data=merl \
    renderer=realcapture_area_emitter \
    material=merl_brdf_model \
    experiment_name=MERL_Stage-1_Adam8bit_Bonn-Aligned_Latent-24_run_1 \
    model.optimizer.name=Adam8bit \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.lr=2e-3 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.035 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=100 \
    model.trainer.check_val_every_n_epoch=5 \
    model.trainer.log_every_n_steps=10 \
    model.trainer.limit_train_batches=292 \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.decoder.smooth_reg=False \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.different_decoder=False \
    data.switch_iters=2000 \
    data.chunk_size=20 \
    data.filter_observations=False \
    data.rays_num=500000 \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/Stage-1_ReLU_Overfit-Material-0_run_1/training/model_0.20_0.20/last.ckpt
