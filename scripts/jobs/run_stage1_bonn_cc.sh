#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

python main.py \
    dataset_folder=/home/zla247/scratch/data/Bonn/train \
    data=bonn \
    data.use_pan=True \
    data.use_lls=True \
    data.debug=False \
    data.debug_num=200 \
    data.rays_num=100000 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name=Stage-1_Logrel_Softplus_Fir_decoder-lr-1e-4_All-data_Latent-24_Color_All-RGB-Pan-LLS_run_1 \
    output_folder=/home/zla247/scratch/output/BRDF \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.name=Adam \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.loss.pan_weight=0.5 \
    model.loss.lls_weight=0.5 \
    model.lls_spp=16 \
    model.stage=1 \
    model.test=False \
    model.continue_training=True \
    model.trainer.max_epochs=4000 \
    model.optimizer.decoder_lr=1e-4 \
    model.trainer.limit_train_batches=512 \
    model.trainer.check_val_every_n_epoch=500 \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False \
    # model.ckpt_path=/home/zla247/scratch/output/BRDF/Bonn-Theia2/Stage-1_Logrel_Softplus_Fir_decoder-lr-1e-4_All-data_Latent-24_Color_No-Chunk-All-RGB-data_run_2/training/model_0.20_0.20/last.ckpt
