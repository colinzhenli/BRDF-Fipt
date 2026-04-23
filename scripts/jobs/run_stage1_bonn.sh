#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python main.py \
    dataset_folder=/media/raid/cloth/Bonn_train \
    data=bonn \
    data.num_load_workers=8 \
    data.use_pan=True \
    data.use_lls=True \
    data.debug=False \
    data.debug_num=200 \
    data.rays_num=1000000 \
    data.point_subsample_ratio=0.1 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name=Stage-1_Bonn_Subsample-0.1_run_1 \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.name=Adam8bit \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.loss.pan_weight=1.0 \
    model.loss.lls_weight=1.0 \
    model.lls_spp=4 \
    model.stage=1 \
    model.test=False \
    model.continue_training=False \
    model.trainer.max_epochs=100 \
    model.optimizer.lr=3e-3 \
    model.optimizer.decoder_lr=3e-4 \
    model.trainer.limit_train_batches=8000 \
    model.trainer.check_val_every_n_epoch=20 \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False \
    # model.ckpt_path=/home/zla247/scratch/output/BRDF/Bonn-Theia2/Stage-1_Logrel_Softplus_Fir_decoder-lr-1e-4_All-data_Latent-24_Color_No-Chunk-All-RGB-data_run_2/training/model_0.20_0.20/last.ckpt
