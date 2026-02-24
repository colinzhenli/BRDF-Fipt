#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python main.py \
    dataset_folder=/media/raid/cloth/Bonn_train \
    data=bonn \
    data.rays_num=100000 \
    data.use_pan=True \
    data.use_lls=True \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name=Stage-1_Bonn-RGB-All-data_Forced-swap-10K-iters_Same-decoder_run_1 \
    model.loss.recon_loss.name=l2 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=3000 \
    model.trainer.check_val_every_n_epoch=100 \
    model.trainer.limit_train_batches=512 \
    material.decoder.use_skip_connection=True \
    material.latent_dim=16 \
    material.decoder.degree=3 \
    material.different_decoder=False \
    material.decoder.use_skip_connection=True \
    data.chunk_size=10 \
    data.switch_iters=10000 \
    data.filter_observations=False \
    data.debug=False
