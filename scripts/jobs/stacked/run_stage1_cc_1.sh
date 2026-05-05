#!/bin/bash

# export CUDA_VISIBLE_DEVICES=3

python main.py \
    dataset_folder=/home/zla247/scratch/data/capture_data \
    output_folder=/home/zla247/scratch/output/BRDF \
    data=points_dense \
    renderer=multiarea_emitter \
    renderer.spp.train=32 \
    material=multi_material_latent \
    experiment_name=Sweep_Stage1_Spp_32 \
    model.optimizer.name=Adam8bit \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.decoder_lr=1e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=1000 \
    model.trainer.check_val_every_n_epoch=500 \
    model.trainer.log_every_n_steps=50 \
    model.trainer.limit_train_batches=512 \
    material.decoder.use_skip_connection=True \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False \
    data.point_subsample_ratio=0.1 \
    data.switch_iters=100 \
    data.chunk_size=2
