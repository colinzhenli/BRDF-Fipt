#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1

python main.py \
    dataset_folder=/home/zla247/scratch/data/227 \
    data=real \
    data.rays_num=262,144 \
    renderer=multiarea_emitter \
    renderer.spp.train=4 \
    renderer.emitter.direction_json=./emitter_calibration.json \
    material=ani_latent_texture_model \
    material.learnable_factor=True \
    material.mono_brdf=False \
    material.different_decoder=False \
    material.neural_geometry.factor=0.04 \
    material.decoder.use_skip_connection=True \
    experiment_name=Stage-2_Fir_L1-loss_Material-227_New-decoder_From-Real_Latent-dim-24_Degree-3_Fixed-Val-227_run_1 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    material.use_latent_bank=False \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    model.freeze_decoder=True \
    model.loss.recon_loss.name=l1 \
    model.loss.reg_loss.weight=0.0 \
    model.trainer.max_epochs=2000 \
    model.trainer.check_val_every_n_epoch=100 \
    model.trainer.limit_train_batches=512 \
    data.switch_iters=3000 \
    data.chunk_size=200 \
    data.debug=False \
    model.ckpt_path=/home/zla247/scratch/checkpoints/Stage1_from_real_last_decoder_only.ckpt
    