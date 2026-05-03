#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

DATASET_FOLDER=/media/raid/cloth/Dataset_submission/300
MAT_ID=${DATASET_FOLDER##*/}

python main.py \
    dataset_folder=${DATASET_FOLDER} \
    data=real \
    data.rays_num=65535 \
    data.use_fixed_val=False \
    data.switch_iters=3000 \
    data.chunk_size=200 \
    data.debug=False \
    renderer=multiarea_emitter \
    renderer.spp.train=4 \
    renderer.emitter.direction_json=/media/raid/cloth/capture_data/Dataset_Nov11/emitter_calibration.json \
    material=ani_latent_texture_model \
    material.texture_resolution=2048 \
    material.learnable_factor=True \
    material.mono_brdf=False \
    material.different_decoder=False \
    material.use_latent_bank=False \
    material.latent_dim=24 \
    material.neural_geometry.factor=0.04 \
    material.decoder.use_skip_connection=True \
    material.decoder.degree=3 \
    experiment_name=Stage2_Ours${MAT_ID}_from_Ours_run_1 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    model.freeze_decoder=True \
    model.loss.recon_loss.name=l1 \
    model.loss.reg_loss.weight=0.0 \
    model.trainer.max_epochs=2000 \
    model.trainer.check_val_every_n_epoch=50 \
    model.trainer.limit_train_batches=512 \
    model.ckpt_path=/media/raid/cloth/output/BRDF/Stage-1-Finals/Ours-442_480K.ckpt
