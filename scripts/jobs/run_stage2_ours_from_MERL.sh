#!/bin/bash

# Production stage-2 run with the dense real-image dataloader
# (RealImageDenseDataset). Drops chunk_size / switch_iters / random_chunks
# overrides — all images are preloaded into RAM once.

export CUDA_VISIBLE_DEVICES=2

DATASET_FOLDER=/media/raid/cloth/HDD/Dataset_submission/227
MAT_ID=${DATASET_FOLDER##*/}

python main.py \
    dataset_folder=${DATASET_FOLDER} \
    data=real_dense \
    data.rays_num=400000 \
    data.use_fixed_val=False \
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
    material.neural_geometry.factor=0.08 \
    material.decoder.use_skip_connection=True \
    material.decoder.degree=3 \
    experiment_name=Stage2_Adam8bit_Ours${MAT_ID}_from_MERL_dense_run_1 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    model.freeze_decoder=True \
    model.optimizer.name=Adam8bit \
    model.optimizer.lr=0.002 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.trainer.max_epochs=100 \
    model.trainer.check_val_every_n_epoch=2 \
    model.trainer.limit_train_batches=8000 \
    model.ckpt_path=/media/raid/cloth/output/BRDF/Stage-1-Finals/MERL_480K.ckpt
