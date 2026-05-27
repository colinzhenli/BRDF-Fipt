#!/bin/bash

# Compute Canada stage-2 run with the dense real-image dataloader
# (RealImageDenseDataset). Drops chunk_size / switch_iters / random_chunks
# overrides — all images are preloaded into RAM once.


DATASET_FOLDER=/home/zla247/scratch/data/capture_data/9
MAT_ID=${DATASET_FOLDER##*/}

python main.py \
    output_folder=/home/zla247/scratch/output/BRDF \
    dataset_folder=${DATASET_FOLDER} \
    renderer.emitter.direction_json=/home/zla247/scratch/data/capture_data/emitter_calibration.json \
    data=real_dense \
    data.rays_num=400000 \
    data.use_fixed_val=False \
    data.debug=False \
    renderer=multiarea_emitter \
    renderer.spp.train=4 \
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
    experiment_name=Stage2_Adam8bit_Ours${MAT_ID}_from_Ours_Grazing-angle_run_1 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    model.freeze_decoder=True \
    model.optimizer.name=Adam8bit \
    model.optimizer.lr=0.002 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.trainer.max_epochs=80 \
    model.trainer.check_val_every_n_epoch=2 \
    model.trainer.limit_train_batches=8000 \
    model.ckpt_path=/home/zla247/scratch/checkpoints/Ours.ckpt
