#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python main.py \
    dataset_folder=/media/raid/cloth/Bonn_val \
    data=bonn \
    data.overfit_mat_id=3 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    material.different_decoder=False \
    material.decoder.use_skip_connection=True \
    material.learnable_factor=True \
    experiment_name=Stage-2_Bonn-3_Lr-1e-3-Decoder-lr-1e-2_Correct-decoder-path_From-Real_run_1 \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.name=Adam \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    model.trainer.max_epochs=4000 \
    model.trainer.check_val_every_n_epoch=200 \
    model.optimizer.decoder_lr=1e-2 \
    model.freeze_decoder=True \
    model.optimizer.lr=0.001 \
    model.trainer.limit_train_batches=512 \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    model.ckpt_path=/media/raid/cloth/output/BRDF/Bonn-Theia2/Stage1_Chunk_Adam8bit_Softplus_Logrel_materials/training/model_0.20_0.20/last_decoder_only.ckpt


