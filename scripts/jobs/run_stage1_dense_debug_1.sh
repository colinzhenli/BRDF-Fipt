#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11 \
    data=points_dense \
    renderer=multiarea_emitter \
    material=multi_material_latent \
    experiment_name=Stage1_No-Chunk_Adam8bit_Dense_Debug_10materials \
    model.optimizer.name=Adam8bit \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.decoder_lr=1e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=200 \
    model.trainer.check_val_every_n_epoch=5 \
    model.trainer.limit_train_batches=32 \
    model.trainer.log_every_n_steps=2 \
    material.decoder.use_skip_connection=True \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False \
    data.switch_iters=100 \
    data.chunk_size=2 \
    data.training_list_path=/home/zla247/projects/BRDF-Fipt/scripts/reformat_data/debug_training_list_10.txt
