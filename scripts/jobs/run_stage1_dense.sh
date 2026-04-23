#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11 \
    data.training_list_path=/media/raid/cloth/capture_data/Dataset_Nov11/training_list_420.txt \
    data=points_dense \
    data.rays_num=1000000 \
    data.point_subsample_ratio=0.1 \
    renderer=multiarea_emitter \
    material=multi_material_latent \
    experiment_name=Stage1_Ours-420_Subsample-0.1_run_1 \
    model.optimizer.name=Adam8bit \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.lr=3e-3 \
    model.optimizer.decoder_lr=3e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=100 \
    model.trainer.check_val_every_n_epoch=20 \
    model.trainer.log_every_n_steps=50 \
    model.trainer.limit_train_batches=8000 \
    material.decoder.use_skip_connection=True \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False \
    data.switch_iters=100 \
    data.chunk_size=2 \
    renderer.spp.train=4 \

