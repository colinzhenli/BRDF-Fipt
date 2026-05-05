#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python main.py \
    dataset_folder=/home/zla247/scratch/data/capture_data \
    data.training_list_path=/home/zla247/scratch/data/capture_data/training_list_442.txt \
    output_folder=/home/zla247/scratch/output/BRDF \
    data=points_dense \
    data.rays_num=500000 \
    data.point_subsample_ratio=0.1 \
    renderer=multiarea_emitter \
    material=multi_material_latent \
    experiment_name=Stage1_Fir_Larger-decoder_Ours-442-all_Subsample-0.1_Batch-500K_run_1 \
    model.optimizer.name=Adam8bit \
    model.continue_training=False \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.lr=2e-3 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=100 \
    model.trainer.check_val_every_n_epoch=10 \
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
    # model.ckpt_path=/home/zla247/scratch/output/BRDF/Bonn-Theia2/Continue_Stage1_Fir_Ours-300-all_Subsample-0.1_Batch-500K_run_1/training/model_0.20_0.20/last.ckpt

