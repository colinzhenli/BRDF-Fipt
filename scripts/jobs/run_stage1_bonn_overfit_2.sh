#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python main.py \
    dataset_folder=/media/raid/cloth/Bonn_train \
    data=bonn \
    data.rays_num=65536 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name=Smooth-reg_Color-decomposition_Overfit_2_debug_1 \
    model.optimizer.name=SparseAdam \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.stage=1 \
    model.test=False \
    model.loss.recon_loss.name=l2 \
    model.trainer.limit_train_batches=512 \
    model.trainer.enable_checkpointing=False \
    model.loss.reg_loss.weight=0.02 \
    material.decoder.smooth_reg=True \
    material.decoder.smooth_reg_eps=0.01 \
    material.decoder.use_skip_connection=True \
    data.debug_num=2 \
    data.debug_rotate=False \
    data.debug_swap_channels=False \
    data.filter_observations=False \
    data.debug=True \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/Bonn-Theia2/Stage-1_Softplus_SparseAdam_Theia-2_decoder-lr-1e-4_Overfit-100_No-Chunk-All-RGB-data_run_1/training/model_0.20_0.20/epoch_799.ckpt

