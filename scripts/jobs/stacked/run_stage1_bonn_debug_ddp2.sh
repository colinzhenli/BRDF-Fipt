#!/bin/bash
# Stage-1 Bonn DDP debug.
# Mirrors run_stage1_bonn.sh but:
#   - DDP on 2 GPUs (1,2). gpu0 left free.
#   - rays_num halved to 250K so effective batch = 2 x 250K = 500K matches the
#     single-GPU baseline. lr / decoder_lr unchanged.
#   - limit_train_batches kept at 8000 -> same optimizer steps per epoch.
#   - point_subsample_ratio reduced 0.1 -> 0.02 for fast debug iteration.
#   - max_epochs reduced 100 -> 5, check_val_every_n_epoch 10 -> 1.
#   - num_load_workers 8 -> 16 so 2 GPUs don't starve.
#   - continue_training disabled and ckpt_path commented out (clean debug start).

export CUDA_VISIBLE_DEVICES=1,2

python main.py \
    dataset_folder=/media/raid/cloth/Bonn_train \
    data=bonn \
    data.num_load_workers=16 \
    data.use_pan=True \
    data.use_lls=True \
    data.debug=False \
    data.debug_num=200 \
    data.rays_num=250000 \
    data.point_subsample_ratio=0.02 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name=Debug_DDP2_Stage-1_Bonn_Subsample-0.02_Batch-500K-eff_run_1 \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.name=Adam8bit \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.loss.pan_weight=1.0 \
    model.loss.lls_weight=1.0 \
    model.lls_spp=4 \
    model.stage=1 \
    model.apply_cosine_weight=False \
    model.test=False \
    model.continue_training=False \
    model.trainer.devices=2 \
    model.trainer.strategy=ddp_find_unused_parameters_true \
    model.trainer.max_epochs=5 \
    model.optimizer.lr=2e-3 \
    model.optimizer.decoder_lr=2e-4 \
    model.trainer.limit_train_batches=8000 \
    model.trainer.check_val_every_n_epoch=1 \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/Bonn-Theia2/Stage-1_VML_Bonn_Subsample-0.1_Batch-500K_run_1/training/model_0.20_0.20/last.ckpt
