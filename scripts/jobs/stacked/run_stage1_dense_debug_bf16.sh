#!/bin/bash
# Debug A/B counterpart to run_stage1_dense_debug.sh
# Same config except precision=bf16 (mixed BF16). Compare loss/PSNR/it/s vs the
# FP32 debug run (DEBUG_Stage1_Ours-3mat_run) to decide if BF16 is safe to ship.
export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=offline

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11 \
    data.training_list_path=/media/raid/cloth/capture_data/Dataset_Nov11/training_list_debug3.txt \
    data=points_dense \
    data.rays_num=500000 \
    data.point_subsample_ratio=0.1 \
    renderer=multiarea_emitter \
    material=multi_material_latent \
    experiment_name=DEBUG_Stage1_Ours-3mat_bf16_run \
    model.optimizer.name=Adam8bit \
    model.continue_training=False \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.lr=2e-3 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=2 \
    model.trainer.check_val_every_n_epoch=2 \
    model.trainer.log_every_n_steps=1 \
    model.trainer.limit_train_batches=50 \
    +model.trainer.precision=bf16 \
    material.decoder.use_skip_connection=True \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False \
    data.switch_iters=100 \
    data.chunk_size=2 \
    renderer.spp.train=4
