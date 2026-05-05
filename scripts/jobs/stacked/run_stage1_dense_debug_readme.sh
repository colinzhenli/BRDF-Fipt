#!/bin/bash
#
# Debug script for Stage-1 dense training on the Dataset_Nov11 capture data.
# Loads only 5 materials (listed in scripts/reformat_data/debug_training_list.txt)
# so the full pipeline (dataset, renderer, optimizer, validation) can be exercised
# in a few minutes. Use this to validate the environment before launching the
# full run in scripts/jobs/run_stage1_dense.sh.
#
# Expected output:
#   - Data loading reports 5 materials from debug_training_list.txt
#   - Training logs show train/total_loss decreasing
#   - First validation runs within ~5 epochs
#

export CUDA_VISIBLE_DEVICES=2

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11 \
    data=points_dense \
    renderer=multiarea_emitter \
    material=multi_material_latent \
    experiment_name=DEBUG_Stage1_Dense_5materials \
    model.optimizer.name=Adam8bit \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.decoder_lr=1e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=10 \
    model.trainer.check_val_every_n_epoch=5 \
    model.trainer.log_every_n_steps=2 \
    model.trainer.limit_train_batches=32 \
    model.trainer.enable_checkpointing=False \
    material.decoder.use_skip_connection=True \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False \
    data.switch_iters=100 \
    data.chunk_size=2 \
    data.training_list_path=./scripts/reformat_data/debug_training_list.txt
