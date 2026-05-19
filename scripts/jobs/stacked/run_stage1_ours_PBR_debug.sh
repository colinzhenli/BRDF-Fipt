#!/bin/bash
# Sanity check: optimize the analytical PBRDecoder inside
# MultiMaterialPBRLatentBRDF on a tiny (3-material) debug list. Logs are
# offline (WANDB_MODE=offline) and outputs land in /tmp.
#
# Verify in stdout:
#   - `train/recon_loss` finite and trending down across a few epochs
#   - run completes without exception
#
# Run: bash scripts/jobs/stacked/run_stage1_ours_PBR_debug.sh

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11 \
    data.training_list_path=/media/raid/cloth/capture_data/Dataset_Nov11/training_list_debug3.txt \
    output_folder=/tmp/brdf_debug_output \
    data=points_dense \
    data.legacy_swap_indexing=False \
    data.rays_num=50000 \
    data.point_subsample_ratio=0.1 \
    renderer=multiarea_emitter \
    material=multi_material_pbr_latent \
    experiment_name=DEBUG_PBR_ours \
    model.optimizer.name=Adam8bit \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.lr=2e-3 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.continue_training=False \
    model.trainer.enable_checkpointing=False \
    model.trainer.max_epochs=4 \
    model.trainer.check_val_every_n_epoch=4 \
    model.trainer.limit_train_batches=20 \
    model.trainer.log_every_n_steps=1 \
    model.grazing_ratio=0.0 \
    data.filter_observations=False \
    data.switch_iters=100 \
    data.chunk_size=2 \
    renderer.spp.train=4
