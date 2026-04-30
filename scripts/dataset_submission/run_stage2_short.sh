#!/bin/bash
# Run a short Stage-2-from-Real training (single epoch, ~N steps) against a
# given dataset_folder, with a CSV logger and deterministic seed. Used by
# compare_train_curves.sh to A/B the original vs cropped HDR.
#
# Usage:
#   bash run_stage2_short.sh <dataset_folder> <experiment_name> <gpu_index> [max_steps]
#
# Example:
#   bash run_stage2_short.sh \
#       /media/raid/cloth/capture_data/Dataset_Nov11/227 eq_orig 0 100
#
# Outputs:
#   /media/raid/cloth/Dataset_submission/_train_compare/<experiment_name>/
#       lightning_logs/version_*/metrics.csv

set -e
DATASET="$1"
EXP_NAME="$2"
GPU_IDX="${3:-0}"
MAX_STEPS="${4:-100}"
PYTHON="${PYTHON:-/mnt/data/colin/colin/anaconda3/envs/fipt_copy/bin/python}"

OUTPUT_ROOT="/media/raid/cloth/Dataset_submission/_train_compare"
mkdir -p "$OUTPUT_ROOT"

cd "$(dirname "$(readlink -f "$0")")/../.."

export CUDA_VISIBLE_DEVICES="$GPU_IDX"
export WANDB_MODE=offline
export WANDB_DIR=/tmp

"$PYTHON" main.py \
    output_folder="$OUTPUT_ROOT" \
    dataset_folder="$DATASET" \
    data=real \
    data.rays_num=65535 \
    data.use_fixed_val=True \
    data.chunk_size=200 \
    data.switch_iters=3000 \
    data.debug=False \
    renderer=multiarea_emitter \
    renderer.spp.train=4 \
    renderer.emitter.direction_json=/media/raid/cloth/capture_data/Dataset_Nov11/emitter_calibration.json \
    material=ani_latent_texture_model \
    material.learnable_factor=True \
    material.mono_brdf=False \
    material.different_decoder=False \
    material.neural_geometry.factor=0.04 \
    material.decoder.use_skip_connection=True \
    material.use_latent_bank=False \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    experiment_name="$EXP_NAME" \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    model.freeze_decoder=True \
    model.loss.recon_loss.name=l1 \
    model.loss.reg_loss.weight=0.0 \
    model.trainer.max_epochs=1 \
    +model.trainer.max_steps="$MAX_STEPS" \
    model.trainer.limit_train_batches="$MAX_STEPS" \
    +model.trainer.limit_val_batches=0 \
    model.trainer.check_val_every_n_epoch=999 \
    model.trainer.num_sanity_val_steps=0 \
    model.trainer.log_every_n_steps=1 \
    model.trainer.enable_checkpointing=False \
    'model.logger._target_=pytorch_lightning.loggers.CSVLogger' \
    'model.logger.name=null' \
    '~model.logger.project' \
    model.ckpt_path=/media/raid/cloth/output/BRDF/Bonn-Theia2/Stage1_Chunk_Adam8bit_Softplus_Logrel_materials/training/model_0.20_0.20/last_decoder_only.ckpt
