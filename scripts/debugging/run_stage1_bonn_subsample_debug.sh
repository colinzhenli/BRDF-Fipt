#!/bin/bash
# Debug runner for the new Bonn point-subsample feature.
#
#   1. Runs the standalone consistency unit-test
#      (verifies BonnDataset / BonnValDataset / BonnLatentBRDF agree on the
#      per-material point counts and that latent indices stay in range).
#
#   2. Kicks off a *short* stage-1 training (1 epoch, 4 batches) over
#      ${DEBUG_NUM} materials with ${RATIO} subsample so the full
#      train+val loop runs end-to-end with the new flag.
#
# Usage:
#   bash scripts/debugging/run_stage1_bonn_subsample_debug.sh [RATIO] [DEBUG_NUM]
#
# Defaults: RATIO=0.1, DEBUG_NUM=10

set -euo pipefail

RATIO="${1:-0.1}"
DEBUG_NUM="${2:-10}"
DATASET="${DATASET:-/media/raid/cloth/Bonn_train}"
EXPERIMENT="${EXPERIMENT:-Stage-1_Bonn_Subsample_Debug}"

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../.. && pwd )"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================================"
echo "Bonn point-subsample debug"
echo "  dataset      = $DATASET"
echo "  ratio        = $RATIO"
echo "  debug_num    = $DEBUG_NUM"
echo "  experiment   = $EXPERIMENT"
echo "  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# ----------------------------------------------------------------------
# 1. Standalone consistency unit test
# ----------------------------------------------------------------------
echo ""
echo "[1/2] Running consistency unit test ..."
conda run -n fipt_copy --no-capture-output python \
    scripts/debugging/test_bonn_subsample_consistency.py \
    --dataset_folder "$DATASET" \
    --debug_num "$DEBUG_NUM" \
    --ratio "$RATIO"

# ----------------------------------------------------------------------
# 2. End-to-end short training run (1 epoch, 4 batches)
# ----------------------------------------------------------------------
echo ""
echo "[2/2] Running short stage-1 training (1 epoch, 4 batches) ..."
conda run -n fipt_copy --no-capture-output python main.py \
    dataset_folder="$DATASET" \
    data=bonn \
    data.num_load_workers=0 \
    data.use_pan=True \
    data.use_lls=True \
    data.debug=True \
    data.debug_num="$DEBUG_NUM" \
    data.point_subsample_ratio="$RATIO" \
    data.rays_num=8192 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name="$EXPERIMENT" \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.name=Adam \
    model.loss.recon_loss.name=logrel \
    # model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.recon_loss.log_space.logrel_ref=0.02 \
    model.loss.reg_loss.weight=0.0 \
    model.loss.pan_weight=0.5 \
    model.loss.lls_weight=0.5 \
    model.lls_spp=4 \
    model.stage=1 \
    model.test=False \
    model.continue_training=False \
    model.trainer.max_epochs=1 \
    model.optimizer.decoder_lr=1e-4 \
    model.trainer.limit_train_batches=4 \
    model.trainer.check_val_every_n_epoch=1 \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False

echo ""
echo "All debug steps completed."
