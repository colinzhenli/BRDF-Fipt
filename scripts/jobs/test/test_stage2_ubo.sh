#!/bin/bash
# Usage: bash scripts/jobs/test/test_stage2_ubo.sh <source: bonn|ours> <material> <gpu>
# Example: bash scripts/jobs/test/test_stage2_ubo.sh ours fabric09 3
#
#   source = bonn  -> from-Bonn-Epoch-60   (stage-1 pretrained on Bonn)
#   source = ours  -> from-Real-442        (stage-1 pretrained on Real-442)

set -e

SOURCE="${1:?Usage: $0 <source: bonn|ours> <material> <gpu>}"
MATERIAL="${2:?Usage: $0 <source: bonn|ours> <material> <gpu>}"
GPU="${3:?Usage: $0 <source: bonn|ours> <material> <gpu>}"

case "$SOURCE" in
    bonn) SOURCE_TAG="from-Bonn-Epoch-60" ;;
    ours) SOURCE_TAG="from-Real-442"      ;;
    *) echo "Error: source must be 'bonn' or 'ours' (got '$SOURCE')" >&2; exit 1 ;;
esac

EXP_BASE="Stage-2_UBO_${MATERIAL}_${SOURCE_TAG}_Subsample-0.1"
RUN_EXP="${EXP_BASE}_run_1"
TEST_EXP="${EXP_BASE}_test_1"
BTF="${MATERIAL}_W400xH400_L151xV151.btf"
CKPT="/media/raid/cloth/output/BRDF/BTF/Bonn-Theia2/${RUN_EXP}/training/model_0.20_0.20/last.ckpt"

if [ ! -f "$CKPT" ]; then
    echo "Error: checkpoint not found: $CKPT" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU

python test.py \
    dataset_folder=/media/raid/cloth/BTF \
    output_folder=/media/raid/cloth/output/BRDF/BTF \
    data=ubo \
    data.btf_filename=$BTF \
    data.rays_num=500000 \
    data.valid_num=5 \
    renderer=multiarea_emitter \
    material=ubo_latent \
    material.latent_dim=24 \
    material.learnable_factor=True \
    material.predict_frame=True \
    material.different_decoder=False \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    experiment_name=$TEST_EXP \
    model.optimizer.name=Adam \
    model.optimizer.lr=0.002 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=2 \
    model.test=False \
    model.freeze_decoder=True \
    model.continue_training=False \
    model.trainer.max_epochs=100 \
    model.trainer.check_val_every_n_epoch=4 \
    model.trainer.limit_train_batches=5838 \
    model.ckpt_path=$CKPT
