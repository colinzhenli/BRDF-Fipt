#!/bin/bash
# Usage: bash scripts/jobs/test/test_stage2_ubo_PBR.sh <material> <gpu>
# Example: bash scripts/jobs/test/test_stage2_ubo_PBR.sh felt05 2
#
# PBR-Disney has no pretrain source — only one variant per material.

set -e

MATERIAL="${1:?Usage: $0 <material> <gpu>}"
GPU="${2:?Usage: $0 <material> <gpu>}"

EXP_BASE="Stage-2_PBR-Disney_UBO_${MATERIAL}_Cosine"
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
    material=ubo_pbr_latent \
    material.learnable_factor=True \
    material.disney=True \
    material.anisotropic=True \
    material.soft_constraint=True \
    material.predict_frame=True \
    experiment_name=$TEST_EXP \
    model.optimizer.name=Adam \
    model.optimizer.lr=0.002 \
    model.optimizer.decoder_lr=2e-4 \
    model.apply_cosine_weight=True \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    model.trainer.max_epochs=100 \
    model.trainer.check_val_every_n_epoch=1 \
    model.freeze_decoder=False \
    model.trainer.limit_train_batches=5838 \
    model.ckpt_path=$CKPT
