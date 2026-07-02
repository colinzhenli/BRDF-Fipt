#!/bin/bash

# Continue (resume FULL training state) a stage-2 UBO "from-Real" run from its
# OWN stage-2 checkpoint (model.continue_training=True -> trainer.fit(ckpt_path=...)).
# Keep all model/material/optimizer settings identical to the original run
# (freeze_decoder=True); only experiment_name, continue_training and ckpt_path change.
#
# NOTE: the original (non-continue) script names the run with an "Ablation_"
# prefix, but the Leonardo-copied runs on disk are named WITHOUT it
# (Stage-2_UBO_<btf>_from-Real-New-500_Grazing-angle_run_1). Set ORIG_EXP to the
# ACTUAL folder name you want to resume before running.

export CUDA_VISIBLE_DEVICES=0

BTF_FILE=felt01_W400xH400_L151xV151.btf
BTF_NAME=${BTF_FILE%%_*}

ORIG_ROOT=/media/raid/cloth/output/BRDF/Bonn-Theia2
ORIG_EXP=Stage-2_UBO_${BTF_NAME}_from-Real-New-500_Grazing-angle_run_1   # <-- verify against disk
RESUME_CKPT=${ORIG_ROOT}/${ORIG_EXP}/training/model_0.20_0.20/last.ckpt

python main.py \
    dataset_folder=/media/raid/cloth/BTF \
    output_folder=/media/raid/cloth/output/BRDF \
    data=ubo \
    data.btf_filename=${BTF_FILE} \
    data.rays_num=500000 \
    data.valid_num=20 \
    renderer=multiarea_emitter \
    material=ubo_latent \
    material.learnable_factor=True \
    material.predict_frame=True \
    material.latent_dim=24 \
    material.different_decoder=False \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    experiment_name=Continue_${ORIG_EXP} \
    model.optimizer.name=Adam \
    model.optimizer.lr=0.002 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=2 \
    model.test=False \
    model.continue_training=True \
    model.freeze_decoder=True \
    model.apply_cosine_weight=True \
    model.trainer.max_epochs=60 \
    model.trainer.check_val_every_n_epoch=2 \
    model.trainer.limit_train_batches=5838 \
    model.ckpt_path=${RESUME_CKPT}
