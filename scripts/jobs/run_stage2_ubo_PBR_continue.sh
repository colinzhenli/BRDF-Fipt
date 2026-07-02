#!/bin/bash

# Continue (resume FULL training state) a stage-2 UBO PBR-Disney run from its
# OWN stage-2 checkpoint. Unlike the from-Bonn/MERL/Real scripts (which load a
# stage-1 decoder and start stage-2 fresh), this resumes everything ---
# weights, optimizer state, epoch and global_step --- via PyTorch Lightning's
# native resume: model.continue_training=True makes main.py pass the checkpoint
# to trainer.fit(ckpt_path=...).
#
# Requirements for a clean resume: every model/material/optimizer setting below
# MUST match the original run (same architecture -> state_dict + optimizer state
# line up). Only experiment_name, continue_training and ckpt_path change.

export CUDA_VISIBLE_DEVICES=0

BTF_FILE=carpet12_W400xH400_L151xV151.btf
BTF_NAME=${BTF_FILE%%_*}

# Original run whose last.ckpt we resume from (the epoch-~13 checkpoint).
ORIG_ROOT=/media/raid/cloth/output/BRDF/Bonn-Theia2
ORIG_EXP=Stage-2_PBR-Disney_UBO_${BTF_NAME}_Cosine_run_1
RESUME_CKPT=${ORIG_ROOT}/${ORIG_EXP}/training/model_0.20_0.20/last.ckpt

python main.py \
    dataset_folder=/media/raid/cloth/BTF \
    output_folder=/media/raid/cloth/output/BRDF \
    data=ubo \
    data.btf_filename=${BTF_FILE} \
    data.rays_num=500000 \
    data.valid_num=20 \
    renderer=multiarea_emitter \
    material=ubo_pbr_latent \
    material.learnable_factor=True \
    material.predict_frame=True \
    material.disney=True \
    material.anisotropic=True \
    material.soft_constraint=True \
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
    model.freeze_decoder=False \
    model.apply_cosine_weight=True \
    model.trainer.max_epochs=60 \
    model.trainer.check_val_every_n_epoch=2 \
    model.trainer.limit_train_batches=5838 \
    model.ckpt_path=${RESUME_CKPT}
