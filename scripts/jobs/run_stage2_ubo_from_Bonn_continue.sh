#!/bin/bash

# Continue (resume FULL training state) a stage-2 UBO "from-Bonn" run from its
# OWN stage-2 checkpoint via PyTorch Lightning native resume
# (model.continue_training=True -> trainer.fit(ckpt_path=...) restores weights,
# optimizer state, epoch and global_step).
#
# IMPORTANT: every model/material/optimizer setting below MUST match the
# original run so the state_dict + optimizer state line up. Only experiment_name,
# continue_training and ckpt_path change. In particular keep freeze_decoder=True
# (same optimizer param groups as the original).

export CUDA_VISIBLE_DEVICES=3

BTF_FILE=fabric09_W400xH400_L151xV151.btf
BTF_NAME=${BTF_FILE%%_*}

# Original run to resume. Verify ORIG_EXP matches the actual folder on disk.
ORIG_ROOT=/media/raid/cloth/output/BRDF/Bonn-Theia2
ORIG_EXP=Stage-2_UBO_${BTF_NAME}_from_Bonn_Grazing-angle_run_1
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
