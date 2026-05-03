#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python main.py \
    dataset_folder=/media/raid/cloth/BTF \
    output_folder=/media/raid/cloth/output/BRDF/BTF \
    data=ubo \
    data.btf_filename=felt05_W400xH400_L151xV151.btf \
    data.rays_num=500000 \
    data.valid_num=20 \
    renderer=multiarea_emitter \
    material=ubo_pbr_latent \
    material.learnable_factor=True \
    material.disney=True \
    material.anisotropic=True \
    material.soft_constraint=True \
    material.predict_frame=False \
    experiment_name=Stage-2_PBR-Disney_UBO_felt05_run_1 \
    model.optimizer.name=Adam \
    model.optimizer.lr=0.002 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    model.trainer.max_epochs=100 \
    model.trainer.check_val_every_n_epoch=1 \
    model.freeze_decoder=False \
    model.trainer.limit_train_batches=5838
