#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

BTF_FILE=felt09_W400xH400_L151xV151.btf
BTF_NAME=${BTF_FILE%%_*}

python main.py \
    dataset_folder=/media/raid/cloth/BTF \
    output_folder=/media/raid/cloth/output/BRDF/BTF \
    data=ubo \
    data.btf_filename=${BTF_FILE} \
    data.rays_num=100000 \
    data.valid_num=20 \
    renderer=multiarea_emitter \
    material=ubo_latent \
    material.learnable_factor=True \
    model.factor_init=0.01 \
    material.predict_frame=True \
    material.latent_dim=24 \
    material.different_decoder=False \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    experiment_name=Stage-2_UBO_${BTF_NAME}_from-MERL_run_1 \
    model.optimizer.name=Adam \
    model.optimizer.lr=0.0002 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    model.freeze_decoder=True \
    model.trainer.max_epochs=100 \
    model.trainer.check_val_every_n_epoch=5 \
    model.trainer.limit_train_batches=5838 \
    model.ckpt_path='/media/raid/cloth/output/BRDF/Stage-1-Finals/MERL_480K.ckpt'
