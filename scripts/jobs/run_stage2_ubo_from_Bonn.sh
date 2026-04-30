#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python main.py \
    dataset_folder=/media/raid/cloth/BTF \
    data=ubo \
    data.btf_filename=fabric10_W400xH400_L151xV151.btf \
    data.rays_num=500000 \
    data.valid_num=20 \
    renderer=multiarea_emitter \
    material=ubo_latent \
    material.learnable_factor=True \
    material.predict_frame=True \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    experiment_name=Stage-2_UBO_fabric10_from-Bonn-Epoch-60_Subsample-0.1_run_1 \
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
    model.trainer.check_val_every_n_epoch=10 \
    model.trainer.limit_train_batches=5000 \
    material.latent_dim=24 \
    material.different_decoder=False \
    model.ckpt_path='/media/raid/cloth/output/BRDF/Bonn_VML/output/Bonn-Theia2/Stage-1_VML_Bonn_Subsample-0.1_Batch-500K_run_1/training/model_0.20_0.20/last.ckpt'
