#!/bin/bash
# Joint-training overfit test for the spherical-Gaussian (SG) basis decoder
# on a single UBO BTF.  Optimises basis lobes (mu, log_lambda), D_psi,
# diffuse_proj, and per-texel latents jointly.

export CUDA_VISIBLE_DEVICES=0

BTF_FILE=felt09_W400xH400_L151xV151.btf
BTF_NAME=${BTF_FILE%%_*}

python main.py \
    dataset_folder=/media/raid/cloth/BTF \
    output_folder=/media/raid/cloth/output/BRDF/BTF \
    data=ubo \
    data.btf_filename=${BTF_FILE} \
    data.rays_num=500000 \
    data.valid_num=20 \
    renderer=multiarea_emitter \
    material=ubo_sg \
    material.latent_dim=32 \
    material.predict_frame=False \
    material.learnable_factor=False \
    material.decoder.direct_coefficients=False \
    material.decoder.use_diffuse=True \
    material.decoder.q_source=halfvec \
    material.decoder.K=32 \
    material.decoder.init_lambda=5.0 \
    material.decoder.smooth_reg=False \
    experiment_name=Stage-2_Basis-SG_UBO_${BTF_NAME}_Joint_run_1 \
    model.optimizer.name=Adam \
    model.optimizer.lr=0.002 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    model.freeze_decoder=False \
    model.apply_cosine_weight=True \
    model.trainer.max_epochs=100 \
    model.trainer.check_val_every_n_epoch=1 \
    model.trainer.limit_train_batches=5838
