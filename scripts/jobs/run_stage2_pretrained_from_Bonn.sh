#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python main.py \
    dataset_folder=/media/raid/cloth/Dataset_submission/400 \
    data=real \
    data.rays_num=65535 \
    data.use_fixed_val=False \
    renderer=multiarea_emitter \
    renderer.spp.train=4 \
    renderer.emitter.direction_json=/media/raid/cloth/capture_data/Dataset_Nov11/emitter_calibration.json \
    material=ani_latent_texture_model \
    material.texture_resolution=2048 \
    material.learnable_factor=True \
    material.mono_brdf=False \
    material.different_decoder=False \
    material.neural_geometry.factor=0.04 \
    material.decoder.use_skip_connection=True \
    experiment_name=Stage-2_Low-Res_L1-loss_Material-400_From-Bonn-Epoch-60_run_1 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    material.use_latent_bank=False \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    model.freeze_decoder=True \
    model.loss.recon_loss.name=l1 \
    model.loss.reg_loss.weight=0.0 \
    model.trainer.max_epochs=2000 \
    model.trainer.check_val_every_n_epoch=50 \
    model.trainer.limit_train_batches=512 \
    data.switch_iters=3000 \
    data.chunk_size=200 \
    data.debug=False \
    model.ckpt_path='/media/raid/cloth/output/BRDF/Bonn_VML/output/Bonn-Theia2/Stage-1_VML_Bonn_Subsample-0.1_Batch-500K_run_1/training/model_0.20_0.20/last.ckpt'
    