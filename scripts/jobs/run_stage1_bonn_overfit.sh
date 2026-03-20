#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python main.py \
    dataset_folder=/media/raid/cloth/Bonn_train \
    data=bonn \
    data.use_pan=False \
    data.use_lls=False \
    data.debug=True \
    data.debug_num=100 \
    data.random_sample_material_number=100 \
    data.rays_num=131072 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name=Stage-1_Correct-mask_Color-Decomp_Film_Softplus_SparseAdam_Theia-2_decoder-lr-1e-4_Overfit-100_Latent-dim-16_No-Chunk-All-RGB-data_run_1 \
    model.optimizer.reset_latent_momentum_on_chunk_switch=True \
    model.optimizer.name=SparseAdam \
    model.loss.recon_loss.name=l2 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=3000 \
    model.optimizer.decoder_lr=1e-4 \
    model.optimizer.lr=0.001 \
    model.trainer.limit_train_batches=512 \
    material.decoder.use_skip_connection=True \
    model.decoder.use_color_decomp=True \
    model.decoder.use_film=True \
    material.latent_dim=16 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/Stage-1_ReLU_Overfit-Material-0_run_1/training/model_0.20_0.20/last.ckpt

