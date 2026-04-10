#!/bin/bash
# Ablation: same as run_stage1.sh but with SH degree 5 instead of 3.
# Purpose: test whether SH degree is the bottleneck for high-frequency BRDF lobes.
# Compare brdf_lobes output with the degree-3 baseline.

export CUDA_VISIBLE_DEVICES=1
python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11 \
    data=points \
    renderer=multiarea_emitter \
    material=multi_material_latent \
    experiment_name=Stage1_Softplus_SparseAdam_All_260-materials_Latent-dim-24_Logrel_decoder_SH-degree-5_ablation \
    model.optimizer.name=SparseAdam \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.decoder_lr=1e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=3000 \
    model.trainer.check_val_every_n_epoch=200 \
    model.trainer.limit_train_batches=512 \
    material.decoder.use_skip_connection=True \
    material.latent_dim=24 \
    material.decoder.degree=5 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.switch_iters=5000 \
    data.chunk_size=20 \
    data.filter_observations=False
