#!/bin/bash

# Validate-on-Stage-1 (dense): freeze decoder loaded from a saved Stage-1 ckpt
# and optimise fresh latents on the val-camera split. Lets us compare losses
# across checkpoints to check whether the decoder has converged.

export CUDA_VISIBLE_DEVICES=3
python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11 \
    data.training_list_path=/media/raid/cloth/capture_data/Dataset_Nov11/test_list_420.txt \
    data=points_dense \
    data.rays_num=500000 \
    data.point_subsample_ratio=0.02 \
    renderer=multiarea_emitter \
    material=multi_material_latent \
    experiment_name=Validate_Stage1_Ours-420_Subsample-0.02_run_1 \
    model.optimizer.name=Adam8bit \
    model.continue_training=False \
    model.validate_on_stage1=True \
    model.freeze_decoder=True \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.lr=2e-3 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=20 \
    model.trainer.check_val_every_n_epoch=10000 \
    model.trainer.num_sanity_val_steps=0 \
    model.trainer.limit_train_batches=2000 \
    model.trainer.log_every_n_steps=20 \
    model.trainer.enable_checkpointing=False \
    material.decoder.use_skip_connection=True \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False \
    data.switch_iters=100 \
    data.chunk_size=2 \
    renderer.spp.train=4 \
    model.ckpt_path=/media/raid/cloth/output/BRDF/Bonn-Theia2/Stage1_VML_Ours-420_Subsample-0.1_run_1/training/model_0.20_0.20/last.ckpt
