#!/bin/bash
# Debug variant of run_stage1_bonn_validate.sh
# - Loads decoder-only checkpoint, freezes decoder, optimises latents on Bonn_val
# - data.debug=True, debug_num=1 (single material)
# - tiny subsample ratio, tiny batch, few train batches / 2 epochs
# - WandB offline

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export WANDB_MODE=offline

python main.py \
    dataset_folder=/media/raid/cloth/Bonn_val \
    data=bonn \
    data.num_load_workers=0 \
    data.use_pan=True \
    data.use_lls=True \
    data.debug=True \
    data.debug_num=1 \
    data.rays_num=2048 \
    data.point_subsample_ratio=0.0001 \
    renderer=multiarea_emitter \
    material=bonn_latent \
    experiment_name=DEBUG_Validate_Stage-1_Bonn_run \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.name=Adam8bit \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.loss.pan_weight=1.0 \
    model.loss.lls_weight=1.0 \
    model.lls_spp=4 \
    model.stage=1 \
    model.apply_cosine_weight=False \
    model.test=False \
    model.continue_training=False \
    model.validate_on_stage1=True \
    model.freeze_decoder=True \
    model.trainer.max_epochs=2 \
    model.optimizer.lr=2e-3 \
    model.optimizer.decoder_lr=2e-4 \
    model.trainer.limit_train_batches=10 \
    model.trainer.check_val_every_n_epoch=10000 \
    model.trainer.num_sanity_val_steps=0 \
    model.trainer.log_every_n_steps=1 \
    model.trainer.enable_checkpointing=False \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False \
    model.ckpt_path=/media/raid/cloth/output/BRDF/Bonn_VML/output/Bonn-Theia2/Stage1_VML_Ours-420_Subsample-0.1_run_1/training/model_0.20_0.20/last_decoder_only.ckpt
