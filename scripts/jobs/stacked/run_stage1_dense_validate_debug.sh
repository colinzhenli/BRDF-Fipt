#!/bin/bash
# Debug variant of run_stage1_dense_validate.sh
# - Loads decoder-only checkpoint, freezes decoder, optimises latents on val cams
# - 3 materials (training_list_debug3.txt), tiny subsample ratio, tiny batch
# - Few train batches / 2 epochs total
# - WandB offline

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export WANDB_MODE=offline

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11 \
    data.training_list_path=/media/raid/cloth/capture_data/Dataset_Nov11/training_list_debug3.txt \
    data=points_dense \
    data.rays_num=2048 \
    data.point_subsample_ratio=0.0001 \
    renderer=multiarea_emitter \
    material=multi_material_latent \
    experiment_name=DEBUG_Validate_Stage1_Ours-3mat_run \
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
    model.trainer.max_epochs=2 \
    model.trainer.check_val_every_n_epoch=10000 \
    model.trainer.num_sanity_val_steps=0 \
    model.trainer.log_every_n_steps=1 \
    model.trainer.limit_train_batches=10 \
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
    model.ckpt_path=/media/raid/cloth/output/BRDF/Bonn_VML/output/Bonn-Theia2/Stage1_VML_Ours-420_Subsample-0.1_run_1/training/model_0.20_0.20/last_decoder_only.ckpt
