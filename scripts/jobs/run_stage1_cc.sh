#!/bin/bash

# export CUDA_VISIBLE_DEVICES=3

python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11 \
    data=points \
    renderer=multiarea_emitter \
    material=multi_material_latent \
    experiment_name=Stage-1_SGD_All-materials_Latent-dim-16_Large-batch_training_Single-color-head_degree-5_run_2 \
    model.loss.recon_loss.name=l2 \
    model.optimizer.name=SGD \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=3000 \
    model.trainer.limit_train_batches=512 \
    material.decoder.use_skip_connection=True \
    material.latent_dim=16 \
    material.decoder.degree=5 \
    material.different_decoder=False \
    data.switch_iters=2000 \
    data.chunk_size=20 \
    data.filter_observations=False \
    # model.ckpt_path=/media/raid/cloth/output/BRDF/points/Stage-1_ReLU_Overfit-Material-0_run_1/training/model_0.20_0.20/last.ckpt

