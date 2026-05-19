#!/bin/bash
# Stage-1 PBR sanity check: optimize analytical PBRDecoder
# (MultiMaterialPBRLatentBRDF) instead of the neural BRDFDecoder.
# - material=multi_material_pbr_latent (latent dim fixed by anisotropic/disney
#   flags inside the config: 6 / 9 / 12)
# - grazing_ratio=0.0 (PBR's NoL gating already zeros at grazing; the grazing
#   pseudo-obs is redundant for an analytical PBR model)
# - material.decoder.* / latent_dim / different_decoder overrides removed
#   (PBRDecoder has no MLP knobs)
export CUDA_VISIBLE_DEVICES=0
python main.py \
    dataset_folder=/media/raid/cloth/capture_data/Dataset_Nov11 \
    data.training_list_path=/media/raid/cloth/capture_data/Dataset_Nov11/training_list_500.txt \
    output_folder=/media/raid/cloth/output/BRDF \
    data=points_dense \
    data.legacy_swap_indexing=False \
    data.rays_num=500000 \
    data.point_subsample_ratio=0.1 \
    renderer=multiarea_emitter \
    material=multi_material_pbr_latent \
    experiment_name=Stage1_Fir_Our-500_PBR-Disney_run_1 \
    model.optimizer.name=Adam8bit \
    model.continue_training=False \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.lr=2e-3 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=100 \
    model.trainer.check_val_every_n_epoch=5 \
    model.trainer.limit_train_batches=8000 \
    model.grazing_ratio=0.0 \
    data.filter_observations=False \
    data.switch_iters=100 \
    data.chunk_size=2 \
    renderer.spp.train=4 \
    # model.ckpt_path=/home/zla247/scratch/output/BRDF/Bonn-Theia2/Stage1_Nibi_Ours-500-all_Subsample-0.1_Batch-500K_run_1/training/model_0.20_0.20/last.ckpt
