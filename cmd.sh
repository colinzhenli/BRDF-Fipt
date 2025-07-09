CUDA_VISIBLE_DEVICES=0 \
python main.py \
    renderer=realcapture_emitter \
    material=ani_latent_texture_model \
    experiment_name=Theia2_Time-record_Ani_Pre-generated-images_Single-chunk_Permed-Reweighted-IS_run_1 \
    output_folder=output \
    model.loss.recon_loss.name=l1 \
    data.importance_sampling=True \
    data.use_single_chunk_sampling=True \
    data.uniform_sampling=False \
    model.pbr_texture_path=/mnt/data/haoran/BRDF-Flit/pbr_texture \
    dataset_folder=/mnt/data/haoran/BRDF-Flit/dataset \
    model.trainer.max_epochs=2 \
    model.trainer.limit_train_batches=128 \
    data.rays_num=512