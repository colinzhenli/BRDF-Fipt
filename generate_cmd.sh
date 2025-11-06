CUDA_VISIBLE_DEVICES=2 \
python generate_images.py \
    renderer=realcapture_emitter \
    material=ani_latent_texture_model \
    experiment_name=Theia2_Time-record_Ani_Pre-generated-images_Single-chunk_Permed-Reweighted-IS_run_1 \
    output_folder=output \
    model.loss.recon_loss.name=l1 \
    data.importance_sampling=False \
    data.use_single_chunk_sampling=False \
    data.uniform_sampling=False \
    dataset_folder=/mnt/data/haoran/BRDF-Flit/dataset \
    model.trainer.max_epochs=150 \
    model.trainer.limit_train_batches=512 \
    data.rays_num=32768

