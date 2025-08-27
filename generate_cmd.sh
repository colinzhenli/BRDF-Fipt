CUDA_VISIBLE_DEVICES=0 \
python generate_images.py \
    renderer=realcapture_area_emitter \
    material=learnablesvpbr \
    experiment_name=Theia2_Time-record_Ani_Pre-generated-images_Single-chunk_Permed-Reweighted-IS_run_1 \
    output_folder=output \
    model.loss.recon_loss.name=l1 \
    data.importance_sampling=False \
    data.use_single_chunk_sampling=False \
    data.uniform_sampling=False \
    dataset_folder=/mnt/data/haoran/BRDF-Flit/dataset_area \
    model.trainer.max_epochs=150 \
    model.trainer.limit_train_batches=512 \
    data.rays_num=32768

