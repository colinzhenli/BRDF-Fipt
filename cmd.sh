CUDA_VISIBLE_DEVICES=2 \
python main.py \
    renderer=realcapture_area_emitter \
    material=moe_lantent \
    experiment_name=moe_lantent_small_gate \
    output_folder=/media/raid/cloth/capture_data/output_h/moe_lantent_small_gate \
    model.loss.recon_loss.name=l1 \
    data.importance_sampling=False \
    data.use_single_chunk_sampling=False \
    data.uniform_sampling=False \
    dataset_folder=/media/raid/cloth/No_cable_capture_Sep22/BRDF_recon_Sep30/Controlled_light/hdr \
    model.trainer.max_epochs=300 \
    model.trainer.limit_train_batches=512 \
    data.rays_num=65536

    #dataset_folder=/media/raid/cloth/No_cable_capture_Sep22/BRDF_recon_Sep30/Controlled_light/hdr \


