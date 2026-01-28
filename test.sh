CUDA_VISIBLE_DEVICES=0 \
python main.py \
    renderer=realcapture_area_emitter \
    material=merl_brdf_model \
    experiment_name=universal_decoder \
    output_folder=/home/featurize/data/output \
    model.loss.recon_loss.name=l1 \
    data.importance_sampling=False \
    data.use_single_chunk_sampling=False \
    data.uniform_sampling=False \
    dataset_folder=/home/featurize/data/train \
    model.stage=1 \
    model.freeze_decoder=False \
    model.trainer.max_epochs=50 \
    model.trainer.limit_train_batches=512 \
    data.rays_num=32768
    

