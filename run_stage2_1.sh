CUDA_VISIBLE_DEVICES=0 \
python main.py \
    renderer=realcapture_area_emitter \
    material=merl_brdf_model \
    experiment_name=autodecoder_test_100_beige-fabric\
    output_folder=/home/featurize/data/output \
    model.loss.recon_loss.name=l1 \
    data.importance_sampling=False \
    data.use_single_chunk_sampling=False \
    data.uniform_sampling=False \
    dataset_folder=/home/featurize/data/demo \
    model.stage=2 \
    material.use_latent_bank=True \
    model.freeze_decoder=True \
    model.ckpt_path=/home/featurize/data/output/points/universal_decoder/training/model_0.20_0.20/merl.ckpt \
    #model.ckpt_path=/home/featurize/data/output/points/universal_decoder_hd/training/model_0.20_0.20/epoch=14-v1.ckpt \
    model.trainer.max_epochs=10000 \
    model.trainer.limit_train_batches=512 \
    data.rays_num=32768
    

