# Running the BRDF Fitting Experiment

## Instructions

### 1. Generate Synthetic Images and Metadata

Run the following script to generate synthetic images along with emitter and camera position metadata:

```bash
python generate_images.py
````

This script will also incorporate your captured images and metadata.

* Set the path to PBR textures via:

  ```bash
  model.pbr_texture_path=<PBR_TEXTURE_FOLDER>
  ```

* You can download the required PBR textures from:

  [Download PBR Textures](https://drive.google.com/file/d/1IQMEKRRNF5Iuqa01aTf1INhzZHu4m8Wv/view?usp=drive_link)

* Set the dataset output folder path:

  ```bash
  dataset_folder=<DATASET_FOLDER>
  ```

---

### 2. Run the BRDF Fitting Experiment

Use the command below, replacing the placeholders with appropriate values:

```bash
CUDA_VISIBLE_DEVICES=<AVAILABLE_GPU_ID> \
python main.py \
    renderer=realcapture_emitter \
    material=ani_latent_texture_model \
    experiment_name=Theia2_Time-record_Ani_Pre-generated-images_Single-chunk_Permed-Reweighted-IS_run_1 \
    output_folder=<OUTPUT_FOLDER_PATH> \
    model.loss.recon_loss.name=l1 \
    data.importance_sampling=True \
    data.use_single_chunk_sampling=True \
    data.uniform_sampling=False \
    model.pbr_texture_path=<PBR_TEXTURE_FOLDER> \
    dataset_folder=<DATASET_FOLDER> \
    model.trainer.max_epochs=<MAX_EPOCHS> \
    model.trainer.limit_train_batches=<ITERATIONS_PER_EPOCH> \
    data.rays_num=<RAYS_PER_BATCH>
```

#### Placeholder Descriptions:

* `<AVAILABLE_GPU_ID>`: ID of the GPU to use (e.g., `0`)
* `<OUTPUT_FOLDER_PATH>`: Where to save experiment logs and results
* `<PBR_TEXTURE_FOLDER>`: Folder containing downloaded PBR textures
* `<DATASET_FOLDER>`: Folder where synthetic images and metadata were saved
* `<MAX_EPOCHS>`: Total number of training epochs
* `<ITERATIONS_PER_EPOCH>`: Number of training iterations per epoch
* `<RAYS_PER_BATCH>`: Number of rays sampled per training batch

---

### 3. Profiling with VizTracer

The code snippet for profiling in `main.py` is:

```python
from viztracer import VizTracer

tracer = VizTracer()
tracer.start()
trainer.fit(model, train_loader, val_loader)
tracer.stop()
tracer.save("is_all-pixels_tracer.json")
```

To view the profiling result:

```bash
vizviewer is_all-pixels_tracer.json
```

Try to locate and measure the time spent on the following operation in the data loader:

```python
chunk_indices = torch.multinomial(chunk_pdf, N_sample, replacement=True)
```

This operation happens inside the class `SphereImageDataset(IterableDataset)` and can be a performance bottleneck if the number of chunks (`chunk_pdf`) is large.

---

Let me know if you want this converted into a `.sh` script or a `.md` file ready for GitHub.
