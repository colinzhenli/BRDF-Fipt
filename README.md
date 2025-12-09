
# Material-Capture Project

This is the code to overfit a single material. 

## 1. Installation

Set up the environment with the following commands:

```bash
conda create --name material-capture python=3.10 pip
conda activate material-capture

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html 
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -r requirements.txt
pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/
pip install trimesh
conda install conda-forge::hydra-core
pip install viztracer pnoise imageio wandb
```

---

## 2. Dataset Structure

```
${dataset_folder}/
├── hdr/                          # HDR images folder (gt_folder)
│   ├── image_0000.png            # 16-bit PNG images (or .exr)
│   ├── image_0001.png
│   └── ...
├── scan_log.json                 # Main metadata file (metadata_path)
├── rotated_camera.json           # Camera poses after rotation alignment (camera_metadata_path)
└── unmatched_scan_ids.json       # (Optional) List of scan IDs to exclude
```

---

## 3. WandB Logger Setup

Before training, set up Weights & Biases for experiment tracking:

```bash
pip install wandb
wandb login
```

The logger creates a project named `brdf-capture` on your wandb dashboard. Each experiment run is named according to your `experiment_name` argument.

**Logged Metrics:**
- `train/recon_loss` - Reconstruction loss
- `train/total_loss` - Total training loss  
- `train/psnr` - Peak Signal-to-Noise Ratio
- `val/loss` - Validation loss
- `val/emitter_radiance` - Learned emitter radiance
- `val/psnr` - Validation PSNR

---

## 4. Training

```bash
# Specify a GPU
# Example dataset_folder: New_center_Nov25/0
CUDA_VISIBLE_DEVICES=0 python main.py \
    dataset_folder=${DATASET_PATH} \
    renderer=realcapture_area_emitter \
    material=ani_latent_texture_model \
    experiment_name=${EXPERIMENT_NAME} \
    model.test=False
```

**Training Configuration:**
- **Max epochs:** 1000
- **Validation frequency:** Every 2 epochs
- **Training batches per epoch:** 512

---

## 5. Testing on Validation Data

```bash
# Specify a GPU
# Example dataset_folder: New_center_Nov25/0
CUDA_VISIBLE_DEVICES=0 python main.py \
    dataset_folder=${DATASET_PATH} \
    renderer=realcapture_area_emitter \
    material=ani_latent_texture_model \
    experiment_name=${EXPERIMENT_NAME} \
    model.test=True \
    model.ckpt_path=${CHECKPOINT_PATH}
```

---

## 6. Output Structure

**Checkpoints:** `${exp_output_root_path}/training/model_0.20_0.20/`

| File | Description |
|------|-------------|
| `epoch={N}.ckpt` | Checkpoint saved at epoch N |
| `last.ckpt` | Symlink to the latest checkpoint |

**Rendered Images:** `${exp_output_root_path}/images/`

| File | Description |
|------|-------------|
| `gt_view_{batch_idx}_{b}.png` | Ground truth 16-bit PNG image |
| `result_view_{batch_idx}_{b}.png` | Rendered result 16-bit PNG image |
| `u_offset_{batch_idx}_{b}.png` | U offset grayscale visualization |
| `v_offset_{batch_idx}_{b}.png` | V offset grayscale visualization |
| `uv_offset_color_{batch_idx}_{b}.png` | Combined UV offset color visualization (R=U, G=V, B=127) |

