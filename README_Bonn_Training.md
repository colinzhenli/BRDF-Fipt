# Bonn SVBRDF Training

## 1. Environment Setup (Compute Canada)

```bash
module load python/3.10
module load StdEnv/2023 intel/2023.2.1 cuda/11.8

source ./envs/fipt/bin/activate

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -r requirements.txt
```

---

## 2. Preprocess: Generate Bonn Metadata

Before training, generate `bonn_point_metadata.json` which the `BonnLatentBRDF` model requires at runtime. This scans all `matXXXX_xyz_rot000.exr` files and records each material's spatial dimensions.

```bash
python scripts/generate_bonn_metadata.py <BONN_DATASET_PATH>
```

For example:

```bash
python scripts/generate_bonn_metadata.py /home/zla247/scratch/data/Bonn/train
```

This writes `bonn_point_metadata.json` inside the dataset folder.

---

## 3. Run Stage-1 Training

Use the provided job script:

```bash
bash scripts/jobs/run_stage1_bonn_cc.sh
```

Before running, edit the three paths in `scripts/jobs/run_stage1_bonn_cc.sh`:

| Variable | Description |
|----------|-------------|
| `dataset_folder` | Path to the Bonn training data (e.g. `/home/zla247/scratch/data/Bonn/train`) |
| `output_folder` | Directory where checkpoints and logs are saved |
| `model.ckpt_path` | Path to an existing checkpoint to resume from (set `model.continue_training=True`) |
