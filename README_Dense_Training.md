# Dense SVBRDF Training (Dataset_Nov11)

Instructions for running Stage-1 dense training on our captured dataset using
[scripts/jobs/run_stage1_dense.sh](scripts/jobs/run_stage1_dense.sh).

## 1. Environment Setup

The conda environment is already installed — just activate it:

```bash
source /home/zla247/anaconda3/etc/profile.d/conda.sh
conda activate fipt_copy
```

Install the one extra package required by the `Adam8bit` optimizer:

```bash
pip install bitsandbytes
```

Verify:

```bash
python -c "import bitsandbytes as bnb; print('bitsandbytes ok at', bnb.__path__[0])"
```

---

## 2. Debug Training (Sanity Check)

Runs the same pipeline as the full job but on 5 materials — completes in
minutes, use it to validate your setup before launching the full run.

**Before running, edit [scripts/jobs/run_stage1_dense_debug_readme.sh](scripts/jobs/run_stage1_dense_debug_readme.sh) and replace these paths:**

| Line | Field | Replace with |
|------|-------|--------------|
| `export CUDA_VISIBLE_DEVICES=2` | GPU index | a free GPU on your machine |
| `dataset_folder=...` | Dataset root | your path to `Dataset_Nov11` |
| `data.training_list_path=...` | Debug material list | your path to `debug_training_list.txt` (ships with the repo) |

Then launch:

```bash
bash scripts/jobs/run_stage1_dense_debug_readme.sh
```

---

## 3. Full Stage-1 Training

**Before running, edit [scripts/jobs/run_stage1_dense.sh](scripts/jobs/run_stage1_dense.sh) and replace these paths:**

| Line | Field | Replace with |
|------|-------|--------------|
| `export CUDA_VISIBLE_DEVICES=0` | GPU index | a free GPU on your machine |
| `dataset_folder=...` | Dataset root | your path to `Dataset_Nov11` |
| `experiment_name=...` | Output subfolder name | a unique name for this run |

The output root (`output_folder` in [config/config.yaml](config/config.yaml))
defaults to `/media/raid/cloth/output/BRDF`. Checkpoints and logs land in
`${output_folder}/Bonn-Theia2/${experiment_name}/`.

Then launch:

```bash
bash scripts/jobs/run_stage1_dense.sh
```

To resume from a checkpoint, append to the command in the script:

```bash
    model.ckpt_path=/path/to/last.ckpt \
    model.continue_training=True \
```
