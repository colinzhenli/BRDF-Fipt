# Full Comparison Experiments (Baselines)

This document covers the **baseline / comparison experiments** from the paper:
decoders pretrained on other datasets (Bonn UBOFAB19, MERL) and stage-2
transfer to the two external test sets (Bonn held-out materials and UBO2014
BTFs). For the core pipeline on our RoboCloth dataset — environment setup,
data layout, stage-1/stage-2 training and the in-domain evaluation — see
[milestone_instruction.md](milestone_instruction.md) first; the environment
and conventions are identical.

All scripts live in `scripts/full_experiments/`. Released checkpoints and
datasets: https://huggingface.co/datasets/koalapenguin/RoboCloth

```
checkpoints/stage1/{Ours,Bonn,MERL}.ckpt              # stage-1 decoders (by training source)
checkpoints/stage2/RoboCloth/<mat>/<Model>_epoch<N>.ckpt
checkpoints/stage2/UBO/<mat>/<Model>_epoch<N>.ckpt
checkpoints/stage2/Bonn/<mat>/<Model>_epoch<N>.ckpt
datasets/MERL/brdfs/*.binary                          # MERL measured BRDFs (103)
datasets/UBO2014/<mat>_W400xH400_L151xV151.btf        # 12 held-out UBO2014 BTFs
```

Extra dependency on top of the `robocloth` env (only needed for UBO2014):

```bash
pip install Cython && pip install btf_extractor==1.7.0 --no-build-isolation
```

---

## 1. Datasets

| Dataset | Role | How to get it |
|---|---|---|
| RoboCloth (ours) | stage-1 source + in-domain test | see milestone_instruction.md |
| MERL | stage-1 source | HF `datasets/MERL/brdfs/` |
| Bonn (UBOFAB19) | stage-1 source + cross-dataset test | download from Uni Bonn (below) |
| UBO2014 | cross-dataset test (BTF ground truth) | HF `datasets/UBO2014/` |

### Bonn (UBOFAB19) download + preprocessing

The training/validation SVBRDF data is downloaded directly from the
University of Bonn servers with the scripts in
`scripts/full_experiments/bonn_data/`:

```bash
mkdir Bonn_train && cd Bonn_train
bash .../bonn_data/get_UBOFAB19_train_meas.sh            # measurements: mat####_{poly,pan,lls,xyz_*}.exr + calibration .mat
bash .../bonn_data/get_UBOFAB19_train_axf_svfresnel.sh   # per-material sv-Fresnel AxF files

mkdir Bonn_val && cd Bonn_val
# validation materials (test split for stage 2):
bash .../bonn_data/get_UBOFAB19_val_axf_svfresnel.sh     # + download the val 'meas' list analogously
```

**Preprocessing (required once per folder):** generate the per-material
point-count index the loader needs:

```bash
python scripts/generate_bonn_metadata.py /path/to/Bonn_train
python scripts/generate_bonn_metadata.py /path/to/Bonn_val
```

This writes `bonn_point_metadata.json` (mat_id → number of surface points,
read from the `mat####_xyz_rot000.exr` headers).

---

## 2. Stage-1 prior training (one run per training source)

Each run trains the shared BRDF decoder + per-point latent prior on one
source dataset. The resulting decoder is the "column" of the comparison
tables. All three released checkpoints are under `checkpoints/stage1/`.

| Source (column) | Script | DATA_ROOT | Epochs (paper) | Released ckpt |
|---|---|---|---|---|
| RoboCloth ("Ours") | `scripts/milestone3/train_stage1.sh` | RoboCloth capture data | 100 (ckpt @60) | `stage1/Ours.ckpt` |
| Bonn | `scripts/full_experiments/train_stage1_bonn.sh` | `Bonn_train/` | 100 | `stage1/Bonn.ckpt` |
| MERL | `scripts/full_experiments/train_stage1_merl.sh` | `datasets/MERL/brdfs/` | 70 | `stage1/MERL.ckpt` |

```bash
DATA_ROOT=/path/to/Bonn_train  OUTPUT_ROOT=/path/to/out bash scripts/full_experiments/train_stage1_bonn.sh
DATA_ROOT=/path/to/MERL/brdfs OUTPUT_ROOT=/path/to/out bash scripts/full_experiments/train_stage1_merl.sh
```

Checkpoints land in `$OUTPUT_ROOT/$EXP_NAME/training/model_0.20_0.20/last.ckpt`.
Smoke test: append `data.debug=True data.debug_num=5 MAX_EPOCHS=2` (Bonn) —
loss should drop steeply within the first epoch.

---

## 3. Stage-2 comparison fitting (one run per test material × decoder source)

Every run fits a per-material latent representation with a **frozen** decoder
selected by `MODEL` (or trains the analytic Disney-PBR baseline from scratch,
`MODEL=PBR`). `STAGE1_CKPT` points at the matching `checkpoints/stage1/<src>.ckpt`.

### 3.1 On RoboCloth test materials (145 226 314 370 452)

Covered in milestone_instruction.md §6 (`scripts/milestone3/train_stage2.sh`,
`MODEL=Ours|Bonn|MERL|PBR`). Released: `checkpoints/stage2/RoboCloth/`.

### 3.2 On Bonn held-out materials (318 377 32 226 37) — 120 epochs

```bash
DATA_ROOT=/path/to/Bonn_val STAGE1_CKPT=.../stage1/Bonn.ckpt MODEL=Bonn \
    bash scripts/full_experiments/train_stage2_bonn.sh 318
```

`MODEL=Ours|Bonn|MERL|PBR`. Released: `checkpoints/stage2/Bonn/`.

### 3.3 On UBO2014 materials (12: fabric02/04/09/11, felt01/03/05/10, carpet02/07/09/12) — 60 epochs

```bash
DATA_ROOT=/path/to/datasets/UBO2014 STAGE1_CKPT=.../stage1/Bonn.ckpt MODEL=Bonn \
    bash scripts/full_experiments/train_stage2_ubo.sh felt01
```

`MODEL=Ours|Bonn|MERL|PBR`; for `MODEL=MERL` the per-channel scale β is
initialized to 0.1 (as in the paper). Released: `checkpoints/stage2/UBO/`.
Smoke test: `MAX_EPOCHS=4 ... model.trainer.limit_train_batches=500` — loss
decreases and validation renders appear after epoch 2.

---

## 4. Evaluation — reproducing the UBO2014 transfer table

`eval_stage2_ubo.sh` evaluates a trained UBO stage-2 checkpoint with the exact
training-time validation step (`main.py model.test=True`): the fixed held-out
20% of BTF angle combinations (seed 42) is re-rendered, `val/psnr` is averaged
over it, and the same GT/prediction visualizations are saved.

```bash
# single checkpoint
DATA_ROOT=/path/to/datasets/UBO2014 \
    bash scripts/full_experiments/eval_stage2_ubo.sh felt01 \
         /path/to/checkpoints/stage2/UBO/felt01/Bonn_epoch60.ckpt

# full table (12 materials x 4 models; resumable; subset via MATERIALS=/MODELS=)
DATA_ROOT=/path/to/datasets/UBO2014 CKPT_ROOT=/path/to/checkpoints/stage2/UBO \
    bash scripts/full_experiments/run_table_ubo.sh
```

The final step prints the reproduced table next to the paper values
("Cross-dataset transfer to UBO2014": RoboCloth 36.76 > PBR 33.05 >
Bonn 32.27 > MERL 30.64 dB on average).

The same pattern applies to the Bonn test materials (`data=bonn`,
`data.overfit_mat_id=<id>`; see `train_stage2_bonn.sh` for the exact
overrides) — note that the Bonn table in the paper additionally reports
poly/gray image-count-weighted PSNR from the per-image validation logs.
