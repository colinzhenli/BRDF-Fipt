# Artifact Reproduction Guide

How to reproduce the paper's main results from scratch:

1. **Stage-1 training** — shared BRDF decoder + per-point latent prior on the RoboCloth training materials (§5).
2. **Stage-2 training** — dense latent-texture fit of a single captured material with the frozen stage-1 decoder (§6).
3. **Evaluation** — evaluate the released stage-2 checkpoints and reproduce the per-material PSNR table on our held-out test set (§7).
4. **Qualitative rendering** — relight trained materials on a cloth mesh under an environment map with Mitsuba 3, and render a custom scene like the paper teaser (§8–10).

Steps 1–3 use this repository; step 4 uses the companion rendering repository (`SGHyperMaterials`).

The **baseline comparison experiments** (decoders pretrained on Bonn/MERL; stage-2 transfer to the Bonn and UBO2014 test sets) are documented in [full_experiments.md](full_experiments.md), with scripts under `scripts/full_experiments/`.

---

## 1. Released assets

Checkpoints, comparison datasets, and rendering assets:
**https://huggingface.co/datasets/koalapenguin/RoboCloth**

| Path in repo | Contents |
|---|---|
| `checkpoints/stage1/{Ours,Bonn,MERL}.ckpt` | pretrained stage-1 decoders (by training source) |
| `checkpoints/stage2/RoboCloth/<mat>/{Ours,Bonn,MERL,PBR}_epoch<N>.ckpt` | trained stage-2 models for test materials 145, 226, 314, 370, 452 |
| `checkpoints/stage2/{UBO,Bonn}/...` | stage-2 models for the comparison experiments ([full_experiments.md](full_experiments.md)) |
| `render_assets/{onBars01_st_hp.ply, pole_spheres_v4.obj}` | draped-cloth + bar meshes for the qualitative renders (§9) |
| `datasets/{MERL,UBO2014}/` | comparison datasets ([full_experiments.md](full_experiments.md)) |

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download koalapenguin/RoboCloth --repo-type dataset \
    --include "checkpoints/*" "render_assets/*" --local-dir robocloth_artifacts
```

The RoboCloth capture data itself is hosted at
**https://huggingface.co/datasets/koalapenguin/cloth-brdf**:
`materials/<mat_id>/` holds the per-material files (with the `hdr/` images
tarred as `hdr.tar`), `globals/` holds the train/test lists and calibration
files. To build `DATA_ROOT` (layout in §4), e.g. for the 5 test materials
(~9 GB each; stage-1 training instead needs the materials listed in
`training_list_500.txt`):

```bash
huggingface-cli download koalapenguin/cloth-brdf --repo-type dataset \
    --include "globals/*" "materials/145/*" "materials/226/*" "materials/314/*" \
              "materials/370/*" "materials/452/*" --local-dir cloth_brdf

mkdir -p DATA_ROOT && cp cloth_brdf/globals/* DATA_ROOT/
for m in 145 226 314 370 452; do
    mv cloth_brdf/materials/$m DATA_ROOT/$m
    tar -xf DATA_ROOT/$m/hdr.tar -C DATA_ROOT/$m   # -> DATA_ROOT/<mat>/hdr/*.png
done
```

## 2. Hardware

* **GPU** — paper trainings used one 80 GB GPU; a 48 GB GPU suffices for all
  trainings at the paper batch sizes, all evaluations (~35 GB peak) and all
  renders. Reduce `RAYS_NUM` on smaller GPUs.
* **CPU RAM** — stage 1 preloads all training-list observations (~1.3 GB per
  material; ~500 GB for the full 400-material list). Stage 2 preloads all
  training views and needs a large-memory node (> 512 GB). Evaluation needs
  < 64 GB.

## 3. Environment (training repository)

Linux, NVIDIA driver ≥ 550. These pins mirror the actual paper-run
environment; in particular **pytorch-lightning must stay 1.9.x**
(`pl_bolts` is incompatible with 2.x). Do not use `requirements.txt` (stale
superset; mitsuba/tiny-cuda-nn are *not* needed — the renderer is a pure
PyTorch path tracer).

```bash
conda create -n robocloth python=3.10 -y
conda activate robocloth

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==1.9.5 torchmetrics==1.8.2
pip install lightning-bolts==0.7.0 --no-deps
pip install bitsandbytes==0.49.2 hydra-core==1.3.2 omegaconf==2.3.0 numpy==1.26.4 \
    opencv-python pillow imageio matplotlib scipy scikit-learn tqdm \
    torchviz viztracer pnoise wandb pyexr openexr_numpy tensorboardX \
    "setuptools<81" tensorboard pandas

python -c "import torch, pytorch_lightning, pl_bolts, bitsandbytes, cv2, hydra; \
           print(torch.__version__, torch.cuda.is_available())"
```

All commands below assume `conda activate robocloth` from the repository
root. Training logs go to wandb; the scripts default to `WANDB_MODE=offline`.

## 4. Data layout (`DATA_ROOT`)

```
DATA_ROOT/
├── training_list_500.txt          # stage-1 material list (one id per line)
├── camera_factor.json             # per-material exposure segments (stage 1)
├── emitter_calibration.json       # LED angular-calibration table
├── 145/                           # one folder per material id
│   ├── observations_structured.npz    # dense per-point observations (stage 1)
│   ├── point_metadata.json            # surface-point count (stage 1)
│   ├── scan_log.json                  # per-image camera/light metadata
│   ├── rotated_camera.json            # refined camera poses
│   ├── bbox.json                      # sample rectangle bounds
│   └── hdr/*.png                      # merged linear 16-bit captures (stage 2)
└── ...
```

## 5. Stage-1 training

```bash
DATA_ROOT=/path/to/capture_data OUTPUT_ROOT=/path/to/outputs \
    bash scripts/milestone3/train_stage1.sh
```

This runs the exact paper configuration. Customizable variables:

* `TRAINING_LIST` — material list (default `$DATA_ROOT/training_list_500.txt`).
* `MAX_EPOCHS` — default 100; the paper checkpoint is taken at epoch 60.
* `RAYS_NUM` — batch size in rays; reduce on GPU OOM.
* `EXP_NAME` — experiment/output folder name.

Output: `$OUTPUT_ROOT/$EXP_NAME/training/model_0.20_0.20/{epoch=N.ckpt,last.ckpt}`.
The released `checkpoints/stage1/Ours.ckpt` is this run's result — you can
skip stage 1 and use it directly.

*Smoke test* (~20 min): point `TRAINING_LIST` at a file with ~5 material ids,
`MAX_EPOCHS=3 model.trainer.limit_train_batches=300` — the loss should drop
steeply within the first epoch.

## 6. Stage-2 training

```bash
DATA_ROOT=/path/to/capture_data OUTPUT_ROOT=/path/to/outputs \
STAGE1_CKPT=/path/to/checkpoints/stage1/Ours.ckpt \
    bash scripts/milestone3/train_stage2.sh 145
```

Decoder-only warm-start from `STAGE1_CKPT` (then frozen); optimizes the 2048²
latent texture, neural geometry, and per-channel scale β. Customizable
variables:

* `MODEL=Bonn|MERL` — warm-start from the corresponding stage-1 decoder;
  `MODEL=PBR` — analytic Disney baseline (no checkpoint).
* `MAX_EPOCHS` — paper: 145→120, 226→100, 314/370/452→80 (default 100).

Outputs: checkpoints as in §5, plus validation renders
(`images/gt_view_<i>_0.png`, `images/result_view_<i>_0_psnr<PSNR>.png`) every
2 epochs.

*Smoke test* (~1 h): `MAX_EPOCHS=4` — loss drops sharply and the epoch-2
validation renders already resemble the GT views.

## 7. Evaluation — reproducing the paper table

Evaluate a trained stage-2 checkpoint on its held-out validation views:

```bash
# single checkpoint
DATA_ROOT=... OUTPUT_ROOT=... bash scripts/milestone3/eval_stage2.sh 145 \
    /path/to/checkpoints/stage2/RoboCloth/145/Ours_epoch112.ckpt

# full table: 5 materials x 4 models (~10 min each on a 48 GB GPU; resumable)
DATA_ROOT=... OUTPUT_ROOT=... CKPT_ROOT=/path/to/checkpoints/stage2/RoboCloth \
    bash scripts/milestone3/run_table_ours.sh
```

Outputs, per checkpoint:

* the validation PSNR (the paper's metric), printed and saved to
  `$OUTPUT_ROOT/eval_results/<mat>_<model>.json`;
* GT / prediction renders of the validation views under
  `$OUTPUT_ROOT/Eval_Ours<mat>_<model>/images/`.

`run_table_ours.sh` finishes by printing the reproduced numbers next to the
reference values — they should match **Table "Per-material reconstruction
PSNR" (our held-out test set block)** in the paper.

## 8. Environment (rendering repository — SGHyperMaterials)

The Mitsuba rendering repository is included as a git submodule at
`SGHyperMaterials/`:

```bash
git submodule update --init SGHyperMaterials
cd SGHyperMaterials          # all §9-§10 commands run from here
```

```bash
conda create -n fipt-mitsuba -c conda-forge python=3.11 -y
conda activate fipt-mitsuba
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install mitsuba==3.8.0 drjit==1.3.1
pip install pytorch-lightning==2.6.1 torchmetrics hydra-core==1.3.2 omegaconf==2.3.0
pip install opencv-python pillow imageio imageio-ffmpeg openexr pyexr simplejpeg trimesh
```

(The submodule's own `requirements.txt` is stale — do not use it.) Two
branches: `gt_render` (single-object relighting, §9 — the submodule's default)
and `teaser-structured` (room scene, §10).

## 9. Qualitative renders — environment light on a cloth mesh (`gt_render`)

The submodule is already on the `gt_render` branch. Quick start
(`data/cloth.obj` + `data/envmap.exr` ship with the branch):

```bash
python rendering.py \
    renderer=ours_anisotropic material_id="'145'" \
    ckpt_path=/path/to/checkpoints/stage2/RoboCloth/145/Ours_epoch112.ckpt \
    lighting=envmap +scene.mesh=cloth +scene.mesh_scale=0.2 \
    camera.origin='[0,0,-1.5]' camera.target='[0,0,0]' \
    render.width=512 render.height=512 render.spp=16 render.batch_spp=4 \
    output_base=/path/to/render_output +output_suffix=_debug
```

Output: `<output_base>/145/ours_anisotropic/envmap_debug.png` (raw linear sRGB).

**Paper-figure scene** (draped cloth on a chrome bar; meshes from
`render_assets/`), paper quality = 2048², spp 512:

```bash
python rendering.py \
    renderer=ours_anisotropic material_id="'145'" ckpt_path=<ckpt> \
    lighting=envmap \
    +scene.mesh=/path/to/render_assets/onBars01_st_hp.ply +scene.mesh_scale=0.3 \
    +bar.enabled=true +bar.mesh=/path/to/render_assets/pole_spheres_v4.obj \
    +bar.material=Cr +bar.alpha=0.1 \
    camera.origin='[0,0,-1.5]' camera.target='[0,0,0]' \
    render.width=2048 render.height=2048 render.spp=512 render.batch_spp=2 \
    output_base=/path/to/render_output
```

Notes: `renderer=ours_anisotropic` for neural checkpoints, `renderer=ours_pbr`
for PBR baselines; `+envmap_path=... +envmap_scale=N` swaps the environment
map; `+scene.mesh=/abs/path.obj` renders any mesh (auto-centered/scaled);
`batch_spp ≤ 4` is the GPU-memory ceiling — total spp accumulates in chunks.

## 10. Custom scene — the teaser (`teaser-structured`)

```bash
git checkout teaser-structured
export BRDF_CKPT_ROOT=/path/to/checkpoints/stage2      # stage-2 checkpoints
export TEASER_SCENE_ROOT=/path/to/teaser_assets        # contains room_2/room/room.xml

# debug (~minutes)
python rendering.py render.spp=16 render.batch_spp=4 resx=960 resy=540 \
    output_base=/path/to/render_output output_name=teaser_debug
# paper quality: defaults (3840x2160, spp 1024)
python rendering.py output_base=/path/to/render_output output_name=teaser
```

The `objects:` map in `config/config.yaml` assigns a renderer config (=
checkpoint) to each shape id; swap one with e.g.
`objects.elm__2=ours_370`. For your own scene, point `scene.xml_path` at any
Mitsuba XML and list your shape ids under `objects:` — unlisted shapes keep
their XML BSDF.

## 11. Troubleshooting

* **GPU OOM** — training: lower `RAYS_NUM` (or `material.texture_resolution`);
  rendering: lower `render.batch_spp`.
* **`emitter_calibration.json` not found** — scripts look in the material
  folder, then the dataset root; set `EMITTER_CALIB=/path/to/file` explicitly.

## 12. Script reference

| Script | Purpose |
|---|---|
| `scripts/milestone3/train_stage1.sh` | stage-1 prior training |
| `scripts/milestone3/train_stage2.sh <mat>` | stage-2 fitting (`MODEL=Ours/Bonn/MERL/PBR`) |
| `scripts/milestone3/eval_stage2.sh <mat> <ckpt> [tag]` | evaluate a stage-2 checkpoint (visualizations + JSON) |
| `scripts/milestone3/run_table_ours.sh` | all 5×4 evaluations + comparison table |
| `scripts/milestone3/collect_eval_results.py` | (re-)print the table from accumulated results |
| `scripts/full_experiments/*` | baseline comparisons — see [full_experiments.md](full_experiments.md) |

The original Compute-Canada job scripts used for the paper runs are preserved
unmodified under `scripts/jobs/`; the scripts above differ only in path
handling.
