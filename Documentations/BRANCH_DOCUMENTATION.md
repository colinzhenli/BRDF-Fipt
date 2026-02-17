# BRDF-Fipt Branch Documentation

This document provides a comprehensive overview of all development branches in the BRDF-Fipt project, their key features, important changes, and usage commands.

## Table of Contents
1. [Early Development Branches](#early-development-branches)
2. [Real Image Training Branches](#real-image-training-branches)
3. [Optimization and Advanced Features](#optimization-and-advanced-features)
4. [Latest Development](#latest-development)

---

## Early Development Branches

### `texturelatent`
**Base Branch:** Initial implementation

**Key Features:**
- Single-scale latent texture using bilinear interpolation
- Basic texture-based BRDF representation

**Training Command:**
```bash
python main.py \
    renderer=dynamicpoint_emitter \
    material=latent_texture_model \
    experiment_name=L1-loss_Latenttexture-run_2 \
    test_on_train=False
```

**Testing Command:**
```bash
python pointlight_test.py \
    renderer=dynamicpoint_emitter \
    material=latent_texture_model \
    experiment_name=Test-L1-loss_Latenttexture-run_2 \
    model.ckpt_path=<checkpoint_path> \
    test_on_train=False
```

---

### `svlatent`
**Base Branch:** From `texturelatent`

**Key Features:**
- Spatially-varying (SV) latents using a neural field
- Reference rendering from PBR texture (roughness, metallic, albedo map, and global sphere normal)

**Notes:**
- Introduced neural field representation for more flexible BRDF modeling

---

### `uniform-training-data`
**Base Branch:** From `svlatent`

**Key Features:**
- Data trained on uniformly sampled queries
- Validation and testing performed on rays (not queries)

**Training Command:**
```bash
python main.py \
    renderer=dynamicpoint_emitter \
    material=svlatent_model \
    experiment_name=Uniformly-trained_run_2 \
    test_on_train=False
```

**Testing Command:**
```bash
python pointlight_test.py \
    renderer=dynamicpoint_emitter \
    material=svlatent_model \
    experiment_name=Test_Debug_3 \
    model.ckpt_path=<checkpoint_path> \
    test_on_train=False
```

**Example Testing Path:**
```bash
model.ckpt_path=/localhome/zla247/theia2_data/output/BRDF-Fipt/Over-fit_experiments_2/sphere/Debug_3/training/model_0.20_0.20/last.ckpt
```

---

### `is_texture_latent`
**Base Branch:** From `texturelatent`

**Key Features:**
- Importance sampled rays for supervision
- Introduced sampling strategies for training efficiency

**Training Command:**
```bash
python main.py \
    renderer=dynamicpoint_emitter \
    material=svlatent_model \
    experiment_name=Theia2_Correct-rays_200-epoches_L2-loss-Neural-field_Baseline-4-point-lights_Uniform-Importance-sampling-8192-rays_all-pixels_run_1 \
    model.loss.recon_loss.name=l2 \
    data.importance_sampling=False \
    data.uniform_sampling=True
```

**Configuration Options:**
- `data.importance_sampling`: Whether to use importance sampling from pixel brightness
- `data.uniform_sampling`: Whether the expectation is from uniformly sampled BRDF queries (applies importance weight for pixels)

---

### `is-complex-svbrdf`
**Base Branch:** From `is_texture_latent`

**Key Features:**
- Added normal map support
- More complex spatially-varying BRDF representation

**Configuration:**
- Additional configuration options in `config/material/svpbr.yaml`

---

### `anisotropic`
**Base Branch:** From `is-complex-svbrdf`

**Key Features:**
- Added anisotropic material support
- Different emitter for different pixels
- Predicts local frames from additional 6-D latent representation

**Training Command:**
```bash
python main.py \
    renderer=dynamicpoint_emitter \
    material=ani_latent_texture_model \
    experiment_name=Anisotropic_Debug_1
```

**Configuration:**
```yaml
# config/material/svpbr.yaml
pbr_texture_path: '/mnt/data/colin/colin/BRDF-Fipt/denim_fabric_03_4k/textures'
```

**Notes:**
- Anisotropic materials exhibit direction-dependent reflectance (e.g., brushed metal, fabric)
- Local frame prediction allows modeling of tangent-space anisotropy

---

### `is-all-images`
**Base Branch:** From `anisotropic`

**Key Features:**
- Precomputes all rendered images
- Applies importance sampling (IS) on all pixels

**Notes:**
- Improves training efficiency by avoiding redundant rendering
- Full-image importance sampling for better convergence

---

## Real Image Training Branches

### `nvdiffrast-initialized-mesh`
**Base Branch:** Major milestone for real geometry

**Key Features:**
1. Load mesh in `scene_loader.py`
2. Mesh initialization with `initial_mesh.py`: Uses nvdiffrast to optimize mesh & PBR material
3. Mesh processing with `process_mesh.py`: Uses Blender (bpy) to generate UV and tangent space

**Mesh Initialization Command:**
```bash
python initial_mesh.py \
    renderer=realcapture_emitter \
    material=latent_texture_model \
    experiment_name=Load-mesh-pbr-constrain-No-normal-z-constrain_1 \
    renderer.mesh.path="/media/raid/zla247/BRDF-Fipt/cube_with_uv_transformed.obj" \
    dataset_folder='/media/raid/zla247/BRDF-Fipt/reference_images_isotropic'
```

**Implementation Details:**
- `self.load_mesh`: Loads mesh and fixes vertex positions
- `self.init_mesh`: Initializes a rectangle mesh with learnable vertices

---

### `calibrations`
**Base Branch:** Major infrastructure branch

**Key Features:**
1. Added 2DGS (2D Gaussian Splatting) to the repository
2. Added nvdiffrast integration
3. Comprehensive COLMAP and calibration pipeline

**Resources:**
- [Notion Documentation](https://www.notion.so/Material-Acquisition-Project-2cf2e2352fde433ab1b00f408743125c?source=copy_link#2583b65d71eb8093be13e9bbc21d5a08)

**Notes:**
- Fundamental branch for real-world capture calibration
- Enables camera pose estimation and scene reconstruction

---

### `real-images-training`
**Base Branch:** From `calibrations`

**Key Features:**
- Adjusted model to train on real captured images
- Integration of calibration pipeline with training

**Notes:**
- Bridge between synthetic and real-world data
- Applies calibrated camera parameters to training

---

### `reconstructed-mesh`
**Base Branch:** From `nvdiffrast-initialized-mesh` and `calibrations`

**Key Features:**
- Complete pipeline for mesh reconstruction
- Chunked importance sampling on all pixels
- Image generation for reconstructed meshes
- COLMAP integration for mesh extraction

**Launch Configuration (COLMAP Quick):**
```bash
python colmap_quick.py \
    --images /mnt/data/colin/colin/BRDF-Fipt/south-building/images \
    --workdir /mnt/data/colin/colin/BRDF-Fipt/south-building/colmap_output \
    --camera_file /mnt/data/colin/colin/BRDF-Fipt/south-building/sparse/cameras.txt \
    --output_mesh /mnt/data/colin/colin/BRDF-Fipt/south-building/mesh.ply
```

**Key Commits:**
- `b7d7c9e`: Finished chunked importance sampling on all pixels
- `9b7228d`: Finished images generation
- `4612b0a`: Finished anisotropic material

---

### `training-with-turntable`
**Base Branch:** From `calibrations`

**Key Features:**
- Support for turntable rotation during capture
- Turntable mask generation
- Correct turntable rotation for emitter
- Multiple area light support
- Height map support

**Shape Matching Command:**
```bash
python ./recon/calibration/shape_matching.py \
    --scan_log_path /media/raid/cloth/New_turntable_Sep15/Lower_exposure_BRDF_recon/scan_log_0918_reindexed.json \
    --mesh_path None \
    --model_path /media/raid/cloth/New_turntable_Sep15/Lower_exposure_BRDF_recon/undistorted/sparse/0
```

**Key Commits:**
- `a9f519a`: Added turntable code without debug
- `c6315e4`: Added turntable mask
- `7d19468`: Correct turntable rotation for emitter
- `0c40c5f`: Added height map
- `d2889f9`: Updated to multiple area light
- `b6234e9`: Initial real area emitter and MIS path-tracing

**Notes:**
- Essential for 360° object capture workflows
- Turntable calibration ensures consistent object rotation

---

## Optimization and Advanced Features

### `speed-optimize_is-complex-svbrdf`
**Base Branch:** From `is-complex-svbrdf`

**Key Features:**
- Speed optimization with clean history
- Added MLP BRDF model
- Point light rendering optimizations

**Training Command:**
```bash
python main.py \
    renderer=dynamicpoint_emitter \
    material=svlatent_model \
    experiment_name=Time_recording_brdf-class_read-pbr-texture_5-epochs-128-iter_baseline
```

**Key Commit:**
- `4cc2b7a`: Speed-optimize – clean history

**Notes:**
- Major performance improvements for complex SVBRDF training
- Baseline for timing comparisons

---

### `multi_views`
**Base Branch:** Focus on multi-view capture

**Key Features:**
- Added SH (Spherical Harmonics) encoding
- Multi-point emitter support
- Evenly distributed camera views

**Testing Command:**
```bash
python test.py \
    renderer=envmap_emitter \
    material=mlp_pbr \
    experiment_name=Eight-point-light-test_1 \
    model.ckpt_path=<checkpoint_path>
```

**Example Checkpoint:**
```bash
model.ckpt_path=/localhome/zla247/theia2_data/output/BRDF-Fipt/Over-fit_experiments_2/sphere/Eight-point-lights_Larger-MLP_pos-enc_Local-coordinate_Evenly-camera-views_L1-loss_point-emitter_run_1/training/model_0.20_0.20/last-v1.ckpt
```

**Key Commits:**
- `de44b09`: Added SH encoding and multi-point emitter
- `ea0bc32`: Updates for multi-view support

---

### `colorful-anisotropic`
**Base Branch:** From `anisotropic`

**Key Features:**
- Added three designs for color texture
- Enhanced anisotropic material with color support
- Object rotation and cleaned emitter code

**Training Command:**
```bash
python main.py \
    renderer=dynamicpoint_emitter \
    material=latent_texture_model \
    experiment_name=Debug_1
```

**Key Commits:**
- `81ae599`: Added three designs for color texture
- `ed254a2`: Rotate the object and clean the emitter code

---

### `flat-material`
**Base Branch:** Simplified material branch

**Key Features:**
- Simplified flat (non-spatially-varying) material model
- Useful for homogeneous materials

**Training Command:**
```bash
python main.py \
    renderer=dynamicpoint_emitter \
    material=latent_texture_model \
    experiment_name=Debug_1
```

---

### `multi-light-render-one`
**Base Branch:** Optimized rendering branch

**Key Features:**
- Added rotation video trajectory generation
- Optimize rendering from separated point lights
- SH encoding for lighting

**Testing Command:**
```bash
python pointlight_test.py \
    renderer=dynamicpoint_emitter \
    material=latent_texture_model \
    experiment_name=Latent-Texture-2048*2048*8_test_1 \
    model.ckpt_path=<checkpoint_path>
```

**Example Checkpoint:**
```bash
model.ckpt_path=/mnt/data/colin/colin/output/BRDF-Fipt/Over-fit_experiments_2/sphere/Theia2_Scale-1.0_Correct-cameras_Pixel-center-rays_Latent-grid-2048-2048-8_run_1/training/model_0.20_0.20/last.ckpt
```

**Key Commits:**
- `418bc00`: Added rotation video trajectory
- `8729aaa`: Optimize from separated point lights

---

### `mipmap`
**Base Branch:** From `training-with-turntable`

**Key Features:**
- Mipmap support for texture filtering
- Neural geometry representation
- Streaming data loading for large datasets
- SH encoding improvements
- Correct camera axis estimation

**Training Command:**
```bash
python main.py \
    dataset_folder=/media/raid/cloth/No_cable_capture_Sep22/BRDF_recon_Sep30/Controlled_light/hdr \
    renderer=realcapture_area_emitter \
    material=ani_latent_texture_model \
    experiment_name=Continue_debug_2 \
    model.ckpt_path=<checkpoint_path>
```

**Example Checkpoint:**
```bash
model.ckpt_path=/media/raid/cloth/output/BRDF/real/All-images_Correct-cam_Neural-geometry_Stream-training-chunk-400-iter-5000_run_1/training/model_0.20_0.20/last.ckpt
```

**Material Configuration:**
```yaml
# config/material/mipmap_learnablesvpbr.yaml
# Mipmap-specific settings for learnable SVPBR
```

**Key Commits:**
- `9657661`: Add neural geometry & streaming data loading
- `b3d5913`: Update Mipmap and axis estimation
- `0c40c5f`: Added height map

**Notes:**
- Mipmapping reduces aliasing and improves rendering quality
- Streaming data loader handles datasets too large for memory

---

## Latest Development

### `trainig-time-debug` (Current Active Branch)
**Base Branch:** From `mipmap` and `training-with-turntable`

**Key Features:**
- **Training time optimization with double memory allocation**
- Color correction matrix implementation
- Enhanced SH encoding
- Neural geometry with streaming data loading
- Turntable support improvements
- Height map integration
- Multiple area light rendering

**Color Correction Command:**
```bash
python ./recon/calibration/color_matrix.py \
    --lab_file ColorChecker24_After_Nov2014.txt \
    --images_folder /media/raid/cloth/capture_data/BRDF_recon_Oct12_color_checker/hdr \
    --out_suffix _ccorr.png
```

**Training Command:**
```bash
python main.py \
    dataset_folder=<path_to_hdr_images> \
    renderer=realcapture_area_emitter \
    material=ani_latent_texture_model \
    experiment_name=Remove-duplicated_SH-time-record_1
```

**Example Launch Configuration:**
```bash
python main.py \
    dataset_folder=/media/raid/cloth/No_cable_capture_Sep22/BRDF_recon_Sep30/Controlled_light/hdr \
    renderer=realcapture_area_emitter \
    material=ani_latent_texture_model \
    experiment_name=Remove-duplicated_SH-time-record_1
```

**Key Commits & Changes:**
- `ddc6f82`: Update color correction matrix code
  - Added `ColorChecker24_After_Nov2014.txt` for color calibration
  - New color correction pipeline in `recon/calibration/color_matrix.py`
- `a1afe84`: Fix dimension issue (critical bug fix)
- `21e4162`: Update the SH encoding
- `9657661`: Add neural geometry & streaming data loading
  - Streaming data loader for handling large datasets
  - Neural geometry representation for better shape modeling
  - Enhanced `config/data/real.yaml` with streaming options
- `7d19468`: Correct turntable rotation for emitter
- `0c40c5f`: Added height map
- `b3d5913`: Update Mipmap and axis estimation

**Major Optimization (Primary Focus):**
- **Double memory allocation** to speed up training significantly
- Improved data loading pipeline with streaming
- Reduced bottlenecks in BRDF evaluation
- See `BOTTLENECK_FIX_EXPLANATION.md` for detailed performance analysis

**Calibration Features:**
- Color correction using ColorChecker24 reference
- Shape matching for geometry alignment
- Turntable axis estimation in COLMAP space
- Per-image reprojection error tracking (`per_image_errors.txt`)

**Important Files:**
- `ColorChecker24_After_Nov2014.txt`: Color calibration reference values
- `TRAINING_LOG.md`: Training session logs and timing records
- `BOTTLENECK_FIX_EXPLANATION.md`: Performance optimization documentation
- `per_image_errors.txt`: Reprojection error metrics

**Notes:**
- This branch focuses heavily on **production-scale training efficiency**
- Double memory technique trades memory for speed (ensure sufficient GPU RAM)
- Color correction is essential for consistent appearance across captures
- Streaming data loader allows training on datasets exceeding GPU memory
- **Critical for real-world deployment with large-scale captures**

---

## Branch Dependency Tree

```
texturelatent
├── svlatent
│   └── uniform-training-data
└── is_texture_latent
    └── is-complex-svbrdf
        ├── anisotropic
        │   ├── is-all-images
        │   └── colorful-anisotropic
        └── speed-optimize_is-complex-svbrdf

nvdiffrast-initialized-mesh
└── calibrations
    ├── real-images-training
    ├── reconstructed-mesh
    └── training-with-turntable
        └── mipmap
            └── trainig-time-debug (CURRENT - Training Optimization)

multi_views (standalone - multi-view focus)
flat-material (standalone - simplified material)
multi-light-render-one (standalone - optimized rendering)
```

---

## Common Configuration Patterns

### Renderer Options
- `dynamicpoint_emitter`: Point light sources with dynamic positioning
- `realcapture_emitter`: Real captured lighting (point lights)
- `realcapture_area_emitter`: Real captured area lights (more realistic)
- `envmap_emitter`: Environment map lighting

### Material Models
- `latent_texture_model`: Single-scale latent texture
- `svlatent_model`: Spatially-varying latent neural field
- `ani_latent_texture_model`: Anisotropic latent texture
- `mlp_pbr`: MLP-based PBR model
- `svpbr`: Spatially-varying PBR
- `mipmap_learnablesvpbr`: Mipmap-enabled learnable SVPBR

### Common Parameters
- `experiment_name`: Name for the experiment output
- `test_on_train`: Whether to test on training data
- `model.ckpt_path`: Path to checkpoint for resuming/testing
- `dataset_folder`: Path to input images
- `model.loss.recon_loss.name`: Loss function (l1, l2, etc.)
- `data.importance_sampling`: Enable importance sampling
- `data.uniform_sampling`: Use uniform BRDF query sampling

---

## Tips for Development

1. **Starting New Experiments:**
   - Always specify a clear `experiment_name`
   - Use descriptive names that include key parameters (e.g., loss type, sampling strategy)
   
2. **Checkpointing:**
   - Checkpoints are saved in `model_<roughness>_<metallic>/last.ckpt`
   - Always verify checkpoint paths before long training runs

3. **Calibration Workflow:**
   - For real captures: `calibrations` → `training-with-turntable` → `trainig-time-debug`
   - Always run color correction before training on real images
   - Verify turntable axis if using rotating capture setup

4. **Performance Optimization:**
   - Use `trainig-time-debug` branch for production training (optimized)
   - Enable streaming data loader for large datasets
   - Monitor GPU memory usage with double memory allocation

5. **Material Selection:**
   - Isotropic materials: Use `latent_texture_model` or `svlatent_model`
   - Anisotropic materials: Use `ani_latent_texture_model`
   - Complex real-world materials: Use `mipmap_learnablesvpbr` with neural geometry

---

## Version History Summary

| Version | Branch | Key Innovation |
|---------|--------|----------------|
| 1.0 | `texturelatent` | Basic latent texture |
| 2.0 | `svlatent` | Neural field representation |
| 3.0 | `is_texture_latent` | Importance sampling |
| 4.0 | `anisotropic` | Anisotropic materials |
| 5.0 | `calibrations` | Real capture pipeline |
| 6.0 | `training-with-turntable` | Turntable support |
| 7.0 | `mipmap` | Mipmap + neural geometry |
| 8.0 | `trainig-time-debug` | **Training optimization + color correction** |

---

## Contact & Resources

For more detailed information about specific branches, check:
- Individual commit messages: `git log origin/<branch-name>`
- Configuration files in `config/` directory
- Launch configurations in `.vscode/launch.json` (per-branch)
- [Project Notion](https://www.notion.so/Material-Acquisition-Project-2cf2e2352fde433ab1b00f408743125c)

---

*Last Updated: October 15, 2025*
*Current Active Branch: `trainig-time-debug`*

