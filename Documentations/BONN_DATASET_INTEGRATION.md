# Bonn SVBRDF Dataset - Structure & Integration Plan

## 1. Dataset Source & Documentation

- **Official webpage**: https://cg.cs.uni-bonn.de/btf/bonn_svbrdf_database.html
- **APPBENCH paper (2020)**: https://cg.cs.uni-bonn.de/backend/v1/files/publications/merzbach2020bonn.pdf
- **Original paper**: Merzbach et al., "Learned Fitting of Spatially Varying BRDFs", Computer Graphics Forum, 2019
- **Dataset name**: UBOFAB19 (fabric materials, release version 1)
- **Scanner**: TAC7 commercial appearance scanner by X-Rite

### Data locations on disk
- Training: `/media/raid/cloth/Bonn_train/` (312 materials, 2801 files)
- Validation: `/media/raid/cloth/Bonn_val/` (68 materials, 605 files)
- No AxF SVBRDF files downloaded (only measurement data)

### Citation
```bibtex
@article{merzbach2019learned,
    author = {Merzbach, Sebastian and Hermann, Max and Rump, Martin and Klein, Reinhard},
    title = {Learned Fitting of Spatially Varying BRDFs},
    journal = {Computer Graphics Forum},
    volume = {38}, number = {4}, year = {2019}, month = jul,
    url = {https://cg.cs.uni-bonn.de/svbrdfs/}
}
```

---

## 2. Per-Material File Structure

Each material `matXXXX` has 8-9 files:

| File | Size (example) | Description |
|------|---------------|-------------|
| `matXXXX.txt` | ~150B | Metadata: fabric type, glossiness, Fresnel F0, pixel & physical dimensions |
| `matXXXX_calibration.mat` | ~24KB | Light/camera positions per turntable rotation, LLS corners, poly-to-pan mappings |
| `matXXXX_pan.exr` | ~140MB | 388 panchromatic (grayscale) HDR images, shape `(H, W, 388)` float16 |
| `matXXXX_poly.exr` | ~129MB | 100 polychromatic (RGB) HDR images, shape `(H, W, 300)` float16 |
| `matXXXX_lls.exr` | ~100MB | 280 linear light source (grayscale) HDR images, shape `(H, W, 280)` float16 |
| `matXXXX_xyz_rot000.exr` | ~2.2MB | Per-pixel 3D world coordinates (mm) for reference rotation, shape `(H, W, 3)` float32 |
| `matXXXX_xyz_others.exr` | ~8.1MB | Per-pixel 3D coords for turntable rotations 45/90/135/180 |
| `matXXXX_xyz_refined_rot000.exr` | ~2.3MB | Refined pixel coords (from Pantora fitting, for evaluation only) |
| `matXXXX_xyz_refined_others.exr` | ~8.3MB | Refined coords for other rotations |

Example metadata (`mat0001.txt`):
```
name:       material 0001
fabric type:    lycra
glossiness:     low
Fresnel F0:     0.0927
pixel dimensions:   512 px x 512 px
physical dimensions:    3.50 cm x 3.49 cm
```

---

## 3. Image Organization

### 3.1 TAC7 Scanner Setup
- **4 panchromatic cameras**: cv01 (top), cv02, cv03, cv04 (oblique)
- **29 white point-like LEDs**: il01-il24 (unfiltered), il26-il28, il31-il32 (color-filtered)
- **5 turntable rotations**: 0, 45, 90, 135, 180 degrees
- **Linear Light Source (LLS)**: rectangular diffuser, 14 inclination angles

### 3.2 Panchromatic Images (388 channels)
- 4 cameras × 24 unfiltered LEDs × 3 rotations (0, 90, 180) = 288
- 4 cameras × 5 filtered LEDs × 5 rotations (0, 45, 90, 135, 180) = 100
- Total = 388 grayscale channels
- Channel naming: `pan_cv01_il009_rot000` (single float16 per pixel)

### 3.3 Polychromatic Images (100 images = 300 channels)
- 4 cameras × 5 color-filtered LEDs (il026, il027, il028, il031, il032) × 5 rotations = 100
- Each image has R, G, B channels (3 channels per image)
- Channel naming: `poly_cv01_il026_rot000_R`, `..._G`, `..._B`
- Color space: **linear sRGB under illuminant E**

### 3.4 LLS Images (280 channels)
- 4 cameras × 14 LLS angles × 5 rotations = 280 grayscale channels
- LLS angles: [-2, 4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76] degrees
- Channel naming: `lls_cv01_lls01_la22.00_rot045`

### 3.5 Key Property: Pixel Reprojection
**All images are reprojected onto the top camera's pixel grid.** This means:
- Pixel (i, j) in ANY image channel corresponds to the SAME physical surface point
- One single xyz map (`xyz_rot000.exr`) gives 3D positions for ALL images
- Different turntable rotations are handled by rotating light/camera positions (not the surface)
- These are NOT raw camera images; they are warped observations in a common coordinate system
- Occluded pixels are marked with value **-1**

---

## 4. Calibration Data (`_calibration.mat`)

### Top-level keys:
- `rot000`, `rot045`, `rot090`, `rot135`, `rot180` — per-rotation light/camera positions
- `llsAnglesDegrees` — array of 14 LLS inclination angles
- `poly2panIndices` — mapping from poly image index to pan channel index (**1-based**)
- `poly2panWeights` — RGB-to-grayscale conversion weights per camera-LED pair

### Per-rotation dict (e.g., `rot000`):
- **29 LED positions**: `il01`..`il24`, `il26`, `il27`, `il28`, `il31`, `il32` — each `(3,)` float32 in **world coords (mm)**
- **4 camera positions**: `cv01`..`cv04` — each `(3,)` float32
- **`llsCorners`**: shape `(3, 4, 14)` — 4 corner positions of LLS quad for each of 14 angles

### Direction computation (from documentation):
```python
import pyexr
xyz = pyexr.read('mat0003_xyz_rot000.exr')  # (H, W, 3) in mm

# For turntable rotation 45°, camera 1, LED 9:
cam_pos = calib['rot045']['cv01']   # (3,) in mm
led_pos = calib['rot045']['il09']   # (3,) in mm

# Per-pixel directions:
wi = normalize(led_pos[None, None, :] - xyz)  # (H, W, 3) light direction
wo = normalize(cam_pos[None, None, :] - xyz)  # (H, W, 3) view direction
```

---

## 5. Radiometric Calibration — Why No Emitter Modeling is Needed

### White-frame calibration (from paper and documentation)
> "All images are radiometrically calibrated, i.e. all camera non-linearities are calibrated out of the data during HDR combination. The same holds for illumination or camera effects like light falloff or lens vignetting, which are removed via white-frame calibration."

White-frame calibration divides each pixel's measurement by a Lambertian white reference captured under the **exact same geometry** (same pixel, same LED, same camera, same rotation).

### For point-lit images (pan + poly):
Each pixel sees one LED from one direction. Everything cancels:
```
calibrated(x,y) = [BRDF_sample(wi,wo) × Le × cos_θi × G × cam_response]
                / [BRDF_white         × Le × cos_θi × G × cam_response]
                = BRDF_sample(wi,wo) / BRDF_white
```
**Result**: Le, cos_θi, G (distance), camera response — ALL cancel. Pixel value ∝ BRDF(wi, wo).

### Forward model for point-lit images:
```python
predicted = BRDF(wi, wo)    # Just BRDF evaluation, nothing else
loss = (predicted - calibrated_pixel) ** 2
```

**No emitter radiance (Le), no geometry term (G), no cos(θ_i), no distance falloff, no PDF.**

### For LLS images:
The LLS is a rectangular diffuser (approximately Lambertian emission). Each pixel integrates over the LLS solid angle. After white-frame calibration:
```
calibrated_LLS(x,y) = C × [Σ_k BRDF(wi_k, wo) × w_k] / [Σ_k w_k]
```
where `w_k = cos_θi_k × G_k` are geometric weights, and C is a constant.

This is a **geometry-weighted average of BRDF** values. The cos_θi terms act as weights (not multiplicative factors) — they determine how much each direction on the LLS contributes to the measurement.

### Forward model for LLS images:
```python
# Sample K points on the LLS quad
quad_samples = sample_quad(corners, K)
w_k = cos_theta_i_k / dist_k**2          # geometric weight per sample
predicted = sum(BRDF(wi_k, wo) * w_k) / sum(w_k)  # weighted average
```

### Confidence maps (for loss weighting, NOT in forward model):
The paper defines confidence maps: `w_i = min(m_i, max(0, <n,l> · <n,v>))`
- `m_i` = masking term (0 if occluded/shadowed)
- `<n,l> · <n,v>` = cos(θ_light) × cos(θ_view)
- These are **evaluation weights** to downweight unreliable grazing-angle pixels
- Used in loss: `loss = confidence × (predicted - measured)²`
- NOT part of the forward model

---

## 6. Integration into Current Point-Based Training Pipeline

### 6.1 Current Pipeline Structure (`MultiMaterialPointDataset`)
Each observation in the current pipeline:
```
[x, y, z, image_id, pixel_x, pixel_y, r, g, b, point_id]
```
- Points from COLMAP sparse reconstruction
- Camera rays generated from camera intrinsics + c2w matrices
- Emitter modeled explicitly (RealAreaEmitter/MultiAreaEmitter with directional distribution)
- Path tracing: `L = BRDF × Le × G / pdf`

### 6.2 Bonn Dataset Equivalent
For Bonn, each observation would be:
- `x, y, z` — from `xyz_rot000.exr` at pixel (i, j), in mm
- `image_id` — index into channel list (encodes camera + LED + rotation)
- `pixel_x, pixel_y` — pixel position in the common reprojected grid
- `r, g, b` — calibrated measurement (for poly images) or grayscale (for pan images)
- `point_id` — derived from (pixel_x, pixel_y)
- `wi` — computed as `normalize(LED_pos - xyz)` (no camera ray generation needed)
- `wo` — computed as `normalize(cam_pos - xyz)` (no camera intrinsics needed)

### 6.3 Key Simplifications vs Current Pipeline
| Aspect | Current Pipeline | Bonn Dataset |
|--------|-----------------|--------------|
| Point source | COLMAP sparse 3D points | Dense regular pixel grid from xyz map |
| Camera model | Intrinsics + c2w + ray generation | Direct: `wo = normalize(cam_pos - xyz)` |
| Light direction | Emitter sampling + intersection | Direct: `wi = normalize(LED_pos - xyz)` |
| Emitter radiance | Complex model (cos^m, calibration table) | **Not needed (Le = 1)** |
| Geometry term G | `cos_θi / dist²` explicitly computed | **Not needed (calibrated out)** |
| cos(θ_i) | Part of rendering equation | **Not needed (calibrated out)** |
| Forward model | `L = BRDF × Le × G / pdf` | **`L = BRDF(wi, wo)`** |
| Path tracing | Full emitter sampling + MIS | **Direct BRDF evaluation only** |
| Occluded pixels | Not applicable (only visible points) | Marked as -1, filter during loading |

### 6.4 Recommended Approach
**Option A — Direct BRDF evaluation (simplest, recommended to start):**
- Bypass path tracing entirely
- Compute wi, wo from calibration data + xyz map
- Call `material_net.eval_brdf(pos, wi, wo, ...)` directly
- Compare output to calibrated pixel value
- No emitter class needed

**Option B — Reuse path tracing code with minimal emitter:**
If keeping the existing code structure is desired, create a trivial emitter class:
- `sample_emitter()`: for point LED, return the LED position; for LLS, sample uniformly on quad
- `eval_emitter()`: return constant `Le = 1.0`
- No directional distribution, no distance falloff, no calibration tables
- Set G = 1, pdf = 1 in path tracing code (or skip those computations)

### 6.5 Emitter Modeling Details

#### Point LED Emitter (for 488 point-lit images per material)
- Each (camera, LED, rotation) = one emitter_id
- Store: `emitter_id → LED_position (3,)` and `emitter_id → cam_position (3,)`
- `eval_emitter()` returns `Le = 1.0` (constant)
- 388 panchromatic + 100 polychromatic = 488 unique (camera, LED, rotation) combinations
- This gives dense angular coverage — sufficient for BRDF fitting without LLS

#### LLS Emitter (for 280 LLS images per material, optional/later)
- Each (camera, LLS_angle, rotation) = one emitter_id
- Store: `emitter_id → 4 quad corners (3, 4)` from `llsCorners`
- `sample_emitter()`: sample K points uniformly on the quad
- `eval_emitter()`: return `Le = 1.0` (constant, diffuser assumption)
- Forward model: geometry-weighted average of BRDF values over quad
- Provides additional angular coverage, especially near-grazing directions

---

## 7. Data Loading Pseudocode

```python
import pyexr
import scipy.io as spio
import numpy as np

def load_bonn_material(mat_path, mat_id='mat0003'):
    """Load all data for one Bonn material."""
    
    # 1. Load xyz position map (shared by ALL images)
    xyz = pyexr.read(f'{mat_path}/{mat_id}_xyz_rot000.exr')  # (H, W, 3) float32, mm
    
    # 2. Load calibration (light/camera positions per rotation)
    calib = loadmat(f'{mat_path}/{mat_id}_calibration.mat')
    
    # 3. Load panchromatic images (388 grayscale channels)
    pan_file = pyexr.open(f'{mat_path}/{mat_id}_pan.exr')
    pan_channels = pan_file.channel_map['all']  # list of channel names
    pan_data = pan_file.get(group='all', precision=pyexr.HALF)  # (H, W, 388) float16
    
    # 4. Load polychromatic images (300 channels = 100 RGB images)
    poly_file = pyexr.open(f'{mat_path}/{mat_id}_poly.exr')
    poly_channels = poly_file.channel_map['all']
    poly_data = poly_file.get(group='all', precision=pyexr.HALF)  # (H, W, 300) float16
    
    # 5. Parse channel names to build (camera, LED, rotation) → channel_index mapping
    # Channel name format: 'pan_cv01_il009_rot000' or 'poly_cv01_il026_rot000_R'
    
    # 6. For each observation, compute:
    #    - position = xyz[i, j]                           (from xyz map)
    #    - wi = normalize(calib[rotXXX][ilYY] - position) (light direction)
    #    - wo = normalize(calib[rotXXX][cvZZ] - position) (view direction)
    #    - rgb = poly_data[i, j, ch:ch+3] or pan_data[i, j, ch] (measurement)
    #    - Filter out pixels where value == -1 (occluded)
    
    return observations
```

---

## 8. Notes & Caveats

1. **World coordinates are in millimeters.** LED/camera positions from calibration.mat are also in mm.
2. **Occluded pixels are marked as -1.** Must filter these during data loading.
3. **poly2panIndices uses 1-based indexing** (Matlab convention).
4. **Each material has different spatial resolution** (e.g., mat0001 is 512×512, mat0003 is 1306×1199).
5. **The xyz_others.exr file** contains xyz for rotations 45/90/135/180, but using the convention of rotating light/camera positions instead means you only need xyz_rot000.exr.
6. **Refined xyz coordinates** (xyz_refined_*.exr) are from Pantora fitting — use for evaluation only, not as input.
7. **No surface normal is provided directly.** If needed, compute from the xyz map via finite differences.
8. **Start with point-lit images only** (488 per material). Add LLS later if needed.
9. **Memory consideration**: Each material's pan.exr is ~140MB. Loading all 312 training materials simultaneously would require ~43GB. Use chunked/streaming loading similar to current pipeline.
10. **The Bonn paper uses Ward BRDF model** with spatially varying Fresnel for their baseline fits. Our neural BRDF model should be more expressive.
