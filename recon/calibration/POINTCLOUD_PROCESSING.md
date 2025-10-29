# Pointcloud Processing Documentation

## Overview

Added functionality to process Colmap sparse pointcloud (`points3D.ply`) with transformation, cropping, outlier removal, and bounding box computation.

## New Function: `process_pointcloud_to_base`

Located in `shape_matching.py`, this function performs the following operations:

### Pipeline Steps

1. **Load Pointcloud**: Loads the `points3D.ply` file from Colmap sparse reconstruction
2. **Transform to Base Frame**: Applies the same `T_BW` transformation used for mesh alignment
3. **XY Cropping**: Crops points based on the rectangle defined in the renderer config (`mesh.rectangle`)
   - Uses `center`, `width` (X), and `length` (Y) from config
   - Creates a rectangular selection region
4. **Z Outlier Removal**: Removes top and bottom percentiles to filter noise
   - Default: removes 5% from top and 5% from bottom (configurable)
   - Ensures only cloth surface points remain
5. **Bounding Box Computation**: Calculates axis-aligned bounding box
6. **Save Results**: Outputs filtered pointcloud and bounding box info

### Function Signature

```python
def process_pointcloud_to_base(pointcloud_path, T_BW, cfg, output_path=None, z_outlier_percentile=5.0):
    """
    Args:
        pointcloud_path: Path to point3d.ply file from Colmap
        T_BW: Transformation matrix from world to base frame
        cfg: Config dict containing mesh.rectangle parameters
        output_path: Path to save filtered pointcloud (optional)
        z_outlier_percentile: Percentage of points to remove from top/bottom (default 5%)
    
    Returns:
        tuple: (filtered_pcd, bbox_min, bbox_max, bbox_center, bbox_size)
    """
```

### Output Files

When processing a pointcloud at path `/path/to/points3D.ply`:

1. **Filtered Pointcloud**: `/path/to/points3D_transformed_filtered.ply`
   - Contains only the cloth surface points after all filtering
   - Same format as input (PLY with colors if available)

2. **Bounding Box JSON**: `/path/to/points3D_bbox.json`
   - JSON file with bounding box information:
   ```json
   {
     "bbox_min": [x_min, y_min, z_min],
     "bbox_max": [x_max, y_max, z_max],
     "bbox_center": [x_center, y_center, z_center],
     "bbox_size": [width, height, depth],
     "num_points": 12345
   }
   ```

### Console Output

The function prints detailed progress information:
```
Loaded 50000 points from /path/to/points3D.ply
Transformed pointcloud to base frame
Cropping rectangle: X=[0.080, 0.240], Y=[-0.210, -0.010]
After XY cropping: 15234 points (30.5%)
After Z outlier removal (5.0% top/bottom): 13710 points (90.0% of cropped)
Z range: [-0.065123, -0.061234]

============================================================
Axis-Aligned Bounding Box (Base Frame):
============================================================
Min:    [ 0.082456, -0.208765,  -0.065123]
Max:    [ 0.238901, -0.011234,  -0.061234]
Center: [ 0.160679, -0.109999,  -0.063179]
Size:   [ 0.156445,  0.197531,   0.003889]
============================================================

Filtered pointcloud saved to: /path/to/points3D_transformed_filtered.ply
Bounding box info saved to: /path/to/points3D_bbox.json
```

## Integration with Main Script

The `main_process` function now accepts two additional parameters:
- `pointcloud_path`: Path to the Colmap pointcloud (optional)
- `z_outlier_percentile`: Outlier removal percentage (default: 5.0)

The main function automatically looks for `points3D.ply` in the sparse model directory:
```python
pointcloud_path = str(Path(model_path) / "points3D.ply")
```

## Configuration

The function uses the renderer config (`config/renderer/realcapture_area_emitter.yaml`):

```yaml
mesh:
  rectangle:
    center: [0.16, -0.11, -0.063]  # XYZ center of cropping region
    width: 0.16   # X dimension (sample_size)
    length: 0.20  # Y dimension (sample_size)
```

These parameters define the rectangular region to keep in the XY plane.

## Usage Examples

### Standalone Usage

```python
from shape_matching import process_pointcloud_to_base, estimate_world2base
from read_write_model import read_model
from pathlib import Path
import hydra

# Load config
cfg = hydra.compose(config_name="realcapture_area_emitter")

# Get transformation matrix
model_path = "/path/to/sparse"
cameras, images = read_model(model_path, ext=".bin")
T_BW, cam_c2w, scan_id = estimate_world2base(...)

# Process pointcloud
pointcloud_path = Path(model_path) / "points3D.ply"
output_path = pointcloud_path.with_name("points3D_transformed_filtered.ply")

pcd, bbox_min, bbox_max, bbox_center, bbox_size = process_pointcloud_to_base(
    pointcloud_path=pointcloud_path,
    T_BW=T_BW,
    cfg=cfg,
    output_path=output_path,
    z_outlier_percentile=5.0  # Remove 5% from top and bottom
)

print(f"Bounding box center: {bbox_center}")
print(f"Bounding box size: {bbox_size}")
```

### Adjusting Outlier Removal

If you need more or less aggressive filtering:

```python
# More aggressive: remove 10% from top/bottom (keep only middle 80%)
process_pointcloud_to_base(..., z_outlier_percentile=10.0)

# Less aggressive: remove 2% from top/bottom (keep middle 96%)
process_pointcloud_to_base(..., z_outlier_percentile=2.0)
```

## Workflow

1. Run Colmap reconstruction to get `sparse/points3D.ply`
2. Run the shape matching script (automatically processes pointcloud)
3. Check console output for bounding box dimensions
4. Use filtered pointcloud for further processing
5. Load bbox JSON for integration with other tools

## Notes

- The function preserves point colors if available in the input PLY
- All coordinates are in the robot base frame after transformation
- The XY crop ensures only points within the sample region are kept
- Z outlier removal handles noise above/below the cloth surface
- Empty results after cropping will raise an error with helpful message

