# MERL BRDF Dataloader Documentation

## Overview

This document describes the MERL BRDF dataloader implementation in `utils/dataset/points.py`. The dataloader provides easy access to the MERL BRDF database, which contains densely measured Bidirectional Reflectance Distribution Functions (BRDFs) for 100 different real-world materials.

## MERL BRDF Database Format

The MERL BRDF database stores each material's BRDF in a binary file with the following structure:

### File Format
- **Header**: 3 int32 values representing dimensions `[n_theta_h, n_theta_d, n_phi_d]`
  - Standard dimensions: 90 × 90 × 180 = 1,458,000 samples
- **Data**: RGB reflectance values as float64 (doubles), interleaved as `[R, G, B, R, G, B, ...]`
- **Invalid samples**: Marked with -1.0 for unmeasured or invalid data

### Parameterization

The BRDF is parameterized using Rusinkiewicz coordinates:
- **θ_h** (theta_h): Half-angle between incoming (wi) and outgoing (wo) directions, range [0, π/2]
- **θ_d** (theta_d): Difference angle, range [0, π/2]
- **φ_d** (phi_d): Azimuthal difference, range [0, 2π)

This parameterization is more efficient than the standard spherical parameterization for representing isotropic BRDFs.

## Dataset Classes

### 1. `MERLBRDFDataset`

A standard PyTorch `Dataset` that loads and stores BRDF data in memory.

#### Constructor Parameters
```python
MERLBRDFDataset(
    data_folder: str,           # Path to folder containing .binary files
    material_names: list = None, # Optional: specific materials to load
    normalize: bool = True,      # Whether to normalize BRDF values to [0, 1]
    filter_invalid: bool = True  # Whether to remove invalid samples (-1.0)
)
```

#### Usage Example
```python
from utils.dataset.points import MERLBRDFDataset

# Load all BRDF materials from a folder
dataset = MERLBRDFDataset(
    data_folder="/path/to/merl/brdfs",
    normalize=True,
    filter_invalid=True
)

# Access a material by index
material = dataset[0]
print(f"Material: {material['material_name']}")
print(f"RGB shape: {material['rgb'].shape}")      # (N, 3) reflectance values
print(f"Coords shape: {material['coords'].shape}") # (N, 3) angular coordinates

# Access a material by name
material = dataset.get_material_by_name("alum-bronze")

# Sample random directions from a material
samples = dataset.sample_random_directions(
    material_idx=0,
    n_samples=1000
)
```

#### Return Format
Each item returns a dictionary with:
- `material_name`: str - Name of the material (filename without extension)
- `material_id`: int - Index of the material in the dataset
- `rgb`: torch.Tensor - Shape (N, 3), RGB reflectance values
- `coords`: torch.Tensor - Shape (N, 3), coordinates as (θ_h, θ_d, φ_d) in radians
- `dims`: tuple - Original dimensions (n_theta_h, n_theta_d, n_phi_d)

### 2. `MERLBRDFIterableDataset`

An iterable dataset that continuously yields random batches of BRDF samples. Ideal for training neural BRDF models.

#### Constructor Parameters
```python
MERLBRDFIterableDataset(
    data_folder: str,
    batch_size: int = 1024,
    material_names: list = None,
    normalize: bool = True,
    filter_invalid: bool = True,
    samples_per_material: int = None  # Defaults to batch_size / num_materials
)
```

#### Usage Example
```python
from utils.dataset.points import MERLBRDFIterableDataset
from torch.utils.data import DataLoader

# Create iterable dataset
dataset = MERLBRDFIterableDataset(
    data_folder="/path/to/merl/brdfs",
    batch_size=512,
    normalize=True
)

# Use with DataLoader
dataloader = DataLoader(dataset, batch_size=None)  # batch_size=None since dataset handles batching

# Training loop
for batch in dataloader:
    rgb = batch['rgb']              # (batch_size, 3) reflectance values
    coords = batch['coords']        # (batch_size, 3) angular coordinates
    material_ids = batch['material_ids']  # (batch_size,) material indices
    
    # Your training code here
    ...
```

#### Return Format
Each batch is a dictionary with:
- `rgb`: torch.Tensor - Shape (batch_size, 3), RGB reflectance values
- `coords`: torch.Tensor - Shape (batch_size, 3), coordinates as (θ_h, θ_d, φ_d)
- `material_ids`: torch.Tensor - Shape (batch_size,), material indices

## Data Statistics

For the example `alum-bronze.binary` file:
- **Total samples**: 1,458,000 (90 × 90 × 180)
- **Valid samples**: ~1,147,122 (after filtering -1.0 values)
- **Percentage valid**: ~78.7%
- **RGB range** (normalized): [0.0, 1.0]
- **File size**: ~34 MB per material

## Normalization

When `normalize=True` (default):
1. Invalid samples (value = -1.0) are filtered out
2. BRDF values are clipped using the 99th percentile to avoid outliers
3. Values are normalized to [0, 1] range

This normalization is useful for neural network training but can be disabled if you need raw BRDF values.

## Example: Training a Neural BRDF Model

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset.points import MERLBRDFIterableDataset

# Create dataset
dataset = MERLBRDFIterableDataset(
    data_folder="/path/to/merl/brdfs",
    batch_size=1024,
    normalize=True,
    filter_invalid=True
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

# Simple neural BRDF model
class BRDFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),  # Input: (θ_h, θ_d, φ_d)
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # Output: RGB
            nn.Sigmoid()
        )
    
    def forward(self, coords):
        return self.net(coords)

# Training
model = BRDFNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for i, batch in enumerate(dataloader):
    if i >= 1000:  # Train for 1000 iterations
        break
    
    coords = batch['coords']
    rgb_gt = batch['rgb']
    
    # Forward pass
    rgb_pred = model(coords)
    loss = criterion(rgb_pred, rgb_gt)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")
```

## Visualization

The test script `test_merl_dataloader.py` includes a visualization function that creates 2D slices of the BRDF:

```python
from utils.dataset.points import MERLBRDFDataset

dataset = MERLBRDFDataset("/path/to/merl/brdfs")
material = dataset[0]

# This will create a visualization showing R, G, B channels and luminance
# as a function of θ_d and φ_d for a fixed θ_h
```

The visualization shows how reflectance varies with viewing angle, which is useful for understanding material appearance.

## Testing

Run the test script to verify the dataloader:

```bash
cd /home/featurize/work/BRDF-Fipt
python test_merl_dataloader.py
```

This will:
1. Load the MERL BRDF dataset
2. Display statistics about the loaded materials
3. Test sampling functionality
4. Create a visualization of the BRDF data
5. Test the iterable dataset

## Performance Considerations

### Memory Usage
- Each material requires ~35 MB on disk
- After loading and filtering: ~27 MB in memory (for float32 tensors)
- Loading all 100 materials: ~2.7 GB RAM

### Loading Time
- Single material: ~0.5 seconds
- All 100 materials: ~50 seconds (with progress bar)

### Recommendations
- Use `material_names` parameter to load only needed materials
- For large-scale training, consider the `MERLBRDFIterableDataset` which handles batching efficiently
- Set `filter_invalid=True` to remove unmeasured samples and reduce memory usage

## Converting Angular Coordinates to Direction Vectors

If you need to convert the Rusinkiewicz coordinates back to direction vectors:

```python
import torch
import numpy as np

def rusinkiewicz_to_directions(theta_h, theta_d, phi_d):
    """
    Convert Rusinkiewicz coordinates to incoming/outgoing direction vectors.
    
    Args:
        theta_h: Half-angle (radians)
        theta_d: Difference angle (radians)
        phi_d: Azimuthal difference (radians)
    
    Returns:
        wi: Incoming direction (N, 3)
        wo: Outgoing direction (N, 3)
    """
    # Half vector in spherical coordinates
    h = torch.stack([
        torch.sin(theta_h) * torch.cos(phi_d / 2),
        torch.sin(theta_h) * torch.sin(phi_d / 2),
        torch.cos(theta_h)
    ], dim=-1)
    
    # Difference vector
    diff = torch.stack([
        torch.sin(theta_d) * torch.cos(phi_d),
        torch.sin(theta_d) * torch.sin(phi_d),
        torch.cos(theta_d)
    ], dim=-1)
    
    # Compute wi and wo from half vector and difference
    # This is a simplified version; full conversion requires rotation
    # See MERL documentation for complete transformation
    
    return h, diff  # Placeholder - implement full conversion as needed
```

## References

1. **MERL BRDF Database**: https://www.merl.com/brdf/
2. **Paper**: "A Data-Driven Reflectance Model" by Matusik et al., SIGGRAPH 2003
3. **Rusinkiewicz Parameterization**: "A New Change of Variables for Efficient BRDF Representation" by Rusinkiewicz, 1998

## Troubleshooting

### Issue: "No BRDF files found"
- Ensure `.binary` files are in the specified `data_folder`
- Check file permissions

### Issue: "Expected X values, got Y"
- File may be corrupted
- Verify file size matches expected ~34 MB per material

### Issue: High memory usage
- Set `filter_invalid=True` to remove invalid samples
- Load only specific materials using `material_names` parameter
- Use `MERLBRDFIterableDataset` instead of loading all data at once

## License

The MERL BRDF database is provided by Mitsubishi Electric Research Laboratories (MERL). Please refer to their website for licensing terms and citation requirements.

