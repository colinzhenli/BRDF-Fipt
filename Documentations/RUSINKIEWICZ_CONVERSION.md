# Rusinkiewicz Parameterization Implementation

## Overview

Added `directions_to_rusinkiewicz()` method to convert incoming (wi) and outgoing (wo) light directions to Rusinkiewicz parameterization used in the MERL BRDF database.

## Location

File: `model/neural_brdf_refactored.py`
Class: `MERLBRDF`
Method: `directions_to_rusinkiewicz(wi, wo)`

## What is Rusinkiewicz Parameterization?

Rusinkiewicz parameterization is a more efficient way to represent isotropic BRDFs compared to standard spherical coordinates. It uses three angles:

### Parameters

1. **theta_h** (θ_h): Half-angle between the half-vector and surface normal
   - Range: [0, π/2]
   - The half-vector h = normalize(wi + wo)
   - For flat surfaces with normal [0,0,1], this is `acos(h.z)`

2. **theta_d** (θ_d): Difference angle (half the angle between wi and wo)
   - Range: [0, π/2]
   - Computed as: `acos(wi · wo) / 2`
   - Measures how far apart the light and view directions are

3. **phi_d** (φ_d): Azimuthal angle difference in the plane perpendicular to the half-vector
   - Range: [0, 2π)
   - Captures the rotational difference between wi and wo around the surface normal

### Why Use This Parameterization?

For **isotropic materials** (materials that look the same when rotated around the surface normal):
- Only needs 3 parameters instead of 4 (no need for absolute phi_h)
- More uniform sampling distribution
- Better aligned with the symmetries of isotropic BRDFs
- Used by MERL BRDF database (90 × 90 × 180 = 1.46M samples)

## Implementation Details

### Function Signature

```python
def directions_to_rusinkiewicz(self, wi, wo):
    """
    Convert incoming (wi) and outgoing (wo) directions to Rusinkiewicz parameterization.
    
    Args:
        wi: [B, 3] incoming light directions (normalized)
        wo: [B, 3] outgoing view directions (normalized)
    
    Returns:
        theta_h: [B] half-angle [0, pi/2]
        theta_d: [B] difference angle [0, pi/2]
        phi_d: [B] azimuthal difference [0, 2*pi)
    """
```

### Algorithm Steps

1. **Normalize input directions**
   ```python
   wi = F.normalize(wi, dim=-1)
   wo = F.normalize(wo, dim=-1)
   ```

2. **Compute half-vector**
   ```python
   h = F.normalize(wi + wo, dim=-1)
   ```

3. **Calculate theta_h** (angle between h and normal [0,0,1])
   ```python
   theta_h = torch.acos(torch.clamp(h[..., 2], -1.0, 1.0))
   ```

4. **Calculate theta_d** (half the angle between wi and wo)
   ```python
   cos_diff = torch.sum(wi * wo, dim=-1)
   theta_d = torch.acos(torch.clamp(cos_diff, -1.0, 1.0)) / 2.0
   ```

5. **Calculate phi_d** (azimuthal difference)
   - Project wi and wo onto plane perpendicular to h
   - Compute angle between projections using atan2
   - Ensure result is in [0, 2π)

### Usage in eval_brdf_wiwo

The function is used in `eval_brdf_wiwo()` to convert input directions to the parameterization expected by the BRDF decoder:

```python
def eval_brdf_wiwo(self, wi, wo, material_id):
    # Convert directions to Rusinkiewicz coords
    theta_h, theta_d, phi_d = self.directions_to_rusinkiewicz(wi, wo)
    
    # Stack into [B, 3] tensor
    combined_dir = torch.stack([theta_h, theta_d, phi_d], dim=-1)
    
    # Retrieve latent codes
    latent = self.point_latent_bank(material_id)
    
    # Decode BRDF
    brdf = self.decoder(combined_dir, latent[:, :self.latent_dim])
    
    return brdf
```

## Assumptions

1. **Surface normal is [0, 0, 1]**: The implementation assumes the surface normal points in the +Z direction. If your wi/wo directions are in a different coordinate system, they need to be transformed to local tangent space first.

2. **Normalized directions**: Input wi and wo should be normalized (or will be normalized by the function).

3. **Isotropic materials**: This parameterization is designed for isotropic materials. For anisotropic materials, you would need the full 4-parameter representation including phi_h.

## Testing

To verify the conversion works correctly, you can:

1. Load MERL BRDF data with known coordinates
2. Convert back to wi/wo using inverse transformation
3. Apply `directions_to_rusinkiewicz` and compare with original coords

Example test:
```python
# Original Rusinkiewicz coords from MERL data
theta_h_orig = batch['coords'][:, 0]
theta_d_orig = batch['coords'][:, 1]
phi_d_orig = batch['coords'][:, 2]

# Convert to wi/wo (inverse transformation - needs to be implemented)
wi, wo = rusinkiewicz_to_directions(theta_h_orig, theta_d_orig, phi_d_orig)

# Convert back
theta_h_new, theta_d_new, phi_d_new = model.directions_to_rusinkiewicz(wi, wo)

# Compare (should be very close)
assert torch.allclose(theta_h_orig, theta_h_new, atol=1e-5)
assert torch.allclose(theta_d_orig, theta_d_new, atol=1e-5)
assert torch.allclose(phi_d_orig, phi_d_new, atol=1e-5)
```

## References

- Rusinkiewicz, S. (1998). "A New Change of Variables for Efficient BRDF Representation"
- MERL BRDF Database: https://www.merl.com/brdf/
- The parameterization reduces the dimensionality for isotropic BRDFs by exploiting rotational symmetry

## Related Files

- `utils/dataset/points.py`: MERL BRDF dataset loader that provides data in Rusinkiewicz coords
- `MERL_BRDF_DATALOADER_README.md`: Documentation of the MERL BRDF dataset format




