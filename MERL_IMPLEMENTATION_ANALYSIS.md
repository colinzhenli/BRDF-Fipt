# MERL BRDF Implementation Analysis

## Critical Issues Found

After comparing with the official MERL C++ reference implementation (`BRDFRead.cpp`), I've identified several issues:

### Issue 1: ❌ Incorrect `theta_d` Calculation

**Our Implementation (WRONG):**
```python
# Line 1549-1550 in neural_brdf_refactored.py
cos_diff = torch.sum(wi * wo, dim=-1)
theta_d = torch.acos(torch.clamp(cos_diff, -1.0, 1.0)) / 2.0
```

This computes theta_d as **half the angle between wi and wo**, which is incorrect!

**Correct MERL Implementation (from BRDFRead.cpp lines 119-124):**
```cpp
// Rotate incoming vector into half-vector coordinate frame
rotate_vector(in, normal, -fi_half, temp);
rotate_vector(temp, bi_normal, -theta_half, diff);
theta_diff = acos(diff[2]);
```

The correct theta_d is computed by:
1. Rotating the incoming vector by -phi_half around the z-axis (normal)
2. Then rotating by -theta_half around the y-axis (binormal)  
3. Taking the polar angle of the resulting difference vector

### Issue 2: ⚠️ phi_d Range and Reciprocity

**MERL C++ Implementation (lines 167-174):**
```cpp
// Because of reciprocity, the BRDF is unchanged under
// phi_diff -> phi_diff + M_PI
if (phi_diff < 0.0)
    phi_diff += M_PI;

// In: phi_diff in [0 .. pi]
// Out: tmp in [0 .. 179]
int tmp = int(phi_diff / M_PI * BRDF_SAMPLING_RES_PHI_D / 2);
```

Key points:
- MERL only stores phi_d in **[0, π]**, not [0, 2π)
- Due to reciprocity: phi_d and phi_d + π give the same BRDF value
- The actual file has 180 phi_d samples (half of 360), not 360

**Our Data Loading (lines 734 in points.py):**
```python
phi_d_vals = np.linspace(0, 2 * np.pi, n_phi_d, endpoint=False)
```

This is technically **WRONG** but might work if n_phi_d = 180 (not 360).

### Issue 3: ❌ Non-linear theta_h Sampling

**MERL C++ Implementation (lines 134-145):**
```cpp
inline int theta_half_index(double theta_half)
{
    double theta_half_deg = ((theta_half / (M_PI/2.0))*BRDF_SAMPLING_RES_THETA_H);
    double temp = theta_half_deg*BRDF_SAMPLING_RES_THETA_H;
    temp = sqrt(temp);  // <-- NON-LINEAR MAPPING!
    int ret_val = (int)temp;
    // ...
}
```

MERL uses a **non-linear (square root) mapping** for theta_h, not linear!

**Our Data Loading (line 732):**
```python
theta_h_vals = np.linspace(0, np.pi / 2, n_theta_h)  # LINEAR - WRONG!
```

This is **WRONG**! We're using linear sampling when MERL uses sqrt-based sampling.

## Summary of Issues

| Component | Issue | Severity | Status |
|-----------|-------|----------|--------|
| Data Loading - theta_h | Linear instead of sqrt mapping | 🔴 Critical | Need to fix |
| Data Loading - phi_d | Range [0, 2π) instead of [0, π] | 🟡 Medium | Need to verify n_phi_d |
| Conversion - theta_d | Using wi·wo/2 instead of rotation method | 🔴 Critical | Need to fix |
| Conversion - phi_d | Range and reciprocity handling | 🟡 Medium | Need to verify |

## Correct Implementation

### 1. Data Loading (points.py)

```python
# Correct theta_h sampling (non-linear)
theta_h_indices = np.arange(n_theta_h)
theta_h_vals = (theta_h_indices ** 2) / (n_theta_h ** 2) * (np.pi / 2)

# Correct theta_d sampling (linear)
theta_d_vals = np.linspace(0, np.pi / 2, n_theta_d)

# Correct phi_d sampling - only [0, π] due to reciprocity
phi_d_vals = np.linspace(0, np.pi, n_phi_d)
```

### 2. Rusinkiewicz Conversion (neural_brdf_refactored.py)

```python
def directions_to_rusinkiewicz(self, wi, wo):
    # Normalize
    wi = F.normalize(wi, dim=-1)
    wo = F.normalize(wo, dim=-1)
    
    # Half-vector
    h = F.normalize(wi + wo, dim=-1)
    
    # theta_h and phi_h
    theta_h = torch.acos(torch.clamp(h[..., 2], -1.0, 1.0))
    phi_h = torch.atan2(h[..., 1], h[..., 0])
    
    # Rotate wi into half-vector coordinate frame
    # First rotate by -phi_h around z-axis
    cos_ph = torch.cos(-phi_h)
    sin_ph = torch.sin(-phi_h)
    wi_rot1_x = wi[..., 0] * cos_ph - wi[..., 1] * sin_ph
    wi_rot1_y = wi[..., 0] * sin_ph + wi[..., 1] * cos_ph
    wi_rot1_z = wi[..., 2]
    
    # Then rotate by -theta_h around y-axis
    cos_th = torch.cos(-theta_h)
    sin_th = torch.sin(-theta_h)
    diff_x = wi_rot1_x * cos_th + wi_rot1_z * sin_th
    diff_y = wi_rot1_y
    diff_z = -wi_rot1_x * sin_th + wi_rot1_z * cos_th
    
    # theta_d and phi_d from diff vector
    theta_d = torch.acos(torch.clamp(diff_z, -1.0, 1.0))
    phi_d = torch.atan2(diff_y, diff_x)
    
    # Handle reciprocity: map to [0, π]
    phi_d = torch.where(phi_d < 0, phi_d + math.pi, phi_d)
    phi_d = torch.where(phi_d > math.pi, phi_d - math.pi, phi_d)
    
    return theta_h, theta_d, phi_d
```

## Testing Recommendations

1. Load a known MERL material
2. Convert wi/wo to Rusinkiewicz coords using correct method
3. Look up BRDF value from dataset
4. Compare with lookup using original MERL C++ code
5. Values should match within numerical precision

## References

- MERL BRDF Database: http://www.merl.com/brdf/
- Rusinkiewicz, S. (1998). "A New Change of Variables for Efficient BRDF Representation"
- BRDFRead.cpp: Official MERL reference implementation



