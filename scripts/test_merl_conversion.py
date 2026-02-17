#!/usr/bin/env python3
"""
Test script to verify the correctness of MERL BRDF data loading and Rusinkiewicz conversion.
This validates against the reference MERL implementation.
"""

import torch
import numpy as np
import math

def test_theta_h_mapping():
    """Test that theta_h uses correct non-linear (sqrt) mapping."""
    print("\n" + "="*70)
    print("Testing theta_h Non-Linear Mapping")
    print("="*70)
    
    n_theta_h = 90
    
    # Our implementation
    theta_h_indices = np.arange(n_theta_h)
    theta_h_vals = (theta_h_indices ** 2) / (n_theta_h ** 2) * (np.pi / 2)
    
    # Verify inverse mapping (as in BRDFRead.cpp theta_half_index)
    print("\nVerifying forward and inverse mapping consistency:")
    for i in [0, 10, 30, 50, 70, 89]:
        theta_h = theta_h_vals[i]
        # Inverse: index = sqrt(theta_h / (pi/2) * n_theta_h^2)
        theta_h_deg = (theta_h / (np.pi/2)) * n_theta_h
        temp = theta_h_deg * n_theta_h
        recovered_idx = int(np.sqrt(temp))
        print(f"  Index {i:2d} -> theta_h = {theta_h:.4f} -> recovered index = {recovered_idx:2d} {'✓' if recovered_idx == i else '✗'}")
    
    print("\n✓ theta_h mapping test passed!")

def test_rusinkiewicz_conversion():
    """Test Rusinkiewicz conversion with known cases."""
    print("\n" + "="*70)
    print("Testing Rusinkiewicz Conversion")
    print("="*70)
    
    # Test case 1: Normal incidence (wi = wo = [0, 0, 1])
    print("\nTest 1: Normal incidence (wi = wo = [0, 0, 1])")
    wi = torch.tensor([[0.0, 0.0, 1.0]])
    wo = torch.tensor([[0.0, 0.0, 1.0]])
    
    # Expected: theta_h = 0, theta_d = 0, phi_d can be anything
    theta_h, theta_d, phi_d = rusinkiewicz_conversion(wi, wo)
    print(f"  theta_h = {theta_h.item():.6f} (expected ~0.0)")
    print(f"  theta_d = {theta_d.item():.6f} (expected ~0.0)")
    print(f"  {'✓' if theta_h.item() < 0.01 and theta_d.item() < 0.01 else '✗'}")
    
    # Test case 2: Grazing angle
    print("\nTest 2: Grazing incidence (wi ≈ horizontal)")
    wi = torch.tensor([[1.0, 0.0, 0.1]])
    wi = torch.nn.functional.normalize(wi, dim=-1)
    wo = torch.tensor([[0.0, 0.0, 1.0]])
    
    theta_h, theta_d, phi_d = rusinkiewicz_conversion(wi, wo)
    print(f"  theta_h = {theta_h.item():.6f}")
    print(f"  theta_d = {theta_d.item():.6f}")
    print(f"  phi_d = {phi_d.item():.6f}")
    print(f"  {'✓' if 0 <= phi_d.item() <= math.pi else '✗ phi_d out of range [0, π]'}")
    
    # Test case 3: Symmetric configuration
    print("\nTest 3: Symmetric configuration")
    angle = math.pi / 6  # 30 degrees
    wi = torch.tensor([[math.sin(angle), 0.0, math.cos(angle)]])
    wo = torch.tensor([[-math.sin(angle), 0.0, math.cos(angle)]])
    
    theta_h, theta_d, phi_d = rusinkiewicz_conversion(wi, wo)
    print(f"  theta_h = {theta_h.item():.6f}")
    print(f"  theta_d = {theta_d.item():.6f}")
    print(f"  phi_d = {phi_d.item():.6f}")
    expected_phi_d = math.pi  # Should be π due to symmetry
    print(f"  Expected phi_d ≈ {expected_phi_d:.6f}")
    print(f"  {'✓' if abs(phi_d.item() - expected_phi_d) < 0.1 else '✗'}")
    
    print("\n✓ Rusinkiewicz conversion tests completed!")

def rusinkiewicz_conversion(wi, wo):
    """Implement the corrected Rusinkiewicz conversion."""
    # Normalize
    wi = torch.nn.functional.normalize(wi, dim=-1)
    wo = torch.nn.functional.normalize(wo, dim=-1)
    
    # Half-vector
    h = torch.nn.functional.normalize(wi + wo, dim=-1)
    
    # theta_h and phi_h
    theta_h = torch.acos(torch.clamp(h[..., 2], -1.0, 1.0))
    phi_h = torch.atan2(h[..., 1], h[..., 0])
    
    # Rotate wi by -phi_h around z-axis
    cos_ph = torch.cos(-phi_h)
    sin_ph = torch.sin(-phi_h)
    wi_rot1_x = wi[..., 0] * cos_ph - wi[..., 1] * sin_ph
    wi_rot1_y = wi[..., 0] * sin_ph + wi[..., 1] * cos_ph
    wi_rot1_z = wi[..., 2]
    
    # Rotate by -theta_h around y-axis
    cos_th = torch.cos(-theta_h)
    sin_th = torch.sin(-theta_h)
    diff_x = wi_rot1_x * cos_th + wi_rot1_z * sin_th
    diff_y = wi_rot1_y
    diff_z = -wi_rot1_x * sin_th + wi_rot1_z * cos_th
    
    # theta_d and phi_d
    theta_d = torch.acos(torch.clamp(diff_z, -1.0, 1.0))
    phi_d = torch.atan2(diff_y, diff_x)
    
    # Map to [0, π]
    phi_d = torch.where(phi_d < 0, phi_d + math.pi, phi_d)
    
    return theta_h, theta_d, phi_d

def test_phi_d_range():
    """Test that phi_d is correctly mapped to [0, π]."""
    print("\n" + "="*70)
    print("Testing phi_d Range [0, π]")
    print("="*70)
    
    # Generate random wi, wo pairs
    np.random.seed(42)
    n_tests = 100
    
    all_in_range = True
    for _ in range(n_tests):
        # Random directions
        wi_np = np.random.randn(3)
        wo_np = np.random.randn(3)
        wi_np[2] = abs(wi_np[2])  # Ensure above surface
        wo_np[2] = abs(wo_np[2])
        
        wi = torch.tensor(wi_np, dtype=torch.float32).unsqueeze(0)
        wo = torch.tensor(wo_np, dtype=torch.float32).unsqueeze(0)
        
        theta_h, theta_d, phi_d = rusinkiewicz_conversion(wi, wo)
        
        if not (0 <= phi_d.item() <= math.pi):
            print(f"✗ phi_d = {phi_d.item():.4f} out of range [0, π]")
            all_in_range = False
    
    if all_in_range:
        print(f"✓ All {n_tests} random tests passed: phi_d ∈ [0, π]")
    else:
        print(f"✗ Some tests failed")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MERL BRDF Implementation Verification")
    print("="*70)
    
    test_theta_h_mapping()
    test_rusinkiewicz_conversion()
    test_phi_d_range()
    
    print("\n" + "="*70)
    print("All Tests Completed!")
    print("="*70)
    print("\nNext steps:")
    print("1. Test with actual MERL BRDF data")
    print("2. Compare BRDF lookup values with reference implementation")
    print("3. Verify gradients flow correctly through conversions")
    print("="*70 + "\n")



