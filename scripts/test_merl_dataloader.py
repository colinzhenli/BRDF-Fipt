#!/usr/bin/env python3
"""
Test script for MERL BRDF dataloader.

This script demonstrates how to use the MERLBRDFDataset and MERLBRDFIterableDataset
to load and sample from the MERL BRDF database.
"""

import torch
from utils.dataset.points import MERLBRDFDataset, MERLBRDFIterableDataset
import matplotlib.pyplot as plt
import numpy as np


def test_merl_dataset():
    """Test the basic MERLBRDFDataset."""
    print("\n" + "="*60)
    print("Testing MERLBRDFDataset")
    print("="*60)
    
    # Load dataset (should find alum-bronze.binary in the current directory)
    dataset = MERLBRDFDataset(
        data_folder="/home/featurize/work/BRDF-Fipt",
        normalize=True,
        filter_invalid=True
    )
    
    print(f"\nDataset size: {len(dataset)} materials")
    print(f"Material names: {dataset.material_names}")
    
    # Get first material
    material = dataset[0]
    print(f"\nMaterial: {material['material_name']}")
    print(f"  Material ID: {material['material_id']}")
    print(f"  RGB shape: {material['rgb'].shape}")
    print(f"  Coords shape: {material['coords'].shape}")
    print(f"  Dimensions: {material['dims']}")
    print(f"  RGB range: [{material['rgb'].min():.4f}, {material['rgb'].max():.4f}]")
    print(f"  RGB mean: {material['rgb'].mean(dim=0)}")
    
    # Sample random directions
    print("\nSampling 1000 random directions...")
    samples = dataset.sample_random_directions(0, 1000)
    print(f"  Sampled RGB shape: {samples['rgb'].shape}")
    print(f"  Sampled coords shape: {samples['coords'].shape}")
    print(f"  Material: {samples['material_name']}")
    
    # Visualize BRDF slice
    visualize_brdf_slice(material)
    
    return dataset


def test_merl_iterable_dataset():
    """Test the MERLBRDFIterableDataset."""
    print("\n" + "="*60)
    print("Testing MERLBRDFIterableDataset")
    print("="*60)
    
    # Create iterable dataset
    dataset = MERLBRDFIterableDataset(
        data_folder="/home/featurize/work/BRDF-Fipt",
        batch_size=512,
        normalize=True,
        filter_invalid=True
    )
    
    print(f"\nBatch size: {dataset.batch_size}")
    print(f"Samples per material: {dataset.samples_per_material}")
    print(f"Number of materials: {len(dataset.base_dataset)}")
    
    # Get a few batches
    print("\nIterating through 3 batches...")
    iterator = iter(dataset)
    for i in range(3):
        batch = next(iterator)
        print(f"\nBatch {i+1}:")
        print(f"  RGB shape: {batch['rgb'].shape}")
        print(f"  Coords shape: {batch['coords'].shape}")
        print(f"  Material IDs shape: {batch['material_ids'].shape}")
        print(f"  RGB range: [{batch['rgb'].min():.4f}, {batch['rgb'].max():.4f}]")
        print(f"  Unique materials in batch: {torch.unique(batch['material_ids']).tolist()}")
    
    return dataset


def visualize_brdf_slice(material, save_path="merl_brdf_visualization.png"):
    """
    Visualize a 2D slice of the BRDF data.
    
    Shows RGB reflectance as a function of theta_d and phi_d for a fixed theta_h.
    """
    print(f"\nCreating BRDF visualization...")
    
    rgb = material['rgb']
    coords = material['coords']
    dims = material['dims']
    
    # Extract coordinates
    theta_h = coords[:, 0]
    theta_d = coords[:, 1]
    phi_d = coords[:, 2]
    
    # Select a slice at theta_h ≈ 45 degrees (pi/4)
    target_theta_h = np.pi / 4
    theta_h_tolerance = np.pi / 180  # 1 degree tolerance
    
    slice_mask = torch.abs(theta_h - target_theta_h) < theta_h_tolerance
    
    if slice_mask.sum() == 0:
        print("Warning: No samples found for the selected theta_h slice")
        return
    
    slice_rgb = rgb[slice_mask]
    slice_theta_d = theta_d[slice_mask].numpy()
    slice_phi_d = phi_d[slice_mask].numpy()
    
    print(f"  Slice contains {len(slice_rgb)} samples")
    print(f"  theta_h ≈ {np.degrees(target_theta_h):.1f}°")
    
    # Create 2D grid visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Plot R, G, B channels and RGB composite
    channels = ['R', 'G', 'B']
    for i, (ax, channel) in enumerate(zip(axes[:3], channels)):
        scatter = ax.scatter(
            np.degrees(slice_phi_d),
            np.degrees(slice_theta_d),
            c=slice_rgb[:, i].numpy(),
            cmap='hot',
            s=1,
            vmin=0,
            vmax=1
        )
        ax.set_xlabel('φ_d (degrees)')
        ax.set_ylabel('θ_d (degrees)')
        ax.set_title(f'{channel} channel')
        plt.colorbar(scatter, ax=ax, label='Reflectance')
    
    # RGB composite (use luminance)
    luminance = 0.2126 * slice_rgb[:, 0] + 0.7152 * slice_rgb[:, 1] + 0.0722 * slice_rgb[:, 2]
    scatter = axes[3].scatter(
        np.degrees(slice_phi_d),
        np.degrees(slice_theta_d),
        c=luminance.numpy(),
        cmap='hot',
        s=1,
        vmin=0,
        vmax=1
    )
    axes[3].set_xlabel('φ_d (degrees)')
    axes[3].set_ylabel('θ_d (degrees)')
    axes[3].set_title('Luminance')
    plt.colorbar(scatter, ax=axes[3], label='Reflectance')
    
    plt.suptitle(f"BRDF Slice: {material['material_name']} (θ_h = {np.degrees(target_theta_h):.1f}°)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to {save_path}")
    plt.close()


def test_get_material_by_name():
    """Test getting material by name."""
    print("\n" + "="*60)
    print("Testing get_material_by_name")
    print("="*60)
    
    dataset = MERLBRDFDataset(
        data_folder="/home/featurize/work/BRDF-Fipt",
        normalize=True,
        filter_invalid=True
    )
    
    # Get material by name
    material_name = dataset.material_names[0]
    print(f"\nGetting material by name: '{material_name}'")
    material = dataset.get_material_by_name(material_name)
    print(f"  Successfully retrieved: {material['material_name']}")
    print(f"  RGB shape: {material['rgb'].shape}")
    
    # Try invalid name
    try:
        dataset.get_material_by_name("nonexistent-material")
    except ValueError as e:
        print(f"\n  Expected error for invalid name: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MERL BRDF Dataloader Test Suite")
    print("="*60)
    
    # Test basic dataset
    dataset = test_merl_dataset()
    
    # Test iterable dataset
    iterable_dataset = test_merl_iterable_dataset()
    
    # Test get by name
    test_get_material_by_name()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

