#!/usr/bin/env python3
"""
Test script to demonstrate the performance of the simplified GPU-accelerated MERL BRDF dataset.
"""

import time
import torch
from utils.dataset.points import MERLBRDFIterableDataset

def test_dataset_speed(data_folder, batch_size=2048, num_iterations=100):
    """Test the dataset loading and sampling speed."""
    
    print(f"\n{'='*70}")
    print("Testing Simplified MERL BRDF Dataset Speed")
    print(f"{'='*70}")
    print(f"Batch size: {batch_size}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'='*70}\n")
    
    # Initialize dataset (this loads all data to GPU)
    print("Initializing dataset (loading all data to GPU)...")
    start_time = time.time()
    dataset = MERLBRDFIterableDataset(
        data_folder=data_folder,
        batch_size=batch_size,
        normalize=True,
        filter_invalid=True
    )
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.2f} seconds")
    print(f"Total samples loaded: {dataset.n_samples:,}\n")
    
    # Test sampling speed
    print(f"Testing sampling speed over {num_iterations} iterations...")
    iterator = iter(dataset)
    
    # Warmup
    for _ in range(10):
        batch = next(iterator)
    
    # Actual timing
    start_time = time.time()
    for i in range(num_iterations):
        batch = next(iterator)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (i + 1) * batch_size / elapsed
            print(f"  Iteration {i+1}/{num_iterations}: {samples_per_sec:.0f} samples/sec")
    
    total_time = time.time() - start_time
    avg_time_per_batch = total_time / num_iterations
    samples_per_sec = (num_iterations * batch_size) / total_time
    
    print(f"\n{'='*70}")
    print("Results:")
    print(f"{'='*70}")
    print(f"Total sampling time: {total_time:.3f} seconds")
    print(f"Average time per batch: {avg_time_per_batch*1000:.2f} ms")
    print(f"Throughput: {samples_per_sec:.0f} samples/second")
    print(f"{'='*70}\n")
    
    # Print sample batch info
    print("Sample batch info:")
    print(f"  RGB shape: {batch['rgb'].shape}")
    print(f"  Coords shape: {batch['coords'].shape}")
    print(f"  Material IDs shape: {batch['material_ids'].shape}")
    print(f"  RGB device: {batch['rgb'].device}")
    print(f"  Coords device: {batch['coords'].device}")
    print(f"  RGB range: [{batch['rgb'].min():.4f}, {batch['rgb'].max():.4f}]")
    print(f"  Coords range: [{batch['coords'].min():.4f}, {batch['coords'].max():.4f}]")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_dataset_speed.py <path_to_merl_brdf_folder>")
        print("\nExample:")
        print("  python test_dataset_speed.py /path/to/merl/brdfs")
        sys.exit(1)
    
    data_folder = sys.argv[1]
    test_dataset_speed(data_folder)

