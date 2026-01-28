#!/usr/bin/env python3
"""
Simple example demonstrating the simplified MERL BRDF dataset.
"""

import torch
from utils.dataset.points import MERLBRDFIterableDataset

def main():
    # Path to your MERL BRDF data folder
    data_folder = "/path/to/merl/brdfs"  # Change this to your actual path
    
    print("="*70)
    print("Simplified MERL BRDF Dataset Example")
    print("="*70)
    
    # Create dataset - this loads everything to GPU
    dataset = MERLBRDFIterableDataset(
        data_folder=data_folder,
        batch_size=2048,
        normalize=True,
        filter_invalid=True
    )
    
    print(f"\nDataset ready!")
    print(f"  Total samples: {dataset.n_samples:,}")
    print(f"  Batch size: {dataset.batch_size}")
    print(f"  Device: {dataset.device}")
    
    # Get some batches
    print(f"\nSampling 5 batches...")
    iterator = iter(dataset)
    
    for i in range(5):
        batch = next(iterator)
        
        print(f"\nBatch {i+1}:")
        print(f"  RGB shape: {batch['rgb'].shape}")
        print(f"  RGB device: {batch['rgb'].device}")
        print(f"  RGB range: [{batch['rgb'].min():.4f}, {batch['rgb'].max():.4f}]")
        print(f"  Coords shape: {batch['coords'].shape}")
        print(f"  Coords range: [{batch['coords'].min():.4f}, {batch['coords'].max():.4f}]")
        print(f"  Material IDs: {batch['material_ids'].unique().tolist()}")
        print(f"  Unique materials in batch: {len(batch['material_ids'].unique())}")
    
    # Simulate training loop
    print(f"\n" + "="*70)
    print("Simulating Training Loop (1000 iterations)")
    print("="*70)
    
    import time
    
    # Warmup
    for _ in range(10):
        batch = next(iterator)
    
    # Timed iterations
    start = time.time()
    n_iters = 1000
    
    for i in range(n_iters):
        batch = next(iterator)
        
        # Simulate some processing
        loss = (batch['rgb'].mean() - 0.5).pow(2)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            samples_per_sec = (i + 1) * dataset.batch_size / elapsed
            print(f"  Iteration {i+1}/{n_iters}: {samples_per_sec:.0f} samples/sec")
    
    total_time = time.time() - start
    avg_per_batch = total_time / n_iters * 1000  # ms
    throughput = n_iters * dataset.batch_size / total_time
    
    print(f"\n" + "="*70)
    print("Results:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average per batch: {avg_per_batch:.2f} ms")
    print(f"  Throughput: {throughput:,.0f} samples/second")
    print("="*70)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Override data folder from command line
        data_folder = sys.argv[1]
        
        from utils.dataset.points import MERLBRDFIterableDataset
        
        dataset = MERLBRDFIterableDataset(
            data_folder=data_folder,
            batch_size=2048,
            normalize=True,
            filter_invalid=True
        )
        
        print(f"\nDataset Info:")
        print(f"  Total samples: {dataset.n_samples:,}")
        print(f"  Device: {dataset.device}")
        print(f"  RGB tensor size: {dataset.rgb.shape}")
        print(f"  Coords tensor size: {dataset.coords.shape}")
        print(f"  Material IDs tensor size: {dataset.material_ids.shape}")
        
        print("\nGetting 3 sample batches...")
        iterator = iter(dataset)
        for i in range(3):
            batch = next(iterator)
            print(f"\nBatch {i+1}:")
            print(f"  RGB: {batch['rgb'].shape} on {batch['rgb'].device}")
            print(f"  Materials in batch: {len(batch['material_ids'].unique())}")
    else:
        print("Usage: python example_simplified_dataset.py <path_to_merl_brdfs>")
        print("\nExample:")
        print("  python example_simplified_dataset.py /path/to/merl/brdfs")




