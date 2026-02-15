"""
Example usage of BonnPointDataset for training.

This script demonstrates how to use the BonnPointDataset class
to load Bonn BRDF data and iterate through batches.
"""

import torch
from utils.dataset import BonnPointDataset
from dataclasses import dataclass

@dataclass
class Config:
    """Simple config for demonstration."""
    batch_size: int = 256

def main():
    # Configuration
    cfg = Config(batch_size=256)
    
    # Path to Bonn BRDF data
    data_dir = "/home/featurize/work/BRDF-Fipt_merl/utils/dataset/Bonn_test"
    material_name = "mat0001"
    
    # Create training dataset
    train_dataset = BonnPointDataset(
        cfg=cfg,
        data_dir=data_dir,
        material_name=material_name,
        split='train',
        device='cuda'
    )
    
    # Create validation dataset
    val_dataset = BonnPointDataset(
        cfg=cfg,
        data_dir=data_dir,
        material_name=material_name,
        split='val',
        device='cuda'
    )
    
    print("\n" + "="*60)
    print("Training iteration example:")
    print("="*60)
    
    # Example: iterate through a few training batches
    train_iter = iter(train_dataset)
    for i in range(5):
        batch = next(train_iter)
        print(f"\nBatch {i+1}:")
        print(f"  material_id shape: {batch['material_id'].shape}")
        print(f"  wi shape: {batch['wi'].shape}")
        print(f"  wo shape: {batch['wo'].shape}")
        print(f"  rgb shape: {batch['rgb'].shape}")
        print(f"  rgb range: [{batch['rgb'].min().item():.4f}, {batch['rgb'].max().item():.4f}]")
        print(f"  wi norm check: {torch.norm(batch['wi'], dim=-1).mean().item():.4f} (should be ~1.0)")
        print(f"  wo norm check: {torch.norm(batch['wo'], dim=-1).mean().item():.4f} (should be ~1.0)")
    
    print("\n" + "="*60)
    print("Validation iteration example:")
    print("="*60)
    
    # Example: iterate through validation dataset
    val_iter = iter(val_dataset)
    num_val_batches = 0
    for batch in val_iter:
        num_val_batches += 1
        if num_val_batches <= 3:  # Print first 3 batches
            print(f"\nValidation Batch {num_val_batches}:")
            print(f"  material_id shape: {batch['material_id'].shape}")
            print(f"  wi shape: {batch['wi'].shape}")
            print(f"  wo shape: {batch['wo'].shape}")
            print(f"  rgb shape: {batch['rgb'].shape}")
    
    print(f"\nTotal validation batches: {num_val_batches}")
    
    print("\n" + "="*60)
    print("Example training loop:")
    print("="*60)
    
    # Simulated training loop
    train_iter = iter(train_dataset)
    for epoch in range(2):
        print(f"\nEpoch {epoch+1}:")
        for step in range(10):  # 10 steps per epoch
            batch = next(train_iter)
            
            # Your training code here
            # For example:
            # loss = model(batch['wi'], batch['wo'], batch['material_id']) - batch['rgb']
            # loss = loss.mean()
            # loss.backward()
            # optimizer.step()
            
            if step % 5 == 0:
                print(f"  Step {step}: batch_size={batch['rgb'].shape[0]}, "
                      f"rgb_mean={batch['rgb'].mean().item():.4f}")

if __name__ == "__main__":
    main()
