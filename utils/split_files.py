#!/usr/bin/env python3
"""
Split .binary files into train and test folders.

Usage:
    python split_files.py <source_dir> [--train-ratio 0.8] [--seed 42] [--move]
"""

import argparse
import shutil
import random
from pathlib import Path


def split_binary_files(source_dir, train_ratio=0.8, seed=42, move=False):
    """
    Split .binary files into train and test folders.
    
    Args:
        source_dir: Directory containing .binary files
        train_ratio: Ratio of files for training (default: 0.8)
        seed: Random seed for reproducibility (default: 42)
        move: If True, move files instead of copying (default: False)
    """
    source_path = Path(source_dir)
    
    # Find all .binary files
    binary_files = list(source_path.glob("*.binary"))
    
    if len(binary_files) == 0:
        print(f"No .binary files found in {source_dir}")
        return
    
    print(f"Found {len(binary_files)} .binary files")
    
    # Shuffle files for random split
    random.seed(seed)
    random.shuffle(binary_files)
    
    # Calculate split point
    n_train = int(len(binary_files) * train_ratio)
    n_test = len(binary_files) - n_train
    
    train_files = binary_files[:n_train]
    test_files = binary_files[n_train:]
    
    print(f"Split: {n_train} train files ({train_ratio*100:.0f}%), {n_test} test files ({(1-train_ratio)*100:.0f}%)")
    
    # Create train and test directories
    train_dir = source_path / "train"
    test_dir = source_path / "test"
    
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    print(f"\nTrain directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    
    # Copy/move files to train folder
    print(f"\n{'Moving' if move else 'Copying'} files to train folder...")
    for file in train_files:
        dest = train_dir / file.name
        if move:
            shutil.move(str(file), str(dest))
        else:
            shutil.copy2(str(file), str(dest))
        print(f"  {file.name} -> train/")
    
    # Copy/move files to test folder
    print(f"\n{'Moving' if move else 'Copying'} files to test folder...")
    for file in test_files:
        dest = test_dir / file.name
        if move:
            shutil.move(str(file), str(dest))
        else:
            shutil.copy2(str(file), str(dest))
        print(f"  {file.name} -> test/")
    
    print(f"\n✓ Done! Split {len(binary_files)} files into:")
    print(f"  - train/: {n_train} files")
    print(f"  - test/:  {n_test} files")
    
    # List the files in each folder
    print(f"\nTrain files:")
    for file in sorted(train_files):
        print(f"  - {file.name}")
    
    print(f"\nTest files:")
    for file in sorted(test_files):
        print(f"  - {file.name}")


def main():
    source_dir="/home/featurize/data"
    train_ratio=0.8
    seed=42
    move=False
    split_binary_files(
        source_dir,
        train_ratio=train_ratio,
        seed=seed,
        move=move
    )


if __name__ == '__main__':
    main()

