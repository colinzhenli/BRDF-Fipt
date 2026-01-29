#!/usr/bin/env python3
"""
Script to validate observation chunk files in the dataset.
Checks for empty files, corrupted files, and missing observations key.

Usage:
    python check_chunks.py /media/raid/cloth/capture_data/Dataset_Nov11
    
Or to only check files from training_list.txt:
    python check_chunks.py /media/raid/cloth/capture_data/Dataset_Nov11 --training-list
"""

import numpy as np
from pathlib import Path
import argparse
import sys
from tqdm import tqdm


def validate_chunk(chunk_path: Path) -> tuple:
    """
    Validate a single chunk file.
    
    Returns:
        (is_valid, error_type, error_msg)
    """
    # Check if file exists
    if not chunk_path.exists():
        return False, "MISSING", "File does not exist"
    
    # Check file size
    file_size = chunk_path.stat().st_size
    if file_size == 0:
        return False, "EMPTY", "File is 0 bytes"
    
    # Try to load the file
    try:
        data = np.load(chunk_path)
        if 'observations' not in data:
            return False, "NO_DATA", "Missing 'observations' key"
        obs = data['observations']
        if obs.shape[0] == 0:
            return False, "EMPTY_DATA", "Observations array is empty"
        return True, None, None
    except EOFError:
        return False, "EOF_ERROR", f"EOFError - truncated/corrupted file ({file_size} bytes)"
    except Exception as e:
        return False, type(e).__name__, str(e)


def main():
    parser = argparse.ArgumentParser(description='Validate observation chunk files')
    parser.add_argument('dataset_path', type=str, help='Path to dataset folder')
    parser.add_argument('--training-list', action='store_true', 
                        help='Only check folders from training_list.txt')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: only check file sizes, skip loading')
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    # Get list of folders to check
    if args.training_list:
        training_list_path = dataset_path / "training_list.txt"
        if not training_list_path.exists():
            print(f"Error: training_list.txt not found at {training_list_path}")
            sys.exit(1)
        
        with open(training_list_path, 'r') as f:
            indices = [line.strip() for line in f if line.strip()]
        folders = [dataset_path / idx / "observations" for idx in indices]
        print(f"Checking {len(indices)} folders from training_list.txt...")
    else:
        # Find all observation folders
        folders = list(dataset_path.glob("*/observations"))
        print(f"Found {len(folders)} observation folders...")
    
    # Collect all chunk files
    print("Scanning for chunk files...")
    chunk_files = []
    missing_folders = []
    
    for folder in tqdm(folders, desc="Scanning folders"):
        if not folder.exists():
            missing_folders.append(folder.parent.name)
            continue
        chunk_files.extend(sorted(folder.glob("observations_chunk_*.npz")))
    
    print(f"Found {len(chunk_files)} chunk files to validate")
    
    if missing_folders:
        print(f"\nWarning: {len(missing_folders)} folders missing observations/:")
        for f in missing_folders[:10]:
            print(f"  - {f}")
        if len(missing_folders) > 10:
            print(f"  ... and {len(missing_folders) - 10} more")
    
    # Validate chunks
    print("\nValidating chunks...")
    
    corrupted_files = []
    empty_files = []
    other_errors = []
    valid_count = 0
    
    for chunk_path in tqdm(chunk_files, desc="Validating"):
        if args.quick:
            # Quick mode: only check file size
            if chunk_path.stat().st_size == 0:
                empty_files.append(str(chunk_path))
            else:
                valid_count += 1
        else:
            is_valid, error_type, error_msg = validate_chunk(chunk_path)
            if is_valid:
                valid_count += 1
            elif error_type == "EMPTY":
                empty_files.append(str(chunk_path))
            elif error_type == "EOF_ERROR":
                corrupted_files.append((str(chunk_path), error_msg))
            else:
                other_errors.append((str(chunk_path), error_type, error_msg))
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total chunks checked: {len(chunk_files)}")
    print(f"Valid chunks: {valid_count}")
    print(f"Empty files (0 bytes): {len(empty_files)}")
    print(f"Corrupted files (EOFError): {len(corrupted_files)}")
    print(f"Other errors: {len(other_errors)}")
    
    # Print problem files
    if empty_files or corrupted_files or other_errors:
        print("\n" + "="*60)
        print("PROBLEM FILES")
        print("="*60)
        
        if empty_files:
            print(f"\n[EMPTY FILES] ({len(empty_files)}):")
            for f in empty_files:
                print(f"  {f}")
        
        if corrupted_files:
            print(f"\n[CORRUPTED FILES] ({len(corrupted_files)}):")
            for f, msg in corrupted_files:
                print(f"  {f}")
                print(f"    -> {msg}")
        
        if other_errors:
            print(f"\n[OTHER ERRORS] ({len(other_errors)}):")
            for f, err_type, msg in other_errors:
                print(f"  {f}")
                print(f"    -> [{err_type}] {msg}")
        
        # Save problem files to a text file
        output_file = dataset_path / "corrupted_chunks.txt"
        with open(output_file, 'w') as f:
            f.write("# Corrupted/Invalid chunk files\n")
            f.write(f"# Generated by check_chunks.py\n\n")
            
            if empty_files:
                f.write("# Empty files:\n")
                for path in empty_files:
                    f.write(f"{path}\n")
            
            if corrupted_files:
                f.write("\n# Corrupted files (EOFError):\n")
                for path, _ in corrupted_files:
                    f.write(f"{path}\n")
            
            if other_errors:
                f.write("\n# Other errors:\n")
                for path, err_type, msg in other_errors:
                    f.write(f"{path}  # {err_type}: {msg}\n")
        
        print(f"\nProblem files saved to: {output_file}")
        sys.exit(1)
    else:
        print("\n✓ All chunks are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
