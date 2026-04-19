#!/usr/bin/env python3
"""
Script to validate data folder completeness and create train/test splits.

Usage:
    python validate_and_split_data.py <data_folder> --train_ratio 0.8
    python validate_and_split_data.py <data_folder> --training_number 10
"""

import argparse
import os
import random
import sys
from pathlib import Path

# Make registration_check importable regardless of CWD
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "recon" / "calibration"))
from registration_check import registration_ratio, REGISTRATION_THRESHOLD  # noqa: E402


# Required files and folders for stage 1 dense training (run_stage1_dense.sh).
# `sparse/` is kept because the registration-health gate reads sparse/0/images.bin.
REQUIRED_FOLDERS = ['sparse']
REQUIRED_FILES = [
    'observations_structured.npz',
    'point_metadata.json',
    'rotated_camera.json',
    'scan_log.json',
]


def is_folder_not_empty(folder_path: Path) -> bool:
    """Check if a folder exists and is not empty."""
    if not folder_path.exists() or not folder_path.is_dir():
        return False
    # Check if folder has any contents
    return any(folder_path.iterdir())


def is_file_not_empty(file_path: Path) -> bool:
    """Check if a file exists and is not empty."""
    if not file_path.exists() or not file_path.is_file():
        return False
    return file_path.stat().st_size > 0


def check_registration_health(material_folder: Path) -> tuple[bool, str]:
    """
    Check that COLMAP registered enough cameras for this material.

    Returns:
        tuple: (ok, message). ok is False when the registration ratio is
        below REGISTRATION_THRESHOLD or when the underlying inputs are
        missing/unreadable.
    """
    n_reg, n_scans, ratio = registration_ratio(str(material_folder))
    if ratio < 0:
        return False, (
            f"Unreadable registration (n_reg={n_reg}, n_scans={n_scans})"
        )
    pct = ratio * 100.0
    msg = f"registration {n_reg}/{n_scans} ({pct:.1f}%)"
    if ratio < REGISTRATION_THRESHOLD:
        return False, f"Low {msg} < {REGISTRATION_THRESHOLD * 100:.0f}%"
    return True, msg


def validate_material_folder(material_folder: Path) -> tuple[bool, list[str]]:
    """
    Validate a material folder for completeness.

    Returns:
        tuple: (is_valid, list of missing/invalid items)
    """
    issues = []

    # Check required folders
    for folder_name in REQUIRED_FOLDERS:
        folder_path = material_folder / folder_name
        if not is_folder_not_empty(folder_path):
            if not folder_path.exists():
                issues.append(f"Missing folder: {folder_name}")
            elif not folder_path.is_dir():
                issues.append(f"Not a folder: {folder_name}")
            else:
                issues.append(f"Empty folder: {folder_name}")

    # Check required files (must exist and not be empty)
    for file_name in REQUIRED_FILES:
        file_path = material_folder / file_name
        if not is_file_not_empty(file_path):
            if not file_path.exists():
                issues.append(f"Missing file: {file_name}")
            elif not file_path.is_file():
                issues.append(f"Not a file: {file_name}")
            else:
                issues.append(f"Empty file: {file_name}")

    # Registration ratio gate (only meaningful if sparse + scan_log are present;
    # if they are missing the file checks above will already have flagged it).
    sparse_dir = material_folder / "sparse"
    scan_log = material_folder / "scan_log.json"
    if sparse_dir.exists() and scan_log.exists():
        ok, reg_msg = check_registration_health(material_folder)
        if not ok:
            issues.append(reg_msg)

    is_valid = len(issues) == 0
    return is_valid, issues


def get_material_id_from_folder(folder_name: str) -> int | None:
    """
    Extract material id from folder name.
    Valid folder names start with an integer (the material id).
    
    Returns:
        int or None: The material id if valid, None otherwise
    """
    # Reject names with non-digit characters; int() accepts underscores
    # (PEP 515), so e.g. "0_2" would otherwise be parsed as material 2.
    if not folder_name.isdigit():
        return None
    return int(folder_name)


def validate_data_folder(data_folder: Path, verbose: bool = True) -> list[int]:
    """
    Validate all material subfolders in the data folder.
    
    Returns:
        list: List of valid material ids
    """
    valid_ids = []
    invalid_count = 0
    
    if verbose:
        print(f"Validating data folder: {data_folder}")
        print(f"Required folders: {REQUIRED_FOLDERS}")
        print(f"Required files: {REQUIRED_FILES}")
        print("-" * 60)
    
    # Get all subfolders and sort them
    subfolders = sorted([f for f in data_folder.iterdir() if f.is_dir()])
    
    for subfolder in subfolders:
        material_id = get_material_id_from_folder(subfolder.name)
        
        if material_id is None:
            if verbose:
                print(f"Skipping {subfolder.name}: not a valid material id (not an integer)")
            continue
        
        is_valid, issues = validate_material_folder(subfolder)
        
        if is_valid:
            valid_ids.append(material_id)
            if verbose:
                print(f"✓ Material {material_id}: VALID")
        else:
            invalid_count += 1
            if verbose:
                print(f"✗ Material {material_id}: INVALID")
                for issue in issues:
                    print(f"    - {issue}")
    
    if verbose:
        print("-" * 60)
        print(f"Total valid materials: {len(valid_ids)}")
        print(f"Total invalid materials: {invalid_count}")
    
    return valid_ids


def create_train_test_split(valid_ids: list[int], train_ratio: float, seed: int = 42, training_number: int = -1) -> tuple[list[int], list[int]]:
    """
    Create train/test split from valid material ids.
    
    Args:
        valid_ids: List of valid material ids
        train_ratio: Ratio of training data (0.0 to 1.0)
        seed: Random seed for reproducibility
        training_number: If not -1, use this fixed number of training samples instead of ratio
    
    Returns:
        tuple: (training_ids, test_ids)
    """
    # Shuffle with seed for reproducibility
    random.seed(seed)
    shuffled_ids = valid_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Calculate split index based on training_number or train_ratio
    if training_number != -1:
        # Use fixed training number
        split_idx = min(training_number, len(shuffled_ids))
    else:
        # Use ratio
        split_idx = int(len(shuffled_ids) * train_ratio)
    
    training_ids = sorted(shuffled_ids[:split_idx])
    test_ids = sorted(shuffled_ids[split_idx:])
    
    return training_ids, test_ids


def save_ids_to_file(ids: list[int], file_path: Path):
    """
    Save material ids to a text file, one id per line.
    Format matches the expected format in neural_brdf_refactored.py:
        training_list = []
        with open(self.training_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    training_list.append(int(line))
    """
    with open(file_path, 'w') as f:
        for mid in ids:
            f.write(f"{mid}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate data folder completeness and create train/test splits."
    )
    parser.add_argument(
        "data_folder",
        type=str,
        help="Path to the data folder containing material subfolders"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--training_number",
        type=int,
        default=-1,
        help="Fixed number of training materials. If not -1, overrides train_ratio (default: -1)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    data_folder = Path(args.data_folder).resolve()
    
    if not data_folder.exists():
        print(f"Error: Data folder does not exist: {data_folder}")
        return 1
    
    if not data_folder.is_dir():
        print(f"Error: Not a directory: {data_folder}")
        return 1
    
    # Validate all material folders
    valid_ids = validate_data_folder(data_folder, verbose=not args.quiet)
    
    if len(valid_ids) == 0:
        print("Error: No valid material folders found!")
        return 1
    
    # Save all valid ids
    all_valid_ids_path = data_folder / "all_valid_ids.txt"
    save_ids_to_file(valid_ids, all_valid_ids_path)
    print(f"\nSaved {len(valid_ids)} valid material ids to: {all_valid_ids_path}")
    
    # Create train/test split
    training_ids, test_ids = create_train_test_split(
        valid_ids, args.train_ratio, args.seed, args.training_number
    )
    
    # Save training list
    training_list_path = data_folder / "training_list.txt"
    save_ids_to_file(training_ids, training_list_path)
    print(f"Saved {len(training_ids)} training material ids to: {training_list_path}")
    
    # Save test list
    test_list_path = data_folder / "test_list.txt"
    save_ids_to_file(test_ids, test_list_path)
    print(f"Saved {len(test_ids)} test material ids to: {test_list_path}")
    
    # Print summary
    if args.training_number != -1:
        print(f"\nSplit Summary (training_number={args.training_number}, seed={args.seed}):")
    else:
        print(f"\nSplit Summary (train_ratio={args.train_ratio}, seed={args.seed}):")
    print(f"  Total valid: {len(valid_ids)}")
    print(f"  Training: {len(training_ids)} ({len(training_ids)/len(valid_ids)*100:.1f}%)")
    print(f"  Test: {len(test_ids)} ({len(test_ids)/len(valid_ids)*100:.1f}%)")
    
    return 0


if __name__ == "__main__":
    exit(main())
