#!/usr/bin/env python3
"""
Simple job scheduler for COLMAP reconstruction and shape matching.

Usage:
  # COLMAP: process materials 0-9 using GPUs 1,2 (skip 3,5)
  python simple_scheduler.py colmap 0 9 --skip 3,5 --gpus 1,2
  
  # Shape matching: process materials 10-19 (skip 12)
  python simple_scheduler.py shape_matching 10 19 --skip 12
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime


def parse_range_and_skip(start: int, end: int, skip_str: str) -> list:
    """
    Parse ID range and skip list.
    
    Args:
        start: Start ID (inclusive)
        end: End ID (inclusive)
        skip_str: Comma-separated IDs to skip (e.g., "3,5,7")
    
    Returns:
        List of IDs to process
    """
    # Generate full range
    all_ids = list(range(start, end + 1))
    
    # Parse skip list
    skip_ids = set()
    if skip_str:
        for item in skip_str.split(','):
            item = item.strip()
            if item:
                skip_ids.add(int(item))
    
    # Filter out skipped IDs
    ids = [id for id in all_ids if id not in skip_ids]
    
    return ids


def run_colmap(material_ids: list, dataset_root: str, gpu_ids: list):
    """
    Run COLMAP for specified materials, distributing evenly across GPUs.
    
    Args:
        material_ids: List of material IDs to process
        dataset_root: Root folder containing material subfolders
        gpu_ids: List of GPU IDs to use
    """
    print(f"Running COLMAP for {len(material_ids)} materials")
    print(f"Materials: {material_ids}")
    print(f"GPUs: {gpu_ids}")
    print(f"Distribution: ~{len(material_ids) // len(gpu_ids)} materials per GPU\n")
    
    colmap_script = "recon/colmap/colmap.sh"
    
    # Check if script exists
    if not os.path.exists(colmap_script):
        print(f"Error: COLMAP script not found: {colmap_script}")
        sys.exit(1)
    
    processes = []
    
    # Distribute materials evenly across GPUs
    for idx, material_id in enumerate(material_ids):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        folder_path = os.path.join(dataset_root, str(material_id))
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found, skipping: {folder_path}")
            continue
        
        # Prepare command
        cmd = ["bash", colmap_script, folder_path, str(gpu_id)]
        
        # Log file
        log_file = os.path.join(folder_path, "colmap.log")
        
        # Launch process
        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True
                )
            
            processes.append({
                'material_id': material_id,
                'gpu_id': gpu_id,
                'pid': process.pid,
                'process': process
            })
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Launched COLMAP for material {material_id} on GPU {gpu_id} (PID: {process.pid})")
            
            # Small delay to avoid overwhelming the system
            time.sleep(1)
            
        except Exception as e:
            print(f"Error launching COLMAP for material {material_id}: {e}")
    
    print(f"\nLaunched {len(processes)} COLMAP jobs. Waiting for completion...\n")
    
    # Wait for all processes to complete
    completed = 0
    while completed < len(processes):
        time.sleep(10)  # Check every 10 seconds
        
        for proc_info in processes:
            if proc_info['process'].poll() is not None and 'completed' not in proc_info:
                proc_info['completed'] = True
                completed += 1
                returncode = proc_info['process'].returncode
                status = "SUCCESS" if returncode == 0 else f"FAILED (code {returncode})"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Material {proc_info['material_id']} on GPU {proc_info['gpu_id']}: {status} ({completed}/{len(processes)})")
    
    print(f"\nAll COLMAP jobs completed!")


def run_shape_matching(material_ids: list, dataset_root: str, num_workers: int = 16):
    """
    Run shape matching for specified materials sequentially.
    
    Args:
        material_ids: List of material IDs to process
        dataset_root: Root folder containing material subfolders
        num_workers: Number of CPU workers per job (default: 16)
    """
    print(f"Running shape matching for {len(material_ids)} materials")
    print(f"Materials: {material_ids}")
    print(f"Workers per material: {num_workers}\n")
    
    shape_matching_script = "recon/calibration/shape_matching.py"
    
    # Check if script exists
    if not os.path.exists(shape_matching_script):
        print(f"Error: Shape matching script not found: {shape_matching_script}")
        sys.exit(1)
    
    completed = 0
    failed = 0
    
    for material_id in material_ids:
        folder_path = os.path.join(dataset_root, str(material_id))
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found, skipping: {folder_path}")
            continue
        
        # Prepare command with Hydra overrides
        cmd = [
            "python", shape_matching_script,
            f"shape_matching.folder_path={folder_path}",
            f"shape_matching.num_workers={num_workers}",
            "shape_matching.z_outlier_percentile=5.0"
        ]
        
        # Log file
        log_file = os.path.join(folder_path, "shape_matching.log")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running shape matching for material {material_id}...")
        
        # Run process and wait for completion
        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
            
            # Wait for completion
            returncode = process.wait()
            
            if returncode == 0:
                completed += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Material {material_id}: SUCCESS ({completed}/{len(material_ids)})")
            else:
                failed += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Material {material_id}: FAILED (code {returncode})")
        
        except Exception as e:
            failed += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Material {material_id}: ERROR - {e}")
    
    print(f"\nShape matching completed!")
    print(f"Success: {completed}/{len(material_ids)}")
    print(f"Failed: {failed}/{len(material_ids)}")


def main():
    parser = argparse.ArgumentParser(
        description="Simple scheduler for COLMAP and shape matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # COLMAP: process materials 0-9 using GPUs 1,2 (skip 3,5)
  python simple_scheduler.py colmap /path/to/dataset 0 9 --skip 3,5 --gpus 1,2
  
  # Shape matching: process materials 10-19 (skip 12), use 24 workers per job
  python simple_scheduler.py shape_matching /path/to/dataset 10 19 --skip 12 --workers 24
  
  # COLMAP: process all materials 0-99 on GPU 1
  python simple_scheduler.py colmap /path/to/dataset 0 99 --gpus 1
        """
    )
    
    parser.add_argument("job_type", choices=["colmap", "shape_matching"],
                       help="Type of job to run")
    parser.add_argument("dataset_root", type=str,
                       help="Path to dataset root folder")
    parser.add_argument("start_id", type=int,
                       help="Start material ID (inclusive)")
    parser.add_argument("end_id", type=int,
                       help="End material ID (inclusive)")
    parser.add_argument("--skip", type=str, default="",
                       help="Comma-separated material IDs to skip (e.g., '3,5,7')")
    parser.add_argument("--gpus", type=str, default="1,2",
                       help="Comma-separated GPU IDs for COLMAP (default: '1,2')")
    parser.add_argument("--workers", type=int, default=16,
                       help="Number of CPU workers per shape matching job (default: 16)")
    
    args = parser.parse_args()
    
    # Validate dataset root
    if not os.path.exists(args.dataset_root):
        print(f"Error: Dataset root does not exist: {args.dataset_root}")
        sys.exit(1)
    
    # Parse material IDs
    material_ids = parse_range_and_skip(args.start_id, args.end_id, args.skip)
    
    if not material_ids:
        print("Error: No materials to process after applying skip list")
        sys.exit(1)
    
    # Run appropriate job type
    if args.job_type == "colmap":
        # Parse GPU IDs
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
        if not gpu_ids:
            print("Error: At least one GPU ID must be specified")
            sys.exit(1)
        
        run_colmap(material_ids, args.dataset_root, gpu_ids)
    
    elif args.job_type == "shape_matching":
        run_shape_matching(material_ids, args.dataset_root, args.workers)


if __name__ == "__main__":
    main()

