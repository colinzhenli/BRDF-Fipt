#!/usr/bin/env python3
"""
Job scheduler for COLMAP reconstruction and shape matching.

Two modes:
1. Streaming mode: Automatically schedules new jobs as old ones finish
2. Manual mode: Schedules first N folders and waits for completion

Resource constraints:
- 2 GPUs (IDs: 1, 2)
- Max 10 COLMAP jobs per GPU (configurable)
- 128 CPU cores total
- Dynamic CPU allocation based on active jobs
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import psutil

# ==================== Configuration ====================

class Config:
    """Scheduler configuration"""
    GPU_IDS = [1, 2]
    MAX_COLMAP_PER_GPU = 10  # Maximum COLMAP jobs per GPU
    TOTAL_CPU_CORES = 128
    CPU_CORES_PER_COLMAP = 8  # CPU threads allocated per COLMAP job
    MIN_SHAPE_MATCHING_WORKERS = 16  # Minimum workers for shape matching
    MAX_CONCURRENT_SHAPE_MATCHING = 3  # Maximum concurrent shape matching jobs
    MIN_GPU_MEMORY_MB = 2048  # Minimum free GPU memory (MB) to launch COLMAP
    POLL_INTERVAL_SEC = 10  # How often to check job status
    MATERIAL_SCAN_INTERVAL_SEC = 30  # How often to scan for new materials in auto mode
    STATE_FILE_NAME = "scheduler_state.json"  # State file name (saved in dataset folder)
    
    # Paths
    COLMAP_SCRIPT = "recon/colmap/colmap.sh"
    SHAPE_MATCHING_SCRIPT = "recon/calibration/shape_matching.py"
    
    def __init__(self, dataset_root: Optional[str] = None):
        """Initialize config with dataset-specific paths."""
        self.dataset_root = dataset_root
        if dataset_root:
            self.STATE_FILE = os.path.join(dataset_root, self.STATE_FILE_NAME)
        else:
            self.STATE_FILE = self.STATE_FILE_NAME

# ==================== Job Status Enum ====================

class JobStatus:
    NOT_STARTED = "NOT_STARTED"
    COLMAP_QUEUED = "COLMAP_QUEUED"
    COLMAP_RUNNING = "COLMAP_RUNNING"
    COLMAP_DONE = "COLMAP_DONE"
    SHAPE_MATCHING_RUNNING = "SHAPE_MATCHING_RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

# ==================== Utility Functions ====================

def check_gpu_memory(gpu_id: int) -> int:
    """
    Check free GPU memory in MB.
    Returns: Free memory in MB, or -1 on error
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        return int(result.stdout.strip())
    except Exception as e:
        print(f"Error checking GPU {gpu_id} memory: {e}")
        return -1

def get_least_loaded_gpu(state: Dict, config: Config) -> Optional[int]:
    """
    Find GPU with fewest running COLMAP jobs and sufficient memory.
    Returns: GPU ID or None if all GPUs are at capacity or have insufficient memory
    """
    gpu_loads = {gpu_id: 0 for gpu_id in config.GPU_IDS}
    
    # Count running COLMAP jobs per GPU
    for material, info in state["materials"].items():
        if info["status"] == JobStatus.COLMAP_RUNNING and "gpu" in info:
            gpu_loads[info["gpu"]] += 1
    
    # Find GPU with lowest load that has capacity and memory
    candidates = []
    for gpu_id in config.GPU_IDS:
        if gpu_loads[gpu_id] < config.MAX_COLMAP_PER_GPU:
            free_mem = check_gpu_memory(gpu_id)
            if free_mem >= config.MIN_GPU_MEMORY_MB:
                candidates.append((gpu_id, gpu_loads[gpu_id]))
    
    if not candidates:
        return None
    
    # Return GPU with lowest load
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]

def calculate_shape_matching_workers(state: Dict, config: Config) -> int:
    """
    Calculate optimal number of workers for shape matching based on current load.
    
    Strategy:
    - Reserve CPU_CORES_PER_COLMAP * active_colmap_count for COLMAP
    - Allocate remaining to shape matching (with min/max bounds)
    """
    # Count active COLMAP jobs
    active_colmap = sum(
        1 for info in state["materials"].values()
        if info["status"] == JobStatus.COLMAP_RUNNING
    )
    
    # Count active shape matching jobs
    active_shape = sum(
        1 for info in state["materials"].values()
        if info["status"] == JobStatus.SHAPE_MATCHING_RUNNING
    )
    
    # Calculate available cores
    colmap_cores = active_colmap * config.CPU_CORES_PER_COLMAP
    
    # If no shape matching jobs, return default
    if active_shape == 0:
        return max(config.MIN_SHAPE_MATCHING_WORKERS, 
                   config.TOTAL_CPU_CORES - colmap_cores - 8)  # Leave 8 core buffer
    
    # Distribute remaining cores among shape matching jobs
    available_cores = config.TOTAL_CPU_CORES - colmap_cores - 8  # 8 core buffer
    workers_per_job = max(config.MIN_SHAPE_MATCHING_WORKERS, 
                          available_cores // (active_shape + 1))
    
    # Cap at reasonable maximum
    return min(workers_per_job, 48)

def check_cpu_capacity(state: Dict, config: Config, for_shape_matching: bool = False) -> bool:
    """
    Check if there's CPU capacity for a new job.
    
    Args:
        state: Current state
        config: Configuration
        for_shape_matching: True if checking for shape matching, False for COLMAP
    
    Returns: True if capacity available
    """
    active_colmap = sum(
        1 for info in state["materials"].values()
        if info["status"] == JobStatus.COLMAP_RUNNING
    )
    
    active_shape = sum(
        1 for info in state["materials"].values()
        if info["status"] == JobStatus.SHAPE_MATCHING_RUNNING
    )
    
    colmap_cores = active_colmap * config.CPU_CORES_PER_COLMAP
    
    if for_shape_matching:
        # Check if we're at max concurrent shape matching jobs
        if active_shape >= config.MAX_CONCURRENT_SHAPE_MATCHING:
            return False
        
        # Estimate cores needed
        estimated_workers = calculate_shape_matching_workers(state, config)
        estimated_total = colmap_cores + (active_shape + 1) * estimated_workers
        return estimated_total < (config.TOTAL_CPU_CORES - 8)  # Leave 8 core buffer
    else:
        # For COLMAP
        estimated_total = (active_colmap + 1) * config.CPU_CORES_PER_COLMAP
        return estimated_total < (config.TOTAL_CPU_CORES - 40)  # Leave room for shape matching

def is_process_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def is_material_ready(folder_path: str) -> bool:
    """
    Check if a material folder is ready for processing.
    A material is ready if scan_log.json exists and is not empty.
    
    Args:
        folder_path: Path to material folder
    
    Returns: True if material is ready for processing
    """
    scan_log_path = os.path.join(folder_path, "scan_log.json")
    
    if not os.path.exists(scan_log_path):
        return False
    
    try:
        # Check if file is not empty
        if os.path.getsize(scan_log_path) == 0:
            return False
        
        # Try to load as JSON to verify it's valid
        with open(scan_log_path, 'r') as f:
            data = json.load(f)
            return bool(data)  # Return True if not empty dict/list
    except (json.JSONDecodeError, IOError):
        return False

# ==================== State Management ====================

def load_state(state_file: str) -> Dict:
    """Load scheduler state from JSON file."""
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    else:
        return {
            "materials": {},
            "last_updated": None,
            "mode": None
        }

def save_state(state: Dict, state_file: str):
    """Save scheduler state to JSON file."""
    state["last_updated"] = datetime.now().isoformat()
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

def initialize_materials(dataset_root: str, state: Dict, verbose: bool = True) -> Tuple[List[str], int]:
    """
    Scan dataset folder for material subfolders (0, 1, 2, ..., 10, ..., 100, ...).
    Initialize state for new materials.
    
    Args:
        dataset_root: Path to dataset root folder
        state: Current scheduler state
        verbose: Whether to print discovery messages
    
    Returns: 
        Tuple of (sorted list of material folder names, count of new materials found)
    """
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")
    
    # Find all numeric subfolders
    material_folders = []
    for item in dataset_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            material_folders.append(item.name)
    
    # Sort numerically
    material_folders.sort(key=int)
    
    # Initialize state for new materials
    new_count = 0
    for material in material_folders:
        if material not in state["materials"]:
            folder_path = str(dataset_path / material)
            state["materials"][material] = {
                "status": JobStatus.NOT_STARTED,
                "folder_path": folder_path,
                "pid": None,
                "gpu": None,
                "workers": None,
                "colmap_start_time": None,
                "colmap_end_time": None,
                "shape_matching_start_time": None,
                "shape_matching_end_time": None,
                "error": None,
                "ready": is_material_ready(folder_path)
            }
            new_count += 1
            if verbose:
                ready_status = "ready" if state["materials"][material]["ready"] else "not ready (no scan_log.json)"
                print(f"  New material {material}: {ready_status}")
        else:
            # Update ready status for existing materials
            if state["materials"][material]["status"] == JobStatus.NOT_STARTED:
                folder_path = state["materials"][material]["folder_path"]
                old_ready = state["materials"][material].get("ready", False)
                new_ready = is_material_ready(folder_path)
                state["materials"][material]["ready"] = new_ready
                
                # Notify if material became ready
                if not old_ready and new_ready and verbose:
                    print(f"  Material {material} is now ready (scan_log.json detected)")
    
    if verbose:
        if new_count > 0:
            print(f"Found {new_count} new material folder(s)")
        print(f"Total materials tracked: {len(material_folders)}")
    
    return material_folders, new_count

# ==================== Job Launchers ====================

def launch_colmap(material: str, folder_path: str, gpu_id: int, state: Dict, config: Config) -> Optional[int]:
    """
    Launch COLMAP reconstruction for a material.
    Returns: Process PID or None on failure
    """
    try:
        # Set environment with CPU limits
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["OMP_NUM_THREADS"] = str(config.CPU_CORES_PER_COLMAP)
        env["MKL_NUM_THREADS"] = str(config.CPU_CORES_PER_COLMAP)
        
        # Launch COLMAP script
        cmd = ["bash", config.COLMAP_SCRIPT, folder_path, str(gpu_id)]
        
        # Redirect output to log file
        log_file = os.path.join(folder_path, "colmap.log")
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True  # Detach from parent
            )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Launched COLMAP for material {material} on GPU {gpu_id} (PID: {process.pid})")
        
        # Update state
        state["materials"][material]["status"] = JobStatus.COLMAP_RUNNING
        state["materials"][material]["pid"] = process.pid
        state["materials"][material]["gpu"] = gpu_id
        state["materials"][material]["colmap_start_time"] = datetime.now().isoformat()
        
        return process.pid
        
    except Exception as e:
        print(f"Error launching COLMAP for material {material}: {e}")
        state["materials"][material]["status"] = JobStatus.FAILED
        state["materials"][material]["error"] = str(e)
        return None

def launch_shape_matching(material: str, folder_path: str, num_workers: int, state: Dict, config: Config) -> Optional[int]:
    """
    Launch shape matching for a material.
    Returns: Process PID or None on failure
    """
    try:
        # Build command with Hydra overrides
        cmd = [
            "python", config.SHAPE_MATCHING_SCRIPT,
            f"shape_matching.folder_path={folder_path}",
            f"shape_matching.num_workers={num_workers}",
            "shape_matching.z_outlier_percentile=5.0"
        ]
        
        # Redirect output to log file
        log_file = os.path.join(folder_path, "shape_matching.log")
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Launched shape matching for material {material} with {num_workers} workers (PID: {process.pid})")
        
        # Update state
        state["materials"][material]["status"] = JobStatus.SHAPE_MATCHING_RUNNING
        state["materials"][material]["pid"] = process.pid
        state["materials"][material]["workers"] = num_workers
        state["materials"][material]["shape_matching_start_time"] = datetime.now().isoformat()
        
        return process.pid
        
    except Exception as e:
        print(f"Error launching shape matching for material {material}: {e}")
        state["materials"][material]["status"] = JobStatus.FAILED
        state["materials"][material]["error"] = str(e)
        return None

# ==================== Job Monitoring ====================

def check_colmap_completion(folder_path: str) -> bool:
    """
    Check if COLMAP has completed by looking for completion message in log file.
    Returns: True if completed successfully
    """
    log_file = os.path.join(folder_path, "colmap.log")
    if not os.path.exists(log_file):
        return False
    
    try:
        with open(log_file, 'r') as f:
            # Read last few lines to check for completion message
            lines = f.readlines()
            for line in reversed(lines[-20:]):  # Check last 20 lines
                if "== Finished COLMAP reconstruction ==" in line:
                    return True
    except (IOError, OSError):
        pass
    
    return False

def check_shape_matching_completion(folder_path: str) -> bool:
    """
    Check if shape matching has completed by looking for completion message in log file.
    Returns: True if completed successfully
    """
    log_file = os.path.join(folder_path, "shape_matching.log")
    if not os.path.exists(log_file):
        return False
    
    try:
        with open(log_file, 'r') as f:
            # Read last few lines to check for completion message
            lines = f.readlines()
            for line in reversed(lines[-20:]):  # Check last 20 lines
                if "Finished shape matching for" in line:
                    return True
    except (IOError, OSError):
        pass
    
    return False

def update_job_status(state: Dict, config: Config):
    """
    Check status of running jobs and update state.
    Uses log file completion messages to detect successful completion.
    Transitions COLMAP_DONE to SHAPE_MATCHING immediately if capacity allows.
    """
    for material, info in state["materials"].items():
        if info["status"] == JobStatus.COLMAP_RUNNING:
            folder_path = info["folder_path"]
            pid = info.get("pid")
            process_alive = pid and is_process_alive(pid)
            
            # Check for completion flag in log file
            colmap_completed = check_colmap_completion(folder_path)
            
            # If completed OR process is dead, check final status
            if colmap_completed or not process_alive:
                if colmap_completed:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] COLMAP completed for material {material}")
                    info["status"] = JobStatus.COLMAP_DONE
                    info["colmap_end_time"] = datetime.now().isoformat()
                    info["pid"] = None
                    info["gpu"] = None
                else:
                    # Process died but no completion flag - check if output exists
                    sparse_path = os.path.join(folder_path, "sparse", "points3D.ply")
                    if os.path.exists(sparse_path):
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] COLMAP completed for material {material} (detected via output file)")
                        info["status"] = JobStatus.COLMAP_DONE
                        info["colmap_end_time"] = datetime.now().isoformat()
                        info["pid"] = None
                        info["gpu"] = None
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] COLMAP failed for material {material} (process died, no completion flag)")
                        info["status"] = JobStatus.FAILED
                        info["error"] = "COLMAP process died without completion"
                        info["pid"] = None
                        info["gpu"] = None
        
        elif info["status"] == JobStatus.SHAPE_MATCHING_RUNNING:
            folder_path = info["folder_path"]
            pid = info.get("pid")
            process_alive = pid and is_process_alive(pid)
            
            # Check for completion flag in log file
            shape_completed = check_shape_matching_completion(folder_path)
            
            # If completed OR process is dead, check final status
            if shape_completed or not process_alive:
                if shape_completed:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Shape matching completed for material {material}")
                    info["status"] = JobStatus.COMPLETED
                    info["shape_matching_end_time"] = datetime.now().isoformat()
                    info["pid"] = None
                    info["workers"] = None
                else:
                    # Process died but no completion flag - check if output exists
                    obs_folder = os.path.join(folder_path, "sparse", "observations")
                    if os.path.exists(obs_folder) and len(os.listdir(obs_folder)) > 0:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Shape matching completed for material {material} (detected via output folder)")
                        info["status"] = JobStatus.COMPLETED
                        info["shape_matching_end_time"] = datetime.now().isoformat()
                        info["pid"] = None
                        info["workers"] = None
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Shape matching failed for material {material} (process died, no completion flag)")
                        info["status"] = JobStatus.FAILED
                        info["error"] = "Shape matching process died without completion"
                        info["pid"] = None
                        info["workers"] = None

def try_launch_shape_matching_jobs(state: Dict, config: Config):
    """
    Try to launch shape matching for materials that have COLMAP_DONE status.
    Prioritizes immediate launch after COLMAP completion.
    """
    for material, info in state["materials"].items():
        if info["status"] == JobStatus.COLMAP_DONE:
            # Check if we have CPU capacity
            if check_cpu_capacity(state, config, for_shape_matching=True):
                num_workers = calculate_shape_matching_workers(state, config)
                pid = launch_shape_matching(material, info["folder_path"], num_workers, state, config)
                if pid:
                    save_state(state, config.STATE_FILE)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for CPU capacity to launch shape matching for material {material}")
                break  # Wait for next iteration

# ==================== Scheduler Modes ====================

def streaming_mode(dataset_root: str, config: Config, auto_detect: bool = False):
    """
    Streaming mode: Continuously schedule jobs as capacity becomes available.
    Runs until all materials are completed.
    
    Args:
        dataset_root: Path to dataset root folder
        config: Scheduler configuration
        auto_detect: If True, periodically scan for new materials and process them automatically
    """
    print("\n" + "="*60)
    if auto_detect:
        print("STREAMING MODE: Automatic job scheduling with auto-detection")
    else:
        print("STREAMING MODE: Automatic job scheduling")
    print("="*60 + "\n")
    
    state = load_state(config.STATE_FILE)
    state["mode"] = "streaming_auto" if auto_detect else "streaming"
    
    # Initialize materials
    materials, new_count = initialize_materials(dataset_root, state)
    save_state(state, config.STATE_FILE)
    
    if not materials:
        print("No material folders found!")
        if not auto_detect:
            return
        else:
            print("Waiting for new materials...")
    
    print(f"Monitoring {len(materials)} materials...")
    print(f"GPU IDs: {config.GPU_IDS}")
    print(f"Max COLMAP per GPU: {config.MAX_COLMAP_PER_GPU}")
    print(f"Total CPU cores: {config.TOTAL_CPU_CORES}")
    print(f"CPU cores per COLMAP: {config.CPU_CORES_PER_COLMAP}")
    if auto_detect:
        print(f"Auto-detection: Scanning for new materials every {config.MATERIAL_SCAN_INTERVAL_SEC}s")
    print()
    
    try:
        last_scan_time = time.time()
        iteration = 0
        
        while True:
            iteration += 1
            
            # Periodic material scanning in auto-detect mode
            if auto_detect and (time.time() - last_scan_time) >= config.MATERIAL_SCAN_INTERVAL_SEC:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning for new materials...")
                materials, new_count = initialize_materials(dataset_root, state, verbose=True)
                if new_count > 0:
                    save_state(state, config.STATE_FILE)
                last_scan_time = time.time()
            
            # Update status of running jobs
            update_job_status(state, config)
            
            # Try to launch shape matching for completed COLMAP jobs (priority)
            try_launch_shape_matching_jobs(state, config)
            
            # Try to launch new COLMAP jobs
            for material in materials:
                info = state["materials"][material]
                
                if info["status"] == JobStatus.NOT_STARTED:
                    # Check if material is ready (has scan_log.json)
                    if not info.get("ready", False):
                        continue  # Skip materials that aren't ready yet
                    
                    # Check GPU availability
                    gpu_id = get_least_loaded_gpu(state, config)
                    if gpu_id is None:
                        break  # No GPU available
                    
                    # Check CPU capacity
                    if not check_cpu_capacity(state, config, for_shape_matching=False):
                        break  # No CPU capacity
                    
                    # Launch COLMAP
                    pid = launch_colmap(material, info["folder_path"], gpu_id, state, config)
                    if pid:
                        save_state(state, config.STATE_FILE)
            
            # Check if all materials are completed (or should exit)
            statuses = [info["status"] for info in state["materials"].values()]
            completed = statuses.count(JobStatus.COMPLETED)
            failed = statuses.count(JobStatus.FAILED)
            total = len(materials)
            
            # In auto-detect mode, never exit - keep waiting for new materials
            if not auto_detect and completed + failed == total:
                print("\n" + "="*60)
                print(f"All materials processed!")
                print(f"Completed: {completed}/{total}")
                print(f"Failed: {failed}/{total}")
                print("="*60 + "\n")
                break
            
            # Print status summary (every 6 iterations = ~1 minute)
            if iteration % 6 == 1:
                running_colmap = statuses.count(JobStatus.COLMAP_RUNNING)
                running_shape = statuses.count(JobStatus.SHAPE_MATCHING_RUNNING)
                not_started = statuses.count(JobStatus.NOT_STARTED)
                
                # Count ready vs not ready materials
                ready_count = sum(1 for info in state["materials"].values() 
                                 if info["status"] == JobStatus.NOT_STARTED and info.get("ready", False))
                not_ready_count = not_started - ready_count
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: "
                      f"COLMAP: {running_colmap}, "
                      f"Shape: {running_shape}, "
                      f"Ready: {ready_count}, "
                      f"Not ready: {not_ready_count}, "
                      f"Completed: {completed}/{total}, "
                      f"Failed: {failed}")
            
            # Save state periodically
            save_state(state, config.STATE_FILE)
            
            # Wait before next check
            time.sleep(config.POLL_INTERVAL_SEC)
            
    except KeyboardInterrupt:
        print("\n\nScheduler interrupted by user. State saved.")
        save_state(state, config.STATE_FILE)
        sys.exit(0)

def manual_mode(dataset_root: str, n_folders: int, config: Config):
    """
    Manual mode: Schedule first N folders and wait for all to complete.
    Balances jobs across GPUs and CPUs upfront.
    Only schedules materials that are ready (have scan_log.json).
    """
    print("\n" + "="*60)
    print(f"MANUAL MODE: Scheduling first {n_folders} folders")
    print("="*60 + "\n")
    
    state = load_state(config.STATE_FILE)
    state["mode"] = "manual"
    
    # Initialize materials
    materials, new_count = initialize_materials(dataset_root, state)
    save_state(state, config.STATE_FILE)
    
    if not materials:
        print("No material folders found!")
        return
    
    # Select first N ready materials
    ready_materials = [m for m in materials if state["materials"][m].get("ready", False)]
    not_ready_materials = [m for m in materials if not state["materials"][m].get("ready", False)]
    
    if not_ready_materials:
        print(f"Warning: {len(not_ready_materials)} materials not ready (missing scan_log.json):")
        print(f"  {', '.join(not_ready_materials[:10])}")
        if len(not_ready_materials) > 10:
            print(f"  ... and {len(not_ready_materials) - 10} more")
        print()
    
    selected_materials = ready_materials[:n_folders]
    if len(selected_materials) < n_folders:
        print(f"Warning: Only {len(selected_materials)} ready materials available (requested {n_folders})")
    
    print(f"Selected materials: {selected_materials}\n")
    
    if not selected_materials:
        print("No ready materials to process!")
        return
    
    # Launch all N COLMAP jobs with balanced GPU assignment
    launched = 0
    for i, material in enumerate(selected_materials):
        info = state["materials"][material]
        
        if info["status"] == JobStatus.NOT_STARTED:
            # Round-robin GPU assignment
            gpu_id = config.GPU_IDS[i % len(config.GPU_IDS)]
            
            # Check GPU memory
            free_mem = check_gpu_memory(gpu_id)
            if free_mem < config.MIN_GPU_MEMORY_MB:
                print(f"Warning: GPU {gpu_id} has low memory ({free_mem} MB), waiting...")
                time.sleep(5)
                free_mem = check_gpu_memory(gpu_id)
                if free_mem < config.MIN_GPU_MEMORY_MB:
                    print(f"Skipping material {material} due to low GPU memory")
                    continue
            
            # Launch COLMAP
            pid = launch_colmap(material, info["folder_path"], gpu_id, state, config)
            if pid:
                launched += 1
                save_state(state, config.STATE_FILE)
            
            # Small delay to avoid overwhelming the system
            time.sleep(2)
    
    print(f"\nLaunched {launched} COLMAP jobs\n")
    
    # Monitor until all selected materials are completed
    try:
        while True:
            # Update status
            update_job_status(state, config)
            
            # Try to launch shape matching
            try_launch_shape_matching_jobs(state, config)
            
            # Check completion status for selected materials only
            statuses = [state["materials"][m]["status"] for m in selected_materials]
            completed = statuses.count(JobStatus.COMPLETED)
            failed = statuses.count(JobStatus.FAILED)
            running_colmap = statuses.count(JobStatus.COLMAP_RUNNING)
            running_shape = statuses.count(JobStatus.SHAPE_MATCHING_RUNNING)
            
            if completed + failed == len(selected_materials):
                print("\n" + "="*60)
                print(f"All selected materials processed!")
                print(f"Completed: {completed}/{len(selected_materials)}")
                print(f"Failed: {failed}/{len(selected_materials)}")
                print("="*60 + "\n")
                break
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: "
                  f"COLMAP running: {running_colmap}, "
                  f"Shape matching running: {running_shape}, "
                  f"Completed: {completed}/{len(selected_materials)}, "
                  f"Failed: {failed}")
            
            save_state(state, config.STATE_FILE)
            time.sleep(config.POLL_INTERVAL_SEC)
            
    except KeyboardInterrupt:
        print("\n\nScheduler interrupted by user. State saved.")
        save_state(state, config.STATE_FILE)
        sys.exit(0)

# ==================== Status Display ====================

def show_status(config: Config):
    """Display current scheduler status."""
    state = load_state(config.STATE_FILE)
    
    if not state["materials"]:
        print("No materials tracked yet. Run scheduler first.")
        return
    
    print("\n" + "="*60)
    print("SCHEDULER STATUS")
    print("="*60)
    print(f"Mode: {state.get('mode', 'N/A')}")
    print(f"Last updated: {state.get('last_updated', 'N/A')}")
    print()
    
    # Group by status
    by_status = {}
    for material, info in sorted(state["materials"].items(), key=lambda x: int(x[0])):
        status = info["status"]
        if status not in by_status:
            by_status[status] = []
        by_status[status].append(material)
    
    for status in [JobStatus.COLMAP_RUNNING, JobStatus.SHAPE_MATCHING_RUNNING, 
                   JobStatus.COLMAP_DONE, JobStatus.COMPLETED, JobStatus.FAILED, 
                   JobStatus.NOT_STARTED]:
        if status in by_status:
            print(f"{status}: {len(by_status[status])} materials")
            print(f"  {', '.join(by_status[status][:20])}")
            if len(by_status[status]) > 20:
                print(f"  ... and {len(by_status[status]) - 20} more")
            print()
    
    # Show running jobs details
    print("Running Jobs:")
    for material, info in sorted(state["materials"].items(), key=lambda x: int(x[0])):
        if info["status"] in [JobStatus.COLMAP_RUNNING, JobStatus.SHAPE_MATCHING_RUNNING]:
            if info["status"] == JobStatus.COLMAP_RUNNING:
                print(f"  Material {material}: COLMAP on GPU {info.get('gpu', '?')} (PID: {info.get('pid', '?')})")
            else:
                print(f"  Material {material}: Shape matching with {info.get('workers', '?')} workers (PID: {info.get('pid', '?')})")
    
    print("="*60 + "\n")

# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Job scheduler for COLMAP and shape matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Streaming mode (automatic scheduling)
  python job_scheduler.py --dataset /path/to/dataset --mode streaming

  # Streaming mode with auto-detection (continuously monitors for new materials)
  python job_scheduler.py --dataset /path/to/dataset --mode streaming --auto_detect

  # Manual mode (first 5 folders)
  python job_scheduler.py --dataset /path/to/dataset --mode manual --n_folders 5

  # Show current status
  python job_scheduler.py --status --dataset /path/to/dataset

  # Custom configuration
  python job_scheduler.py --dataset /path/to/dataset --mode streaming \\
      --max_colmap_per_gpu 8 --cpu_per_colmap 6
        """
    )
    
    parser.add_argument("--dataset", type=str, help="Path to dataset root folder")
    parser.add_argument("--mode", choices=["streaming", "manual"], help="Scheduling mode")
    parser.add_argument("--n_folders", type=int, help="Number of folders to process (manual mode only)")
    parser.add_argument("--status", action="store_true", help="Show current scheduler status")
    parser.add_argument("--auto_detect", action="store_true", 
                       help="Auto-detect new materials (streaming mode only). Continuously monitors for new material folders.")
    
    # Optional configuration overrides
    parser.add_argument("--max_colmap_per_gpu", type=int, help=f"Max COLMAP jobs per GPU (default: {Config.MAX_COLMAP_PER_GPU})")
    parser.add_argument("--cpu_per_colmap", type=int, help=f"CPU cores per COLMAP (default: {Config.CPU_CORES_PER_COLMAP})")
    parser.add_argument("--state_file", type=str, help=f"State file path (default: saved in dataset folder)")
    
    args = parser.parse_args()
    
    # Create config with dataset root
    config = Config(dataset_root=args.dataset)
    
    # Apply overrides
    if args.max_colmap_per_gpu:
        config.MAX_COLMAP_PER_GPU = args.max_colmap_per_gpu
    if args.cpu_per_colmap:
        config.CPU_CORES_PER_COLMAP = args.cpu_per_colmap
    if args.state_file:
        config.STATE_FILE = args.state_file
    
    # Status display mode
    if args.status:
        if not args.dataset:
            parser.error("--dataset is required for --status")
        show_status(config)
        return
    
    # Validate arguments for scheduling modes
    if not args.dataset:
        parser.error("--dataset is required when not using --status")
    if not args.mode:
        parser.error("--mode is required when not using --status")
    
    if args.mode == "manual" and not args.n_folders:
        parser.error("--n_folders is required for manual mode")
    
    if args.auto_detect and args.mode != "streaming":
        parser.error("--auto_detect can only be used with streaming mode")
    
    # Run scheduler
    if args.mode == "streaming":
        streaming_mode(args.dataset, config, auto_detect=args.auto_detect)
    elif args.mode == "manual":
        manual_mode(args.dataset, args.n_folders, config)

if __name__ == "__main__":
    main()

