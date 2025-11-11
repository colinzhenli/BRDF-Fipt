# Job Scheduler for COLMAP and Shape Matching

Automatically schedules and manages COLMAP reconstruction and shape matching jobs across multiple materials with intelligent GPU and CPU resource management.

## Features

- **Two modes**: Streaming (automatic) and Manual (first N folders)
- **Smart resource management**: Balances 2 GPUs and 128 CPU cores
- **GPU memory monitoring**: Checks available VRAM before launching jobs
- **Dynamic CPU allocation**: Shape matching workers adapt to current load
- **Persistent state tracking**: Survives crashes and allows resumption
- **Priority scheduling**: Shape matching starts immediately after COLMAP completion

## Resource Allocation Strategy

### Default Configuration
- **GPUs**: 2 (IDs: 1, 2)
- **Max COLMAP per GPU**: 10 (configurable)
- **Total CPU cores**: 128
- **CPU per COLMAP**: 8 threads (configurable)
- **Shape matching workers**: Dynamic (16-48 based on load)
- **Max concurrent shape matching**: 3 jobs

### Load Balancing Algorithm

1. **COLMAP CPU allocation**: Fixed 8 cores per job
2. **Shape matching CPU allocation**: Dynamic based on formula:
   ```
   available_cores = 128 - (active_colmap * 8) - 8 (buffer)
   workers_per_job = max(16, available_cores / (active_shape + 1))
   workers_per_job = min(workers_per_job, 48)
   ```
3. **GPU selection**: Round-robin with load balancing
4. **Memory check**: Requires ≥2GB free VRAM before launch

### Example Resource Distribution

| Scenario | COLMAP Jobs | Shape Jobs | COLMAP Cores | Shape Cores | Total |
|----------|-------------|------------|--------------|-------------|-------|
| Early    | 10          | 1          | 80           | 32          | 112   |
| Mid      | 7           | 2          | 56           | 32×2=64     | 120   |
| Late     | 3           | 3          | 24           | 32×3=96     | 120   |

## Usage

### Prerequisites

1. **Directory structure**: Dataset folder with numeric subfolders
   ```
   /path/to/dataset/
   ├── 0/
   │   ├── ldr/          # Images for COLMAP
   │   ├── hdr/          # HDR images for shape matching
   │   └── scan_log.json
   ├── 1/
   ├── 2/
   └── ...
   ```

2. **Required files in each material folder**:
   - `ldr/` - Images for COLMAP feature extraction
   - `hdr/` - HDR images for shape matching
   - `scan_log.json` - Robot log data

### Streaming Mode (Automatic)

Continuously schedules jobs as capacity becomes available. Runs until all materials complete.

```bash
python recon/scheduler/job_scheduler.py \
    --dataset /path/to/dataset \
    --mode streaming
```

**How it works**:
1. Scans for all numeric material folders
2. Launches COLMAP jobs up to GPU/CPU limits
3. Monitors running jobs every 10 seconds
4. When COLMAP completes → immediately launches shape matching (if CPU available)
5. When capacity frees up → launches next COLMAP job
6. Continues until all materials processed

### Manual Mode (First N folders)

Schedules first N folders upfront with balanced GPU assignment, then waits for completion.

```bash
python recon/scheduler/job_scheduler.py \
    --dataset /path/to/dataset \
    --mode manual \
    --n_folders 5
```

**How it works**:
1. Selects first N materials (sorted numerically)
2. Launches all N COLMAP jobs immediately with round-robin GPU assignment
3. Monitors jobs and launches shape matching as COLMAPs complete
4. Exits when all N materials are done

**Use case**: Testing on first few materials before full batch processing.

### Check Status

View current scheduler state without starting jobs:

```bash
python recon/scheduler/job_scheduler.py --status
```

Output shows:
- Materials by status (NOT_STARTED, COLMAP_RUNNING, COMPLETED, etc.)
- Currently running jobs with GPU/worker details
- Last update timestamp

### Custom Configuration

Override default resource limits:

```bash
python recon/scheduler/job_scheduler.py \
    --dataset /path/to/dataset \
    --mode streaming \
    --max_colmap_per_gpu 8 \
    --cpu_per_colmap 6 \
    --state_file custom_state.json
```

## State File

The scheduler maintains persistent state in `scheduler_state.json` (default) containing:

```json
{
  "mode": "streaming",
  "last_updated": "2025-11-10T15:30:45",
  "materials": {
    "0": {
      "status": "COMPLETED",
      "folder_path": "/path/to/dataset/0",
      "gpu": null,
      "pid": null,
      "colmap_start_time": "2025-11-10T14:00:00",
      "colmap_end_time": "2025-11-10T14:45:00",
      "shape_matching_start_time": "2025-11-10T14:45:10",
      "shape_matching_end_time": "2025-11-10T15:30:00"
    },
    "1": {
      "status": "COLMAP_RUNNING",
      "folder_path": "/path/to/dataset/1",
      "gpu": 1,
      "pid": 12345,
      "colmap_start_time": "2025-11-10T14:05:00"
    }
  }
}
```

### Resuming After Crash

The scheduler automatically resumes from saved state:
1. Detects stale PIDs (process no longer running)
2. Marks jobs as FAILED or checks for output files
3. Continues scheduling from where it left off

## Job Status States

```
NOT_STARTED → COLMAP_RUNNING → COLMAP_DONE → SHAPE_MATCHING_RUNNING → COMPLETED
                     ↓                                    ↓
                  FAILED  ←────────────────────────────  FAILED
```

- **NOT_STARTED**: Material not yet processed
- **COLMAP_RUNNING**: COLMAP reconstruction in progress
- **COLMAP_DONE**: COLMAP completed, waiting for shape matching
- **SHAPE_MATCHING_RUNNING**: Shape matching in progress
- **COMPLETED**: All processing done
- **FAILED**: Job failed (check logs in material folder)

## Output Files

For each material, the following are created:

```
material_folder/
├── colmap.log                    # COLMAP stdout/stderr
├── shape_matching.log            # Shape matching stdout/stderr
├── database.db                   # COLMAP database
├── sparse/
│   ├── cameras.bin, images.bin, points3D.bin
│   ├── points3D.ply              # Sparse reconstruction
│   ├── points3D_transformed_filtered.ply
│   ├── points3D_bbox.json        # Bounding box info
│   └── observations/             # 50 NPZ chunks with point-ray data
│       ├── observations_chunk_00.npz
│       ├── observations_chunk_01.npz
│       └── ...
├── undistorted/                  # Undistorted images
└── rotated_camera.json           # Camera poses in base frame
```

## Monitoring

### Real-time Console Output

```
[15:30:45] Status: COLMAP running: 8, Shape matching running: 2, Completed: 45/100, Failed: 0
[15:30:55] COLMAP completed for material 46
[15:30:56] Launched shape matching for material 46 with 32 workers (PID: 67890)
[15:31:05] Status: COLMAP running: 8, Shape matching running: 2, Completed: 45/100, Failed: 0
```

### Check Individual Logs

```bash
# COLMAP log
tail -f /path/to/dataset/0/colmap.log

# Shape matching log
tail -f /path/to/dataset/0/shape_matching.log
```

### GPU Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check specific GPU
nvidia-smi -i 1
```

## Troubleshooting

### Jobs stuck in COLMAP_RUNNING

**Symptom**: Jobs show as running but aren't making progress

**Diagnosis**:
```bash
python recon/scheduler/job_scheduler.py --status
# Check if PIDs are valid
ps aux | grep <pid>
```

**Solution**: 
- Kill stuck processes manually
- Remove state file and restart scheduler
- Check `colmap.log` for errors

### Out of GPU Memory

**Symptom**: COLMAP jobs fail with CUDA out of memory errors

**Solutions**:
1. Reduce `--max_colmap_per_gpu`:
   ```bash
   python recon/scheduler/job_scheduler.py --dataset /path --mode streaming --max_colmap_per_gpu 8
   ```
2. Increase `MIN_GPU_MEMORY_MB` threshold in `Config` class

### CPU Overload

**Symptom**: System becomes unresponsive

**Solutions**:
1. Reduce CPU allocation:
   ```bash
   python recon/scheduler/job_scheduler.py --dataset /path --mode streaming --cpu_per_colmap 6
   ```
2. Adjust `MAX_CONCURRENT_SHAPE_MATCHING` in `Config` class

### Shape Matching Not Starting

**Symptom**: COLMAP completes but shape matching doesn't start

**Diagnosis**: Check if CPU capacity is available
- Need at least 16 cores free
- Reduce concurrent jobs or increase core allocation

## Performance Tips

1. **Start with manual mode on first 3-5 folders** to validate setup
2. **Monitor first few jobs** to tune resource limits
3. **Use SSD storage** for database and sparse reconstruction
4. **Pre-process images** (debayering, HDR generation) before running scheduler
5. **Check logs regularly** during first batch to catch issues early

## Advanced: Modifying Resource Allocation

Edit `Config` class in `job_scheduler.py`:

```python
class Config:
    GPU_IDS = [1, 2]                          # Available GPUs
    MAX_COLMAP_PER_GPU = 10                   # Max COLMAP per GPU
    TOTAL_CPU_CORES = 128                     # Total CPU cores
    CPU_CORES_PER_COLMAP = 8                  # CPU threads per COLMAP
    MIN_SHAPE_MATCHING_WORKERS = 16           # Min workers for shape matching
    MAX_CONCURRENT_SHAPE_MATCHING = 3         # Max concurrent shape matching
    MIN_GPU_MEMORY_MB = 2048                  # Min free GPU memory (MB)
    POLL_INTERVAL_SEC = 10                    # Status check interval
```

## Example Workflow

### Full Dataset Processing

```bash
# 1. Start streaming mode
python recon/scheduler/job_scheduler.py \
    --dataset /media/raid/cloth/capture_data/my_dataset \
    --mode streaming

# 2. Monitor in another terminal
watch -n 5 "python recon/scheduler/job_scheduler.py --status"

# 3. Watch GPU usage
watch -n 1 nvidia-smi

# The scheduler will run until all materials are processed
# State is saved every 10 seconds - safe to interrupt with Ctrl+C
```

### Testing on First 5 Materials

```bash
# Launch first 5 materials only
python recon/scheduler/job_scheduler.py \
    --dataset /media/raid/cloth/capture_data/my_dataset \
    --mode manual \
    --n_folders 5

# Wait for completion, then check results
ls /media/raid/cloth/capture_data/my_dataset/*/sparse/observations/
```

## Requirements

- Python 3.8+
- COLMAP installed and in PATH
- nvidia-smi (for GPU monitoring)
- psutil package: `pip install psutil`
- Hydra package: `pip install hydra-core`
- All dependencies for shape_matching.py (Open3D, OpenCV, NumPy, etc.)

