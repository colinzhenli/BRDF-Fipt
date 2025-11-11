#!/bin/bash
# Example usage scripts for the job scheduler

# ============================================================================
# EXAMPLE 1: Streaming Mode (Automatic Scheduling)
# ============================================================================
# Processes all materials in dataset automatically
# Jobs are scheduled as capacity becomes available
# Runs until all materials complete

echo "Example 1: Streaming mode"
echo "=========================="
echo ""
echo "python recon/scheduler/job_scheduler.py \\"
echo "    --dataset /media/raid/cloth/capture_data/my_dataset \\"
echo "    --mode streaming"
echo ""

# Uncomment to run:
python recon/scheduler/job_scheduler.py \
    --dataset /media/raid/cloth/capture_data/my_dataset \
    --mode streaming


# ============================================================================
# EXAMPLE 2: Manual Mode (First N Folders)
# ============================================================================
# Processes first 5 materials only
# All jobs launched upfront with balanced GPU assignment
# Useful for testing before full batch

echo "Example 2: Manual mode (first 5 folders)"
echo "========================================="
echo ""
echo "python recon/scheduler/job_scheduler.py \\"
echo "    --dataset /media/raid/cloth/capture_data/my_dataset \\"
echo "    --mode manual \\"
echo "    --n_folders 5"
echo ""

# Uncomment to run:
# python recon/scheduler/job_scheduler.py \
#     --dataset /media/raid/cloth/capture_data/my_dataset \
#     --mode manual \
#     --n_folders 5


# ============================================================================
# EXAMPLE 3: Check Status
# ============================================================================
# View current scheduler state without starting jobs

echo "Example 3: Check status"
echo "======================"
echo ""
echo "python recon/scheduler/job_scheduler.py --status"
echo ""

# Uncomment to run:
# python recon/scheduler/job_scheduler.py --status


# ============================================================================
# EXAMPLE 4: Custom Resource Configuration
# ============================================================================
# Override default resource limits
# Useful if you have different hardware or want to be more conservative

echo "Example 4: Custom configuration"
echo "==============================="
echo ""
echo "python recon/scheduler/job_scheduler.py \\"
echo "    --dataset /media/raid/cloth/capture_data/my_dataset \\"
echo "    --mode streaming \\"
echo "    --max_colmap_per_gpu 8 \\"
echo "    --cpu_per_colmap 6 \\"
echo "    --state_file my_custom_state.json"
echo ""

# Uncomment to run:
# python recon/scheduler/job_scheduler.py \
#     --dataset /media/raid/cloth/capture_data/my_dataset \
#     --mode streaming \
#     --max_colmap_per_gpu 8 \
#     --cpu_per_colmap 6 \
#     --state_file my_custom_state.json


# ============================================================================
# EXAMPLE 5: Monitoring in Real-Time (Run in separate terminals)
# ============================================================================

echo "Example 5: Real-time monitoring"
echo "==============================="
echo ""
echo "Terminal 1: Run scheduler"
echo "python recon/scheduler/job_scheduler.py --dataset /path --mode streaming"
echo ""
echo "Terminal 2: Watch status"
echo "watch -n 5 'python recon/scheduler/job_scheduler.py --status'"
echo ""
echo "Terminal 3: Watch GPU usage"
echo "watch -n 1 nvidia-smi"
echo ""
echo "Terminal 4: Watch logs"
echo "tail -f /path/to/dataset/0/colmap.log"
echo ""


# ============================================================================
# EXAMPLE 6: Resume After Interruption
# ============================================================================
# Scheduler automatically resumes from saved state
# Just run the same command again

echo "Example 6: Resume after interruption"
echo "====================================="
echo ""
echo "1. Start scheduler (Ctrl+C to interrupt):"
echo "   python recon/scheduler/job_scheduler.py --dataset /path --mode streaming"
echo ""
echo "2. Resume (same command):"
echo "   python recon/scheduler/job_scheduler.py --dataset /path --mode streaming"
echo ""
echo "State is automatically loaded from scheduler_state.json"
echo ""


# ============================================================================
# TYPICAL WORKFLOW
# ============================================================================

echo ""
echo "============================================================================"
echo "TYPICAL WORKFLOW"
echo "============================================================================"
echo ""
echo "1. Test on first 3 materials:"
echo "   python recon/scheduler/job_scheduler.py --dataset /path --mode manual --n_folders 3"
echo ""
echo "2. Check results:"
echo "   ls /path/to/dataset/*/sparse/observations/"
echo ""
echo "3. If OK, run full batch:"
echo "   python recon/scheduler/job_scheduler.py --dataset /path --mode streaming"
echo ""
echo "4. Monitor progress:"
echo "   watch -n 10 'python recon/scheduler/job_scheduler.py --status'"
echo ""
echo "5. After completion, verify:"
echo "   python recon/scheduler/job_scheduler.py --status"
echo ""

