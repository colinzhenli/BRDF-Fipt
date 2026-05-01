#!/bin/bash
#SBATCH --account=def-ataiya-ab              # Your account (change if needed)
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1  # Request 1 H100 MIG instance (3/8 compute, 40GB memory)
# SBATCH --gpus=h100:1 # 1 H100
#SBATCH --cpus-per-task=8              # CPU cores/threads
#SBATCH --mem=80G                      # Memory per node
#SBATCH --time=48:00:00                # Time limit (HH:MM:SS) - adjust as needed
#SBATCH --output=logs/fipt-%j.out      # STDOUT (%j = job ID)
#SBATCH --error=logs/fipt-%j.err       # STDERR
#SBATCH --job-name=bonn          # Job name

# ---------------------------------------------------------------------
# Job Information
# ---------------------------------------------------------------------
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"
echo ""
# ---------------------------------------------------------------------

# Load modules (packages are in ~/.local/lib/python3.10/site-packages/)
module load python/3.10.13 gcc/12.3 StdEnv/2023 opencv/4.8.1

# Wandb API key - get from https://wandb.ai/authorize
# Replace YOUR_API_KEY_HERE with your actual key, or use offline mode
export WANDB_API_KEY=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
# For offline mode instead, uncomment:
# export WANDB_MODE=offline

# torch, torchvision, bitsandbytes already installed in ~/.local/lib/python3.10/
# To reinstall/upgrade, run on login node: pip install --no-index "torch<2.6" torchvision bitsandbytes

# Show GPU info
nvidia-smi

# Navigate to the project directory
cd /home/zla247/projects/BRDF-Fipt

# Run the debug script
bash ./scripts/jobs/run_stage1_bonn.sh

# ---------------------------------------------------------------------
echo ""
echo "Job finished at: $(date)"
echo "Exit code: $?"
# ---------------------------------------------------------------------
