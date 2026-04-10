#!/bin/bash
# Debug sweep for learning rate on stage 1 training.
# Usage:
#   1. Create the sweep:
#        cd /home/zla247/projects/BRDF-Fipt
#        conda activate fipt_copy
#        wandb sweep scripts/debugging/sweep_lr_config.yaml --project brdf-capture
#      This prints a sweep ID like "username/brdf-capture/abc123"
#
#   2. Launch an agent on a free GPU (one agent = sequential runs):
#        bash scripts/debugging/run_sweep_lr_debug.sh <sweep_id> [gpu_id]
#      e.g.:
#        bash scripts/debugging/run_sweep_lr_debug.sh username/brdf-capture/abc123 1
#
#   3. To run on multiple GPUs in parallel, launch multiple agents:
#        bash scripts/debugging/run_sweep_lr_debug.sh <sweep_id> 0 &
#        bash scripts/debugging/run_sweep_lr_debug.sh <sweep_id> 1 &

set -euo pipefail
cd /home/zla247/projects/BRDF-Fipt

SWEEP_ID="${1:?Usage: $0 <sweep_id> [gpu_id]}"
GPU_ID="${2:-1}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "Starting wandb agent on GPU $GPU_ID for sweep $SWEEP_ID"
wandb agent "$SWEEP_ID"
