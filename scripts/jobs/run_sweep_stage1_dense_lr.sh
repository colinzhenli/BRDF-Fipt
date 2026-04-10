#!/bin/bash
# Sweep decoder learning rate for stage 1 dense training.
#
# Usage:
#   1. Create the sweep (once):
#        cd /home/zla247/projects/BRDF-Fipt
#        wandb sweep scripts/jobs/sweep_stage1_dense_lr.yaml --project brdf-capture
#      This prints a sweep ID like "koala_penguin/brdf-capture/abc123"
#
#   2. Launch agent(s) — one per GPU:
#        bash scripts/jobs/run_sweep_stage1_dense_lr.sh <sweep_id> <gpu_id>
#
#      Examples:
#        # Single GPU:
#        bash scripts/jobs/run_sweep_stage1_dense_lr.sh koala_penguin/brdf-capture/abc123 0
#
#        # Multiple GPUs in parallel:
#        bash scripts/jobs/run_sweep_stage1_dense_lr.sh koala_penguin/brdf-capture/abc123 0 &
#        bash scripts/jobs/run_sweep_stage1_dense_lr.sh koala_penguin/brdf-capture/abc123 1 &

set -euo pipefail
cd /home/zla247/projects/BRDF-Fipt

SWEEP_ID="${1:?Usage: $0 <sweep_id> <gpu_id>}"
GPU_ID="${2:?Usage: $0 <sweep_id> <gpu_id>}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "Starting wandb agent on GPU $GPU_ID for sweep $SWEEP_ID"
wandb agent "$SWEEP_ID"
