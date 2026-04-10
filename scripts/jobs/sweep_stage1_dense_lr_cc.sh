#!/bin/bash
#SBATCH --account=def-ataiya-ab
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sweep-lr-%j.out
#SBATCH --error=logs/sweep-lr-%j.err
#SBATCH --job-name=sweep-lr

# ---------------------------------------------------------------------
# Usage:
#   1. Create the sweep ONCE from login node:
#        wandb sweep scripts/jobs/sweep_stage1_dense_lr.yaml --project brdf-capture
#      This prints a sweep ID like "koala_penguin/brdf-capture/abc123"
#
#   2. Submit N jobs (N = number of parallel agents you want):
#        sbatch scripts/jobs/sweep_stage1_dense_lr_cc.sh koala_penguin/brdf-capture/abc123
#        sbatch scripts/jobs/sweep_stage1_dense_lr_cc.sh koala_penguin/brdf-capture/abc123
#        sbatch scripts/jobs/sweep_stage1_dense_lr_cc.sh koala_penguin/brdf-capture/abc123
#
#      Each job gets its own GPU and runs one agent.
#      The agents share the same sweep queue — no duplicates.
# ---------------------------------------------------------------------

SWEEP_ID="${1:?Error: pass sweep ID as argument, e.g.: sbatch $0 koala_penguin/brdf-capture/abc123}"

echo "Job ID: $SLURM_JOB_ID"
echo "Sweep ID: $SWEEP_ID"
echo "Starting at: $(date)"

# Environment setup
source /home/zla247/envs/fipt/bin/activate
module load python/3.10.13 gcc/12.3 StdEnv/2023 opencv/4.8.1
export WANDB_API_KEY=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
pip install --no-index "torch<2.6" torchvision

nvidia-smi

cd /home/zla247/projects/BRDF-Fipt

# Run the wandb agent — it will pull jobs from the sweep queue
# and train until all configs are done (or this job times out)
wandb agent "$SWEEP_ID"

echo "Job finished at: $(date)"
