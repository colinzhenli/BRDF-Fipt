#!/bin/bash
# End-to-end: convert material 0, validate, then benchmark 10 materials.
# Usage: bash scripts/reformat_data/run_all.sh

set -e
cd "$(dirname "$0")"

DATASET="/media/raid/cloth/capture_data/Dataset_Nov11"
CONDA_ENV="fipt_copy"

echo "============================================================"
echo "Step 1: Convert material 0"
echo "============================================================"
conda run -n $CONDA_ENV --no-capture-output \
    python convert_single.py "$DATASET/0"

echo ""
echo "============================================================"
echo "Step 2: Validate material 0"
echo "============================================================"
conda run -n $CONDA_ENV --no-capture-output \
    python validate.py "$DATASET/0"

echo ""
echo "============================================================"
echo "Step 3: Benchmark 10 materials (different parallelism configs)"
echo "============================================================"
conda run -n $CONDA_ENV --no-capture-output \
    python convert_batch.py "$DATASET" \
        --mat-ids 0 48 71 95 112 121 145 169 192 215 \
        --benchmark

echo ""
echo "Done!"
