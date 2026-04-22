#!/bin/bash
# Benchmark BonnDataset cold-load: serial vs ProcessPoolExecutor.
# Defaults: 4 materials, use_pan=True use_lls=True (matches stage1 bonn workload),
# 16 workers (matches the parallelization plan in project_bonn_loader_bottleneck.md).

set -euo pipefail

DEBUG_NUM=${DEBUG_NUM:-4}
NUM_WORKERS=${NUM_WORKERS:-16}
USE_PAN=${USE_PAN:-True}
USE_LLS=${USE_LLS:-True}
DATASET_FOLDER=${DATASET_FOLDER:-/media/raid/cloth/Bonn_train}

cd "$(dirname "$0")/../.."

CONDA_ENV=${CONDA_ENV:-material-capture}

conda run -n ${CONDA_ENV} --live-stream python scripts/test_bonn_loader_speed.py \
    debug_num=${DEBUG_NUM} \
    num_workers=${NUM_WORKERS} \
    use_pan=${USE_PAN} \
    use_lls=${USE_LLS} \
    dataset_folder=${DATASET_FOLDER}
