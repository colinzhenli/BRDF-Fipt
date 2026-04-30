#!/bin/bash
# Manually relaunch shape_matching for materials the running scheduler considers FAILED.
# State edits are clobbered by the scheduler's in-memory snapshot; this bypasses that
# by launching shape_matching out-of-band. On the next scheduler restart,
# verify_completed_materials() sees the success log and updates state.

set -u
DATASET="/media/raid/cloth/capture_data/Dataset_Nov11"
PROJECT_DIR="/home/zla247/projects/BRDF-Fipt"
MATERIALS=(447 498 504 505 506 507 508 509 510)
MAX_PARALLEL=3
NUM_WORKERS=8

cd "${PROJECT_DIR}"
source /home/zla247/anaconda3/etc/profile.d/conda.sh
conda activate fipt_copy

echo "Launching ${#MATERIALS[@]} shape_matching jobs (max ${MAX_PARALLEL} concurrent, ${NUM_WORKERS} workers each)"
echo "Logs: <dataset>/<id>/shape_matching.log"
echo

run_one() {
    local mid="$1"
    local folder="${DATASET}/${mid}"
    local log="${folder}/shape_matching.log"
    echo "[$(date +%H:%M:%S)] START material ${mid} (log: ${log})"
    python recon/calibration/shape_matching.py \
        shape_matching.folder_path="${folder}" \
        shape_matching.num_workers=${NUM_WORKERS} \
        shape_matching.z_outlier_percentile=5.0 \
        > "${log}" 2>&1
    local rc=$?
    if [ ${rc} -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] OK    material ${mid} (rc=${rc})"
    else
        echo "[$(date +%H:%M:%S)] FAIL  material ${mid} (rc=${rc}) — see ${log}"
    fi
}

# Concurrency limiter: only MAX_PARALLEL jobs at once
for mid in "${MATERIALS[@]}"; do
    while [ "$(jobs -r | wc -l)" -ge ${MAX_PARALLEL} ]; do
        sleep 5
    done
    run_one "${mid}" &
done

wait
echo
echo "All ${#MATERIALS[@]} jobs finished."
