#!/bin/bash
# Transfer Stage 1 dense training files to cs-vml-gpu09.
# Per-material: observations_structured.npz, scan_log.json, rotated_camera.json, point_metadata.json
# Plus training_list_420.txt at the dataset root.
#
# Materials are taken from training_list_420.txt (excludes the likes of 23, 111).

set -euo pipefail

SRC_ROOT="/media/raid/cloth/capture_data/Dataset_Nov11"
DST_USER="zpa34"
DST_HOST="cs-vml-gpu09"
DST_ROOT="/local-scratch2/zpa34/capture_data/dataset_Nov"
TRAINING_LIST="training_list_420.txt"

PER_MAT_FILES=(
    "observations_structured.npz"
    "scan_log.json"
    "rotated_camera.json"
    "point_metadata.json"
)

# 1) Ensure destination root exists
ssh "${DST_USER}@${DST_HOST}" "mkdir -p '${DST_ROOT}'"

# 2) Copy the training list itself
rsync -avh --progress \
    "${SRC_ROOT}/${TRAINING_LIST}" \
    "${DST_USER}@${DST_HOST}:${DST_ROOT}/"

# 3) Build an rsync include-list from the training list, restricted to the 4 files per material
TMP_INCLUDES="$(mktemp)"
trap 'rm -f "${TMP_INCLUDES}"' EXIT

while IFS= read -r mat_id; do
    mat_id="${mat_id//[[:space:]]/}"
    [[ -z "${mat_id}" ]] && continue
    # include the material dir itself, then each required file
    echo "/${mat_id}/" >> "${TMP_INCLUDES}"
    for f in "${PER_MAT_FILES[@]}"; do
        echo "/${mat_id}/${f}" >> "${TMP_INCLUDES}"
    done
done < "${SRC_ROOT}/${TRAINING_LIST}"

# 4) Transfer: include only listed dirs + their 4 files, exclude everything else.
#    -R / relative preserves {mat_id}/file layout on the remote side.
rsync -avh --progress \
    --include-from="${TMP_INCLUDES}" \
    --exclude='*' \
    "${SRC_ROOT}/" \
    "${DST_USER}@${DST_HOST}:${DST_ROOT}/"

echo "Done. Transferred $(wc -l < "${SRC_ROOT}/${TRAINING_LIST}") materials to ${DST_USER}@${DST_HOST}:${DST_ROOT}"
