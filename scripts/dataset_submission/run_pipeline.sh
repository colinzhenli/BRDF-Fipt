#!/bin/bash
# End-to-end submission build for one material.
#
# Steps:
#   1) Crop HDR images by projected sample mask + dilation buffer (writes hdr/
#      and hdr_crop_bboxes.json under <DST>/<material>/).
#   2) Copy non-image files needed by stage 1 / stage 2 training.
#   3) Copy global calibration / list files (run once for the whole dataset).
#   4) Run a static unit test verifying inside-bbox bit-identity.
#
# Usage:
#   bash scripts/dataset_submission/run_pipeline.sh 227

set -e

MATERIAL="${1:-227}"
SRC="${SRC:-/media/raid/cloth/capture_data/Dataset_Nov11}"
DST="${DST:-/media/raid/cloth/Dataset_submission}"
DILATE_PX="${DILATE_PX:-8}"
WORKERS="${WORKERS:-12}"
PYTHON="${PYTHON:-/mnt/data/colin/colin/anaconda3/envs/fipt_copy/bin/python}"

cd "$(dirname "$(readlink -f "$0")")/../.."

echo "=== [1/4] Cropping HDR images for material ${MATERIAL} ==="
"$PYTHON" scripts/dataset_submission/crop_hdr_by_mask.py \
    --src "$SRC" --dst "$DST" --material "$MATERIAL" \
    --dilate_px "$DILATE_PX" --workers "$WORKERS"

echo "=== [2/4] Copying per-material metadata for ${MATERIAL} ==="
"$PYTHON" scripts/dataset_submission/build_submission.py \
    --src "$SRC" --dst "$DST" --material "$MATERIAL" --mode copy

echo "=== [3/4] Copying global calibration / list files ==="
"$PYTHON" scripts/dataset_submission/build_submission.py \
    --src "$SRC" --dst "$DST" --material "$MATERIAL" --mode copy --copy_globals

echo "=== [4/4] Running static unit test (inside-bbox bit-identity) ==="
"$PYTHON" scripts/dataset_submission/test_crop_equivalence.py \
    --orig "$SRC/$MATERIAL" --cropped "$DST/$MATERIAL" --num_samples 12

echo ""
echo "Submission ready at: $DST/$MATERIAL"
echo "To verify training equivalence, point run_stage2_pretrained_from_Real.sh"
echo "at dataset_folder=$DST/$MATERIAL and confirm the loss curve matches the"
echo "original. Pixels outside the projected mask + ${DILATE_PX}px buffer were"
echo "zeroed; the renderer's vis-mask filters them from the loss anyway."
