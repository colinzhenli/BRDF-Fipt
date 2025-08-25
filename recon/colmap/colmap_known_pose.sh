#!/usr/bin/env bash
set -euo pipefail

PROJECT="${1:?usage: $0 <project_dir>}"
gpu_id="$2"
KNOWN_SPARSE="${PROJECT}/sparse/0"     # bin + txt all live here
IMG_DIR="${PROJECT}/images_raw"
DB="${PROJECT}/database.db"
OUT_SPARSE="${PROJECT}/sparse"         # triangulated sparse model goes here
UNDIST_OUT="${PROJECT}/undistorted"          # undistorted workspace (optional)

# Ensure cameras.txt and images.txt exist (fail if missing)
test -f "$KNOWN_SPARSE/cameras.txt" || { echo "cameras.txt missing after conversion"; exit 1; }
test -f "$KNOWN_SPARSE/images.txt"  || { echo "images.txt missing after conversion";  exit 1; }

# Make points3D.txt empty (required by known-poses docs)
echo "== creating points3D.txt =="
touch "$KNOWN_SPARSE/points3D.txt"

echo "== Feature extraction =="
CUDA_VISIBLE_DEVICES=$gpu_id colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$IMG_DIR" \
    --ImageReader.single_camera=true \
    # --SiftExtraction.use_gpu 0

echo "== Matching (choose one) =="
CUDA_VISIBLE_DEVICES=$gpu_id colmap sequential_matcher --database_path "$DB"

echo "== Triangulate with known poses =="
mkdir -p "$OUT_SPARSE"
CUDA_VISIBLE_DEVICES=$gpu_id colmap point_triangulator \
    --database_path "$DB" \
    --image_path "$IMG_DIR" \
    --input_path "$KNOWN_SPARSE" \
    --output_path "$OUT_SPARSE"

echo "== (Optional) Undistort =="
CUDA_VISIBLE_DEVICES=$gpu_id colmap image_undistorter \
    --image_path "$IMG_DIR" \
    --input_path "$OUT_SPARSE" \
    --output_path "$UNDIST_OUT" \
    --output_type COLMAP

echo "Done. Triangulated sparse model: $OUT_SPARSE"

CUDA_VISIBLE_DEVICES=$gpu_id colmap model_converter \
    --input_path "${PROJECT}/sparse" \
    --output_path "${PROJECT}/sparse/points3D.ply" \
    --output_type PLY

echo "== Convert undistorted sparse model to text =="
colmap model_converter \
    --input_path   ${UNDIST_OUT}/sparse \
    --output_path  ${UNDIST_OUT}/sparse \
    --output_type  TXT