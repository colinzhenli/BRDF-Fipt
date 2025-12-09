#!/usr/bin/env bash

PROJECT="${1:?usage: $0 <project_dir>}"
gpu_id="${2:?usage: $0 <gpu_id>}"

IMG_DIR="${PROJECT}/ldr"
DB="${PROJECT}/database.db"

# Models
OUT_SPARSE="${PROJECT}/sparse"         # final refined sparse model
TRI_MODEL="${PROJECT}/sparse_tri"      # raw triangulated model
INIT_MODEL="${PROJECT}/init_model"     # your known-poses model (cameras.txt, images.txt, points3D.txt)
UNDIST_OUT="${PROJECT}/undistorted"    # undistorted workspace (optional)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "== Initialize camera poses from robot data =="
python "${REPO_ROOT}/recon/calibration/init_camera.py" \
    shape_matching.folder_path="${PROJECT}"

# Move cameras.txt from parent folder to INIT_MODEL
PARENT_DIR="$(dirname "${PROJECT}")"
cp "${PARENT_DIR}/cameras.txt" "${INIT_MODEL}/cameras.txt"

# Create empty points3D.txt in INIT_MODEL
touch "${INIT_MODEL}/points3D.txt"

if [ $? -eq 0 ]; then
    echo "== Init camera finished successfully =="
else
    echo "== Init camera failed, exiting =="
    exit 1
fi

echo "== Feature extraction =="
CUDA_VISIBLE_DEVICES=$gpu_id colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$IMG_DIR" \
    --ImageReader.single_camera=true

echo "== Sequential matching =="
CUDA_VISIBLE_DEVICES=$gpu_id colmap sequential_matcher \
    --database_path "$DB"

echo "== Triangulate points from known camera poses =="
mkdir -p "$TRI_MODEL"
CUDA_VISIBLE_DEVICES=$gpu_id colmap point_triangulator \
    --database_path "$DB" \
    --image_path "$IMG_DIR" \
    --input_path "$INIT_MODEL" \
    --output_path "$TRI_MODEL"

echo "== Bundle adjust (refine extrinsics and intrinsics) =="
mkdir -p "$OUT_SPARSE"
colmap bundle_adjuster \
  --input_path  "$TRI_MODEL" \
  --output_path "$OUT_SPARSE" \
  --BundleAdjustment.refine_rig_from_world 1 \
  --BundleAdjustment.refine_sensor_from_rig 0 \
  --BundleAdjustment.refine_focal_length 1 \
  --BundleAdjustment.refine_principal_point 0 \
  --BundleAdjustment.refine_extra_params 1

# echo "== Undistort images =="
# colmap image_undistorter \
#     --image_path=${IMG_DIR} \
#     --input_path=${OUT_SPARSE} \
#     --output_path=${UNDIST_OUT} \
#     --output_type=COLMAP

echo "== Convert sparse model to text =="
colmap model_converter \
    --input_path   "$OUT_SPARSE" \
    --output_path  "$OUT_SPARSE" \
    --output_type  TXT

# echo "== Convert undistorted sparse model to text =="
# colmap model_converter \
#     --input_path   ${UNDIST_OUT}/sparse \
#     --output_path  ${UNDIST_OUT}/sparse \
#     --output_type  TXT

echo "== Convert sparse model to ply =="
CUDA_VISIBLE_DEVICES=$gpu_id colmap model_converter \
    --input_path  "$OUT_SPARSE" \
    --output_path "$OUT_SPARSE/points3D.ply" \
    --output_type PLY

echo "== Finished COLMAP reconstruction with known poses =="
