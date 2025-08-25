#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_colmap_with_known_poses.sh <project_dir> <gpu_id>
PROJECT="${1:?usage: $0 <project_dir> <gpu_id>}"
GPU_ID="${2:?usage: $0 <project_dir> <gpu_id>}"

DB="${PROJECT}/database.db"
IMG_DIR="${PROJECT}/images_raw"
KNOWN_TXT="${PROJECT}/known_model_txt"      # your prepared cameras.txt & images.txt
TRI_MODEL="${PROJECT}/model_triangulated"   # output after triangulation (fixed poses)
REFINED_MODEL="${PROJECT}/model_refinedK"   # output after BA (intrinsics-only)

echo "== Feature extraction =="
CUDA_VISIBLE_DEVICES=$GPU_ID colmap feature_extractor \
  --database_path "$DB" \
  --image_path "$IMG_DIR" \
  --ImageReader.single_camera=true \

echo "== Matching =="
CUDA_VISIBLE_DEVICES=$GPU_ID colmap sequential_matcher \
  --database_path "$DB"

mkdir -p $TRI_MODEL
mkdir -p $REFINED_MODEL
echo "== Triangulate with known poses =="
colmap point_triangulator \
  --database_path "$DB" \
  --image_path "$IMG_DIR" \
  --input_path  "$KNOWN_TXT" \
  --output_path "$TRI_MODEL" \
  --Mapper.tri_ignore_two_view_tracks 1

echo "== Bundle adjust (intrinsics only; extrinsics fixed) =="
colmap bundle_adjuster \
  --input_path  "$TRI_MODEL" \
  --output_path "$REFINED_MODEL" \
  --BundleAdjustment.refine_rig_from_world 0 \
  --BundleAdjustment.refine_sensor_from_rig 0 \
  --BundleAdjustment.refine_focal_length 0 \
  --BundleAdjustment.refine_principal_point 0 \
  --BundleAdjustment.refine_extra_params 0

echo "== (Optional) Export refined model to TXT and undistort =="
colmap model_converter --input_path "$REFINED_MODEL" --output_path "$REFINED_MODEL" --output_type TXT
colmap image_undistorter --image_path "$IMG_DIR" --input_path "$REFINED_MODEL" --output_path "${PROJECT}/undistorted" --output_type COLMAP
colmap model_converter --input_path "${PROJECT}/undistorted/sparse" --output_path "${PROJECT}/undistorted/sparse" --output_type TXT

CUDA_VISIBLE_DEVICES=$GPU_ID colmap model_converter \
    --input_path "$TRI_MODEL" \
    --output_path "${TRI_MODEL}/points3D.ply" \
    --output_type PLY

# For the refined model:
colmap model_analyzer --path "$REFINED_MODEL" 