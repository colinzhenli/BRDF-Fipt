#!/usr/bin/env bash

DATASET_DIR="$1"
gpu_id="$2"
echo "Starting reconstruction pipeline for: $DATASET_DIR"

# Step 1: Convert binary files to text format
echo "== Converting COLMAP binary files to text =="
CUDA_VISIBLE_DEVICES=$gpu_id python3 recon/colmap/convert_bin.py "$DATASET_DIR/sparse/0"

# Step 2: Run COLMAP with known poses
echo "== Running COLMAP reconstruction =="
bash recon/colmap/colmap_known_pose.sh "$DATASET_DIR" "$gpu_id"

# Create the sparse/0 subdirectory and move sparse files
echo "== Organizing undistorted sparse files =="
mkdir -p "$DATASET_DIR/undistorted/sparse/0"
mv "$DATASET_DIR/undistorted/sparse"/* "$DATASET_DIR/undistorted/sparse/0/" 2>/dev/null || true

# Step 2.5: Convert undistorted binary files to text format
echo "== Converting undistorted COLMAP binary files to text =="
CUDA_VISIBLE_DEVICES=$gpu_id python3 recon/colmap/convert_bin.py "$DATASET_DIR/undistorted/sparse/0"

# Step 3: Train 2DGS
echo "== Training 2D Gaussian Splatting =="
cd recon/2dgs
CUDA_VISIBLE_DEVICES=$gpu_id python3 train.py -s "$DATASET_DIR/undistorted"
cd ../..

echo "✅ Reconstruction pipeline completed!"

