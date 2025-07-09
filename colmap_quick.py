#!/usr/bin/env python3
"""
COLMAP CLI wrapper for camera extrinsics estimation and mesh generation.
Uses COLMAP command line interface with known camera intrinsics.
"""

import subprocess
import os
import sys
from pathlib import Path
import argparse
# At the top of main()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # or whatever GPU you want

def run_command(cmd, check=True, capture_output=True):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    if capture_output:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            if check:
                sys.exit(1)
        return result
    else:
        # For long-running processes, don't capture output
        result = subprocess.run(cmd, check=True)
        return result


def main():
    parser = argparse.ArgumentParser(description="COLMAP CLI pipeline for mesh generation")
    parser.add_argument("--images", required=True, help="Path to images folder")
    parser.add_argument("--workdir", required=True, help="Working directory for COLMAP output")
    parser.add_argument("--camera_file", help="Path to camera intrinsics txt file (optional)")
    parser.add_argument("--output_mesh", help="Output mesh path (default: workdir/mesh.ply)")
    parser.add_argument("--unknown_intrinsics", action="store_true", 
                       help="Use automatic camera calibration instead of known intrinsics")
    
    args = parser.parse_args()
    
    images_path = Path(args.images)
    work_dir = Path(args.workdir)
    
    if not images_path.exists():
        sys.exit(f"Images folder not found: {images_path}")
    
    # Check camera file only if not using unknown intrinsics
    if not args.unknown_intrinsics:
        if not args.camera_file:
            sys.exit("Camera file is required when not using --unknown_intrinsics")
        camera_file = Path(args.camera_file)
        if not camera_file.exists():
            sys.exit(f"Camera file not found: {camera_file}")
    
    # Create working directory structure
    work_dir.mkdir(parents=True, exist_ok=True)
    database_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    dense_dir = work_dir / "dense"
    sparse_dir.mkdir(exist_ok=True)
    dense_dir.mkdir(exist_ok=True)
    
    output_mesh = args.output_mesh or str(work_dir / "mesh.ply")
    
    # 1. Feature extraction
    print("=== Feature Extraction ===")
    
    if args.unknown_intrinsics:
        # Automatic camera calibration
        print("Using automatic camera calibration")
        run_command([
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_path),
            "--SiftExtraction.use_gpu", "1"
        ], capture_output=True)  # Don't capture output for this long process
    else:
        # Known camera intrinsics
        print("Using known camera intrinsics")
        # run_command([
        #     "colmap", "feature_extractor",
        #     "--database_path", str(database_path),
        #     "--image_path", str(images_path),
        #     "--ImageReader.camera_model", "SIMPLE_RADIAL",
        #     "--ImageReader.single_camera", "1",
        #     "--ImageReader.camera_params", "2559.68,1536,1152,-0.0204997",
        #     "--SiftExtraction.use_gpu", "1"
        # ], capture_output=False)  # Don't capture output for this long process
    
    # 2. Feature matching
    print("=== Feature Matching ===")
    run_command([
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "1"
    ])
    
    # 3. Sparse reconstruction (SfM)
    print("=== Sparse Reconstruction ===")
    run_command([
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(images_path),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "1"
    ])
    
    # Find the reconstruction folder (usually "0")
    reconstruction_dirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
    if not reconstruction_dirs:
        sys.exit("No reconstruction found in sparse directory")
    
    reconstruction_path = reconstruction_dirs[0]  # Use first/largest reconstruction
    print(f"Using reconstruction: {reconstruction_path}")
    
    # Export camera poses
    print("=== Exporting Camera Poses ===")
    poses_file = work_dir / "cameras.txt"
    images_file = work_dir / "images.txt"
    run_command([
        "colmap", "model_converter",
        "--input_path", str(reconstruction_path),
        "--output_path", str(work_dir),
        "--output_type", "TXT"
    ])
    
    # 4. Image undistortion for dense reconstruction
    print("=== Image Undistortion ===")
    run_command([
        "colmap", "image_undistorter",
        "--image_path", str(images_path),
        "--input_path", str(reconstruction_path),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP"
    ])
    
    # 5. Dense stereo matching
    print("=== Dense Stereo Matching ===")
    run_command([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "1",
        "--PatchMatchStereo.gpu_index", "0"
    ])
    
    # 6. Stereo fusion to create dense point cloud
    print("=== Stereo Fusion ===")
    dense_ply = dense_dir / "fused.ply"
    run_command([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(dense_ply)
    ])
    
    # 7. Poisson surface reconstruction
    print("=== Poisson Meshing ===")
    run_command([
        "colmap", "poisson_mesher",
        "--input_path", str(dense_ply),
        "--output_path", output_mesh
    ])
    
    print(f"\n=== Results ===")
    print(f"Sparse reconstruction: {reconstruction_path}")
    print(f"Camera poses (TXT): {poses_file}")
    print(f"Images info (TXT): {images_file}")
    print(f"Dense point cloud: {dense_ply}")
    print(f"Final mesh: {output_mesh}")


if __name__ == "__main__":
    main()
