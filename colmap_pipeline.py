#!/usr/bin/env python3
"""
pure-PyCOLMAP SfM → dense cloud → Poisson mesh.
Works with the stock `pip install pycolmap` wheels (no private DB calls).

USAGE  ───────────────────────────────────────────────────────────
# (A) unknown intrinsics
python sfm_mesh.py --data_dir path/to/project --unknown_intrinsics

# (B) one shared calibrated camera
python sfm_mesh.py --data_dir path/to/project \
                   --width 1920 --height 1080 \
                   --fx 1450.1 --fy 1450.1 --cx 960 --cy 540
"""

from pathlib import Path
import argparse, json, sys
import pycolmap

# ── helpers ────────────────────────────────────────────────────────────────
def save_poses(rec, out_file):
    with out_file.open("w") as f:
        json.dump({im.name: {"qvec": im.qvec.tolist(),
                             "tvec": im.tvec.tolist()}
                   for im in rec.images.values()}, f, indent=2)

# ── pipeline ───────────────────────────────────────────────────────────────
def main(a):
    images = Path(a.data_dir, "images")
    if not images.is_dir():
        sys.exit("images/ folder not found")

    work      = Path(a.data_dir, "colmap")
    db_path   = work / "database_1.db"
    sparse    = work / "sparse_1"
    mvs       = work / "mvs_1"
    mesh_path = Path(a.data_dir, "mesh_1.ply")
    work.mkdir(exist_ok=True)

    # 1. features & matches --------------------------------------------------
    if a.unknown_intrinsics:
        pycolmap.extract_features(db_path, images)  # COLMAP guesses intrinsics
    else:
        iro = pycolmap.ImageReaderOptions()   # defaults
        iro.camera_model   = a.camera_model   # or SIMPLE_RADIAL / OPENCV …
        iro.camera_params  = f"{a.fx},{a.fy},{a.cx},{a.cy}" # ← your intrinsics as string
        # (order must match the chosen camera model)

        pycolmap.extract_features(
            database_path   = db_path,
            image_path      = images,
            camera_mode     = pycolmap.CameraMode.SINGLE,  # one camera for all images
            reader_options  = iro,
        )
    pycolmap.match_exhaustive(db_path)

    # 2. incremental SfM ----------------------------------------------------
    opts = pycolmap.IncrementalMapperOptions()
    if not a.unknown_intrinsics:
        opts.ba_refine_focal_length    = False
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params    = False

    recs = pycolmap.incremental_mapping(db_path, images, sparse, options=opts)
    if not recs:
        sys.exit("mapper failed")
    rec = max(recs, key=lambda r: len(r.images))

    save_poses(rec, Path(a.data_dir, "extrinsics.json"))
    rec.export_PLY(str(Path(a.data_dir, "sparse.ply")))

    # 3. dense cloud --------------------------------------------------------
    pycolmap.undistort_images(mvs, sparse, images)
    pycolmap.patch_match_stereo(mvs)               # CUDA required in build
    pycolmap.stereo_fusion(mvs / "fused.ply", mvs)

    # 4. mesh (Poisson) -----------------------------------------------------
    if hasattr(pycolmap, "poisson_mesher"):
        pycolmap.poisson_mesher(input_path=mvs / "fused.ply",
                                output_path=mesh_path)
        print("mesh →", mesh_path)
    else:
        print("⚠ pycolmap build lacks Poisson mesher; dense cloud saved.")


# ── CLI --------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="folder with images/")
    p.add_argument("--unknown_intrinsics", action="store_true",
                   help="let COLMAP estimate intrinsics")
    p.add_argument("--camera_model", default="PINHOLE")
    p.add_argument("--width",  type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--fx", type=float)
    p.add_argument("--fy", type=float)
    p.add_argument("--cx", type=float)
    p.add_argument("--cy", type=float)
    args = p.parse_args()

    # sanity-check required params if intrinsics are known
    if not args.unknown_intrinsics:
        for k in ("width", "height", "fx", "fy", "cx", "cy"):
            if getattr(args, k) is None:
                p.error(f"--{k} is required unless --unknown_intrinsics is set")

    main(args)
