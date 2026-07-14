#!/usr/bin/env python3
"""Upload the RoboCloth artifact bundle (checkpoints, comparison datasets,
render assets) to the Hugging Face dataset repo.

Layout created in the repo:
    checkpoints/stage1/{Ours,Bonn,MERL}.ckpt
    checkpoints/stage2/RoboCloth/<mat>/{Ours,Bonn,MERL,PBR}_epoch<N>.ckpt
    checkpoints/stage2/UBO/<mat>/{Ours,Bonn,MERL,PBR}_epoch<N>.ckpt
    checkpoints/stage2/Bonn/<mat>/{Ours,Bonn,MERL,PBR}_epoch<N>.ckpt
    datasets/MERL/brdfs/*.binary
    datasets/UBO2014/<mat>_W400xH400_L151xV151.btf
    render_assets/{onBars01_st_hp.ply,pole_spheres_v4.obj}

Resume-safe: files already present in the repo with the same size are skipped.

Usage:
    python upload_robocloth_artifacts.py [--repo koalapenguin/RoboCloth]
        [--dry-run] [--private]
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import HfApi  # noqa: E402

RAID = os.path.expanduser(os.environ.get("RAID_ROOT", "~/raid_mnt/cloth"))
S1 = f"{RAID}/output/BRDF/Stage-1-Finals"
S2 = f"{RAID}/output/BRDF/Stage-2-Finals"
MESHES = f"{RAID}/output/Misuba_rendering/output/render/felt01/cloth_candidates/meshes"

ROBOCLOTH_MATS = ["145", "226", "314", "370", "452"]
UBO_MATS = ["fabric02", "fabric04", "fabric09", "fabric11",
            "felt01", "felt03", "felt05", "felt10",
            "carpet02", "carpet07", "carpet09", "carpet12"]
BONN_MATS = ["318", "377", "32", "226", "37"]


def build_manifest():
    """Return list of (local_path, path_in_repo)."""
    m = []
    for name in ["Ours", "Bonn", "MERL"]:
        m.append((f"{S1}/{name}.ckpt", f"checkpoints/stage1/{name}.ckpt"))
    for mat in ROBOCLOTH_MATS:
        for f in sorted(glob.glob(f"{S2}/Ours/{mat}/*_epoch*.ckpt")):
            m.append((f, f"checkpoints/stage2/RoboCloth/{mat}/{os.path.basename(f)}"))
    for mat in UBO_MATS:
        for f in sorted(glob.glob(f"{S2}/UBO/{mat}/*_epoch*.ckpt")):
            m.append((f, f"checkpoints/stage2/UBO/{mat}/{os.path.basename(f)}"))
    for mat in BONN_MATS:
        for f in sorted(glob.glob(f"{S2}/Bonn/{mat}/*_epoch*.ckpt")):
            m.append((f, f"checkpoints/stage2/Bonn/{mat}/{os.path.basename(f)}"))
    for f in sorted(glob.glob(f"{RAID}/BRDFDatabase/brdfs/*.binary")):
        m.append((f, f"datasets/MERL/brdfs/{os.path.basename(f)}"))
    for mat in UBO_MATS:
        f = f"{RAID}/BTF/{mat}_W400xH400_L151xV151.btf"
        m.append((f, f"datasets/UBO2014/{os.path.basename(f)}"))
    for f in ["onBars01_st_hp.ply", "pole_spheres_v4.obj"]:
        m.append((f"{MESHES}/{f}", f"render_assets/{f}"))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="koalapenguin/RoboCloth")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    manifest = build_manifest()
    missing = [(l, r) for l, r in manifest if not os.path.isfile(l)]
    if missing:
        print("MISSING local files:")
        for l, _ in missing:
            print("  ", l)
    manifest = [(l, r) for l, r in manifest if os.path.isfile(l)]
    total = sum(os.path.getsize(l) for l, _ in manifest)
    print(f"{len(manifest)} files, {total / 1e9:.1f} GB total")
    if args.dry_run:
        for l, r in manifest:
            print(f"{os.path.getsize(l)/1e9:7.2f} GB  {r}")
        return

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)

    existing = {}
    try:
        for entry in api.list_repo_tree(args.repo, repo_type="dataset", recursive=True):
            if hasattr(entry, "size") and entry.size is not None:
                existing[entry.path] = entry.size
    except Exception as e:
        print(f"(could not list repo tree, uploading everything: {e})")

    done = 0
    for local, remote in manifest:
        size = os.path.getsize(local)
        if existing.get(remote) == size:
            print(f"[skip] {remote} (already uploaded, {size/1e9:.2f} GB)")
            done += size
            continue
        print(f"[upload] {remote} ({size/1e9:.2f} GB) ... ", flush=True)
        for attempt in range(3):
            try:
                api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                                repo_id=args.repo, repo_type="dataset")
                break
            except Exception as e:
                print(f"  retry {attempt+1}/3 after error: {e}", flush=True)
                if attempt == 2:
                    raise
        done += size
        print(f"[ok] {remote}  ({done/1e9:.1f}/{total/1e9:.1f} GB done)", flush=True)

    print("ALL-UPLOADS-DONE")


if __name__ == "__main__":
    main()
