#!/usr/bin/env python3
"""Upload per-material capture files to Compute Canada.

Runs on the file server directly (no NFS). For each material subfolder under
SRC_ROOT that contains ALL of REQUIRED_FILES, it uploads those files to
DEST_HOST:DEST_ROOT/<material>/.

Uses a single persistent SSH connection (ControlMaster) shared by all rsync
calls, so authentication happens once. rsync is used as the "fast protocol"
and provides per-file progress; tqdm shows overall material progress.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm


REQUIRED_FILES = [
    "scan_log.json",
    "rotated_camera.json",
    "bbox.json",
    "point_metadata.json",
    "observations_structured.npz",
]

DEFAULT_SRC = "/media/raid/cloth/capture_data/Dataset_Nov11"
DEFAULT_USER_HOST = "zla247@fir.computecanada.ca"
DEFAULT_DEST = "~/scratch/data/capture_data"
DEFAULT_STATE = "/media/raid/cloth/capture_data/.uploaded_to_computecanada.json"


def load_state(path: Path):
    if not path.exists():
        return {"uploaded": []}
    try:
        with path.open("r") as f:
            data = json.load(f)
        if "uploaded" not in data or not isinstance(data["uploaded"], list):
            return {"uploaded": []}
        return data
    except (json.JSONDecodeError, OSError):
        print(f"WARNING: could not parse state file {path}; starting fresh.")
        return {"uploaded": []}


def save_state(path: Path, uploaded_set):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump({"uploaded": sorted(uploaded_set, key=natural_key)}, f, indent=2)
    tmp.replace(path)


def find_valid_materials(src_root: Path):
    """Return sorted list of material dir names where all REQUIRED_FILES exist."""
    valid = []
    skipped = []
    for entry in sorted(src_root.iterdir()):
        if not entry.is_dir():
            continue
        missing = [f for f in REQUIRED_FILES if not (entry / f).exists()]
        if missing:
            skipped.append((entry.name, missing))
        else:
            valid.append(entry.name)
    return valid, skipped


def natural_key(name: str):
    # Sort 0, 1, 2, ..., 10, 11 naturally; keep non-numeric names after.
    try:
        return (0, int(name), name)
    except ValueError:
        return (1, 0, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=DEFAULT_SRC,
                        help=f"Source root (default: {DEFAULT_SRC})")
    parser.add_argument("--user-host", default=DEFAULT_USER_HOST,
                        help=f"user@host (default: {DEFAULT_USER_HOST})")
    parser.add_argument("--dest", default=DEFAULT_DEST,
                        help=f"Remote destination root (default: {DEFAULT_DEST})")
    parser.add_argument("--dry-run", action="store_true",
                        help="List eligible materials but do not upload.")
    parser.add_argument("--materials", nargs="*", default=None,
                        help="Restrict to specific material names (optional).")
    parser.add_argument("--state-file", default=DEFAULT_STATE,
                        help=f"JSON file tracking uploaded materials "
                             f"(default: {DEFAULT_STATE})")
    parser.add_argument("--force", action="store_true",
                        help="Re-upload materials even if listed in state file.")
    args = parser.parse_args()

    if shutil.which("rsync") is None:
        sys.exit("ERROR: rsync is required but not found in PATH.")
    if shutil.which("ssh") is None:
        sys.exit("ERROR: ssh is required but not found in PATH.")

    src_root = Path(args.src).resolve()
    if not src_root.is_dir():
        sys.exit(f"ERROR: source root does not exist: {src_root}")

    valid, skipped = find_valid_materials(src_root)
    valid.sort(key=natural_key)

    if args.materials:
        requested = set(args.materials)
        valid = [m for m in valid if m in requested]

    state_path = Path(args.state_file)
    state = load_state(state_path)
    already_uploaded = set(state["uploaded"])

    if args.force:
        to_upload = list(valid)
        already_skipped = []
    else:
        to_upload = [m for m in valid if m not in already_uploaded]
        already_skipped = [m for m in valid if m in already_uploaded]

    print(f"Source: {src_root}")
    print(f"Destination: {args.user_host}:{args.dest}")
    print(f"State file: {state_path}")
    print(f"Eligible materials: {len(valid)}")
    print(f"Already uploaded (skipping): {len(already_skipped)}")
    print(f"To upload this run: {len(to_upload)}")
    if skipped:
        print(f"Skipped (missing required files): {len(skipped)}")
        for name, missing in skipped[:10]:
            print(f"  - {name}: missing {missing}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    if not to_upload:
        print("Nothing to upload.")
        return

    if args.dry_run:
        print("Dry run — materials that would be uploaded:")
        for name in to_upload:
            print(f"  {name}")
        return

    # Set up persistent SSH control master so all rsync calls reuse one
    # authenticated TCP connection.
    ctl_dir = tempfile.mkdtemp(prefix="ssh_ctl_")
    ctl_path = os.path.join(ctl_dir, "cm-%r@%h:%p")
    ssh_opts = [
        "-o", f"ControlPath={ctl_path}",
        "-o", "ControlMaster=auto",
        "-o", "ControlPersist=600",
        "-o", "ServerAliveInterval=30",
        "-o", "Compression=no",          # capture files are already compressed
    ]

    def ssh_cmd(*extra):
        return ["ssh", *ssh_opts, *extra]

    rsh = "ssh " + " ".join(ssh_opts)

    try:
        # 1. Open the master connection (prompts for password/2FA once).
        print("\nOpening persistent SSH connection (authenticate once)...")
        master_proc = subprocess.run(
            ssh_cmd("-o", "ControlMaster=yes",
                    "-o", "ControlPersist=600",
                    args.user_host, "echo connected"),
        )
        if master_proc.returncode != 0:
            sys.exit("ERROR: failed to establish SSH master connection.")

        # 2. Ensure remote root exists.
        subprocess.run(
            ssh_cmd(args.user_host, f"mkdir -p {args.dest}"),
            check=True,
        )

        # 3. Upload each material via rsync over the shared connection.
        failures = []
        pbar = tqdm(to_upload, desc="Materials", unit="mat")
        for name in pbar:
            pbar.set_postfix_str(name)
            src_dir = src_root / name
            remote_dir = f"{args.dest}/{name}"

            # Create remote material dir.
            mk = subprocess.run(
                ssh_cmd(args.user_host, f"mkdir -p {remote_dir}"),
            )
            if mk.returncode != 0:
                failures.append((name, "mkdir failed"))
                continue

            src_files = [str(src_dir / f) for f in REQUIRED_FILES]
            # rsync with per-file progress; -a preserves perms/times; no -z
            # because observations_structured.npz is already compressed.
            cmd = [
                "rsync", "-a", "--info=progress2", "--human-readable",
                "-e", rsh,
                *src_files,
                f"{args.user_host}:{remote_dir}/",
            ]
            # Let rsync print its own progress above the tqdm bar.
            pbar.clear()
            result = subprocess.run(cmd)
            pbar.refresh()
            if result.returncode != 0:
                failures.append((name, f"rsync exit {result.returncode}"))
            else:
                # Persist progress immediately so a later interrupt
                # still leaves the state file accurate.
                already_uploaded.add(name)
                save_state(state_path, already_uploaded)

        pbar.close()

        print("\nDone.")
        print(f"Uploaded this run: {len(to_upload) - len(failures)} / {len(to_upload)}")
        print(f"Total uploaded (state file): {len(already_uploaded)}")
        if failures:
            print("Failures:")
            for name, reason in failures:
                print(f"  {name}: {reason}")
            sys.exit(1)

    finally:
        # Tear down the master connection.
        subprocess.run(
            ssh_cmd("-O", "exit", args.user_host),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        shutil.rmtree(ctl_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
