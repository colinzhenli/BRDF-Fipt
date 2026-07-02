#!/usr/bin/env python
"""Decode a wandb run's binary ``run-<id>.wandb`` log into a per-epoch PSNR history.

A wandb run folder has NO per-epoch JSON. ``files/wandb-summary.json`` holds only
the *last* logged value of each metric; the full per-step history lives in the
binary ``run-<id>.wandb`` transaction log. This reads that log offline (no network)
and emits the per-validation-epoch metrics as JSON + CSV so the curve can be plotted
and any epoch can be queried.

Usage:
    python scripts/extract_wandb_psnr_history.py <exp_dir | run_dir | .wandb file> [-o OUT_PREFIX]

On Bonn the image-level "overall" PSNR is the image-count-weighted poly/gray average.
The standard val split (RandomState(42), ratio 0.2, 668 imgs) is 27 poly + 106 gray,
so this also reports  overall = (27*poly + 106*gray) / 133  at each epoch.
See memory: feedback_psnr_from_wandb.
"""
import sys, os, json, glob, argparse
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb

N_POLY, N_GRAY = 27, 106  # Bonn val split

def find_wandb_file(path):
    if path.endswith(".wandb") and os.path.isfile(path):
        return path
    # Pick the newest run dir by NAME (run start-timestamp). mtime is unreliable
    # for downloaded/rsynced runs (it can make an empty, stale resume look newest).
    # For resumed runs with several run-* dirs, the newest-named one usually holds
    # the final epochs -- verify its last epoch against the checkpoint; if it has
    # 0 val rows, fall back to the next newest.
    run_dirs = sorted(glob.glob(os.path.join(path, "wandb", "run-*")) +
                      glob.glob(os.path.join(path, "run-*")))
    for rd in reversed(run_dirs):
        wf = glob.glob(os.path.join(rd, "*.wandb"))
        if wf:
            return wf[0]
    cands = glob.glob(os.path.join(path, "*.wandb"))
    if not cands:
        raise FileNotFoundError(f"no .wandb file under {path}")
    return cands[0]

def read_val_history(wandb_file):
    ds = datastore.DataStore()
    tmp = None
    try:
        ds.open_for_scan(wandb_file)            # opens "r+b" -> needs write perm
    except (PermissionError, OSError):
        # downloaded runs are often read-only; scan a writable temp copy instead
        import tempfile, shutil
        fd, tmp = tempfile.mkstemp(suffix=".wandb")
        os.close(fd)
        shutil.copyfile(wandb_file, tmp)
        ds.open_for_scan(tmp)
    rows = []
    try:
        while True:
            try:
                data = ds.scan_data()
            except Exception:
                break
            if data is None:
                break
            rec = pb.Record()
            rec.ParseFromString(data)
            if rec.WhichOneof("record_type") != "history":
                continue
            d = {}
            for it in rec.history.item:
                key = it.key or ".".join(it.nested_key)
                try:
                    d[key] = json.loads(it.value_json)
                except Exception:
                    d[key] = it.value_json
            if "val/poly_psnr" in d:          # a validation step
                rows.append(d)
    finally:
        if tmp is not None:
            os.remove(tmp)
    return rows

def clean(rows):
    keys = ["epoch", "_step", "val/poly_psnr", "val/gray_psnr", "val/all_psnr",
            "val/poly_mse", "val/gray_mse", "val/all_mse"]
    out = []
    for d in rows:
        r = {k: d.get(k) for k in keys}
        pp, gp = d.get("val/poly_psnr"), d.get("val/gray_psnr")
        if pp is not None and gp is not None:
            r["val/overall_psnr_imgwavg"] = (N_POLY * pp + N_GRAY * gp) / (N_POLY + N_GRAY)
        out.append(r)
    out.sort(key=lambda r: (r["epoch"] if r["epoch"] is not None else -1))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="exp dir, run dir, or .wandb file")
    ap.add_argument("-o", "--out", default=None, help="output prefix; writes <out>.json and <out>.csv")
    args = ap.parse_args()

    wf = find_wandb_file(args.path)
    out = clean(read_val_history(wf))
    if not out:
        print(f"{wf}: no validation rows found"); return
    print(f"{wf}\n  -> {len(out)} validation epochs ({out[0]['epoch']}..{out[-1]['epoch']})")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out + ".json", "w") as f:
            json.dump(out, f, indent=2)
        cols = list(out[0].keys())
        with open(args.out + ".csv", "w") as f:
            f.write(",".join(cols) + "\n")
            for r in out:
                f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
        print(f"  wrote {args.out}.json  and  {args.out}.csv")
    else:
        for r in out:
            ov = r.get("val/overall_psnr_imgwavg", float("nan"))
            print(f"  ep{str(r['epoch']):>4}  overall {ov:6.3f}  poly {r['val/poly_psnr']:.3f}  gray {r['val/gray_psnr']:.3f}")

if __name__ == "__main__":
    main()
