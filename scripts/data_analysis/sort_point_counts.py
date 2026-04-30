"""Sort materials in Dataset_Nov11 by num_points (ascending) and dump a table.

Reads `point_metadata.json` from each material folder under
`/media/raid/cloth/capture_data/Dataset_Nov11/` and prints a three-column
table (mat_id, num_points, num_observations) sorted ascending by num_points.

Usage:
    python scripts/data_analysis/sort_point_counts.py
    python scripts/data_analysis/sort_point_counts.py --max-id 470
    python scripts/data_analysis/sort_point_counts.py --output point_counts_sorted.txt
"""

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path("/media/raid/cloth/capture_data/Dataset_Nov11")


def _list_material_ids(data_dir: Path, min_id: int, max_id: int) -> list[int]:
    ids: list[int] = []
    for p in sorted(data_dir.iterdir()):
        if not p.is_dir():
            continue
        try:
            mat_id = int(p.name)
        except ValueError:
            continue
        if mat_id < min_id or mat_id > max_id:
            continue
        if (p / "point_metadata.json").exists():
            ids.append(mat_id)
    return sorted(ids)


def _read_counts(data_dir: Path, mat_id: int) -> tuple[int, int] | None:
    meta_path = data_dir / str(mat_id) / "point_metadata.json"
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  WARN: mat{mat_id}: {exc}", file=sys.stderr)
        return None
    if "num_points" not in meta:
        return None
    return int(meta["num_points"]), int(meta.get("num_observations", 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--min-id", type=int, default=0)
    ap.add_argument("--max-id", type=int, default=470)
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional output file path; defaults to stdout only.")
    args = ap.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: {args.data_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    mat_ids = _list_material_ids(args.data_dir, args.min_id, args.max_id)
    rows: list[tuple[int, int, int]] = []
    for mid in mat_ids:
        result = _read_counts(args.data_dir, mid)
        if result is None:
            continue
        np_, no_ = result
        rows.append((mid, np_, no_))

    rows.sort(key=lambda r: r[1])

    lines = [f"Found {len(rows)} materials",
             f"  {'mat_id':>6}  {'num_points':>11}  {'num_obs':>13}"]
    for mid, np_, no_ in rows:
        lines.append(f"  {mid:>6d}  {np_:>11,d}  {no_:>13,d}")
    text = "\n".join(lines) + "\n"

    sys.stdout.write(text)

    if args.output is not None:
        args.output.write_text(text)
        print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
