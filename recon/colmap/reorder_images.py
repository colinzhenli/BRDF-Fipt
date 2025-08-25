#!/usr/bin/env python3
import re, sys

inp = sys.argv[1]          # input images.txt (poses-only)
outp = sys.argv[2]         # output images.txt (sorted & renumbered)

def scan_id_from_name(name: str) -> int:
    m = re.search(r"scan-(\d+)-phi", name)
    if not m:
        raise ValueError(f"Bad name format: {name}")
    return int(m.group(1))

headers = []
entries = []  # (scan_id, fields, raw_name)

with open(inp, "r", encoding="utf-8") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].rstrip("\n")
    if not line or line.lstrip().startswith("#"):
        headers.append(line)
        i += 1
        continue

    toks = line.split()
    if len(toks) < 10:
        i += 1
        continue

    # Parse the pose header
    img_id = toks[0]
    qw, qx, qy, qz = toks[1:5]
    tx, ty, tz = toks[5:8]
    cam_id = toks[8]
    name = " ".join(toks[9:])  # support spaces (unlikely here)

    sid = scan_id_from_name(name)
    entries.append((sid, (qw, qx, qy, qz, tx, ty, tz, cam_id, name)))

    # Skip the following (empty) POINTS2D line if present
    if i + 1 < len(lines) and not lines[i+1].strip().startswith("#"):
        i += 2
    else:
        i += 1

# Sort by numeric scan ID
entries.sort(key=lambda e: e[0])

with open(outp, "w", encoding="utf-8") as f:
    # Write two standard header comments like COLMAP style
    if not headers:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#\n")
    else:
        for h in headers[:2]:
            f.write(h + "\n")

    for new_id, (_, fields) in enumerate(entries, start=1):
        qw, qx, qy, qz, tx, ty, tz, cam_id, name = fields
        f.write(f"{new_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {name}\n")
        f.write("\n")  # blank POINTS2D line
print(f"Wrote sorted images.txt → {outp}")
