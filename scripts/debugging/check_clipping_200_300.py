"""Check clipping ratio for materials 200-300 from observations_structured.npz."""
import numpy as np
import os
import sys

base = "/media/raid/cloth/capture_data/Dataset_Nov11"
mats = sorted([int(d) for d in os.listdir(base)
               if os.path.isdir(os.path.join(base, d)) and d.isdigit()
               and 200 <= int(d) <= 300])
print(f"Materials found in 200-300: {mats}")
print()
print(f"{'mat':>5} | {'K':>4} | {'V':>8} | {'valid_obs':>12} | {'median':>8} | {'p99':>8} | {'clipped':>10} | {'% clip':>7}")
print("-" * 80)
sys.stdout.flush()

for mid in mats:
    path = os.path.join(base, str(mid), "observations_structured.npz")
    if not os.path.exists(path):
        print(f"{mid:>5} | MISSING")
        sys.stdout.flush()
        continue
    d = np.load(path)
    rgbs = d["rgbs"]  # (K, V, 3) uint16
    K, V, _ = rgbs.shape
    flat = rgbs.reshape(-1, 3)
    all_zero = flat.sum(axis=1) == 0
    valid = flat[~all_zero]
    n_valid = len(valid)
    any_clip = int((valid == 65535).any(axis=1).sum())
    pct = 100.0 * any_clip / n_valid if n_valid > 0 else 0
    med = int(np.median(valid.max(axis=1)))
    p99 = int(np.percentile(valid.max(axis=1), 99))
    print(f"{mid:>5} | {K:>4} | {V:>8} | {n_valid:>12,} | {med:>8} | {p99:>8} | {any_clip:>10,} | {pct:>6.2f}%")
    sys.stdout.flush()
    del d, rgbs, flat, valid
