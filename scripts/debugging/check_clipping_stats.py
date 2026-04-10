"""Check how many observation pixels hit uint16 max (65535) in observations_structured.npz."""
import numpy as np

mats = [0, 48, 71, 95, 100, 101, 102, 103, 104, 112]
header = f"{'mat':>5} | {'K':>4} | {'V':>8} | {'total_obs':>12} | {'median':>8} | {'p99':>8} | {'==65535':>10} | {'%clipped':>8}"
print(header)
print("-" * len(header))

for mid in mats:
    path = f"/media/raid/cloth/capture_data/Dataset_Nov11/{mid}/observations_structured.npz"
    try:
        d = np.load(path)
        rgbs = d["rgbs"]  # (K, V, 3) uint16
        K, V, _ = rgbs.shape
        total = K * V  # total observation-pixels (each pixel = one valid surface point seen by one image)

        flat = rgbs.reshape(-1, 3)  # (K*V, 3)

        # Per-observation: is ANY channel clipped?
        any_clipped = (flat == 65535).any(axis=1)
        n_clipped = int(any_clipped.sum())
        pct = 100.0 * n_clipped / len(flat)

        # Also count per-channel
        n_r = int((flat[:, 0] == 65535).sum())
        n_g = int((flat[:, 1] == 65535).sum())
        n_b = int((flat[:, 2] == 65535).sum())

        max_ch = flat.max(axis=1)
        med = np.median(max_ch)
        p99 = np.percentile(max_ch, 99)

        print(f"{mid:>5} | {K:>4} | {V:>8} | {total:>12,} | {med:>8.0f} | {p99:>8.0f} | {n_clipped:>10,} | {pct:>7.2f}%")
        print(f"        per-channel clipped:  R={n_r:,}  G={n_g:,}  B={n_b:,}")

        # Also: how many observations are fully zero (occluded)?
        all_zero = (flat.sum(axis=1) == 0)
        n_zero = int(all_zero.sum())
        pct_zero = 100.0 * n_zero / len(flat)
        # Non-zero observations
        nonzero = flat[~all_zero]
        n_clipped_nonzero = int((nonzero == 65535).any(axis=1).sum())
        pct_nonzero = 100.0 * n_clipped_nonzero / len(nonzero) if len(nonzero) > 0 else 0
        print(f"        zero/occluded: {n_zero:,} ({pct_zero:.1f}%) | clipped among valid: {n_clipped_nonzero:,} ({pct_nonzero:.2f}%)")
        print()
    except Exception as e:
        print(f"{mid:>5} | ERROR: {e}")
