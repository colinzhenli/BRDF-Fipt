"""Compute per-channel statistics of the UBO carpet07 BTF dataset (RGB-corrected)."""
import numpy as np
from btf_extractor import Ubo2014

btf_path = "/mnt/data/colin/colin/Bonn_BTF/carpet07_W400xH400_L151xV151.btf"
btf = Ubo2014(btf_path)

angles = sorted(btf.angles_set)
print(f"Total angles: {len(angles)}")

all_means = []
all_mins = []
all_maxs = []
global_sum = np.zeros(3, dtype=np.float64)
global_count = 0

for i, a in enumerate(angles):
    img = btf.angles_to_image(*a)
    img = img[:, :, ::-1].copy()  # BGR → RGB
    np.clip(img, 0.0, None, out=img)

    all_means.append(img.reshape(-1, 3).mean(axis=0))
    all_mins.append(img.min())
    all_maxs.append(img.max())
    global_sum += img.reshape(-1, 3).astype(np.float64).sum(axis=0)
    global_count += img.shape[0] * img.shape[1]

all_means = np.array(all_means)
print(f"\nPer-angle mean (averaged over all angles):")
print(f"  R={all_means[:,0].mean():.5f}  G={all_means[:,1].mean():.5f}  B={all_means[:,2].mean():.5f}")
print(f"  Overall mean: {all_means.mean():.5f}")

print(f"\nGlobal mean (all pixels, all angles):")
gm = global_sum / global_count
print(f"  R={gm[0]:.5f}  G={gm[1]:.5f}  B={gm[2]:.5f}")
print(f"  Overall: {gm.mean():.5f}")

print(f"\nGlobal min:  {min(all_mins):.5f}")
print(f"Global max:  {max(all_maxs):.5f}")

print(f"\nPercentiles of per-angle means:")
for p in [10, 25, 50, 75, 90]:
    print(f"  p{p}: {np.percentile(all_means.mean(axis=1), p):.5f}")
