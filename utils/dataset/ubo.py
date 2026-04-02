"""
UBO2014 BTF Dataset loaders for stage-2 single-material overfitting.

The Bonn UBO2014 BTF dataset stores per-texel BRDF measurements as a
compressed 5-D tensor: (nL, nV, H, W, 3).  We use the ``btf-extractor``
package to read the binary ``.btf`` files and decode images on the fly.

Data characteristics (important for pipeline correctness):
  - Values are **LDR** float32 in roughly [0, 0.7].  No HDR tone-mapping
    is needed; simple gamma for display is sufficient.
  - Angles from btf-extractor are ``(theta_l, phi_l, theta_v, phi_v)``
    in **degrees** (inclination from +Z, azimuth from +X toward +Y).
  - The sample is a flat surface with normal = (0, 0, 1).  All directions
    are already in the **local tangent frame** — no world-to-local
    transform is needed.
  - Images are RGB float32, shape ``(H, W, 3)``.

Directory layout expected:
    btf_folder/
        <material_name>_W400xH400_L151xV151.btf   (or resampled variant)
"""

import math
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Spherical → Cartesian   (physics convention: theta = inclination from +Z)
# ---------------------------------------------------------------------------

def _sph2cart(theta_deg, phi_deg):
    """Convert (theta, phi) in degrees to unit Cartesian (x, y, z).

    Convention matches UBO dome:
      theta = polar angle from +Z  (0° = straight up/down, 90° = horizon)
      phi   = azimuth from +X toward +Y
    Returns (x, y, z) as float32.
    """
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    phi = np.deg2rad(np.asarray(phi_deg, dtype=np.float64))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# UBOBTFTrainDataset  (IterableDataset, single-material stage-2)
# ---------------------------------------------------------------------------

class UBOBTFTrainDataset(IterableDataset):
    """Training dataset for UBO2014 BTF single-material overfitting.

    Loads one ``.btf`` file, caches all angle combos, and yields infinite
    random batches of ``(wi, wo, rgb, point_id)`` tuples.

    Each batch draws ``rays_num`` samples by randomly picking:
      - a random (light, view) angle pair  →  wi, wo
      - a random pixel (row, col)          →  point_id, rgb
    """

    def __init__(self, cfg, btf_path: str, split='train'):
        from btf_extractor import Ubo2014

        self.cfg = cfg
        self.split = split
        self.rays_num = cfg.data.rays_num
        self.val_ratio = getattr(cfg.data, 'val_view_ratio', 0.2)
        self.val_seed = getattr(cfg.data, 'val_seed', 42)

        print(f"\n{'='*60}")
        print(f"UBOBTFTrainDataset ({split})  |  {Path(btf_path).stem}")
        print(f"{'='*60}")

        # Load BTF
        self.btf = Ubo2014(btf_path)
        self.H, self.W, _ = self.btf.img_shape
        self.n_pixels = self.H * self.W

        # All available angle tuples: (theta_l, phi_l, theta_v, phi_v)
        all_angles = sorted(self.btf.angles_set)
        n_total = len(all_angles)

        # Train/val split on angle combos with a fixed seed
        rng = np.random.RandomState(self.val_seed)
        perm = rng.permutation(n_total)
        n_val = max(1, int(n_total * self.val_ratio))
        if split == 'train':
            indices = np.sort(perm[n_val:])
        else:
            indices = np.sort(perm[:n_val])

        self.angles = [all_angles[i] for i in indices]
        self.n_angles = len(self.angles)

        # Precompute direction vectors  (n_angles, 3) each
        theta_l = np.array([a[0] for a in self.angles], dtype=np.float64)
        phi_l = np.array([a[1] for a in self.angles], dtype=np.float64)
        theta_v = np.array([a[2] for a in self.angles], dtype=np.float64)
        phi_v = np.array([a[3] for a in self.angles], dtype=np.float64)

        self.wi_all = _sph2cart(theta_l, phi_l)  # (n_angles, 3)
        self.wo_all = _sph2cart(theta_v, phi_v)  # (n_angles, 3)

        # Preload all valid images into memory directly
        print(f"Loading {self.n_angles} images into memory... (This may take a few minutes)")
        self.rgbs = np.empty((self.n_angles, self.n_pixels, 3), dtype=np.float32)
        for i, a in enumerate(tqdm(self.angles, desc="Loading BTF angles")):
            img = self.btf.angles_to_image(*a)  # returns BGR (OpenCV convention)
            img = img[:, :, ::-1].copy()         # BGR → RGB
            np.clip(img, 0.0, None, out=img)
            self.rgbs[i] = img.reshape(self.n_pixels, 3)
        print("Loading complete.")

        total_obs = self.n_angles * self.n_pixels
        print(f"  {split}: {self.n_pixels:,} pixels × {self.n_angles} angle combos "
              f"= {total_obs:,} observations ({(self.rgbs.nbytes / 1e9):.2f} GB)")
        print(f"  Image shape: {self.H}×{self.W}, LDR float32")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    def set_step(self, step: int):
        """Compatibility with trainer on_train_batch_start."""
        pass

    def __len__(self):
        return 1_000_000

    # ------------------------------------------------------------------
    def _sample_batch(self, n_rays):
        # Random angle indices and pixel indices
        angle_i = np.random.randint(0, self.n_angles, n_rays)
        pix_i = np.random.randint(0, self.n_pixels, n_rays)

        # Gather directions
        wi = self.wi_all[angle_i]       # (n_rays, 3)
        wo = self.wo_all[angle_i]       # (n_rays, 3)

        # Gather RGB values (vectorized from preloaded array)
        rgbs = self.rgbs[angle_i, pix_i]

        # Point IDs are just the flat pixel indices
        point_ids = pix_i.astype(np.int64)

        # Normal is always (0,0,1) for flat BTF samples
        normal = np.zeros((n_rays, 3), dtype=np.float32)
        normal[:, 2] = 1.0

        return {
            'wi':         torch.from_numpy(wi).float(),
            'wo':         torch.from_numpy(wo).float(),
            'rgbs':       torch.from_numpy(rgbs).float(),
            'point_ids':  torch.from_numpy(point_ids).long(),
            'gt_normals': torch.from_numpy(normal).float(),
        }

    # ------------------------------------------------------------------
    def __iter__(self):
        while True:
            yield self._sample_batch(self.rays_num)


# ---------------------------------------------------------------------------
# UBOBTFValDataset  (Dataset, one full image per __getitem__)
# ---------------------------------------------------------------------------

class UBOBTFValDataset(Dataset):
    """Validation dataset for UBO2014 BTF single-material overfitting.

    Uses the held-out angle combos from the same fixed-seed split.
    Each ``__getitem__`` returns **all pixels** (H*W) for one angle combo,
    together with ``img_hw`` for 2-D image reconstruction.
    """

    def __init__(self, cfg, btf_path: str):
        from btf_extractor import Ubo2014

        self.cfg = cfg
        self.valid_num = getattr(cfg.data, 'valid_num', 10)
        self.val_ratio = getattr(cfg.data, 'val_view_ratio', 0.2)
        self.val_seed = getattr(cfg.data, 'val_seed', 42)

        print(f"\n{'='*60}")
        print(f"UBOBTFValDataset  |  {Path(btf_path).stem}")
        print(f"{'='*60}")

        # Load BTF
        btf = Ubo2014(btf_path)
        self.H, self.W, _ = btf.img_shape
        self.n_pixels = self.H * self.W

        # Reproduce the held-out angle split
        all_angles = sorted(btf.angles_set)
        n_total = len(all_angles)
        rng = np.random.RandomState(self.val_seed)
        perm = rng.permutation(n_total)
        n_val = max(1, int(n_total * self.val_ratio))
        val_indices = np.sort(perm[:n_val])

        # Limit to valid_num
        if 0 < self.valid_num < len(val_indices):
            val_indices = val_indices[:self.valid_num]

        print(f"  {len(val_indices)} val angle combos  "
              f"({self.n_pixels:,} pixels each)")

        # Pre-decode all val images and build items
        point_ids = np.arange(self.n_pixels, dtype=np.int64)

        self._items = []
        for idx in val_indices:
            angle = all_angles[idx]
            img = btf.angles_to_image(*angle)  # returns BGR (OpenCV convention)
            img = img[:, :, ::-1].copy()         # BGR → RGB
            np.clip(img, 0.0, None, out=img)
            img_flat = img.reshape(-1, 3)  # (H*W, 3)

            wi_vec = _sph2cart(angle[0], angle[1])  # (3,)
            wo_vec = _sph2cart(angle[2], angle[3])  # (3,)

            # Broadcast to all pixels (same direction for entire image)
            wi = np.broadcast_to(wi_vec[None, :], (self.n_pixels, 3)).copy()
            wo = np.broadcast_to(wo_vec[None, :], (self.n_pixels, 3)).copy()

            normal = np.zeros((self.n_pixels, 3), dtype=np.float32)
            normal[:, 2] = 1.0

            label = f"tl{angle[0]:.0f}_pl{angle[1]:.0f}_tv{angle[2]:.0f}_pv{angle[3]:.0f}"

            self._items.append({
                'wi':         torch.from_numpy(wi).float(),
                'wo':         torch.from_numpy(wo).float(),
                'rgbs':       torch.from_numpy(img_flat.copy()).float(),
                'point_ids':  torch.from_numpy(point_ids.copy()).long(),
                'gt_normals': torch.from_numpy(normal).float(),
                'img_hw':     torch.tensor([self.H, self.W]),
                'label':      label,
            })

        del btf  # free memory
        print(f"UBOBTFValDataset ready  ({len(self._items)} images)\n"
              f"{'='*60}\n")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]
