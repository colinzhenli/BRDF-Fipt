import torch
import numpy as np
import os
import re
import math
import scipy.io as spio
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
import time
from dataclasses import dataclass
import random
import pyexr
import imageio

DTYPE_POLY = 0
DTYPE_PAN = 1
DTYPE_LLS = 2


# ---------------------------------------------------------------------------
# Channel-name parsers
# ---------------------------------------------------------------------------

def _parse_poly_channels(channel_names):
    """Parse poly channel names into per-image descriptors.

    pyexr returns channels sorted alphabetically.  Every 3 consecutive
    channels form one RGB image: (_B, _G, _R) which correspond to
    visual (R, G, B) — matching the official Bonn reference code that
    passes ``poly[:, :, :3]`` directly to ``plt.imshow``.

    Returns: list of dicts with keys camera, led, rotation, ch_start
             (ch_start = starting index of the 3-channel group).
    """
    pattern = re.compile(r'poly_(cv\d+)_(il\d+)_(rot\d+)_[BGR]')
    images = []
    for i in range(0, len(channel_names), 3):
        if i + 2 >= len(channel_names):
            break
        m = pattern.match(channel_names[i])
        if m:
            cam, led, rot = m.groups()
            images.append(dict(camera=cam, led=led, rotation=rot, ch_start=i))
    return images



# ---------------------------------------------------------------------------
# EXR reading helpers  (uses pyexr, matching the official Bonn reference code)
# ---------------------------------------------------------------------------

def _read_exr(filepath):
    """Read all channels from an EXR using pyexr.

    Returns (data (H,W,C) float16, channel_names list, H, W).
    The Bonn EXR files store data natively as half precision,
    so we keep float16 to save memory.
    """
    exr = pyexr.open(str(filepath))
    ch_names = exr.channel_map['all']
    data = exr.get(group='all', precision=pyexr.HALF)  # float16
    H, W = data.shape[:2]
    return data, ch_names, H, W


def _read_xyz_map(filepath):
    """Read xyz_rot000.exr → (xyz (H,W,3) float32, H, W)."""
    xyz = pyexr.read(str(filepath))
    H, W = xyz.shape[:2]
    return xyz, H, W


# ---------------------------------------------------------------------------
# BonnDataset
# ---------------------------------------------------------------------------

class BonnDataset(IterableDataset):
    """Bonn SVBRDF database (UBOFAB19) dataset.

    Loads calibrated HDR measurements from multi-channel EXR files.
    All images are reprojected onto the top camera's pixel grid so
    pixel (i, j) across ALL channels refers to the same surface point.

    Data type indices:
        0 = polychromatic  (RGB, color-filtered LEDs)
        1 = panchromatic   (grayscale, unfiltered LEDs)
        2 = LLS            (grayscale, linear light source)
    """

    DTYPE_POLY = DTYPE_POLY

    # ------------------------------------------------------------------
    @dataclass
    class ChunkData:
        """All materials loaded into memory.

        Each entry in *materials* is a dict from ``_load_single_material``:
        mat_id, xyz (V,3), point_ids (V,),
        rgbs (K,V,3) float16, light_pos (K,3), cam_pos (K,3).
        """
        materials: list            # list of per-material dicts
        total_obs: int             # sum(n_pixels * n_images) across materials

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, cfg, root_folder, split='train'):
        self.cfg = cfg
        self.root_folder = Path(root_folder)
        self.split = split
        self.rays_num = cfg.data.rays_num

        self.debug = getattr(cfg.data, 'debug', False)
        self.debug_num = getattr(cfg.data, 'debug_num', 1)
        self.points_per_material = getattr(cfg.data, 'points_per_material', 2000)
        self.random_observations = getattr(cfg.data, 'random_observations', False)
        self.subsample_ratio = getattr(cfg.data, 'subsample_ratio', 1.0)
        self.val_materials = getattr(cfg.data, 'val_materials', 2)
        self.val_points = getattr(cfg.data, 'val_points', 500)
        self.debug_rotate = getattr(cfg.data, 'debug_rotate', False)
        self.debug_swap_channels = getattr(cfg.data, 'debug_swap_channels', False)
        self.step = 0

        # Approximate luminance weights for pan→scalar projection
        self.pan_weights = np.array([0.34, 0.36, 0.28], dtype=np.float32)

        # Discover materials
        self.mat_ids = self._discover_materials()
        if self.debug:
            self.mat_ids = self.mat_ids[:self.debug_num]
            print(f"[DEBUG] Using {self.debug_num} material(s): {self.mat_ids}")

        print(f"\n{'='*60}")
        print(f"BonnDataset ({split})  |  materials={len(self.mat_ids)}")
        print(f"{'='*60}")

        # Load lightweight calibration for every material
        self.calibrations = {}
        for mid in tqdm(self.mat_ids, desc="Loading calibrations"):
            self.calibrations[mid] = self._load_calibration(mid)

        # Load ALL materials into memory (no chunk switching)
        if split == 'train':
            print(f"Loading ALL {len(self.mat_ids)} materials into memory ...")
            self._all_data = self._load_all_materials()
            print(f"All data loaded: {self._all_data.total_obs:,} total observations "
                  f"from {len(self._all_data.materials)} materials")
        else:
            print("Loading validation subset …")
            self._val_data = self._load_all_materials(val_mode=True)
            print(f"Validation: {self._val_data.total_obs:,} observations")

        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Material discovery
    # ------------------------------------------------------------------
    def _discover_materials(self):
        training_list_path = getattr(self.cfg.data, 'training_list_path', '')
        if training_list_path and os.path.isfile(training_list_path):
            mat_ids = []
            with open(training_list_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        mat_ids.append(int(line))
            print(f"Loaded {len(mat_ids)} material IDs from {training_list_path}")
            return sorted(mat_ids)

        poly_files = sorted(self.root_folder.glob('mat*_poly.exr'))
        mat_ids = []
        for p in poly_files:
            mat_str = p.stem.split('_')[0]       # 'mat0001'
            mat_ids.append(int(mat_str[3:]))      # 1
        print(f"Auto-discovered {len(mat_ids)} materials in {self.root_folder}")
        return sorted(mat_ids)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _mat_prefix(self, mat_id):
        return self.root_folder / f'mat{mat_id:04d}'

    def _load_calibration(self, mat_id):
        path = f'{self._mat_prefix(mat_id)}_calibration.mat'
        raw = spio.loadmat(path)
        calib = {}
        for rot_key in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
            rd = raw[rot_key][0, 0]
            rot_dict = {}
            for field in rd.dtype.names:
                val = rd[field]
                if field == 'llsCorners':
                    rot_dict[field] = np.array(val, dtype=np.float32)   # (3,4,14)
                else:
                    rot_dict[field] = np.array(val, dtype=np.float32).flatten()  # (3,)
                # Calibration uses 2-digit names (il01, cv01) but EXR
                # channels use 3-digit (il001, cv01).  Store both keys
                # so lookups from either convention work.
                m = re.match(r'(il|cv)(\d+)', field)
                if m and len(m.group(2)) < 3:
                    padded = f'{m.group(1)}{int(m.group(2)):03d}'
                    rot_dict[padded] = rot_dict[field]
            calib[rot_key] = rot_dict
        calib['llsAnglesDegrees'] = raw['llsAnglesDegrees'].flatten().astype(np.float64)
        return calib

    # ------------------------------------------------------------------
    # Single-material loader
    # ------------------------------------------------------------------
    def _load_single_material(self, mat_id):
        """Load one Bonn material with ALL valid pixels.

        Uses pyexr (matching the official Bonn reference code) to read
        all EXR data.  Poly channels come in alphabetically sorted groups
        of 3 (_B, _G, _R) which map to visual (R, G, B).

        Returns compact dict or None on failure.
        """
        try:
            prefix = self._mat_prefix(mat_id)
            calib = self.calibrations[mat_id]

            # ---- xyz map ------------------------------------------------
            xyz_map, H, W = _read_xyz_map(f'{prefix}_xyz_rot000.exr')
            n_pixels = H * W
            xyz_pts = xyz_map.reshape(n_pixels, 3).astype(np.float32)
            pids = np.arange(n_pixels, dtype=np.int64)

            # ============================================================
            # Read all channels from each EXR using pyexr
            # ============================================================
            poly_data, poly_ch_names, pH, pW = _read_exr(f'{prefix}_poly.exr')
            assert (pH, pW) == (H, W)

            # ============================================================
            # Assembly (poly only, no emitter_ids / lls / data_type)
            # ============================================================
            rgbs_parts  = []
            light_parts = []
            cam_parts   = []

            # --- Poly (RGB) ---------------------------------------------
            poly_images = _parse_poly_channels(poly_ch_names)
            n_poly = len(poly_images)
            if n_poly > 0:
                poly_rgbs = poly_data.reshape(n_pixels, n_poly, 3)  # (V, K, 3)
                poly_rgbs = poly_rgbs.transpose(1, 0, 2)            # (K, V, 3)

                poly_light = np.array([
                    calib[im['rotation']][im['led']] for im in poly_images])
                poly_cam = np.array([
                    calib[im['rotation']][im['camera']] for im in poly_images])

                rgbs_parts.append(poly_rgbs)
                light_parts.append(poly_light)
                cam_parts.append(poly_cam)
            del poly_data

            # ---- concatenate across source types ------------------------
            if not rgbs_parts:
                return None

            all_rgbs      = np.concatenate(rgbs_parts, axis=0)
            all_light_pos = np.concatenate(light_parts, axis=0)
            all_cam_pos   = np.concatenate(cam_parts, axis=0)

            np.clip(all_rgbs, 0, None, out=all_rgbs)

            return {
                'mat_id':    mat_id,
                'xyz':       xyz_pts,         # (V, 3)   float32
                'point_ids': pids,            # (V,)     int64
                'rgbs':      all_rgbs,        # (K, V, 3) float16
                'light_pos': all_light_pos,   # (K, 3)   float32
                'cam_pos':   all_cam_pos,     # (K, 3)   float32
            }

        except Exception as exc:
            print(f"  [Warning] Failed to load mat{mat_id:04d}: {exc}")
            import traceback; traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def _make_debug_pair(self, mat):
        """Create a modified copy of *mat* as a synthetic second material.

        - debug_rotate:        rotate light_pos and cam_pos 180° around Z.
        - debug_swap_channels: swap R and G channels in rgbs.
        """
        import copy
        mat2 = copy.deepcopy(mat)
        mat2['mat_id'] = mat['mat_id'] + 1  # fake second ID

        if self.debug_rotate:
            # 180° rotation around Z: (x, y, z) -> (-x, -y, z)
            mat2['light_pos'] = mat2['light_pos'].copy()
            mat2['light_pos'][:, 0] *= -1
            mat2['light_pos'][:, 1] *= -1
            mat2['cam_pos'] = mat2['cam_pos'].copy()
            mat2['cam_pos'][:, 0] *= -1
            mat2['cam_pos'][:, 1] *= -1
            print(f"[DEBUG] Created mat {mat2['mat_id']} by rotating light/cam 180° around Z")

        if self.debug_swap_channels:
            # Swap R (idx 0) and G (idx 1)
            rgbs = mat2['rgbs'].copy()          # (K, V, 3) float16
            rgbs[:, :, 0], rgbs[:, :, 1] = mat['rgbs'][:, :, 1].copy(), mat['rgbs'][:, :, 0].copy()
            mat2['rgbs'] = rgbs
            print(f"[DEBUG] Created mat {mat2['mat_id']} by swapping R/G channels")

        return mat2

    # ------------------------------------------------------------------
    # Load all materials
    # ------------------------------------------------------------------
    def _load_all_materials(self, val_mode=False):
        """Load all materials into memory at once (no chunking)."""
        if val_mode:
            selected = self.mat_ids[:min(self.val_materials, len(self.mat_ids))]
        else:
            selected = self.mat_ids  # load ALL materials

        materials = []
        tag = 'val' if val_mode else 'train'
        for mid in tqdm(selected, desc=f"Loading {tag} (all materials)"):
            result = self._load_single_material(mid)
            if result is not None:
                materials.append(result)

        # Debug pair: duplicate first material with modification
        if (self.debug_rotate or self.debug_swap_channels) and materials:
            mat2 = self._make_debug_pair(materials[0])
            materials.append(mat2)

        if not materials:
            return BonnDataset.ChunkData(materials=[], total_obs=0)

        total_obs = sum(m['rgbs'].shape[0] * m['rgbs'].shape[1] for m in materials)
        total_rgb_mb = sum(m['rgbs'].nbytes for m in materials) / 1e6

        print(f"[{tag}] All data loaded: {total_obs:,} observations "
              f"from {len(materials)} materials  "
              f"(rgbs {total_rgb_mb:.0f} MB)")

        return BonnDataset.ChunkData(materials=materials, total_obs=total_obs)

    # ------------------------------------------------------------------
    # Training-loop interface
    # ------------------------------------------------------------------
    def set_step(self, step: int):
        """No-op: chunk switching is disabled, all data is in memory."""
        self.step = step

    def __len__(self):
        if hasattr(self, '_val_data'):
            return max(1, math.ceil(self._val_data.total_obs / self.rays_num))
        return 1_000_000

    # ------------------------------------------------------------------
    def _sample_batch(self, chunk, n_rays):
        """Randomly sample n_rays from chunk.

        Rays are distributed across materials proportionally to their
        observation count (n_images * n_pixels), then random (img, pixel)
        pairs are drawn within each material.
        """
        materials = chunk.materials
        obs_counts = np.array(
            [m['rgbs'].shape[0] * m['rgbs'].shape[1] for m in materials],
            dtype=np.float64)
        weights = obs_counts / obs_counts.sum()
        rays_per_mat = np.round(weights * n_rays).astype(int)
        rays_per_mat[-1] = n_rays - rays_per_mat[:-1].sum()

        parts_xyz   = []
        parts_rgbs  = []
        parts_pids  = []
        parts_mids  = []
        parts_light = []
        parts_cam   = []

        for mi, mat in enumerate(materials):
            n = int(rays_per_mat[mi])
            if n <= 0:
                continue

            n_images = mat['rgbs'].shape[0]
            n_pixels = mat['rgbs'].shape[1]

            img_i = np.random.randint(0, n_images, n)
            pix_i = np.random.randint(0, n_pixels, n)

            parts_xyz.append(mat['xyz'][pix_i])
            parts_rgbs.append(mat['rgbs'][img_i, pix_i].astype(np.float32))
            parts_pids.append(mat['point_ids'][pix_i])
            parts_mids.append(np.full(n, mat['mat_id'], dtype=np.int64))
            parts_light.append(mat['light_pos'][img_i])
            parts_cam.append(mat['cam_pos'][img_i])

        xyz   = np.concatenate(parts_xyz)
        rgbs  = np.concatenate(parts_rgbs)
        light = np.concatenate(parts_light)
        cam   = np.concatenate(parts_cam)

        wi = light - xyz
        wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
        wo = cam - xyz
        wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)

        # Shuffle so rays from different materials are interleaved
        pids = np.concatenate(parts_pids)
        mids = np.concatenate(parts_mids)
        perm = np.random.permutation(n_rays)
        xyz, wi, wo, rgbs, pids, mids = \
            xyz[perm], wi[perm], wo[perm], rgbs[perm], pids[perm], mids[perm]

        return {
            'xyz':          torch.from_numpy(xyz).float(),
            'wi':           torch.from_numpy(wi).float(),
            'wo':           torch.from_numpy(wo).float(),
            'rgbs':         torch.from_numpy(rgbs).float(),
            'point_ids':    torch.from_numpy(pids).long(),
            'material_ids': torch.from_numpy(mids).long(),
            'confidence':   torch.ones(n_rays, dtype=torch.float32),
            'gt_params':    torch.zeros(1),
        }

    # ------------------------------------------------------------------
    def __iter__(self):
        # ----- training (infinite, all data in memory) -----
        if hasattr(self, '_all_data'):
            while True:
                chunk = self._all_data
                if chunk is None or chunk.total_obs == 0:
                    time.sleep(0.01)
                    continue

                yield self._sample_batch(chunk, self.rays_num)

        # ----- validation (finite, sequential) -----
        else:
            chunk = self._val_data
            total = chunk.total_obs
            n_batches = max(1, math.ceil(total / self.rays_num))

            for i in range(n_batches):
                n = min(self.rays_num, total - i * self.rays_num)
                yield self._sample_batch(chunk, n)


# ---------------------------------------------------------------------------
# BonnValDataset  (standard Dataset, one full image per __getitem__)
# ---------------------------------------------------------------------------

class BonnValDataset(Dataset):
    """Validation dataset for Bonn SVBRDF.

    Randomly picks one material, selects ``valid_num`` poly images.
    Each ``__getitem__`` returns **all pixels** (H*W) for one image,
    together with ``img_hw`` so the trainer can reconstruct the 2-D
    image for side-by-side visualisation.

    Return dict keys match the training batch (xyz, wi, wo, rgbs, …)
    so the same forward-model code can be reused.
    """

    def __init__(self, cfg, root_folder):
        self.cfg = cfg
        self.root_folder = Path(root_folder)
        self.valid_num = getattr(cfg.data, 'valid_num', 5)
        self.debug = getattr(cfg.data, 'debug', False)

        # ---- discover materials & pick one ---------------------
        mat_ids = self._discover_materials()
        if self.debug:
            self.mat_id = 1
        else:
            self.mat_id = random.choice(mat_ids)
        prefix = self.root_folder / f'mat{self.mat_id:04d}'

        print(f"\n{'='*60}")
        print(f"BonnValDataset  |  material=mat{self.mat_id:04d}  "
              f"valid_num={self.valid_num}")
        print(f"{'='*60}")

        # ---- load xyz map & calibration ---------------------------------
        self.xyz_map, self.H, self.W = \
            _read_xyz_map(f'{prefix}_xyz_rot000.exr')
        self.n_pixels = self.H * self.W

        calib = self._load_calibration(self.mat_id)

        xyz_flat = self.xyz_map.reshape(self.n_pixels, 3).astype(np.float32)
        pids = np.arange(self.n_pixels, dtype=np.int64)

        # ---- read poly data using pyexr (same as official Bonn code) -----
        poly_data, poly_ch_names, pH, pW = _read_exr(f'{prefix}_poly.exr')
        assert (pH, pW) == (self.H, self.W)
        poly_images = _parse_poly_channels(poly_ch_names)

        n_select = min(self.valid_num, len(poly_images))
        selected = random.sample(poly_images, n_select)
        print(f"Selected {n_select} poly images for validation "
              f"({self.n_pixels} pixels each)")

        # ---- preload every selected image --------------------------------
        self._items = []
        for eid, img in enumerate(selected):
            rot_data = calib[img['rotation']]

            cam_pos = rot_data[img['camera']]
            wo = cam_pos[None, :] - xyz_flat
            wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)

            led_pos = rot_data[img['led']]
            wi = led_pos[None, :] - xyz_flat
            wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)

            idx = img['ch_start']
            rgbs = poly_data[:, :, idx:idx+3].reshape(-1, 3)    # (H*W, 3)
            np.clip(rgbs, 0, None, out=rgbs)

            label = (f"mat{self.mat_id:04d}_{img['camera']}_"
                     f"{img['led']}_{img['rotation']}")

            self._items.append({
                'xyz':          torch.from_numpy(xyz_flat.copy()).float(),
                'wi':           torch.from_numpy(wi).float(),
                'wo':           torch.from_numpy(wo).float(),
                'rgbs':         torch.from_numpy(rgbs).float(),
                'point_ids':    torch.from_numpy(pids.copy()).long(),
                'material_ids': torch.full((self.n_pixels,), self.mat_id,
                                           dtype=torch.long),
                'emitter_ids':  torch.full((self.n_pixels,), eid,
                                           dtype=torch.long),
                'data_type':    torch.full((self.n_pixels,), DTYPE_POLY,
                                           dtype=torch.long),
                'lls_corners':  torch.zeros(self.n_pixels, 4, 3),
                'confidence':   torch.ones(self.n_pixels),
                'img_hw':       torch.tensor([self.H, self.W]),
                'gt_params':    torch.zeros(1),
                'label':        label,
            })
        del poly_data

        print(f"BonnValDataset ready  ({len(self._items)} images)\n"
              f"{'='*60}\n")

    # ------------------------------------------------------------------
    def _discover_materials(self):
        training_list_path = getattr(self.cfg.data, 'training_list_path', '')
        if training_list_path and os.path.isfile(training_list_path):
            ids = []
            with open(training_list_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ids.append(int(line))
            return sorted(ids)
        poly_files = sorted(self.root_folder.glob('mat*_poly.exr'))
        return sorted(int(p.stem.split('_')[0][3:]) for p in poly_files)

    def _load_calibration(self, mat_id):
        path = f'{self.root_folder / f"mat{mat_id:04d}"}_calibration.mat'
        raw = spio.loadmat(path)
        calib = {}
        for rot_key in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
            rd = raw[rot_key][0, 0]
            rot_dict = {}
            for field in rd.dtype.names:
                val = rd[field]
                if field == 'llsCorners':
                    rot_dict[field] = np.array(val, dtype=np.float32)
                else:
                    rot_dict[field] = np.array(val, dtype=np.float32).flatten()
                m = re.match(r'(il|cv)(\d+)', field)
                if m and len(m.group(2)) < 3:
                    padded = f'{m.group(1)}{int(m.group(2)):03d}'
                    rot_dict[padded] = rot_dict[field]
            calib[rot_key] = rot_dict
        calib['llsAnglesDegrees'] = raw['llsAnglesDegrees'].flatten() \
                                         .astype(np.float64)
        return calib

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


# ---------------------------------------------------------------------------
# Single-material full loader (no pixel subsampling)
# ---------------------------------------------------------------------------

def _load_single_material_full(root_folder, mat_id):
    """Load one Bonn material with ALL pixels (no subsampling).

    Same assembly logic as BonnDataset._load_single_material but keeps
    all H*W pixels and also returns H, W for image reconstruction.

    Returns dict or None on failure.
    """
    try:
        root_folder = Path(root_folder)
        prefix = root_folder / f'mat{mat_id:04d}'

        # ---- calibration ------------------------------------------------
        path = f'{prefix}_calibration.mat'
        raw = spio.loadmat(path)
        calib = {}
        for rot_key in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
            rd = raw[rot_key][0, 0]
            rot_dict = {}
            for field in rd.dtype.names:
                val = rd[field]
                if field == 'llsCorners':
                    rot_dict[field] = np.array(val, dtype=np.float32)
                else:
                    rot_dict[field] = np.array(val, dtype=np.float32).flatten()
                m = re.match(r'(il|cv)(\d+)', field)
                if m and len(m.group(2)) < 3:
                    padded = f'{m.group(1)}{int(m.group(2)):03d}'
                    rot_dict[padded] = rot_dict[field]
            calib[rot_key] = rot_dict
        calib['llsAnglesDegrees'] = raw['llsAnglesDegrees'].flatten().astype(np.float64)

        # ---- xyz map ----------------------------------------------------
        xyz_map, H, W = _read_xyz_map(f'{prefix}_xyz_rot000.exr')
        n_pixels = H * W
        xyz_pts = xyz_map.reshape(n_pixels, 3).astype(np.float32)
        pids = np.arange(n_pixels, dtype=np.int64)

        # ---- read EXR data ----------------------------------------------
        poly_data, poly_ch_names, pH, pW = _read_exr(f'{prefix}_poly.exr')
        assert (pH, pW) == (H, W)

        # ---- assembly (no pixel subsampling) ----------------------------
        rgbs_parts      = []
        light_parts     = []
        cam_parts       = []
        dtype_parts     = []
        eid_parts       = []
        lls_corner_parts = []
        eid = 0

        # Poly (RGB)
        poly_images = _parse_poly_channels(poly_ch_names)
        n_poly = len(poly_images)
        if n_poly > 0:
            poly_flat = poly_data.reshape(n_pixels, -1)
            poly_rgbs = poly_flat.reshape(n_pixels, n_poly, 3).transpose(1, 0, 2)
            poly_light = np.array([
                calib[im['rotation']][im['led']] for im in poly_images])
            poly_cam = np.array([
                calib[im['rotation']][im['camera']] for im in poly_images])

            rgbs_parts.append(poly_rgbs)
            light_parts.append(poly_light)
            cam_parts.append(poly_cam)
            dtype_parts.append(np.full(n_poly, DTYPE_POLY, dtype=np.int64))
            eid_parts.append(np.arange(eid, eid + n_poly, dtype=np.int64))
            lls_corner_parts.append(np.zeros((n_poly, 4, 3), dtype=np.float32))
            eid += n_poly
        del poly_data

        if not rgbs_parts:
            return None

        all_rgbs       = np.concatenate(rgbs_parts, axis=0)       # (K, V, 3)
        all_light_pos  = np.concatenate(light_parts, axis=0)      # (K, 3)
        all_cam_pos    = np.concatenate(cam_parts, axis=0)        # (K, 3)
        all_data_type  = np.concatenate(dtype_parts, axis=0)      # (K,)
        all_emitter_id = np.concatenate(eid_parts, axis=0)        # (K,)
        all_lls_corner = np.concatenate(lls_corner_parts, axis=0) # (K, 4, 3)

        np.clip(all_rgbs, 0, None, out=all_rgbs)
        n_images = all_rgbs.shape[0]

        mem_mb = all_rgbs.nbytes / 1e6
        print(f"  mat{mat_id:04d}: {n_pixels:,} pixels × {n_images} images "
              f"= {n_pixels * n_images:,} obs  (rgbs {mem_mb:.0f} MB)")

        return {
            'mat_id':      mat_id,
            'n_pixels':    n_pixels,
            'n_images':    n_images,
            'H': H, 'W': W,
            'xyz':         xyz_pts,           # (V, 3)
            'point_ids':   pids,              # (V,)
            'rgbs':        all_rgbs,          # (K, V, 3)
            'light_pos':   all_light_pos,     # (K, 3)
            'cam_pos':     all_cam_pos,       # (K, 3)
            'data_type':   all_data_type,     # (K,)
            'emitter_ids': all_emitter_id,    # (K,)
            'lls_corners': all_lls_corner,    # (K, 4, 3)
        }

    except Exception as exc:
        print(f"  [Warning] Failed to load mat{mat_id:04d}: {exc}")
        import traceback; traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# BonnSingleMaterialDataset  (IterableDataset, single-material stage-2)
# ---------------------------------------------------------------------------

class BonnSingleMaterialDataset(IterableDataset):
    """Single-material Bonn dataset for stage 2 overfitting.

    Loads ALL pixels for one material into memory (no subsampling,
    no double-buffer chunk switching).  Images are split 80/20 into
    train/val with a fixed seed.  Training yields infinite random
    (image, pixel) pair batches.

    Return dict keys match the stage-1 training batch exactly.
    """

    def __init__(self, cfg, root_folder, split='train'):
        self.cfg = cfg
        self.root_folder = Path(root_folder)
        self.split = split
        self.rays_num = cfg.data.rays_num
        self.mat_id = cfg.data.overfit_mat_id
        self.val_view_ratio = getattr(cfg.data, 'val_view_ratio', 0.2)
        self.val_seed = getattr(cfg.data, 'val_seed', 42)

        print(f"\n{'='*60}")
        print(f"BonnSingleMaterialDataset ({split})  |  mat{self.mat_id:04d}")
        print(f"{'='*60}")

        # Load full material (all pixels, all images)
        mat_data = _load_single_material_full(root_folder, self.mat_id)
        if mat_data is None:
            raise RuntimeError(f"Failed to load mat{self.mat_id:04d}")

        # Split images 80/20 with fixed seed
        n_images = mat_data['n_images']
        rng = np.random.RandomState(self.val_seed)
        perm = rng.permutation(n_images)
        n_val = max(1, int(n_images * self.val_view_ratio))
        val_indices = np.sort(perm[:n_val])
        train_indices = np.sort(perm[n_val:])

        if split == 'train':
            indices = train_indices
        else:
            indices = val_indices

        # Store the split's subset
        self.n_pixels = mat_data['n_pixels']
        self.n_images = len(indices)
        self.H = mat_data['H']
        self.W = mat_data['W']
        self.xyz        = mat_data['xyz']                     # (V, 3)
        self.point_ids  = mat_data['point_ids']               # (V,)
        self.rgbs       = mat_data['rgbs'][indices]           # (K', V, 3)
        self.light_pos  = mat_data['light_pos'][indices]      # (K', 3)
        self.cam_pos    = mat_data['cam_pos'][indices]        # (K', 3)
        self.data_type  = mat_data['data_type'][indices]      # (K',)
        self.emitter_ids = mat_data['emitter_ids'][indices]   # (K',)
        self.lls_corners = mat_data['lls_corners'][indices]   # (K', 4, 3)

        total_obs = self.n_pixels * self.n_images
        mem_mb = self.rgbs.nbytes / 1e6
        print(f"  {split}: {self.n_pixels:,} pixels × {self.n_images} images "
              f"= {total_obs:,} obs  (rgbs {mem_mb:.0f} MB)")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    def set_step(self, step: int):
        pass

    def __len__(self):
        return 1_000_000

    # ------------------------------------------------------------------
    def _sample_batch(self, n_rays):
        img_i = np.random.randint(0, self.n_images, n_rays)
        pix_i = np.random.randint(0, self.n_pixels, n_rays)

        xyz  = self.xyz[pix_i]
        rgbs = self.rgbs[img_i, pix_i]
        light = self.light_pos[img_i]
        cam   = self.cam_pos[img_i]

        wi = light - xyz
        wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
        wo = cam - xyz
        wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)

        return {
            'xyz':          torch.from_numpy(xyz).float(),
            'wi':           torch.from_numpy(wi).float(),
            'wo':           torch.from_numpy(wo).float(),
            'rgbs':         torch.from_numpy(rgbs).float(),
            'point_ids':    torch.from_numpy(self.point_ids[pix_i].copy()).long(),
            'material_ids': torch.full((n_rays,), self.mat_id, dtype=torch.long),
            'emitter_ids':  torch.from_numpy(self.emitter_ids[img_i].copy()).long(),
            'data_type':    torch.from_numpy(self.data_type[img_i].copy()).long(),
            'lls_corners':  torch.from_numpy(self.lls_corners[img_i].copy()).float(),
            'confidence':   torch.ones(n_rays, dtype=torch.float32),
            'gt_params':    torch.zeros(1),
        }

    # ------------------------------------------------------------------
    def __iter__(self):
        while True:
            yield self._sample_batch(self.rays_num)


# ---------------------------------------------------------------------------
# BonnSingleMaterialValDataset  (Dataset, single-material stage-2 val)
# ---------------------------------------------------------------------------

class BonnSingleMaterialValDataset(Dataset):
    """Validation dataset for single-material Bonn stage-2 overfitting.

    Uses the SAME material and the SAME fixed-seed split as
    ``BonnSingleMaterialDataset`` but takes the held-out 20 % of images.
    Each ``__getitem__`` returns **all pixels** (H*W) for one image
    together with ``img_hw`` so the trainer can reconstruct 2-D images.

    Return dict keys match the stage-1 validation batch exactly.
    """

    def __init__(self, cfg, root_folder):
        self.cfg = cfg
        self.root_folder = Path(root_folder)
        self.mat_id = cfg.data.overfit_mat_id
        self.val_view_ratio = getattr(cfg.data, 'val_view_ratio', 0.2)
        self.val_seed = getattr(cfg.data, 'val_seed', 42)
        self.valid_num = getattr(cfg.data, 'valid_num', -1)

        print(f"\n{'='*60}")
        print(f"BonnSingleMaterialValDataset  |  mat{self.mat_id:04d}")
        print(f"{'='*60}")

        # Load full material
        mat_data = _load_single_material_full(root_folder, self.mat_id)
        if mat_data is None:
            raise RuntimeError(f"Failed to load mat{self.mat_id:04d}")

        self.H = mat_data['H']
        self.W = mat_data['W']
        self.n_pixels = mat_data['n_pixels']

        # Reproduce the same split as training
        n_images = mat_data['n_images']
        rng = np.random.RandomState(self.val_seed)
        perm = rng.permutation(n_images)
        n_val = max(1, int(n_images * self.val_view_ratio))
        val_indices = np.sort(perm[:n_val])

        if self.valid_num > 0:
            val_indices = val_indices[:self.valid_num]

        print(f"  {len(val_indices)} val images  ({self.n_pixels:,} pixels each)")

        # Precompute one item per val image (all pixels)
        xyz_flat = mat_data['xyz']      # (V, 3)
        pids     = mat_data['point_ids']  # (V,)

        self._items = []
        for eid_local, k in enumerate(val_indices):
            light = mat_data['light_pos'][k]          # (3,)
            cam   = mat_data['cam_pos'][k]            # (3,)

            wi = light[None, :] - xyz_flat
            wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
            wo = cam[None, :] - xyz_flat
            wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)

            rgbs = mat_data['rgbs'][k]                # (V, 3)

            dtype_val = int(mat_data['data_type'][k])
            eid_val   = int(mat_data['emitter_ids'][k])

            label = f"mat{self.mat_id:04d}_img{k:03d}"

            self._items.append({
                'xyz':          torch.from_numpy(xyz_flat.copy()).float(),
                'wi':           torch.from_numpy(wi).float(),
                'wo':           torch.from_numpy(wo).float(),
                'rgbs':         torch.from_numpy(rgbs.copy()).float(),
                'point_ids':    torch.from_numpy(pids.copy()).long(),
                'material_ids': torch.full((self.n_pixels,), self.mat_id,
                                           dtype=torch.long),
                'emitter_ids':  torch.full((self.n_pixels,), eid_val,
                                           dtype=torch.long),
                'data_type':    torch.full((self.n_pixels,), dtype_val,
                                           dtype=torch.long),
                'lls_corners':  torch.from_numpy(
                    np.broadcast_to(mat_data['lls_corners'][k],
                                    (self.n_pixels, 4, 3)).copy()).float(),
                'confidence':   torch.ones(self.n_pixels),
                'img_hw':       torch.tensor([self.H, self.W]),
                'gt_params':    torch.zeros(1),
                'label':        label,
            })

        print(f"BonnSingleMaterialValDataset ready  ({len(self._items)} images)\n"
              f"{'='*60}\n")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]
