import torch
import numpy as np
import os
import re
import math
import scipy.io as spio
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
import threading, queue, time
from dataclasses import dataclass
import random
import pyexr

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


def _parse_pan_channels(channel_names):
    """Parse pan channel names.

    Channel format: 'pan_cv01_il001_rot000'
    Returns: list of dicts with keys camera, led, rotation, channel
    """
    pattern = re.compile(r'pan_(cv\d+)_(il\d+)_(rot\d+)')
    images = []
    for name in channel_names:
        m = pattern.match(name)
        if m:
            cam, led, rot = m.groups()
            images.append(dict(camera=cam, led=led, rotation=rot, channel=name))
    return images


def _parse_lls_channels(channel_names):
    """Parse LLS channel names.

    Channel format: 'lls_cv01_lls01_la22.00_rot045'
    Returns: list of dicts with keys camera, angle, rotation, channel
    """
    pattern = re.compile(r'lls_(cv\d+)_lls\d+_la([+-]?\d+\.?\d*)_(rot\d+)')
    images = []
    for name in channel_names:
        m = pattern.match(name)
        if m:
            cam, angle, rot = m.groups()
            images.append(dict(camera=cam, angle=float(angle), rotation=rot, channel=name))
    return images


# ---------------------------------------------------------------------------
# EXR reading helpers  (uses pyexr, matching the official Bonn reference code)
# ---------------------------------------------------------------------------

def _read_exr(filepath):
    """Read all channels from an EXR using pyexr.

    Returns (data (H,W,C) float32, channel_names list, H, W).
    """
    exr = pyexr.open(str(filepath))
    ch_names = exr.channel_map['all']
    data = exr.get(group='all', precision=pyexr.HALF).astype(np.float32)
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
    DTYPE_PAN = DTYPE_PAN
    DTYPE_LLS = DTYPE_LLS

    # ------------------------------------------------------------------
    @dataclass
    class ChunkData:
        """Per-material arrays loaded from EXR (no pixel×image expansion).

        Each entry in *materials* is a dict from ``_load_single_material``:
        mat_id, n_pixels, n_images, xyz (H*W,3), point_ids (H*W,),
        rgbs (K,V,3), light_pos (K,3), cam_pos (K,3), data_type (K,),
        emitter_ids (K,), lls_corners (K,4,3).
        """
        materials: list            # list of per-material dicts
        total_obs: int             # sum(n_pixels * n_images) across materials

    # ------------------------------------------------------------------
    class _DoubleBuffer:
        """Two RAM slots with a background thread that fills the inactive slot."""

        def __init__(self, build_fn):
            self.build_fn = build_fn
            self.slots = [None, None]
            self.ready = [threading.Event(), threading.Event()]
            self.active = 0
            self._stop = False
            self._q: queue.Queue = queue.Queue(maxsize=2)
            self._t = threading.Thread(target=self._worker, daemon=True)
            self._t.start()

        def _worker(self):
            while not self._stop:
                try:
                    slot_id = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                self.slots[slot_id] = self.build_fn()
                self.ready[slot_id].set()

        def request_fill(self, slot_id):
            self.ready[slot_id].clear()
            try:
                self._q.put_nowait(slot_id)
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                self._q.put_nowait(slot_id)

        def wait_initial(self):
            self.ready[self.active].wait()

        def try_swap(self):
            nxt = 1 - self.active
            if self.ready[nxt].is_set():
                old = self.active
                self.active = nxt
                self.ready[old].clear()
                return old
            return None

        def force_swap(self):
            nxt = 1 - self.active
            self.ready[nxt].wait()
            old = self.active
            self.active = nxt
            self.ready[old].clear()
            return old

        def current(self):
            self.ready[self.active].wait()
            return self.slots[self.active]

        def stop(self):
            self._stop = True
            self._t.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, cfg, root_folder, split='train'):
        self.cfg = cfg
        self.root_folder = Path(root_folder)
        self.split = split
        self.rays_num = cfg.data.rays_num

        self.use_pan = getattr(cfg.data, 'use_pan', False)
        self.use_lls = getattr(cfg.data, 'use_lls', False)
        self.debug = getattr(cfg.data, 'debug', False)
        self.debug_num = getattr(cfg.data, 'debug_num', 1)
        self.points_per_material = getattr(cfg.data, 'points_per_material', 2000)
        self.random_observations = getattr(cfg.data, 'random_observations', False)
        self.subsample_ratio = getattr(cfg.data, 'subsample_ratio', 1.0)
        self.switch_iters = getattr(cfg.data, 'switch_iters', 3000)
        self.chunk_size = getattr(cfg.data, 'chunk_size', 20)
        self.val_materials = getattr(cfg.data, 'val_materials', 2)
        self.val_points = getattr(cfg.data, 'val_points', 500)
        self.step = 0

        # Approximate luminance weights for pan→scalar projection
        self.pan_weights = np.array([0.34, 0.36, 0.28], dtype=np.float32)

        # Discover materials
        self.mat_ids = self._discover_materials()
        if self.debug:
            self.mat_ids = self.mat_ids[:self.debug_num]
            print(f"[DEBUG] Using single material: mat{self.mat_ids[0]:04d}")

        print(f"\n{'='*60}")
        print(f"BonnDataset ({split})  |  materials={len(self.mat_ids)}  "
              f"use_pan={self.use_pan}  use_lls={self.use_lls}")
        print(f"{'='*60}")

        # Load lightweight calibration for every material
        self.calibrations = {}
        for mid in tqdm(self.mat_ids, desc="Loading calibrations"):
            self.calibrations[mid] = self._load_calibration(mid)

        # Initialise for split
        if split == 'train':
            print(f"Initialising double buffer (reload every {self.switch_iters} iters) ...")
            self._dbuf = BonnDataset._DoubleBuffer(self._load_chunk)
            self._dbuf.request_fill(0)
            self._dbuf.request_fill(1)
            self._dbuf.wait_initial()
            print("Double buffer ready!")
        else:
            print("Loading validation subset …")
            self._val_data = self._load_chunk(val_mode=True)
            print(f"Validation: {self._val_data.xyz.shape[0]:,} observations")

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

            pan_data, pan_ch_names = None, None
            if self.use_pan:
                pan_data, pan_ch_names, panH, panW = _read_exr(f'{prefix}_pan.exr')
                assert (panH, panW) == (H, W)

            lls_data, lls_ch_names = None, None
            lls_angle_to_idx = {}
            if self.use_lls:
                lls_angles = calib['llsAnglesDegrees']
                lls_angle_to_idx = {float(a): i for i, a in enumerate(lls_angles)}
                lls_data, lls_ch_names, lH, lW = _read_exr(f'{prefix}_lls.exr')
                assert (lH, lW) == (H, W)

            # ============================================================
            # Assembly
            # ============================================================
            rgbs_parts      = []
            light_parts     = []
            cam_parts       = []
            dtype_parts     = []
            eid_parts       = []
            lls_corner_parts = []
            eid = 0

            # --- Poly (RGB) ---------------------------------------------
            # pyexr gives (H, W, K*3) with every 3 channels = one image
            # in alphabetical order (_B, _G, _R) = visual (R, G, B)
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
                dtype_parts.append(np.full(n_poly, DTYPE_POLY, dtype=np.int64))
                eid_parts.append(np.arange(eid, eid + n_poly, dtype=np.int64))
                lls_corner_parts.append(np.zeros((n_poly, 4, 3), dtype=np.float32))
                eid += n_poly
            del poly_data

            # --- Pan (grayscale, il001-il024 only) -----------------------
            if self.use_pan and pan_data is not None:
                pan_name_to_idx = {n: i for i, n in enumerate(pan_ch_names)}
                pan_images = [im for im in _parse_pan_channels(pan_ch_names)
                              if int(im['led'][2:]) <= 24]
                n_pan = len(pan_images)
                if n_pan > 0:
                    pan_flat = pan_data.reshape(n_pixels, -1)       # (V, C)
                    ch_ci = np.array([pan_name_to_idx[im['channel']]
                                      for im in pan_images])
                    pan_gray = pan_flat[:, ch_ci].T                  # (K, V)
                    pan_rgbs = np.stack([pan_gray, pan_gray, pan_gray], axis=-1)

                    pan_light = np.array([
                        calib[im['rotation']][im['led']] for im in pan_images])
                    pan_cam = np.array([
                        calib[im['rotation']][im['camera']] for im in pan_images])
                    rgbs_parts.append(pan_rgbs)
                    light_parts.append(pan_light)
                    cam_parts.append(pan_cam)
                    dtype_parts.append(np.full(n_pan, DTYPE_PAN, dtype=np.int64))
                    eid_parts.append(np.arange(eid, eid + n_pan, dtype=np.int64))
                    lls_corner_parts.append(np.zeros((n_pan, 4, 3), dtype=np.float32))
                    eid += n_pan
                del pan_data

            # --- LLS (grayscale) -----------------------------------------
            if self.use_lls and lls_data is not None:
                lls_name_to_idx = {n: i for i, n in enumerate(lls_ch_names)}
                lls_images = _parse_lls_channels(lls_ch_names)
                n_lls = len(lls_images)
                if n_lls > 0:
                    lls_flat = lls_data.reshape(n_pixels, -1)       # (V, C)
                    ch_ci = np.array([lls_name_to_idx[im['channel']]
                                      for im in lls_images])
                    lls_gray = lls_flat[:, ch_ci].T                  # (K, V)
                    lls_rgbs = np.stack([lls_gray, lls_gray, lls_gray], axis=-1)

                    corners_list = []
                    center_list = []
                    for im in lls_images:
                        rd = calib[im['rotation']]
                        ai = lls_angle_to_idx[im['angle']]
                        c = rd['llsCorners'][:, :, ai].T             # (4, 3)
                        corners_list.append(c)
                        center_list.append(c.mean(axis=0))

                    lls_light = np.array(center_list, dtype=np.float32)
                    lls_cam = np.array([
                        calib[im['rotation']][im['camera']] for im in lls_images])
                    lls_corners_arr = np.array(corners_list, dtype=np.float32)

                    rgbs_parts.append(lls_rgbs)
                    light_parts.append(lls_light)
                    cam_parts.append(lls_cam)
                    dtype_parts.append(np.full(n_lls, DTYPE_LLS, dtype=np.int64))
                    eid_parts.append(np.arange(eid, eid + n_lls, dtype=np.int64))
                    lls_corner_parts.append(lls_corners_arr)
                    eid += n_lls
                del lls_data

            # ---- concatenate across source types ------------------------
            if not rgbs_parts:
                return None

            all_rgbs       = np.concatenate(rgbs_parts, axis=0)
            all_light_pos  = np.concatenate(light_parts, axis=0)
            all_cam_pos    = np.concatenate(cam_parts, axis=0)
            all_data_type  = np.concatenate(dtype_parts, axis=0)
            all_emitter_id = np.concatenate(eid_parts, axis=0)
            all_lls_corner = np.concatenate(lls_corner_parts, axis=0)

            np.clip(all_rgbs, 0, None, out=all_rgbs)
            n_images = all_rgbs.shape[0]

            # ----------------------------------------------------------
            # Per-ray flattening (random_observations mode)
            # ----------------------------------------------------------
            if self.random_observations:
                K, V = n_images, n_pixels
                total_rays = K * V

                # rgbs: (K, V, 3) -> (K*V, 3)
                flat_rgbs = all_rgbs.reshape(total_rays, 3)

                # xyz: (V, 3) -> tile K times -> (K*V, 3)
                flat_xyz = np.tile(xyz_pts, (K, 1))

                # point_ids: (V,) -> tile K times -> (K*V,)
                flat_pids = np.tile(pids, K)

                # per-image arrays: repeat each entry V times
                flat_light_pos  = np.repeat(all_light_pos, V, axis=0)    # (K*V, 3)
                flat_cam_pos    = np.repeat(all_cam_pos, V, axis=0)      # (K*V, 3)
                flat_data_type  = np.repeat(all_data_type, V)            # (K*V,)
                flat_emitter_id = np.repeat(all_emitter_id, V)           # (K*V,)
                flat_lls_corner = np.repeat(all_lls_corner, V, axis=0)   # (K*V, 4, 3)

                # subsample rays by ratio
                n_rays_kept = max(1, int(total_rays * self.subsample_ratio))
                if n_rays_kept < total_rays:
                    keep = np.sort(np.random.choice(
                        total_rays, n_rays_kept, replace=False))
                    flat_rgbs       = flat_rgbs[keep]
                    flat_xyz        = flat_xyz[keep]
                    flat_pids       = flat_pids[keep]
                    flat_light_pos  = flat_light_pos[keep]
                    flat_cam_pos    = flat_cam_pos[keep]
                    flat_data_type  = flat_data_type[keep]
                    flat_emitter_id = flat_emitter_id[keep]
                    flat_lls_corner = flat_lls_corner[keep]

                mem_mb = flat_rgbs.nbytes / 1e6
                print(f"  mat{mat_id:04d}: {n_rays_kept:,}/{total_rays:,} rays "
                      f"(from {n_pixels:,} px × {n_images} img)  "
                      f"(rgbs {mem_mb:.0f} MB)")

                return {
                    'mat_id':      mat_id,
                    'n_rays':      n_rays_kept,
                    'flat':        True,
                    'xyz':         flat_xyz,           # (M, 3)
                    'point_ids':   flat_pids,          # (M,)
                    'rgbs':        flat_rgbs,          # (M, 3)
                    'light_pos':   flat_light_pos,     # (M, 3)
                    'cam_pos':     flat_cam_pos,       # (M, 3)
                    'data_type':   flat_data_type,     # (M,)
                    'emitter_ids': flat_emitter_id,    # (M,)
                    'lls_corners': flat_lls_corner,    # (M, 4, 3)
                }

            # ----------------------------------------------------------
            # Per-pixel mode: subsample pixels, keep all K images
            # ----------------------------------------------------------
            n_full = n_pixels
            n_keep = max(1, int(n_pixels * self.subsample_ratio))
            if n_keep < n_pixels:
                keep_idx = np.sort(np.random.choice(n_pixels, n_keep, replace=False))
                xyz_pts  = xyz_pts[keep_idx]
                pids     = pids[keep_idx]
                all_rgbs = all_rgbs[:, keep_idx, :]       # (K, V', 3)
                n_pixels = n_keep

            mem_mb = all_rgbs.nbytes / 1e6
            print(f"  mat{mat_id:04d}: {n_pixels:,}/{n_full:,} pixels "
                  f"({self.subsample_ratio:.0%}) × {n_images} images "
                  f"= {n_pixels * n_images:,} obs  (rgbs {mem_mb:.0f} MB)")

            return {
                'mat_id':      mat_id,
                'n_pixels':    n_pixels,
                'n_images':    n_images,
                'xyz':         xyz_pts,           # (V', 3)  float32
                'point_ids':   pids,              # (V',)    int64
                'rgbs':        all_rgbs,          # (K, V', 3) float32
                'light_pos':   all_light_pos,     # (K, 3)  float32
                'cam_pos':     all_cam_pos,       # (K, 3)  float32
                'data_type':   all_data_type,     # (K,)    int64
                'emitter_ids': all_emitter_id,    # (K,)    int64
                'lls_corners': all_lls_corner,    # (K, 4, 3) float32
            }

        except Exception as exc:
            print(f"  [Warning] Failed to load mat{mat_id:04d}: {exc}")
            import traceback; traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Chunk builder (called from background thread or main thread)
    # ------------------------------------------------------------------
    def _load_chunk(self, val_mode=False):
        if val_mode:
            selected = self.mat_ids[:min(self.val_materials, len(self.mat_ids))]
        elif self.debug:
            selected = self.mat_ids[:self.debug_num]
        else:
            n = min(self.chunk_size, len(self.mat_ids))
            selected = random.sample(self.mat_ids, n)

        materials = []
        tag = 'val' if val_mode else 'train'
        for mid in tqdm(selected, desc=f"Loading {tag} chunk"):
            result = self._load_single_material(mid)
            if result is not None:
                materials.append(result)

        if not materials:
            return BonnDataset.ChunkData(materials=[], total_obs=0)

        is_flat = materials[0].get('flat', False)
        if is_flat:
            total_obs = sum(m['n_rays'] for m in materials)
        else:
            total_obs = sum(m['n_pixels'] * m['n_images'] for m in materials)
        total_rgb_mb = sum(m['rgbs'].nbytes for m in materials) / 1e6

        print(f"[{tag}] Chunk built: {total_obs:,} observations "
              f"from {len(materials)} materials  "
              f"(rgbs {total_rgb_mb:.0f} MB)")

        return BonnDataset.ChunkData(materials=materials, total_obs=total_obs)

    # ------------------------------------------------------------------
    # Training-loop interface
    # ------------------------------------------------------------------
    def set_step(self, step: int):
        self.step = step
        if hasattr(self, '_dbuf') and step > 0 and step % self.switch_iters == 0:
            old_slot = self._dbuf.force_swap()
            print(f"[Step {step}] Forced switch to slot {self._dbuf.active}")
            self._dbuf.request_fill(old_slot)

    def __len__(self):
        if hasattr(self, '_val_data'):
            return max(1, math.ceil(self._val_data.total_obs / self.rays_num))
        return 1_000_000

    # ------------------------------------------------------------------
    def _sample_batch(self, chunk, n_rays):
        """Randomly sample n_rays from chunk.

        Two modes depending on the chunk format:
        - flat (random_observations): each material stores (M, ...) rays;
          draw random indices from the flat ray pool.
        - per-pixel (original): each material stores (K, V, 3) rgbs;
          draw random (img, pixel) pairs.
        """
        materials = chunk.materials
        is_flat = materials[0].get('flat', False)

        if is_flat:
            return self._sample_batch_flat(materials, n_rays)
        return self._sample_batch_perpixel(materials, n_rays)

    def _sample_batch_flat(self, materials, n_rays):
        """Sample from pre-flattened (M, ...) per-material ray pools."""
        obs_counts = np.array(
            [m['n_rays'] for m in materials], dtype=np.float64)
        weights = obs_counts / obs_counts.sum()
        rays_per_mat = np.round(weights * n_rays).astype(int)
        rays_per_mat[-1] = n_rays - rays_per_mat[:-1].sum()

        parts_xyz   = []
        parts_rgbs  = []
        parts_pids  = []
        parts_mids  = []
        parts_dtype = []
        parts_eids  = []
        parts_lls   = []
        parts_light = []
        parts_cam   = []

        for mi, mat in enumerate(materials):
            n = int(rays_per_mat[mi])
            if n <= 0:
                continue
            idx = np.random.randint(0, mat['n_rays'], n)

            parts_xyz.append(mat['xyz'][idx])
            parts_rgbs.append(mat['rgbs'][idx])
            parts_pids.append(mat['point_ids'][idx])
            parts_mids.append(np.full(n, mat['mat_id'], dtype=np.int64))
            parts_dtype.append(mat['data_type'][idx])
            parts_eids.append(mat['emitter_ids'][idx])
            parts_lls.append(mat['lls_corners'][idx])
            parts_light.append(mat['light_pos'][idx])
            parts_cam.append(mat['cam_pos'][idx])

        xyz   = np.concatenate(parts_xyz)
        rgbs  = np.concatenate(parts_rgbs)
        light = np.concatenate(parts_light)
        cam   = np.concatenate(parts_cam)

        wi = light - xyz
        wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
        wo = cam - xyz
        wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)

        return {
            'xyz':          torch.from_numpy(xyz).float(),
            'wi':           torch.from_numpy(wi).float(),
            'wo':           torch.from_numpy(wo).float(),
            'rgbs':         torch.from_numpy(rgbs).float(),
            'point_ids':    torch.from_numpy(np.concatenate(parts_pids)).long(),
            'material_ids': torch.from_numpy(np.concatenate(parts_mids)).long(),
            'emitter_ids':  torch.from_numpy(np.concatenate(parts_eids)).long(),
            'data_type':    torch.from_numpy(np.concatenate(parts_dtype)).long(),
            'lls_corners':  torch.from_numpy(np.concatenate(parts_lls)).float(),
            'confidence':   torch.ones(n_rays, dtype=torch.float32),
            'gt_params':    torch.zeros(1),
        }

    def _sample_batch_perpixel(self, materials, n_rays):
        """Sample by drawing random (img, pixel) pairs per material."""
        obs_counts = np.array(
            [m['n_pixels'] * m['n_images'] for m in materials], dtype=np.float64)
        weights = obs_counts / obs_counts.sum()
        rays_per_mat = np.round(weights * n_rays).astype(int)
        rays_per_mat[-1] = n_rays - rays_per_mat[:-1].sum()

        parts_xyz   = []
        parts_rgbs  = []
        parts_pids  = []
        parts_mids  = []
        parts_dtype = []
        parts_eids  = []
        parts_lls   = []
        parts_light = []
        parts_cam   = []

        for mi, mat in enumerate(materials):
            n = int(rays_per_mat[mi])
            if n <= 0:
                continue
            img_i = np.random.randint(0, mat['n_images'], n)
            pix_i = np.random.randint(0, mat['n_pixels'], n)

            parts_xyz.append(mat['xyz'][pix_i])
            parts_rgbs.append(mat['rgbs'][img_i, pix_i])
            parts_pids.append(mat['point_ids'][pix_i])
            parts_mids.append(np.full(n, mat['mat_id'], dtype=np.int64))
            parts_dtype.append(mat['data_type'][img_i])
            parts_eids.append(mat['emitter_ids'][img_i])
            parts_lls.append(mat['lls_corners'][img_i])
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

        return {
            'xyz':          torch.from_numpy(xyz).float(),
            'wi':           torch.from_numpy(wi).float(),
            'wo':           torch.from_numpy(wo).float(),
            'rgbs':         torch.from_numpy(rgbs).float(),
            'point_ids':    torch.from_numpy(np.concatenate(parts_pids)).long(),
            'material_ids': torch.from_numpy(np.concatenate(parts_mids)).long(),
            'emitter_ids':  torch.from_numpy(np.concatenate(parts_eids)).long(),
            'data_type':    torch.from_numpy(np.concatenate(parts_dtype)).long(),
            'lls_corners':  torch.from_numpy(np.concatenate(parts_lls)).float(),
            'confidence':   torch.ones(n_rays, dtype=torch.float32),
            'gt_params':    torch.zeros(1),
        }

    # ------------------------------------------------------------------
    def __iter__(self):
        # ----- training (infinite, double-buffered) -----
        if hasattr(self, '_dbuf'):
            while True:
                chunk = self._dbuf.current()
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
        self.use_pan = getattr(cfg.data, 'use_pan', False)
        self.use_lls = getattr(cfg.data, 'use_lls', False)
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
