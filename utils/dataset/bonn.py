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

DTYPE_POLY = 0
DTYPE_PAN = 1
DTYPE_LLS = 2


# ---------------------------------------------------------------------------
# Channel-name parsers
# ---------------------------------------------------------------------------

def _parse_poly_channels(channel_names):
    """Group poly channel names into per-image RGB descriptors.

    Channel format: 'poly_cv01_il026_rot000_R'
    Returns: list of dicts with keys camera, led, rotation, ch_r, ch_g, ch_b
    """
    pattern = re.compile(r'poly_(cv\d+)_(il\d+)_(rot\d+)_([RGB])')
    groups = {}
    for name in channel_names:
        m = pattern.match(name)
        if m:
            cam, led, rot, color = m.groups()
            key = (cam, led, rot)
            if key not in groups:
                groups[key] = {}
            groups[key][color] = name

    images = []
    for (cam, led, rot), colors in sorted(groups.items()):
        if len(colors) == 3:
            images.append(dict(camera=cam, led=led, rotation=rot,
                               ch_r=colors['R'], ch_g=colors['G'], ch_b=colors['B']))
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
# EXR reading helpers  (requires OpenEXR + Imath)
# ---------------------------------------------------------------------------

def _open_exr(filepath):
    """Open an EXR file and return (InputFile, H, W, sorted_channel_names)."""
    import OpenEXR
    f = OpenEXR.InputFile(str(filepath))
    hdr = f.header()
    dw = hdr['dataWindow']
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    ch_names = sorted(hdr['channels'].keys())
    return f, H, W, ch_names


def _read_channel(exr_file, ch_name, H, W, precision='half'):
    """Read a single channel from an already-opened EXR file."""
    import Imath
    if precision == 'half':
        pt = Imath.PixelType(Imath.PixelType.HALF)
        dtype = np.float16
    else:
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        dtype = np.float32
    raw = exr_file.channel(ch_name, pt)
    return np.frombuffer(raw, dtype=dtype).reshape(H, W)


def _read_xyz_map(filepath):
    """Read xyz_rot000.exr → (xyz (H,W,3) float32, valid_mask (H,W) bool, H, W)."""
    import Imath
    f, H, W, ch_names = _open_exr(filepath)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = []
    for name in ch_names:
        raw = f.channel(name, pt)
        channels.append(np.frombuffer(raw, dtype=np.float32).reshape(H, W))
    xyz = np.stack(channels, axis=-1)  # (H, W, 3)
    valid_mask = ~np.all(xyz < -0.5, axis=2)
    return xyz, valid_mask, H, W


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
        xyz: torch.Tensor           # (N, 3)  surface position in mm
        wi: torch.Tensor            # (N, 3)  light dir, world space, normalised
        wo: torch.Tensor            # (N, 3)  view  dir, world space, normalised
        rgbs: torch.Tensor          # (N, 3)  calibrated measurement
        point_ids: torch.Tensor     # (N,)    row*W + col  (local per material)
        material_ids: torch.Tensor  # (N,)    material index
        emitter_ids: torch.Tensor   # (N,)    sequential emitter index
        data_type: torch.Tensor     # (N,)    0/1/2
        lls_corners: torch.Tensor   # (N, 4, 3)  quad corners (valid when data_type==2)
        confidence: torch.Tensor    # (N,)    loss weight

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
                return True
            return False

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
        self.points_per_material = getattr(cfg.data, 'points_per_material', 2000)
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
            self.mat_ids = self.mat_ids[:1]
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
            calib[rot_key] = rot_dict
        calib['llsAnglesDegrees'] = raw['llsAnglesDegrees'].flatten().astype(np.float64)
        return calib

    # ------------------------------------------------------------------
    # Single-material loader
    # ------------------------------------------------------------------
    def _load_single_material(self, mat_id, num_points):
        """Load one Bonn material, sample *num_points* pixels, build all
        (pixel × image) observations.  Returns dict of tensors or None."""
        try:
            prefix = self._mat_prefix(mat_id)
            calib = self.calibrations[mat_id]

            # ---- xyz map ------------------------------------------------
            xyz_map, valid_mask, H, W = _read_xyz_map(f'{prefix}_xyz_rot000.exr')
            valid_rows, valid_cols = np.where(valid_mask)
            n_valid = len(valid_rows)
            if n_valid == 0:
                return None

            n_sample = min(num_points, n_valid)
            chosen = np.random.choice(n_valid, n_sample, replace=False)
            srows = valid_rows[chosen]
            scols = valid_cols[chosen]
            xyz_pts = xyz_map[srows, scols]                     # (P, 3)
            pids = (srows * W + scols).astype(np.int64)         # (P,)

            # ---- collect image descriptors ------------------------------
            images = []   # list of dicts describing each measurement image
            eid = 0       # running emitter_id counter

            # Poly
            poly_exr, pH, pW, poly_ch_names = _open_exr(f'{prefix}_poly.exr')
            assert (pH, pW) == (H, W)
            for img in _parse_poly_channels(poly_ch_names):
                img['type'] = DTYPE_POLY
                img['emitter_id'] = eid; eid += 1
                images.append(img)

            # Pan (optional, unfiltered LEDs only: il001-il024)
            pan_exr = None
            if self.use_pan:
                pan_exr, panH, panW, pan_ch_names = _open_exr(f'{prefix}_pan.exr')
                assert (panH, panW) == (H, W)
                for img in _parse_pan_channels(pan_ch_names):
                    led_num = int(img['led'][2:])
                    if led_num > 24:
                        continue
                    img['type'] = DTYPE_PAN
                    img['emitter_id'] = eid; eid += 1
                    images.append(img)

            # LLS (optional)
            lls_exr = None
            lls_angle_to_idx = {}
            if self.use_lls:
                lls_angles = calib['llsAnglesDegrees']
                lls_angle_to_idx = {float(a): i for i, a in enumerate(lls_angles)}
                lls_exr, lH, lW, lls_ch_names = _open_exr(f'{prefix}_lls.exr')
                assert (lH, lW) == (H, W)
                for img in _parse_lls_channels(lls_ch_names):
                    img['type'] = DTYPE_LLS
                    img['emitter_id'] = eid; eid += 1
                    images.append(img)

            n_images = len(images)
            n_obs = n_sample * n_images

            # ---- pre-allocate outputs -----------------------------------
            out_xyz        = np.tile(xyz_pts, (n_images, 1))               # (N, 3)
            out_wi         = np.zeros((n_obs, 3), dtype=np.float32)
            out_wo         = np.zeros((n_obs, 3), dtype=np.float32)
            out_rgbs       = np.zeros((n_obs, 3), dtype=np.float32)
            out_pids       = np.tile(pids, n_images)                       # (N,)
            out_eid        = np.zeros(n_obs, dtype=np.int64)
            out_dtype      = np.zeros(n_obs, dtype=np.int64)
            out_lls_corner = np.zeros((n_obs, 4, 3), dtype=np.float32)
            out_conf       = np.ones(n_obs, dtype=np.float32)

            # ---- per-image processing -----------------------------------
            for img_idx, img in enumerate(images):
                s = img_idx * n_sample
                e = s + n_sample
                rot_data = calib[img['rotation']]

                # View direction  wo = normalize(cam_pos - xyz)
                cam_pos = rot_data[img['camera']]               # (3,)
                wo = cam_pos[None, :] - xyz_pts                 # (P, 3)
                wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)
                out_wo[s:e] = wo

                out_eid[s:e] = img['emitter_id']
                out_dtype[s:e] = img['type']

                if img['type'] == DTYPE_POLY:
                    led_pos = rot_data[img['led']]
                    wi = led_pos[None, :] - xyz_pts
                    wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
                    out_wi[s:e] = wi
                    r = _read_channel(poly_exr, img['ch_r'], H, W).astype(np.float32)
                    g = _read_channel(poly_exr, img['ch_g'], H, W).astype(np.float32)
                    b = _read_channel(poly_exr, img['ch_b'], H, W).astype(np.float32)
                    out_rgbs[s:e, 0] = r[srows, scols]
                    out_rgbs[s:e, 1] = g[srows, scols]
                    out_rgbs[s:e, 2] = b[srows, scols]

                elif img['type'] == DTYPE_PAN:
                    led_pos = rot_data[img['led']]
                    wi = led_pos[None, :] - xyz_pts
                    wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
                    out_wi[s:e] = wi
                    gray = _read_channel(pan_exr, img['channel'], H, W).astype(np.float32)
                    g = gray[srows, scols]
                    out_rgbs[s:e, 0] = g
                    out_rgbs[s:e, 1] = g
                    out_rgbs[s:e, 2] = g

                elif img['type'] == DTYPE_LLS:
                    angle_idx = lls_angle_to_idx[img['angle']]
                    corners = rot_data['llsCorners'][:, :, angle_idx].T   # (4, 3)
                    lls_center = corners.mean(axis=0)                     # (3,)
                    wi = lls_center[None, :] - xyz_pts
                    wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
                    out_wi[s:e] = wi
                    gray = _read_channel(lls_exr, img['channel'], H, W).astype(np.float32)
                    g = gray[srows, scols]
                    out_rgbs[s:e, 0] = g
                    out_rgbs[s:e, 1] = g
                    out_rgbs[s:e, 2] = g
                    out_lls_corner[s:e] = corners[None, :, :]

            # ---- filter negatives (rare, but safety) --------------------
            keep = np.all(out_rgbs >= -0.5, axis=1)

            return {
                'xyz':          torch.from_numpy(out_xyz[keep]).float(),
                'wi':           torch.from_numpy(out_wi[keep]).float(),
                'wo':           torch.from_numpy(out_wo[keep]).float(),
                'rgbs':         torch.from_numpy(np.clip(out_rgbs[keep], 0, None)).float(),
                'point_ids':    torch.from_numpy(out_pids[keep]).long(),
                'material_ids': torch.full((int(keep.sum()),), mat_id, dtype=torch.long),
                'emitter_ids':  torch.from_numpy(out_eid[keep]).long(),
                'data_type':    torch.from_numpy(out_dtype[keep]).long(),
                'lls_corners':  torch.from_numpy(out_lls_corner[keep]).float(),
                'confidence':   torch.from_numpy(out_conf[keep]).float(),
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
            pts = self.val_points
        elif self.debug:
            selected = self.mat_ids[:1]
            pts = self.points_per_material
        else:
            n = min(self.chunk_size, len(self.mat_ids))
            selected = random.sample(self.mat_ids, n)
            pts = self.points_per_material

        keys = ['xyz', 'wi', 'wo', 'rgbs', 'point_ids',
                'material_ids', 'emitter_ids', 'data_type',
                'lls_corners', 'confidence']
        accum = {k: [] for k in keys}

        tag = 'val' if val_mode else 'train'
        for mid in tqdm(selected, desc=f"Loading {tag} chunk"):
            result = self._load_single_material(mid, pts)
            if result is not None:
                for k in keys:
                    accum[k].append(result[k])

        if all(len(v) > 0 for v in accum.values()):
            merged = {k: torch.cat(v, 0) for k, v in accum.items()}
        else:
            merged = {k: torch.zeros(0) for k in keys}

        n_obs = merged['xyz'].shape[0] if merged['xyz'].dim() > 0 else 0
        print(f"[{tag}] Chunk built: {n_obs:,} observations "
              f"from {len(selected)} materials")
        return BonnDataset.ChunkData(**merged)

    # ------------------------------------------------------------------
    # Training-loop interface
    # ------------------------------------------------------------------
    def set_step(self, step: int):
        self.step = step
        if hasattr(self, '_dbuf') and step > 0 and step % self.switch_iters == 0:
            next_slot = 1 - self._dbuf.active
            print(f"[Step {step}] Requesting new chunk → slot {next_slot}")
            self._dbuf.request_fill(next_slot)

    def __len__(self):
        if hasattr(self, '_val_data'):
            n = self._val_data.xyz.shape[0]
            return max(1, math.ceil(n / self.rays_num))
        return 1_000_000

    def __iter__(self):
        # ----- training (infinite, double-buffered) -----
        if hasattr(self, '_dbuf'):
            while True:
                if self._dbuf.try_swap():
                    print(f"[Step {self.step}] Switched to new chunk "
                          f"(slot {self._dbuf.active})")

                chunk = self._dbuf.current()
                if chunk is None or chunk.xyz.numel() == 0:
                    time.sleep(0.01)
                    continue

                total = chunk.xyz.shape[0]
                idx = torch.randint(0, total, (self.rays_num,))

                yield {
                    'xyz':          chunk.xyz[idx],
                    'wi':           chunk.wi[idx],
                    'wo':           chunk.wo[idx],
                    'rgbs':         chunk.rgbs[idx],
                    'point_ids':    chunk.point_ids[idx],
                    'material_ids': chunk.material_ids[idx],
                    'emitter_ids':  chunk.emitter_ids[idx],
                    'data_type':    chunk.data_type[idx],
                    'lls_corners':  chunk.lls_corners[idx],
                    'confidence':   chunk.confidence[idx],
                    'gt_params':    torch.zeros(1),
                }

        # ----- validation (finite, sequential) -----
        else:
            data = self._val_data
            total = data.xyz.shape[0]
            n_batches = max(1, math.ceil(total / self.rays_num))

            for i in range(n_batches):
                s = i * self.rays_num
                e = min(s + self.rays_num, total)
                yield {
                    'xyz':          data.xyz[s:e],
                    'wi':           data.wi[s:e],
                    'wo':           data.wo[s:e],
                    'rgbs':         data.rgbs[s:e],
                    'point_ids':    data.point_ids[s:e],
                    'material_ids': data.material_ids[s:e],
                    'emitter_ids':  data.emitter_ids[s:e],
                    'data_type':    data.data_type[s:e],
                    'lls_corners':  data.lls_corners[s:e],
                    'confidence':   data.confidence[s:e],
                    'gt_params':    torch.zeros(1),
                }


# ---------------------------------------------------------------------------
# BonnValDataset  (standard Dataset, one full image per __getitem__)
# ---------------------------------------------------------------------------

class BonnValDataset(Dataset):
    """Validation dataset for Bonn SVBRDF.

    Randomly picks one material, selects ``valid_num`` poly images.
    Each ``__getitem__`` returns **all valid pixels** for one image,
    together with ``valid_mask`` and ``img_hw`` so the trainer can
    reconstruct the 2-D image for side-by-side visualisation.

    Return dict keys match the training batch (xyz, wi, wo, rgbs, …)
    so the same forward-model code can be reused.
    """

    def __init__(self, cfg, root_folder):
        self.cfg = cfg
        self.root_folder = Path(root_folder)
        self.valid_num = getattr(cfg.data, 'valid_num', 5)
        self.use_pan = getattr(cfg.data, 'use_pan', False)
        self.use_lls = getattr(cfg.data, 'use_lls', False)

        # ---- discover materials & pick one randomly ---------------------
        mat_ids = self._discover_materials()
        self.mat_id = random.choice(mat_ids)
        prefix = self.root_folder / f'mat{self.mat_id:04d}'

        print(f"\n{'='*60}")
        print(f"BonnValDataset  |  material=mat{self.mat_id:04d}  "
              f"valid_num={self.valid_num}")
        print(f"{'='*60}")

        # ---- load xyz map & calibration ---------------------------------
        self.xyz_map, self.valid_mask_2d, self.H, self.W = \
            _read_xyz_map(f'{prefix}_xyz_rot000.exr')
        self.valid_mask_flat = self.valid_mask_2d.reshape(-1)   # (H*W,)
        self.n_valid = int(self.valid_mask_flat.sum())

        calib = self._load_calibration(self.mat_id)

        # valid pixel positions  (n_valid, 3)
        vrows, vcols = np.where(self.valid_mask_2d)
        xyz_valid = self.xyz_map[vrows, vcols]                  # (V, 3)
        pids_valid = (vrows * self.W + vcols).astype(np.int64)  # (V,)

        # ---- parse poly images ------------------------------------------
        poly_exr, pH, pW, poly_ch_names = _open_exr(f'{prefix}_poly.exr')
        assert (pH, pW) == (self.H, self.W)
        poly_images = _parse_poly_channels(poly_ch_names)

        n_select = min(self.valid_num, len(poly_images))
        selected = random.sample(poly_images, n_select)
        print(f"Selected {n_select} poly images for validation "
              f"({self.n_valid} valid pixels each)")

        # ---- preload every selected image --------------------------------
        self._items = []
        for eid, img in enumerate(selected):
            rot_data = calib[img['rotation']]

            # wo = normalize(cam_pos - xyz)
            cam_pos = rot_data[img['camera']]
            wo = cam_pos[None, :] - xyz_valid
            wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)

            # wi = normalize(led_pos - xyz)
            led_pos = rot_data[img['led']]
            wi = led_pos[None, :] - xyz_valid
            wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)

            # read RGB channels  (full H×W, then index valid)
            r = _read_channel(poly_exr, img['ch_r'], self.H, self.W) \
                .astype(np.float32)[vrows, vcols]
            g = _read_channel(poly_exr, img['ch_g'], self.H, self.W) \
                .astype(np.float32)[vrows, vcols]
            b = _read_channel(poly_exr, img['ch_b'], self.H, self.W) \
                .astype(np.float32)[vrows, vcols]
            rgbs = np.stack([r, g, b], axis=-1)                 # (V, 3)
            np.clip(rgbs, 0, None, out=rgbs)

            label = (f"mat{self.mat_id:04d}_{img['camera']}_"
                     f"{img['led']}_{img['rotation']}")

            self._items.append({
                'xyz':          torch.from_numpy(xyz_valid.copy()).float(),
                'wi':           torch.from_numpy(wi).float(),
                'wo':           torch.from_numpy(wo).float(),
                'rgbs':         torch.from_numpy(rgbs).float(),
                'point_ids':    torch.from_numpy(pids_valid.copy()).long(),
                'material_ids': torch.full((self.n_valid,), self.mat_id,
                                           dtype=torch.long),
                'emitter_ids':  torch.full((self.n_valid,), eid,
                                           dtype=torch.long),
                'data_type':    torch.full((self.n_valid,), DTYPE_POLY,
                                           dtype=torch.long),
                'lls_corners':  torch.zeros(self.n_valid, 4, 3),
                'confidence':   torch.ones(self.n_valid),
                'valid_mask':   torch.from_numpy(
                                    self.valid_mask_flat.copy()).bool(),
                'img_hw':       torch.tensor([self.H, self.W]),
                'gt_params':    torch.zeros(1),
                'label':        label,
            })

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
            calib[rot_key] = rot_dict
        calib['llsAnglesDegrees'] = raw['llsAnglesDegrees'].flatten() \
                                         .astype(np.float64)
        return calib

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]
