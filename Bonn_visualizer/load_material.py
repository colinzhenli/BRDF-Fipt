"""Load one Bonn material's XYZ map, poly EXR stack, and per-view light/camera poses.

Aligns with ``utils/dataset/bonn.py``: ``_parse_poly_channels``, calibration keys,
and ``poly_*`` channel grouping (B,G,R per view → same slice as training).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as spio

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _bonn_module():
    """Load ``utils/dataset/bonn.py`` without importing ``utils.dataset`` (avoids optional cv2)."""
    path = _REPO_ROOT / "utils" / "dataset" / "bonn.py"
    spec = importlib.util.spec_from_file_location("_bonn_visualizer_bonn", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_bonn_cached = None


def _bonn():
    global _bonn_cached
    if _bonn_cached is None:
        _bonn_cached = _bonn_module()
    return _bonn_cached


def _load_calibration(mat_prefix: Path) -> dict:
    """Same structure as ``BonnDataset._load_calibration`` / ``_load_single_material_full``."""
    path = f"{mat_prefix}_calibration.mat"
    raw = spio.loadmat(path)
    calib = {}
    for rot_key in ["rot000", "rot045", "rot090", "rot135", "rot180"]:
        rd = raw[rot_key][0, 0]
        rot_dict = {}
        for field in rd.dtype.names:
            val = rd[field]
            if field == "llsCorners":
                rot_dict[field] = np.array(val, dtype=np.float32)
            else:
                rot_dict[field] = np.array(val, dtype=np.float32).flatten()
            m = re.match(r"(il|cv)(\d+)", field)
            if m and len(m.group(2)) < 3:
                padded = f"{m.group(1)}{int(m.group(2)):03d}"
                rot_dict[padded] = rot_dict[field]
        calib[rot_key] = rot_dict
    calib["llsAnglesDegrees"] = raw["llsAnglesDegrees"].flatten().astype(np.float64)
    return calib


def load_bonn_poly_material(root_folder: str | Path, mat_id: int) -> dict:
    """Load poly views only (RGB EXR layers + poses), full resolution.

    Returns
    -------
    dict with keys:
        mat_id, H, W,
        xyz (H*W, 3) float32 — surface points (same row-major order as EXR pixels),
        xyz_map (H, W, 3),
        rgb_stack (K, H, W, 3) float32 — per-view HDR RGB,
        light_pos (K, 3), cam_pos (K, 3),
        labels list[str] — e.g. ``cv01_il027_rot090``,
        valid_mask (H*W,) bool — finite XYZ
    """
    b = _bonn()
    read_exr, parse_poly, read_xyz = b._read_exr, b._parse_poly_channels, b._read_xyz_map

    root_folder = Path(root_folder)
    prefix = root_folder / f"mat{mat_id:04d}"
    calib = _load_calibration(prefix)

    xyz_map, H, W = read_xyz(f"{prefix}_xyz_rot000.exr")
    poly_data, poly_ch_names, pH, pW = read_exr(f"{prefix}_poly.exr")
    if (pH, pW) != (H, W):
        raise ValueError(f"poly shape {(pH, pW)} != xyz shape {(H, W)}")

    poly_images = parse_poly(poly_ch_names)
    if not poly_images:
        raise ValueError(f"No poly channels parsed in {prefix}_poly.exr")

    K = len(poly_images)
    light_pos = np.array(
        [calib[im["rotation"]][im["led"]] for im in poly_images], dtype=np.float32
    )
    cam_pos = np.array(
        [calib[im["rotation"]][im["camera"]] for im in poly_images], dtype=np.float32
    )
    labels = [
        f"{im['camera']}_{im['led']}_{im['rotation']}" for im in poly_images
    ]

    rgb_stack = np.stack(
        [
            poly_data[:, :, im["ch_start"] : im["ch_start"] + 3].astype(np.float32)
            for im in poly_images
        ],
        axis=0,
    )
    np.clip(rgb_stack, 0, None, out=rgb_stack)
    del poly_data

    xyz_flat = xyz_map.reshape(-1, 3).astype(np.float32)
    valid_mask = np.all(np.isfinite(xyz_flat), axis=1)

    return {
        "mat_id": mat_id,
        "H": H,
        "W": W,
        "xyz": xyz_flat,
        "xyz_map": xyz_map.astype(np.float32),
        "rgb_stack": rgb_stack,
        "light_pos": light_pos,
        "cam_pos": cam_pos,
        "labels": labels,
        "valid_mask": valid_mask,
        "n_views": K,
    }
