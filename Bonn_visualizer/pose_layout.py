"""Bonn poly batch layout: 100 views = 4 cameras × 5 LEDs × 5 turntable angles.

Camera pose depends only on (camera id, turntable rotation); each such pose is repeated
for every color LED. LED pose depends only on (LED id, turntable rotation); repeated for
each camera. Dedupe + turntable polylines recover the 4×5 camera grid and 5×5 light grid.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

_ROT_ORDER = {"rot000": 0, "rot045": 1, "rot090": 2, "rot135": 3, "rot180": 4}
_ROT_KEYS = ("rot000", "rot045", "rot090", "rot135", "rot180")


def unique_rows_first(pts: np.ndarray, decimals: int = 3) -> np.ndarray:
    """Deduplicate N×3 rows that coincide after rounding (float-safe)."""
    r = np.round(pts, decimals)
    _, idx = np.unique(r, axis=0, return_index=True)
    return pts[np.sort(idx)].astype(np.float64)


def _cam_pos_by_cv_rot(cam_pos: np.ndarray, labels: Sequence[str]) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], np.ndarray] = {}
    for i, lab in enumerate(labels):
        m = re.match(r"(cv\d+)_(il\d+)_(rot\d+)", lab)
        if not m:
            continue
        key = (m.group(1), m.group(3))
        if key not in out:
            out[key] = np.asarray(cam_pos[i], dtype=np.float64)
    return out


def _light_pos_by_il_rot(light_pos: np.ndarray, labels: Sequence[str]) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], np.ndarray] = {}
    for i, lab in enumerate(labels):
        m = re.match(r"(cv\d+)_(il\d+)_(rot\d+)", lab)
        if not m:
            continue
        key = (m.group(2), m.group(3))
        if key not in out:
            out[key] = np.asarray(light_pos[i], dtype=np.float64)
    return out


def merged_camera_turntable_lineset(
    cam_pos: np.ndarray,
    labels: Sequence[str],
    color_rgb: tuple[float, float, float] = (0.2, 0.55, 0.92),
):
    """One open polyline per camera through the five turntable angles (azimuth)."""
    import open3d as o3d

    by_cr = _cam_pos_by_cv_rot(cam_pos, labels)
    all_pts: list[np.ndarray] = []
    all_lines: list[list[int]] = []
    offset = 0
    for cv in ("cv01", "cv02", "cv03", "cv04"):
        ring: list[np.ndarray] = []
        for rk in _ROT_KEYS:
            p = by_cr.get((cv, rk))
            if p is not None:
                ring.append(p)
        if len(ring) < 2:
            continue
        pts = np.stack(ring, axis=0)
        n = len(pts)
        all_pts.append(pts)
        for j in range(n - 1):
            all_lines.append([offset + j, offset + j + 1])
        offset += n
    if not all_pts:
        return o3d.geometry.LineSet()
    P = np.vstack(all_pts)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(P),
        lines=o3d.utility.Vector2iVector(all_lines),
    )
    ls.colors = o3d.utility.Vector3dVector([list(color_rgb) for _ in all_lines])
    return ls


def merged_led_turntable_lineset(
    light_pos: np.ndarray,
    labels: Sequence[str],
    color_rgb: tuple[float, float, float] = (0.95, 0.72, 0.12),
):
    """One open polyline per color LED through the five turntable angles."""
    import open3d as o3d

    by_ir = _light_pos_by_il_rot(light_pos, labels)
    il_ids = sorted({k[0] for k in by_ir})
    all_pts: list[np.ndarray] = []
    all_lines: list[list[int]] = []
    offset = 0
    for il in il_ids:
        ring: list[np.ndarray] = []
        for rk in _ROT_KEYS:
            p = by_ir.get((il, rk))
            if p is not None:
                ring.append(p)
        if len(ring) < 2:
            continue
        pts = np.stack(ring, axis=0)
        n = len(pts)
        all_pts.append(pts)
        for j in range(n - 1):
            all_lines.append([offset + j, offset + j + 1])
        offset += n
    if not all_pts:
        return o3d.geometry.LineSet()
    P = np.vstack(all_pts)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(P),
        lines=o3d.utility.Vector2iVector(all_lines),
    )
    ls.colors = o3d.utility.Vector3dVector([list(color_rgb) for _ in all_lines])
    return ls


def iter_camera_ring_polylines(cam_pos: np.ndarray, labels: Sequence[str]):
    """Yield (N,3) arrays, one open polyline per camera through turntable angles."""
    by_cr = _cam_pos_by_cv_rot(cam_pos, labels)
    for cv in ("cv01", "cv02", "cv03", "cv04"):
        ring = [by_cr[(cv, rk)] for rk in _ROT_KEYS if (cv, rk) in by_cr]
        if len(ring) >= 2:
            yield np.stack(ring, axis=0)


def iter_led_ring_polylines(light_pos: np.ndarray, labels: Sequence[str]):
    """Yield (N,3) arrays, one open polyline per LED through turntable angles."""
    by_ir = _light_pos_by_il_rot(light_pos, labels)
    il_ids = sorted({k[0] for k in by_ir})
    for il in il_ids:
        ring = [by_ir[(il, rk)] for rk in _ROT_KEYS if (il, rk) in by_ir]
        if len(ring) >= 2:
            yield np.stack(ring, axis=0)
