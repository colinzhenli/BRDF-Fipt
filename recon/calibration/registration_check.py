"""
Single source of truth for COLMAP registration health checks.

A material is considered "well-registered" when COLMAP successfully placed
at least REGISTRATION_THRESHOLD * len(scan_log) images in sparse/0/images.bin.
Materials below the threshold need to be re-run with exhaustive_matcher (see
recon/colmap/colmap_exhaustive.sh).

Used by:
  - recon/scheduler/job_scheduler.py  (live + restart sanity check)
  - scripts/validate_and_split_data.py (post-processing gate)
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Make read_write_model importable regardless of CWD
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from read_write_model import read_images_binary  # noqa: E402

# Materials with registered/K below this are unhealthy and must be re-run.
# Empirically, healthy materials register 95-100% of scans; the 7 worst failure
# cases registered <2%, the next tier 13-58%, and the marginal cases sit at 93-95%.
REGISTRATION_THRESHOLD = 0.95


def count_registered_images(material_dir) -> int:
    """Return the number of images in sparse/0/images.bin, or -1 if unreadable."""
    img_bin = Path(material_dir) / "sparse" / "0" / "images.bin"
    if not img_bin.exists():
        return -1
    try:
        return len(read_images_binary(str(img_bin)))
    except Exception:
        return -1


def count_scan_log_entries(material_dir) -> int:
    """Return the number of entries in scan_log.json, or -1 if unreadable."""
    sl = Path(material_dir) / "scan_log.json"
    if not sl.exists():
        return -1
    try:
        with open(sl) as f:
            return len(json.load(f))
    except Exception:
        return -1


def registration_ratio(material_dir) -> Tuple[int, int, float]:
    """
    Return (n_registered, n_scans, ratio).
    Ratio is in [0, 1], or -1.0 if either side is missing/unreadable.
    """
    n_reg = count_registered_images(material_dir)
    n_scans = count_scan_log_entries(material_dir)
    if n_reg < 0 or n_scans <= 0:
        return n_reg, n_scans, -1.0
    return n_reg, n_scans, n_reg / n_scans


def is_well_registered(material_dir, threshold: Optional[float] = None) -> bool:
    """Return True iff registration ratio >= threshold (default REGISTRATION_THRESHOLD)."""
    if threshold is None:
        threshold = REGISTRATION_THRESHOLD
    _, _, ratio = registration_ratio(material_dir)
    return ratio >= threshold
