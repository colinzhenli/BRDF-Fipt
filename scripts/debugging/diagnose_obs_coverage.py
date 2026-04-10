"""
Diagnostic: Compare angular coverage and observation density between
Dataset_Nov11 and Bonn datasets.

Metrics:
  - Number of images (observations) per surface point
  - Angular coverage: distribution of (theta_h, theta_d) pairs
  - Angular gaps: largest gap in theta_h sampling

Usage:
    conda activate fipt_copy
    python scripts/debugging/diagnose_obs_coverage.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio
import os

# ── paths ────────────────────────────────────────────────────────────────────
DATASET_NOV11_ROOT = Path("/media/raid/cloth/capture_data/Dataset_Nov11")
BONN_ROOT = Path("/media/raid/cloth/Bonn_train")
OUTPUT_DIR = Path("/media/raid/cloth/output/BRDF/diagnostics/obs_coverage")

NOV11_MATERIAL_IDS = [0, 50, 100]
BONN_MATERIAL_IDS = [1, 2, 5]


def compute_rusinkiewicz(wi, wo):
    """Compute Rusinkiewicz half/difference parametrization.

    Args:
        wi: (K, 3) normalized incident directions
        wo: (K, 3) normalized outgoing directions

    Returns:
        theta_h: (K,) half-vector polar angle in radians
        theta_d: (K,) difference polar angle in radians
        phi_d:   (K,) difference azimuthal angle in radians
    """
    h = wi + wo
    h_norm = np.linalg.norm(h, axis=-1, keepdims=True)
    h = h / np.maximum(h_norm, 1e-8)

    # theta_h: angle between half-vector and normal [0,0,1]
    cos_theta_h = np.clip(h[:, 2], -1.0, 1.0)
    theta_h = np.arccos(cos_theta_h)

    # Build frame from h
    # tangent: h x [0,0,1] (or h x [1,0,0] if h is near [0,0,1])
    up = np.array([0.0, 0.0, 1.0])
    cross = np.cross(h, up[None, :])
    cross_norm = np.linalg.norm(cross, axis=-1, keepdims=True)
    # fallback for h ~= [0,0,1]
    fallback = np.cross(h, np.array([[1.0, 0.0, 0.0]]))
    use_fallback = (cross_norm.squeeze() < 1e-6)
    cross[use_fallback] = fallback[use_fallback]
    cross_norm = np.linalg.norm(cross, axis=-1, keepdims=True)
    t = cross / np.maximum(cross_norm, 1e-8)
    b = np.cross(h, t)

    # wi in half-vector frame
    wi_h = np.stack([
        np.einsum("ij,ij->i", wi, t),
        np.einsum("ij,ij->i", wi, b),
        np.einsum("ij,ij->i", wi, h),
    ], axis=-1)

    theta_d = np.arccos(np.clip(wi_h[:, 2], -1.0, 1.0))
    phi_d = np.arctan2(wi_h[:, 1], wi_h[:, 0])

    return theta_h, theta_d, phi_d


def load_nov11_directions(mat_id):
    """Load directions for a Dataset_Nov11 material. Returns wi, wo arrays."""
    path = DATASET_NOV11_ROOT / str(mat_id) / "observations_structured.npz"
    if not path.exists():
        return None, None, None
    d = np.load(path)
    xyz = d["xyz"]       # (V, 3)
    cam_pos = d["cam_pos"]    # (K, 3)
    light_pos = d["light_pos"]  # (K, 3)
    K = cam_pos.shape[0]
    V = xyz.shape[0]

    # Pick a representative point (centroid of the surface)
    center_idx = np.argmin(np.linalg.norm(xyz - xyz.mean(axis=0, keepdims=True), axis=1))
    pt = xyz[center_idx]

    wi = light_pos - pt[None, :]  # (K, 3)
    wi = wi / np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
    wo = cam_pos - pt[None, :]
    wo = wo / np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)

    return wi, wo, K


def load_bonn_directions(mat_id):
    """Load directions for a Bonn material. Returns wi, wo arrays."""
    prefix = BONN_ROOT / f"mat{mat_id:04d}"
    cal_path = f"{prefix}_calibration.mat"
    xyz_path = f"{prefix}_xyz_rot000.exr"

    if not os.path.exists(cal_path):
        return None, None, None

    try:
        import pyexr
        xyz_exr = pyexr.open(str(xyz_path))
        xyz_map = xyz_exr.get()
        H, W, _ = xyz_map.shape
        xyz = xyz_map.reshape(-1, 3).astype(np.float32)
    except Exception:
        # Fallback: use origin
        xyz = np.zeros((1, 3), dtype=np.float32)

    center_idx = np.argmin(np.linalg.norm(xyz - xyz.mean(axis=0, keepdims=True), axis=1))
    pt = xyz[center_idx]

    raw_cal = sio.loadmat(cal_path)
    rot = raw_cal["rot000"][0, 0]
    led_names = sorted([n for n in rot.dtype.names if n.startswith("il")])
    cam_names = sorted([n for n in rot.dtype.names if n.startswith("cv")])

    # Build all (cam, led) pairs
    wi_list, wo_list = [], []
    for cam_name in cam_names:
        cam_p = rot[cam_name].flatten().astype(np.float32)
        for led_name in led_names:
            led_p = rot[led_name].flatten().astype(np.float32)
            wi_vec = led_p - pt
            wo_vec = cam_p - pt
            wi_list.append(wi_vec / max(np.linalg.norm(wi_vec), 1e-8))
            wo_list.append(wo_vec / max(np.linalg.norm(wo_vec), 1e-8))

    wi = np.array(wi_list)
    wo = np.array(wo_list)
    K = len(wi)

    return wi, wo, K


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(NOV11_MATERIAL_IDS) + len(BONN_MATERIAL_IDS), 3,
                             figsize=(18, 4 * (len(NOV11_MATERIAL_IDS) + len(BONN_MATERIAL_IDS))),
                             squeeze=False)

    row = 0
    summary_lines = []

    # ── Dataset_Nov11 ──
    print("=" * 60)
    print("Dataset_Nov11 angular coverage")
    print("=" * 60)
    for mat_id in NOV11_MATERIAL_IDS:
        wi, wo, K = load_nov11_directions(mat_id)
        if wi is None:
            print(f"  mat {mat_id}: SKIP")
            continue

        theta_h, theta_d, phi_d = compute_rusinkiewicz(wi, wo)
        theta_h_deg = np.degrees(theta_h)
        theta_d_deg = np.degrees(theta_d)

        # Stats
        sorted_th = np.sort(theta_h_deg)
        gaps = np.diff(sorted_th)
        max_gap = gaps.max() if len(gaps) > 0 else 0

        line = (f"  Nov11 mat {mat_id}: K={K} images, "
                f"theta_h range=[{theta_h_deg.min():.1f}, {theta_h_deg.max():.1f}]°, "
                f"max_gap={max_gap:.1f}°, "
                f"theta_d range=[{theta_d_deg.min():.1f}, {theta_d_deg.max():.1f}]°")
        print(line)
        summary_lines.append(line)

        # Plot: theta_h histogram
        axes[row, 0].hist(theta_h_deg, bins=50, color="steelblue", alpha=0.7)
        axes[row, 0].set_title(f"Nov11 mat {mat_id}: theta_h distribution (K={K})")
        axes[row, 0].set_xlabel("theta_h (degrees)")

        # Plot: theta_d histogram
        axes[row, 1].hist(theta_d_deg, bins=50, color="coral", alpha=0.7)
        axes[row, 1].set_title(f"Nov11 mat {mat_id}: theta_d distribution")
        axes[row, 1].set_xlabel("theta_d (degrees)")

        # Plot: 2D scatter theta_h vs theta_d
        axes[row, 2].scatter(theta_h_deg, theta_d_deg, s=5, alpha=0.5)
        axes[row, 2].set_title(f"Nov11 mat {mat_id}: angular coverage")
        axes[row, 2].set_xlabel("theta_h (degrees)")
        axes[row, 2].set_ylabel("theta_d (degrees)")
        axes[row, 2].set_xlim(0, 90)
        axes[row, 2].set_ylim(0, 90)
        row += 1

    # ── Bonn ──
    print("=" * 60)
    print("Bonn angular coverage")
    print("=" * 60)
    for mat_id in BONN_MATERIAL_IDS:
        wi, wo, K = load_bonn_directions(mat_id)
        if wi is None:
            print(f"  mat {mat_id}: SKIP")
            continue

        theta_h, theta_d, phi_d = compute_rusinkiewicz(wi, wo)
        theta_h_deg = np.degrees(theta_h)
        theta_d_deg = np.degrees(theta_d)

        sorted_th = np.sort(theta_h_deg)
        gaps = np.diff(sorted_th)
        max_gap = gaps.max() if len(gaps) > 0 else 0

        line = (f"  Bonn mat {mat_id}: K={K} images, "
                f"theta_h range=[{theta_h_deg.min():.1f}, {theta_h_deg.max():.1f}]°, "
                f"max_gap={max_gap:.1f}°, "
                f"theta_d range=[{theta_d_deg.min():.1f}, {theta_d_deg.max():.1f}]°")
        print(line)
        summary_lines.append(line)

        axes[row, 0].hist(theta_h_deg, bins=50, color="steelblue", alpha=0.7)
        axes[row, 0].set_title(f"Bonn mat {mat_id}: theta_h distribution (K={K})")
        axes[row, 0].set_xlabel("theta_h (degrees)")

        axes[row, 1].hist(theta_d_deg, bins=50, color="coral", alpha=0.7)
        axes[row, 1].set_title(f"Bonn mat {mat_id}: theta_d distribution")
        axes[row, 1].set_xlabel("theta_d (degrees)")

        axes[row, 2].scatter(theta_h_deg, theta_d_deg, s=5, alpha=0.5)
        axes[row, 2].set_title(f"Bonn mat {mat_id}: angular coverage")
        axes[row, 2].set_xlabel("theta_h (degrees)")
        axes[row, 2].set_ylabel("theta_d (degrees)")
        axes[row, 2].set_xlim(0, 90)
        axes[row, 2].set_ylim(0, 90)
        row += 1

    plt.tight_layout()
    out_path = OUTPUT_DIR / "angular_coverage_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()

    # Save summary
    summary_path = OUTPUT_DIR / "coverage_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
