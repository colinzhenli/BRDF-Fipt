"""
Diagnostic: Compare raw BRDF observation sharpness between Dataset_Nov11 and Bonn.

For each dataset, pick a surface point, compute the half-vector angle (theta_h)
for every observation, and plot intensity vs theta_h.  Sharp specular materials
should show a narrow peak near theta_h=0.

Usage:
    conda activate fipt_copy
    python scripts/debugging/diagnose_brdf_sharpness.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio
import json
import os
import sys

# ── paths ────────────────────────────────────────────────────────────────────
DATASET_NOV11_ROOT = Path("/media/raid/cloth/capture_data/Dataset_Nov11")
BONN_ROOT = Path("/media/raid/cloth/Bonn_train")
OUTPUT_DIR = Path("/media/raid/cloth/output/BRDF/diagnostics/brdf_sharpness")

# CCM for Dataset_Nov11 (from config/data/base.yaml)
CCM = np.array([
    [3.6724617, -0.94800931, 0.08428962],
    [-0.44629176, 2.96095854, -1.17898539],
    [-0.47909694, -0.39991418, 2.10705124]
])

# Materials to inspect (pick a few from each dataset)
NOV11_MATERIAL_IDS = [0, 50, 100]
BONN_MATERIAL_IDS = [1, 2, 5]

# Number of surface points to sample per material
N_POINTS = 5


def compute_half_angle(wi, wo):
    """Compute half-vector angle theta_h (radians) for arrays of wi, wo."""
    h = wi + wo
    h_norm = np.linalg.norm(h, axis=-1, keepdims=True)
    h_norm = np.maximum(h_norm, 1e-8)
    h = h / h_norm
    # theta_h = angle between half-vector and surface normal [0,0,1]
    cos_theta_h = np.clip(h[..., 2], -1.0, 1.0)
    return np.arccos(cos_theta_h)


def load_nov11_material(mat_id):
    """Load structured NPZ for a Dataset_Nov11 material.

    Returns: xyz (V,3), rgbs (K,V,3) float32, cam_pos (K,3), light_pos (K,3)
    """
    path = DATASET_NOV11_ROOT / str(mat_id) / "observations_structured.npz"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return None
    d = np.load(path)
    xyz = d["xyz"]  # (V, 3)
    rgbs = d["rgbs"].astype(np.float64)  # (K, V, 3) uint16 -> float64
    # Apply CCM
    K, V, _ = rgbs.shape
    rgbs_flat = rgbs.reshape(-1, 3) @ CCM
    rgbs = np.clip(rgbs_flat.reshape(K, V, 3), 0, None)
    cam_pos = d["cam_pos"]  # (K, 3)
    light_pos = d["light_pos"]  # (K, 3)
    return xyz, rgbs, cam_pos, light_pos


def load_bonn_material(mat_id):
    """Load Bonn material: calibration + poly EXR.

    Returns: xyz (V,3), rgbs (K,V,3) float32, cam_pos (K,3), light_pos (K,3)
    """
    try:
        import pyexr
    except ImportError:
        print("  [ERROR] pyexr not installed, cannot load Bonn EXR files")
        return None

    prefix = BONN_ROOT / f"mat{mat_id:04d}"
    xyz_path = f"{prefix}_xyz_rot000.exr"
    poly_path = f"{prefix}_poly.exr"
    cal_path = f"{prefix}_calibration.mat"

    if not os.path.exists(poly_path):
        print(f"  [SKIP] {poly_path} not found")
        return None

    # ── xyz ──
    xyz_exr = pyexr.open(xyz_path)
    xyz_map = xyz_exr.get()  # (H, W, 3)
    H, W, _ = xyz_map.shape
    xyz = xyz_map.reshape(-1, 3).astype(np.float32)

    # ── calibration ──
    raw_cal = sio.loadmat(cal_path)
    rot = raw_cal["rot000"][0, 0]

    # Get LED names (il*) and camera names (cv*)
    led_names = sorted([n for n in rot.dtype.names if n.startswith("il")])
    cam_names = sorted([n for n in rot.dtype.names if n.startswith("cv")])

    led_positions = {n: rot[n].flatten().astype(np.float32) for n in led_names}
    cam_positions = {n: rot[n].flatten().astype(np.float32) for n in cam_names}

    # ── poly channels ──
    poly_exr = pyexr.open(poly_path)
    poly_data = poly_exr.get()  # (H, W, C)
    ch_names = poly_exr.channel_map
    # channels come in alphabetical groups of 3: _B, _G, _R
    # parse image names
    image_names = sorted(set(ch.rsplit(".", 1)[0] if "." in ch else ch.rsplit("_", 1)[0] for ch in ch_names))

    # Deduce the poly image structure: (cam, led) pairs
    # Bonn poly images: 4 cameras x 25-29 LEDs per rotation = ~100 images
    # Channel ordering: alphabetical → group by 3 (B,G,R)
    n_channels = poly_data.shape[2]
    n_images = n_channels // 3
    V = H * W

    rgbs = poly_data.reshape(V, n_images, 3)  # (V, K, 3)
    rgbs = rgbs.transpose(1, 0, 2)  # (K, V, 3)
    np.clip(rgbs, 0, None, out=rgbs)

    # Build light/cam position arrays for each poly image
    # Poly images cycle: for each camera, iterate over LEDs
    light_pos_list = []
    cam_pos_list = []
    for cam_name in cam_names:
        for led_name in led_names:
            light_pos_list.append(led_positions[led_name])
            cam_pos_list.append(cam_positions[cam_name])
            if len(light_pos_list) >= n_images:
                break
        if len(light_pos_list) >= n_images:
            break

    # If we couldn't match exactly, just use first camera + all LEDs pattern
    if len(light_pos_list) != n_images:
        print(f"  [WARN] Expected {n_images} images but got {len(light_pos_list)} (cam,led) pairs. "
              f"Using cyclic assignment: {len(cam_names)} cams x {len(led_names)} LEDs")
        light_pos_list = []
        cam_pos_list = []
        for ci, cam_name in enumerate(cam_names):
            for li, led_name in enumerate(led_names):
                light_pos_list.append(led_positions[led_name])
                cam_pos_list.append(cam_positions[cam_name])

        # Trim or warn
        light_pos_list = light_pos_list[:n_images]
        cam_pos_list = cam_pos_list[:n_images]

    light_pos = np.array(light_pos_list)
    cam_pos = np.array(cam_pos_list)

    return xyz, rgbs.astype(np.float32), cam_pos, light_pos


def analyze_material(xyz, rgbs, cam_pos, light_pos, dataset_name, mat_id):
    """For a few surface points, compute and return (theta_h, intensity) arrays."""
    K, V, _ = rgbs.shape
    results = []

    # Pick N_POINTS random points with decent signal
    mean_intensity = rgbs.mean(axis=(0, 2))  # (V,)
    # Choose points with above-median intensity (more likely to have specular)
    above_median = np.where(mean_intensity > np.median(mean_intensity))[0]
    if len(above_median) < N_POINTS:
        above_median = np.arange(V)
    rng = np.random.RandomState(42)
    chosen = rng.choice(above_median, size=min(N_POINTS, len(above_median)), replace=False)

    for pt_idx in chosen:
        pt_xyz = xyz[pt_idx]  # (3,)

        # Compute wi, wo for all K images
        wi = light_pos - pt_xyz[None, :]  # (K, 3)
        wi_norm = np.linalg.norm(wi, axis=1, keepdims=True)
        wi = wi / np.maximum(wi_norm, 1e-8)

        wo = cam_pos - pt_xyz[None, :]  # (K, 3)
        wo_norm = np.linalg.norm(wo, axis=1, keepdims=True)
        wo = wo / np.maximum(wo_norm, 1e-8)

        theta_h = compute_half_angle(wi, wo)  # (K,)
        intensity = rgbs[:, pt_idx, :].mean(axis=1)  # (K,) avg over RGB

        # Filter out zero-intensity (occluded)
        valid = intensity > 0
        results.append({
            "theta_h_deg": np.degrees(theta_h[valid]),
            "intensity": intensity[valid],
            "point_idx": pt_idx,
        })

    return results


def plot_comparison(nov11_results, bonn_results, output_path):
    """Plot side-by-side theta_h vs intensity for both datasets."""
    n_nov11 = len(nov11_results)
    n_bonn = len(bonn_results)
    n_rows = max(n_nov11, n_bonn)

    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows), squeeze=False)
    fig.suptitle("Raw BRDF Observations: Intensity vs Half-Angle (theta_h)\n"
                 "Left = Dataset_Nov11, Right = Bonn", fontsize=14, y=1.01)

    for i in range(n_rows):
        # Nov11
        ax = axes[i, 0]
        if i < n_nov11:
            mat_id, pt_data_list = nov11_results[i]
            for pd in pt_data_list:
                ax.scatter(pd["theta_h_deg"], pd["intensity"], s=3, alpha=0.5,
                           label=f"pt {pd['point_idx']}")
            ax.set_title(f"Dataset_Nov11 mat={mat_id}")
            ax.set_xlabel("theta_h (degrees)")
            ax.set_ylabel("Intensity (CCM-corrected)")
            ax.set_xlim(0, 90)
            ax.legend(fontsize=6, markerscale=3)
        else:
            ax.axis("off")

        # Bonn
        ax = axes[i, 1]
        if i < n_bonn:
            mat_id, pt_data_list = bonn_results[i]
            for pd in pt_data_list:
                ax.scatter(pd["theta_h_deg"], pd["intensity"], s=3, alpha=0.5,
                           label=f"pt {pd['point_idx']}")
            ax.set_title(f"Bonn mat={mat_id}")
            ax.set_xlabel("theta_h (degrees)")
            ax.set_ylabel("Intensity (raw)")
            ax.set_xlim(0, 90)
            ax.legend(fontsize=6, markerscale=3)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_peak_sharpness_comparison(nov11_results, bonn_results, output_path):
    """For each material, compute a sharpness metric (peak / median ratio)
    and the angle at peak. Summarize in a bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, results, name in [(axes[0], nov11_results, "Dataset_Nov11"),
                               (axes[1], bonn_results, "Bonn")]:
        mat_labels = []
        peak_ratios = []
        peak_angles = []

        for mat_id, pt_data_list in results:
            # Aggregate all points for this material
            all_theta = np.concatenate([pd["theta_h_deg"] for pd in pt_data_list])
            all_intensity = np.concatenate([pd["intensity"] for pd in pt_data_list])

            if len(all_intensity) == 0:
                continue

            median_i = np.median(all_intensity)
            peak_i = np.percentile(all_intensity, 99)  # 99th percentile to avoid outliers
            ratio = peak_i / max(median_i, 1e-8)

            # Find angle at peak
            top_mask = all_intensity >= np.percentile(all_intensity, 95)
            peak_angle = np.median(all_theta[top_mask])

            mat_labels.append(f"mat {mat_id}")
            peak_ratios.append(ratio)
            peak_angles.append(peak_angle)

        x = np.arange(len(mat_labels))
        bars = ax.bar(x, peak_ratios, color="steelblue", alpha=0.8)
        # Annotate with peak angle
        for xi, (r, a) in enumerate(zip(peak_ratios, peak_angles)):
            ax.text(xi, r + 0.5, f"{a:.0f}°", ha="center", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(mat_labels)
        ax.set_ylabel("Peak/Median Intensity Ratio")
        ax.set_title(f"{name}\n(number = angle at peak)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    # ── Dataset_Nov11 ──
    print("=" * 60)
    print("Loading Dataset_Nov11 materials...")
    print("=" * 60)
    nov11_results = []
    for mat_id in NOV11_MATERIAL_IDS:
        print(f"  Loading mat {mat_id}...")
        data = load_nov11_material(mat_id)
        if data is None:
            continue
        xyz, rgbs, cam_pos, light_pos = data
        print(f"    {rgbs.shape[0]} images, {rgbs.shape[1]} points")
        pt_data = analyze_material(xyz, rgbs, cam_pos, light_pos, "nov11", mat_id)
        nov11_results.append((mat_id, pt_data))
        # Free memory
        del xyz, rgbs, cam_pos, light_pos, data

    # ── Bonn ──
    print("=" * 60)
    print("Loading Bonn materials...")
    print("=" * 60)
    bonn_results = []
    for mat_id in BONN_MATERIAL_IDS:
        print(f"  Loading mat {mat_id}...")
        data = load_bonn_material(mat_id)
        if data is None:
            continue
        xyz, rgbs, cam_pos, light_pos = data
        print(f"    {rgbs.shape[0]} images, {rgbs.shape[1]} points")
        pt_data = analyze_material(xyz, rgbs, cam_pos, light_pos, "bonn", mat_id)
        bonn_results.append((mat_id, pt_data))
        del xyz, rgbs, cam_pos, light_pos, data

    # ── Plot ──
    if nov11_results or bonn_results:
        plot_comparison(nov11_results, bonn_results,
                        OUTPUT_DIR / "raw_intensity_vs_theta_h.png")
        plot_peak_sharpness_comparison(nov11_results, bonn_results,
                                       OUTPUT_DIR / "peak_sharpness_comparison.png")
    else:
        print("No data loaded, nothing to plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()
