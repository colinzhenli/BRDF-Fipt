"""Bonn poly viewer: default Open3D (MeshLab-like 3D) + optional all-matplotlib UI.

See ``open3d_view.py`` for the preferred 3D interaction (orbit / zoom / pan).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from Bonn_visualizer.hdr_display import ColorDisplayMode, hdr_to_display_uint8
from Bonn_visualizer.pose_layout import (
    iter_camera_ring_polylines,
    iter_led_ring_polylines,
    unique_rows_first,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _subsample_indices(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def run_viewer(
    root_folder: Path,
    mat_id: int,
    max_points: int = 80_000,
    seed: int = 0,
    save_preview: Path | None = None,
    slider_debounce_ms: int = 75,
    display_color: ColorDisplayMode = "tonemap",
    linear_percentile: float = 99.5,
) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.widgets import Button, Slider
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    from Bonn_visualizer.load_material import load_bonn_poly_material

    data = load_bonn_poly_material(root_folder, mat_id)
    H, W = data["H"], data["W"]
    xyz = data["xyz"]
    valid = data["valid_mask"]
    rgb_stack = data["rgb_stack"]
    light_pos = data["light_pos"]
    cam_pos = data["cam_pos"]
    labels = data["labels"]
    K = data["n_views"]

    # One-time: convert HDR to uint8 per view for fast slider updates.
    display_stack = np.empty((K, H, W, 3), dtype=np.uint8)
    for i in range(K):
        display_stack[i] = hdr_to_display_uint8(
            rgb_stack[i], display_color, linear_percentile=linear_percentile
        )
    data.pop("rgb_stack", None)
    del rgb_stack

    rng = np.random.default_rng(seed)
    valid_idx = np.nonzero(valid)[0]
    if valid_idx.size == 0:
        raise RuntimeError("No finite XYZ points in map.")
    pick = _subsample_indices(valid_idx.size, max_points, rng)
    sel = valid_idx[pick]
    pts = xyz[sel]

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.14], width_ratios=[1.0, 1.0], hspace=0.25, wspace=0.2)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_im = fig.add_subplot(gs[0, 1])
    gs_ctrl = gs[1, :].subgridspec(1, 3, width_ratios=[0.07, 1.0, 0.07], wspace=0.25)
    ax_btn_m = fig.add_subplot(gs_ctrl[0, 0])
    ax_slider = fig.add_subplot(gs_ctrl[0, 1])
    ax_btn_p = fig.add_subplot(gs_ctrl[0, 2])

    # Single Line3D with pixel marker is much faster than scatter() for ~10^5 points.
    ax3d.plot(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        ",",
        color="#4a6fa5",
        alpha=0.35,
        markersize=2,
        linestyle="",
        zorder=1,
    )
    cam_u = unique_rows_first(np.asarray(cam_pos))
    led_u = unique_rows_first(np.asarray(light_pos))
    ax3d.scatter(
        cam_u[:, 0],
        cam_u[:, 1],
        cam_u[:, 2],
        s=14,
        c="#4a90d9",
        alpha=0.75,
        label=f"cameras ({len(cam_u)} sites)",
        depthshade=False,
    )
    ax3d.scatter(
        led_u[:, 0],
        led_u[:, 1],
        led_u[:, 2],
        s=14,
        c="#e6b800",
        alpha=0.75,
        label=f"lights ({len(led_u)} sites)",
        depthshade=False,
    )
    for ring in iter_camera_ring_polylines(np.asarray(cam_pos), labels):
        ax3d.plot(
            ring[:, 0],
            ring[:, 1],
            ring[:, 2],
            "-",
            color="#6ab0ff",
            linewidth=1.2,
            alpha=0.65,
            zorder=2,
        )
    for ring in iter_led_ring_polylines(np.asarray(light_pos), labels):
        ax3d.plot(
            ring[:, 0],
            ring[:, 1],
            ring[:, 2],
            "-",
            color="#e6a020",
            linewidth=1.2,
            alpha=0.65,
            zorder=2,
        )

    centroid = np.mean(xyz[valid], axis=0)
    c0, ell0 = cam_pos[0], light_pos[0]
    (sel_cam_pt,) = ax3d.plot(
        [c0[0]], [c0[1]], [c0[2]], "c*", markersize=14, markeredgecolor="k", markeredgewidth=0.4, zorder=10
    )
    (sel_led_pt,) = ax3d.plot(
        [ell0[0]],
        [ell0[1]],
        [ell0[2]],
        "m*",
        markersize=14,
        markeredgecolor="k",
        markeredgewidth=0.4,
        zorder=10,
    )
    (line_cam,) = ax3d.plot(
        [centroid[0], c0[0]],
        [centroid[1], c0[1]],
        [centroid[2], c0[2]],
        "c-",
        linewidth=1.5,
        alpha=0.85,
        zorder=9,
    )
    (line_led,) = ax3d.plot(
        [centroid[0], ell0[0]],
        [centroid[1], ell0[1]],
        [centroid[2], ell0[2]],
        "m-",
        linewidth=1.5,
        alpha=0.85,
        zorder=9,
    )

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title(f"mat{mat_id:04d}  |  {K} poly views  |  {len(pts):,} / {valid.sum():,} points")
    ax3d.legend(loc="upper left", fontsize=7, markerscale=0.8)

    _set_equal_aspect_3d(ax3d, np.vstack([pts, cam_u, led_u]))

    im_artist = ax_im.imshow(np.zeros((H, W, 3), dtype=np.uint8))
    ax_im.set_title(
        "Poly (linear, no γ)" if display_color == "linear" else "Poly (tone-mapped)"
    )
    ax_im.axis("off")
    text_info = fig.text(0.5, 0.02, "", ha="center", fontsize=9, family="monospace")

    for bax in (ax_btn_m, ax_btn_p):
        bax.set_navigate(False)
        bax.set_xticks([])
        bax.set_yticks([])
    btn_minus = Button(ax_btn_m, "−", color="0.88", hovercolor="0.78")
    btn_plus = Button(ax_btn_p, "+", color="0.88", hovercolor="0.78")

    slider = Slider(ax_slider, "view", 0, K - 1, valinit=0, valstep=1)

    pending_k = [0]

    def apply_view(k: int, draw: bool) -> None:
        k = int(np.clip(k, 0, K - 1))
        im_artist.set_data(display_stack[k])
        c = cam_pos[k]
        ell = light_pos[k]
        sel_cam_pt.set_data_3d([c[0]], [c[1]], [c[2]])
        sel_led_pt.set_data_3d([ell[0]], [ell[1]], [ell[2]])
        line_cam.set_data_3d([centroid[0], c[0]], [centroid[1], c[1]], [centroid[2], c[2]])
        line_led.set_data_3d([centroid[0], ell[0]], [centroid[1], ell[1]], [centroid[2], ell[2]])
        text_info.set_text(f"[{k:3d} / {K}]  {labels[k]}")
        if draw:
            fig.canvas.draw_idle()

    def set_slider_silent(k: int) -> None:
        k = int(np.clip(k, 0, K - 1))
        slider.eventson = False
        slider.set_val(k)
        slider.eventson = True

    deb_timer = fig.canvas.new_timer(interval=max(1, int(slider_debounce_ms)))

    def _debounced_draw() -> None:
        apply_view(pending_k[0], draw=True)

    deb_timer.add_callback(_debounced_draw)
    deb_timer.single_shot = True

    def on_slider_changed(v: float) -> None:
        pending_k[0] = int(v)
        deb_timer.stop()
        deb_timer.start()

    def bump(delta: int) -> None:
        k = int(np.clip(slider.val + delta, 0, K - 1))
        pending_k[0] = k
        set_slider_silent(k)
        deb_timer.stop()
        apply_view(k, draw=True)

    slider.on_changed(on_slider_changed)
    btn_minus.on_clicked(lambda _e: bump(-1))
    btn_plus.on_clicked(lambda _e: bump(1))

    def on_fig_key(ev) -> None:
        k = (ev.key or "").lower()
        if k in ("left", "arrowleft"):
            bump(-1)
        elif k in ("right", "arrowright"):
            bump(1)

    fig.canvas.mpl_connect("key_press_event", on_fig_key)

    apply_view(0, draw=False)
    set_slider_silent(0)
    if save_preview is not None:
        save_preview = Path(save_preview)
        save_preview.parent.mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()
        fig.savefig(save_preview, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return
    fig.canvas.draw()
    plt.show()


def _set_equal_aspect_3d(ax, pts: np.ndarray) -> None:
    """Rough equal scale for mplot3d (matplotlib has no true equal aspect)."""
    lim = pts
    x0, x1 = lim[:, 0].min(), lim[:, 0].max()
    y0, y1 = lim[:, 1].min(), lim[:, 1].max()
    z0, z1 = lim[:, 2].min(), lim[:, 2].max()
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    r = max(x1 - x0, y1 - y0, z1 - z0) / 2 * 0.55
    if r < 1e-6:
        r = 1.0
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bonn poly viewer: Open3D 3D (default) or matplotlib-only; poly HDR image per view."
    )
    p.add_argument("--root", type=Path, required=True, help="Bonn dataset root (contains matXXXX_* files).")
    p.add_argument("--mat", type=int, required=True, help="Material id, e.g. 1 for mat0001.")
    p.add_argument(
        "--backend",
        choices=("open3d", "matplotlib"),
        default="open3d",
        help="open3d: MeshLab-like 3D + small matplotlib image window. matplotlib: single-window UI.",
    )
    p.add_argument(
        "--material-points",
        type=int,
        default=1_200,
        help="(open3d) Random surface points to show material extent; cam/light are emphasized.",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=80_000,
        help="(matplotlib) Max surface points in 3D subplot.",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling.")
    p.add_argument(
        "--slider-debounce-ms",
        type=int,
        default=75,
        help="(matplotlib) Delay (ms) after last slider movement before redrawing.",
    )
    p.add_argument(
        "--save-preview",
        type=Path,
        default=None,
        metavar="PNG",
        help="matplotlib: full layout PNG. open3d: first-view image only (no Open3D snapshot).",
    )
    p.add_argument(
        "--display-color",
        choices=("tonemap", "linear"),
        default="tonemap",
        help="tonemap: Reinhard + sRGB γ (default). linear: measured linear RGB scaled to 8-bit, no γ.",
    )
    p.add_argument(
        "--linear-percentile",
        type=float,
        default=99.5,
        help="With --display-color linear: scale so this percentile maps to white (avoids hot pixels).",
    )
    args = p.parse_args()
    if args.backend == "open3d":
        from Bonn_visualizer.open3d_view import run_open3d_viewer, save_image_only_preview

        if args.save_preview is not None:
            save_image_only_preview(
                args.root,
                args.mat,
                args.save_preview,
                display_color=args.display_color,
                linear_percentile=args.linear_percentile,
            )
            return
        run_open3d_viewer(
            args.root,
            args.mat,
            material_points=args.material_points,
            seed=args.seed,
            display_color=args.display_color,
            linear_percentile=args.linear_percentile,
        )
        return
    run_viewer(
        args.root,
        args.mat,
        max_points=args.max_points,
        seed=args.seed,
        save_preview=args.save_preview,
        slider_debounce_ms=args.slider_debounce_ms,
        display_color=args.display_color,
        linear_percentile=args.linear_percentile,
    )


if __name__ == "__main__":
    main()
