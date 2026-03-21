"""Open3D 3D viewer (MeshLab-like mouse orbit / zoom / pan) + small matplotlib window for poly image."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from Bonn_visualizer.hdr_display import ColorDisplayMode, hdr_to_display_uint8
from Bonn_visualizer.pose_layout import (
    merged_camera_turntable_lineset,
    merged_led_turntable_lineset,
    unique_rows_first,
)


def _scene_extent(xyz_valid: np.ndarray, cam_pos: np.ndarray, light_pos: np.ndarray) -> float:
    pts = np.vstack([xyz_valid, cam_pos, light_pos])
    e = float(np.ptp(pts, axis=0).max())
    return max(e, 1e-4)


def _sphere_at(center: np.ndarray, radius: float, color_rgb: tuple[float, float, float], resolution: int = 10):
    import open3d as o3d

    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    s.translate(center.astype(np.float64))
    s.paint_uniform_color(list(color_rgb))
    s.compute_vertex_normals()
    return s


def _merge_spheres(centers: np.ndarray, radius: float, color_rgb: tuple[float, float, float], resolution: int = 10):
    import open3d as o3d

    if len(centers) == 0:
        return o3d.geometry.TriangleMesh()
    m = _sphere_at(centers[0], radius, color_rgb, resolution)
    for i in range(1, len(centers)):
        m += _sphere_at(centers[i], radius, color_rgb, resolution)
    return m


# GLFW key codes (Open3D / Filament viewer)
_GLFW_KEY_RIGHT = 262
_GLFW_KEY_LEFT = 263


def _print_help_keys() -> None:
    print(
        "Keys:  ← / → = prev / next image   [ or ]   N / P   0-9 = jump min(n*10,K-1)   H = help\n"
        "Mouse: drag = orbit (MeshLab-style), wheel = zoom, Shift+drag = pan\n"
        "Note: 100 views reuse 20 camera + 25 LED poses; blue/yellow arcs = turntable motion."
    )


def run_open3d_viewer(
    root_folder: Path,
    mat_id: int,
    material_points: int = 1_200,
    seed: int = 0,
    display_color: ColorDisplayMode = "tonemap",
    linear_percentile: float = 99.5,
) -> None:
    import open3d as o3d
    from matplotlib import pyplot as plt

    from Bonn_visualizer.load_material import load_bonn_poly_material

    data = load_bonn_poly_material(root_folder, mat_id)
    H, W = data["H"], data["W"]
    xyz = data["xyz"]
    valid = data["valid_mask"]
    rgb_stack = data["rgb_stack"]
    light_pos = data["light_pos"].astype(np.float64)
    cam_pos = data["cam_pos"].astype(np.float64)
    labels = data["labels"]
    K = data["n_views"]

    display_stack = np.empty((K, H, W, 3), dtype=np.uint8)
    for i in range(K):
        display_stack[i] = hdr_to_display_uint8(
            rgb_stack[i], display_color, linear_percentile=linear_percentile
        )
    data.pop("rgb_stack", None)

    valid_idx = np.nonzero(valid)[0]
    if valid_idx.size == 0:
        raise RuntimeError("No finite XYZ points in map.")
    rng = np.random.default_rng(seed)
    n_mat = min(int(material_points), valid_idx.size)
    mat_ix = rng.choice(valid_idx, size=n_mat, replace=False)
    mat_pts = xyz[mat_ix].astype(np.float64)

    centroid = np.mean(xyz[valid], axis=0).astype(np.float64)
    extent = _scene_extent(xyz[valid], cam_pos, light_pos)

    r_led = extent * 0.018
    r_cam = extent * 0.014
    r_sel = extent * 0.028
    axis_len = extent * 0.22

    # 100 views = same camera pose repeated for each color LED → dedupe for layout like MeshLab (4×5).
    cam_unique = unique_rows_first(cam_pos)
    led_unique = unique_rows_first(light_pos)
    cam_rings = merged_camera_turntable_lineset(cam_pos, labels)
    led_rings = merged_led_turntable_lineset(light_pos, labels)
    print(
        f"Bonn poly: {K} views = 4 cams × 5 LEDs × 5 turntable angles.\n"
        f"Drawing {len(cam_unique)} camera sites, {len(led_unique)} LED sites (deduplicated); "
        f"blue lines = turntable arc per camera, gold lines = per LED."
    )

    mat_pcd = o3d.geometry.PointCloud()
    mat_pcd.points = o3d.utility.Vector3dVector(mat_pts)
    mat_pcd.paint_uniform_color([0.45, 0.48, 0.52])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len, origin=centroid)

    cam_mesh = _merge_spheres(cam_unique, r_cam, (0.15, 0.45, 0.95), resolution=10)
    led_mesh = _merge_spheres(led_unique, r_led, (0.95, 0.75, 0.12), resolution=10)

    geoms_static = [mat_pcd, frame, cam_mesh, led_mesh]
    if len(cam_rings.lines) > 0:
        geoms_static.append(cam_rings)
    if len(led_rings.lines) > 0:
        geoms_static.append(led_rings)

    def selection_geoms(k: int):
        k = int(np.clip(k, 0, K - 1))
        sc = _sphere_at(cam_pos[k], r_sel, (0.2, 1.0, 1.0), resolution=12)
        sl = _sphere_at(light_pos[k], r_sel, (1.0, 0.25, 1.0), resolution=12)
        pts = np.vstack([centroid, cam_pos[k], centroid, light_pos[k]]).astype(np.float64)
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector([[0, 1], [2, 3]]),
        )
        ls.colors = o3d.utility.Vector3dVector([[0.2, 0.9, 0.9], [0.95, 0.3, 0.95]])
        return sc, sl, ls

    sel_cam, sel_led, lines = selection_geoms(0)
    state: dict = {"k": 0, "sel_cam": sel_cam, "sel_led": sel_led, "lines": lines}

    plt.ion()
    fig_img, ax_img = plt.subplots(1, 1, figsize=(6.0, 6.0), num=f"Bonn poly — mat{mat_id:04d}")
    try:
        fig_img.canvas.manager.set_window_title(f"Bonn poly image — mat{mat_id:04d}")
    except Exception:
        pass
    im_artist = ax_img.imshow(display_stack[0])
    ax_img.axis("off")
    _im_caption = "linear (no γ)" if display_color == "linear" else "tone-mapped"
    ax_img.set_title(_im_caption, fontsize=9, pad=4)
    title = fig_img.suptitle(f"[0 / {K}]  {labels[0]}", fontsize=10, fontfamily="monospace")

    def refresh_image(k: int) -> None:
        k = int(np.clip(k, 0, K - 1))
        im_artist.set_data(display_stack[k])
        title.set_text(f"[{k} / {K}]  {labels[k]}")
        fig_img.canvas.draw_idle()
        fig_img.canvas.flush_events()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"Bonn 3D — mat{mat_id:04d}  |  ←→ images  drag=orbit  wheel=zoom",
        width=1280,
        height=900,
    )
    ro = vis.get_render_option()
    ro.background_color = np.array([0.08, 0.10, 0.16])
    ro.point_size = 3.0
    ro.line_width = 3.0

    for g in geoms_static:
        vis.add_geometry(g, reset_bounding_box=True)
    vis.add_geometry(state["sel_cam"], reset_bounding_box=False)
    vis.add_geometry(state["sel_led"], reset_bounding_box=False)
    vis.add_geometry(state["lines"], reset_bounding_box=False)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.65)

    def apply_view_index(k: int) -> None:
        k = int(np.clip(k, 0, K - 1))
        vis.remove_geometry(state["sel_cam"], reset_bounding_box=False)
        vis.remove_geometry(state["sel_led"], reset_bounding_box=False)
        vis.remove_geometry(state["lines"], reset_bounding_box=False)
        state["k"] = k
        state["sel_cam"], state["sel_led"], state["lines"] = selection_geoms(k)
        vis.add_geometry(state["sel_cam"], reset_bounding_box=False)
        vis.add_geometry(state["sel_led"], reset_bounding_box=False)
        vis.add_geometry(state["lines"], reset_bounding_box=False)
        refresh_image(k)
        vis.poll_events()
        vis.update_renderer()

    def next_view(_vis) -> bool:
        apply_view_index((state["k"] + 1) % K)
        return False

    def prev_view(_vis) -> bool:
        apply_view_index((state["k"] - 1 + K) % K)
        return False

    def help_keys(_vis) -> bool:
        _print_help_keys()
        return False

    vis.register_key_callback(_GLFW_KEY_LEFT, prev_view)
    vis.register_key_callback(_GLFW_KEY_RIGHT, next_view)
    vis.register_key_callback(ord("]"), next_view)
    vis.register_key_callback(ord("N"), next_view)
    vis.register_key_callback(ord("["), prev_view)
    vis.register_key_callback(ord("P"), prev_view)
    vis.register_key_callback(ord("H"), help_keys)

    for digit in range(10):

        def make_jump(d):
            def jump(_vis):
                apply_view_index(min(d * 10, K - 1))
                return False

            return jump

        vis.register_key_callback(ord(str(digit)), make_jump(digit))

    def on_image_window_key(ev) -> None:
        k = (ev.key or "").lower()
        if k in ("left", "arrowleft"):
            apply_view_index((state["k"] - 1 + K) % K)
        elif k in ("right", "arrowright"):
            apply_view_index((state["k"] + 1) % K)

    fig_img.canvas.mpl_connect("key_press_event", on_image_window_key)

    _print_help_keys()
    refresh_image(0)

    vis.run()
    plt.close(fig_img)


def save_image_only_preview(
    root_folder: Path,
    mat_id: int,
    save_preview: Path,
    display_color: ColorDisplayMode = "tonemap",
    linear_percentile: float = 99.5,
) -> None:
    """When Open3D cannot run headless, save only the first-view image panel."""
    from matplotlib import pyplot as plt

    from Bonn_visualizer.load_material import load_bonn_poly_material

    data = load_bonn_poly_material(root_folder, mat_id)
    rgb_stack = data["rgb_stack"]
    K = data["n_views"]
    img0 = hdr_to_display_uint8(
        rgb_stack[0], display_color, linear_percentile=linear_percentile
    )
    mode_s = "linear" if display_color == "linear" else "tonemap"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img0)
    ax.axis("off")
    ax.set_title(f"mat{mat_id:04d} view 0/{K}  ({mode_s})")
    save_preview = Path(save_preview)
    save_preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_preview, dpi=140, bbox_inches="tight")
    plt.close(fig)
