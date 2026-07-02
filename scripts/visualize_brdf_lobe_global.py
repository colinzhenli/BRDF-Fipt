"""
Polar BRDF lobe (fix wo, vary wi; trainer Viz4) rendered in a *global* frame so
the lobes are comparable across checkpoints with no frame/latent gauge ambiguity.

See trainers/stage2_trainer.py::visualize_brdf_lobe for the original renderer.
The original evaluates the decoder in each model's own canonical local frame and
discards the predicted frame.  Here, per checkpoint we read the predicted
(normal, tangent) from the latent's last 6 dims, build wi/wo in a single shared
ANCHOR frame (the MAIN checkpoint's predicted surface frame for the point),
convert to world, then into each checkpoint's OWN predicted local frame before
decoding.  Plots use the shared (global) angles -> gauge-equivalent models render
identically; only genuine world-space differences show.

Four checkpoints are compared.  Three (Ours / NearZeroBRDF / ZeroExact) share the
post-recapture point indexing; one (No-grazing) was trained on an earlier dataset
state with a DIFFERENT global point indexing.  Because the 10% subsample is seeded
per-material by material_id alone (utils/dataset/points.py:415), any material whose
total point count is unchanged maps identically: g_old = offset_old[mat] + local_id
where (mat, local_id) come from offset_new.  We restrict comparison points to such
"safe" (count-unchanged) materials, so all four panels show the same physical point.

We render ONLY Viz4 (polar BRDF*cos, fix wo vary wi), one high-resolution figure
PER checkpoint, with the radial scale shared across the four for comparability.
"""

import os
import sys
import gc
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.neural_brdf_refactored import BRDFDecoder  # noqa: E402


STAGE1 = "/media/raid/cloth/output/BRDF/Stage-1-Finals"
DATASET_DIR = "/media/raid/cloth/capture_data/Dataset_Nov11"
# (name, path, indexing)  indexing: "new" = post-recapture bank, "old" = No-grazing bank
CHECKPOINTS = [
    ("Ours",         os.path.join(STAGE1, "Ours.ckpt"),                          "new"),
    ("NearZeroBRDF", os.path.join(STAGE1, "Ablations/Ours_NearZeroBRDF.ckpt"),   "new"),
    ("ZeroExact",    os.path.join(STAGE1, "Ablations/Ours_ZeroExact.ckpt"),      "new"),
    ("No-grazing",   os.path.join(STAGE1, "Ablations/Ours_480K_Cosine_No-grazing-angle.ckpt"), "old"),
]
ANCHOR_NAME = "Ours"
CKPT_COLORS = {"Ours": "#1f77b4", "NearZeroBRDF": "#d62728",
               "ZeroExact": "#2ca02c", "No-grazing": "#9467bd"}


# ----------------------------------------------------------------------------
# Frame helpers (replicated EXACTLY from MultiMaterialLatentBRDF).
# ----------------------------------------------------------------------------
def extract_frame_from_latent(latent):
    n = F.normalize(latent[..., -6:-3], dim=-1)
    t = F.normalize(latent[..., -3:], dim=-1)
    t = t - (t * n).sum(-1, keepdim=True) * n
    t = F.normalize(t, dim=-1)
    return n, t


def world_to_local(v, normal, tangent):
    tangent = tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-8)
    bitangent = torch.cross(normal, tangent, dim=-1)
    return torch.stack(
        [(v * tangent).sum(-1), (v * bitangent).sum(-1), (v * normal).sum(-1)], dim=-1)


def anchor_to_local(anchor_dir, anchor, n, t):
    aT, aB, aN = anchor
    world = anchor_dir[..., 0:1] * aT + anchor_dir[..., 1:2] * aB + anchor_dir[..., 2:3] * aN
    M = world.shape[0]
    return world_to_local(world, n.expand(M, 3), t.expand(M, 3))


def local_normal_like(x):
    ln = torch.zeros_like(x); ln[..., 2] = 1.0
    return ln


# ----------------------------------------------------------------------------
# Old<->new global point id remap (uses the saved material_offset_tensor).
# ----------------------------------------------------------------------------
class Remap:
    """Map a NEW global point id -> OLD global id for the same physical point.

    Valid only for materials whose total point count is unchanged between the two
    dataset states (per-material subsample seed = material_id, so unchanged count
    => identical local-id ordering).  Returns None for changed/unsafe materials.
    """
    def __init__(self, off_new, off_old, train, N_new, N_old):
        self.train = train
        self.on = np.array([int(off_new[m]) for m in train], dtype=np.int64)
        self.oo = np.array([int(off_old[m]) for m in train], dtype=np.int64)
        self.cn = np.append(np.diff(self.on), N_new - self.on[-1])
        self.co = np.append(np.diff(self.oo), N_old - self.oo[-1])
        self.n_changed = int((self.cn != self.co).sum())

    def _locate(self, g_new):
        i = int(np.searchsorted(self.on, g_new, side="right")) - 1
        if i < 0 or g_new - self.on[i] >= self.cn[i]:
            return None
        return i, int(g_new - self.on[i])

    def is_safe(self, g_new):
        loc = self._locate(g_new)
        return loc is not None and self.cn[loc[0]] == self.co[loc[0]]

    def to_old(self, g_new):
        loc = self._locate(g_new)
        if loc is None:
            return None
        i, local = loc
        if self.cn[i] != self.co[i]:
            return None
        return int(self.oo[i] + local), int(self.train[i])


# ----------------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------------
def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mat = ckpt["hyper_parameters"].material
    latent_dim = int(mat.latent_dim)
    brdf_latent_dim = latent_dim * 3 if bool(mat.different_decoder) else latent_dim
    decoder = BRDFDecoder(cfg=mat.decoder, latent_dim=latent_dim,
                          use_pos_enc=bool(mat.use_pos_enc),
                          different_decoder=bool(mat.different_decoder))
    prefix = "material.decoder."
    decoder.load_state_dict({k[len(prefix):]: v for k, v in ckpt["state_dict"].items()
                             if k.startswith(prefix)}, strict=False)
    decoder.eval().to(device)
    for p in decoder.parameters():
        p.requires_grad_(False)
    bank = ckpt["state_dict"]["material.point_latent_bank.weight"]
    offset = ckpt["state_dict"]["material.material_offset_tensor"].numpy()
    del ckpt
    gc.collect()
    return decoder, bank, latent_dim, brdf_latent_dim, offset


@torch.no_grad()
def decode_mean(d, wi_local, wo_local):
    enc = d["decoder"].encode_directions(wi_local, wo_local, local_normal_like(wi_local))
    return d["decoder"](enc, d["brdf_lat"].expand(wi_local.shape[0], -1)).mean(-1)


# ----------------------------------------------------------------------------
# Point selection (anchor checkpoint), restricted to SAFE materials.
# ----------------------------------------------------------------------------
def select_points(decoder, bank, brdf_latent_dim, remap, device, num_points, seed,
                  pool=8000, min_gap=100_000, avoid=()):
    avoid = list(avoid)
    g = torch.Generator().manual_seed(seed)
    cand = torch.randint(0, bank.shape[0], (pool,), generator=g).numpy()
    cand = np.array([int(c) for c in cand if remap.is_safe(int(c))], dtype=np.int64)
    lat = bank[cand].to(device)
    brdf_lat = lat[:, :brdf_latent_dim]
    res = 24
    th = torch.linspace(0, np.pi / 2, res); ph = torch.linspace(0, 2 * np.pi, res)
    TH, PH = torch.meshgrid(th, ph, indexing="ij")
    wi = torch.stack([torch.sin(TH) * torch.cos(PH), torch.sin(TH) * torch.sin(PH),
                      torch.cos(TH)], -1).reshape(-1, 3).to(device)
    wo = torch.tensor([[np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)]],
                      device=device).expand(wi.shape[0], 3)
    ln = local_normal_like(wi)
    scores = np.zeros(len(cand))
    with torch.no_grad():
        enc = decoder.encode_directions(wi, wo, ln)
        for i in range(len(cand)):
            b = decoder(enc, brdf_lat[i:i + 1].expand(wi.shape[0], -1)).mean(-1)
            scores[i] = float(b.max() / (b.mean() + 1e-8)) if torch.isfinite(b).all() else -1
    order = np.argsort(scores)  # ascending specularity
    order = order[scores[order] > 0]
    n = len(order)
    chosen_gids, used = [], set()
    for p in np.linspace(0.45, 0.97, num_points):
        start = int(round(p * (n - 1)))
        for off in range(n):
            placed = False
            for j in (start + off, start - off):
                if 0 <= j < n and j not in used:
                    gid = int(cand[order[j]])
                    if all(abs(gid - g2) >= min_gap for g2 in chosen_gids + avoid):
                        used.add(j); chosen_gids.append(gid); placed = True; break
            if placed:
                break
    print(f"  selected NEW gids: {chosen_gids}")
    return chosen_gids


# ----------------------------------------------------------------------------
# Viz4 : polar BRDF*cos, fix theta_o vary theta_i.  One figure PER checkpoint.
# ----------------------------------------------------------------------------
def viz4_per_checkpoint(point_data, anchor, g_new, p_idx, out_dir, resolution=128,
                        dpi=300, theta_o_values=(15.0, 30.0, 45.0, 60.0)):
    device = anchor[0].device
    ti = np.linspace(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, resolution * 2)
    wi_np = np.zeros((len(ti), 3), np.float32)
    pos = ti >= 0
    wi_np[pos, 0] = -np.sin(ti[pos]); wi_np[pos, 2] = np.cos(ti[pos])
    wi_np[~pos, 0] = np.sin(-ti[~pos]); wi_np[~pos, 2] = np.cos(-ti[~pos])
    wi_anchor = torch.from_numpy(wi_np).to(device)

    # compute all curves first -> shared radial scale across checkpoints
    all_curves, rmax = {}, 0.0
    for d in point_data:
        cset = []
        for to_deg in theta_o_values:
            to = np.radians(to_deg)
            wo_anchor = torch.tensor([[np.sin(to), 0.0, np.cos(to)]],
                                     device=device).expand(len(ti), 3)
            wi_l = anchor_to_local(wi_anchor, anchor, d["n"], d["t"])
            wo_l = anchor_to_local(wo_anchor, anchor, d["n"], d["t"])
            cos_i = wi_l[:, 2].clamp(min=0).cpu().numpy()
            b = decode_mean(d, wi_l, wo_l).cpu().numpy() * cos_i
            cset.append((to_deg, b)); rmax = max(rmax, float(b.max()))
        all_curves[d["name"]] = cset
    rmax *= 1.05

    pdir = os.path.join(out_dir, f"point{p_idx}_gid{g_new}")
    os.makedirs(pdir, exist_ok=True)
    for d in point_data:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="polar")
        for to_deg, b in all_curves[d["name"]]:
            line, = ax.plot(ti, b, lw=2.2, label=f"θ_o={to_deg:.0f}°")
            ax.axvline(x=np.radians(to_deg), color=line.get_color(), ls="--", alpha=0.45)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(1)
        ax.set_thetamin(-90); ax.set_thetamax(90)
        ax.set_rmax(rmax)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_rlabel_position(255); ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=10)
        gid_txt = f"gid={d['gid']}" + ("" if d["indexing"] == "new" else f" (old idx; new={g_new})")
        title_col = CKPT_COLORS.get(d["name"], "k")
        ax.set_title(f"{d['name']}   [{d['indexing']} idx]\npoint #{p_idx}  {gid_txt}\n"
                     f"BRDF×cos, fix wo vary wi | 0°=anchor N, dashed=specular",
                     fontsize=12, color=title_col)
        ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.08), fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, f"{d['name']}.png"), dpi=dpi)
        plt.close(fig)
    return pdir


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",
                    default="/media/raid/cloth/output/BRDF/Qualitative/Ablations/Grazing_loss")
    ap.add_argument("--num_points", type=int, default=5)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start_idx", type=int, default=0,
                    help="starting point index for naming; appends to existing manifest")
    ap.add_argument("--gids", type=int, nargs="*", default=None, help="explicit NEW gids")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={device}  out_dir={args.out_dir}")

    loaded = {}
    for name, path, indexing in CHECKPOINTS:
        print(f"loading {name} ({indexing}) ...")
        dec, bank, ldim, bdim, offset = load_checkpoint(path, device)
        loaded[name] = dict(decoder=dec, bank=bank, brdf_latent_dim=bdim,
                            offset=offset, indexing=indexing)

    # build remap from anchor(new) and the old-indexed checkpoint
    train = [int(x) for x in open(os.path.join(DATASET_DIR, "training_list_500.txt")).read().split()]
    old_name = next(n for n, _, idx in CHECKPOINTS if idx == "old")
    remap = Remap(loaded[ANCHOR_NAME]["offset"], loaded[old_name]["offset"], train,
                  loaded[ANCHOR_NAME]["bank"].shape[0], loaded[old_name]["bank"].shape[0])
    print(f"remap: {remap.n_changed}/{len(train)} materials changed (unsafe); "
          f"identity-safe pool < g={int(remap.on[np.argmax(remap.on != remap.oo)])}")

    # load existing manifest to append to / avoid duplicate points
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    if args.start_idx > 0 and os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        manifest.setdefault("points", [])
    else:
        manifest = {"checkpoints": {n: p for n, p, _ in CHECKPOINTS}, "anchor": ANCHOR_NAME,
                    "remap_note": "No-grazing uses OLD indexing; g_old=offset_old[mat]+local_id",
                    "points": []}
    existing_gids = [int(p["new_gid"]) for p in manifest["points"]]

    if args.gids:
        gids = [g for g in args.gids if remap.is_safe(g)]
        skipped = [g for g in args.gids if not remap.is_safe(g)]
        if skipped:
            print(f"  WARNING: dropped unsafe (changed-material) gids: {skipped}")
    else:
        print(f"selecting {args.num_points} points (safe materials only, "
              f"avoiding {len(existing_gids)} existing) ...")
        gids = select_points(loaded[ANCHOR_NAME]["decoder"], loaded[ANCHOR_NAME]["bank"],
                             loaded[ANCHOR_NAME]["brdf_latent_dim"], remap, device,
                             args.num_points, args.seed, avoid=existing_gids)

    for i, g_new in enumerate(gids):
        p_idx = args.start_idx + i
        g_old, mat_id = remap.to_old(g_new)
        a_lat = loaded[ANCHOR_NAME]["bank"][g_new:g_new + 1].to(device)
        a_n, a_t = extract_frame_from_latent(a_lat)
        anchor = (a_t[0], torch.cross(a_n[0], a_t[0], dim=-1), a_n[0])

        point_data, rec = [], {"new_gid": int(g_new), "old_gid": int(g_old),
                               "material_id": mat_id, "normal_tilt_vs_anchor_deg": {}}
        for name, _, indexing in CHECKPOINTS:
            gid = g_new if indexing == "new" else g_old
            lat = loaded[name]["bank"][gid:gid + 1].to(device)
            n, t = extract_frame_from_latent(lat)
            bdim = loaded[name]["brdf_latent_dim"]
            point_data.append(dict(name=name, indexing=indexing, gid=int(gid),
                                   decoder=loaded[name]["decoder"],
                                   brdf_lat=lat[:, :bdim], n=n[0], t=t[0]))
            cosang = float((n[0] * a_n[0]).sum().clamp(-1, 1).cpu())
            rec["normal_tilt_vs_anchor_deg"][name] = round(float(np.degrees(np.arccos(cosang))), 2)
        manifest["points"].append(rec)

        pdir = viz4_per_checkpoint(point_data, anchor, g_new, p_idx, args.out_dir,
                                   resolution=args.resolution, dpi=args.dpi)
        print(f"[point {p_idx}] new_gid={g_new} -> old_gid={g_old} (mat {mat_id})  "
              f"normals vs anchor: "
              f"{ {k: rec['normal_tilt_vs_anchor_deg'][k] for k in ['NearZeroBRDF','ZeroExact','No-grazing']} }  "
              f"-> {os.path.basename(pdir)}/")

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. {len(gids)} points x {len(CHECKPOINTS)} checkpoints -> {args.out_dir}")


if __name__ == "__main__":
    main()
