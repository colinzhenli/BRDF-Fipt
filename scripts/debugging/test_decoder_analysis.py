#!/usr/bin/env python
"""
Diagnostic script for analyzing pretrained BRDF decoders and optimized latents.

Tests:
  1. ReLU vs LeakyReLU intermediate activation mismatch
  2. Decoder output distribution (positive/negative range)
  3. Angular coverage: training data vs Bonn data (same-half-space issue)
  4. Comparison of two checkpoints (Real-pretrained vs Bonn-pretrained decoders)
  5. Specular highlight capability of each decoder

Usage:
    conda run -n fipt_copy python scripts/debugging/test_decoder_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as NF
import numpy as np
import math
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DECODER_ONLY_CKPT = (
    "/media/raid/cloth/output/BRDF/points/"
    "All-materials_training_Single-color-head_run_1/training/"
    "model_0.20_0.20/last_decoder_only.ckpt"
)
BONN_FROM_REAL_CKPT = (
    "/media/raid/cloth/output/BRDF/Bonn-Theia2/"
    "Stage-2_Bonn-1_3D-factor_Logrel_Lr-0.01_Decoder-lr-1e-2_From-Real_run_1/"
    "training/model_0.20_0.20/last.ckpt"
)
BONN_FROM_BONN_CKPT = (
    "/media/raid/cloth/output/BRDF/Bonn-Theia2/"
    "Stage-2_Bonn-1_from_Fir-Bonn_stage1-trainer_run_1/"
    "training/model_0.20_0.20/last.ckpt"
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers: build a BRDFDecoder with explicit activation choice
# ---------------------------------------------------------------------------
from model.neural_brdf_refactored import (
    BRDFDecoder,
    components_from_spherical_harmonics,
    num_sh_bases,
)


class DecoderConfig:
    """Minimal config object to construct BRDFDecoder."""
    def __init__(self, hidden_layers, activation, degree=3, output_channels=3,
                 use_skip_connection=True, skip_layer=2, use_film=False,
                 use_color_decomp=False, color_latent_dim=6):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_channels = output_channels
        self.degree = degree
        self.use_skip_connection = use_skip_connection
        self.skip_layer = skip_layer
        self.use_film = use_film
        self.use_color_decomp = use_color_decomp
        self.color_latent_dim = color_latent_dim


def load_decoder_state(ckpt_path, prefix="material.decoder."):
    """Load decoder weights from a checkpoint (strip the prefix)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    decoder_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith(prefix):
            decoder_sd[k[len(prefix):]] = v
    return decoder_sd


def load_latent_bank(ckpt_path, key="material.point_latent_bank.weight"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    if key in sd:
        return sd[key]
    return None


def load_factor(ckpt_path, key="material.factor"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    if key in sd:
        return sd[key]
    return None


def build_decoder(hidden_layers, activation, latent_dim=16):
    cfg = DecoderConfig(
        hidden_layers=hidden_layers,
        activation=activation,
        degree=3,
        output_channels=3,
        use_skip_connection=True,
        skip_layer=2,
    )
    return BRDFDecoder(cfg, latent_dim=latent_dim, use_pos_enc=True, different_decoder=False)


def make_directions_grid(n_theta=64, n_phi=64):
    """Generate a grid of upper-hemisphere directions (local space)."""
    theta = torch.linspace(0.01, math.pi / 2 - 0.01, n_theta)
    phi = torch.linspace(0, 2 * math.pi, n_phi)
    TH, PH = torch.meshgrid(theta, phi, indexing="ij")
    dirs = torch.stack([
        torch.sin(TH) * torch.cos(PH),
        torch.sin(TH) * torch.sin(PH),
        torch.cos(TH),
    ], dim=-1)
    return dirs.reshape(-1, 3)


# =====================================================================
# TEST 1: ReLU vs LeakyReLU intermediate activation mismatch
# =====================================================================
def test_relu_vs_leakyrelu_mismatch():
    """
    The pretrained decoder (merl branch) used ReLU as intermediate activation
    in _forward_with_skip, while the current bonn_no_chunk branch uses LeakyReLU.
    This test loads the same weights into both configurations and measures
    the output discrepancy.
    """
    print("\n" + "=" * 72)
    print("TEST 1: ReLU vs LeakyReLU intermediate activation mismatch")
    print("=" * 72)

    decoder_sd = load_decoder_state(DECODER_ONLY_CKPT)
    print(f"  Loaded {len(decoder_sd)} decoder parameters from decoder-only ckpt")

    inferred_hidden = []
    i = 0
    while f"mlp.{i}.weight" in decoder_sd:
        w = decoder_sd[f"mlp.{i}.weight"]
        inferred_hidden.append(w.shape[0])
        i += 1
    last_layer_idx = i - 1
    inferred_hidden.pop()
    print(f"  Inferred hidden layers: {inferred_hidden}")
    print(f"  Output layer (mlp.{last_layer_idx}): out_features={decoder_sd[f'mlp.{last_layer_idx}.weight'].shape[0]}")

    # Build two decoders: one with the ORIGINAL ReLU, one with LeakyReLU
    # We need to manually swap the activation after building
    decoder_relu = build_decoder(inferred_hidden, "leakyrelu")
    decoder_leaky = build_decoder(inferred_hidden, "leakyrelu")

    decoder_relu.load_state_dict(decoder_sd, strict=False)
    decoder_leaky.load_state_dict(decoder_sd, strict=False)

    # Swap intermediate activation on decoder_relu to ReLU (what merl branch used)
    decoder_relu.activation = nn.ReLU()
    # decoder_leaky keeps nn.LeakyReLU() (what bonn_no_chunk uses)

    decoder_relu.eval()
    decoder_leaky.eval()

    # Generate test directions
    torch.manual_seed(42)
    N = 4096
    wi = NF.normalize(torch.randn(N, 3), dim=-1)
    wo = NF.normalize(torch.randn(N, 3), dim=-1)
    wi[:, 2] = wi[:, 2].abs()
    wo[:, 2] = wo[:, 2].abs()
    wi = NF.normalize(wi, dim=-1)
    wo = NF.normalize(wo, dim=-1)
    normal = torch.zeros(N, 3)
    normal[:, 2] = 1.0
    latent = torch.randn(N, 16) * 0.1

    with torch.no_grad():
        enc = decoder_relu.encode_directions(wi, wo, normal)
        out_relu = decoder_relu(enc, latent)
        out_leaky = decoder_leaky(enc, latent)

    diff = (out_relu - out_leaky).abs()
    rel_diff = diff / (out_relu.abs() + 1e-8)

    print(f"\n  Output stats (ReLU)   : mean={out_relu.mean():.6f}, std={out_relu.std():.6f}, "
          f"min={out_relu.min():.6f}, max={out_relu.max():.6f}")
    print(f"  Output stats (LeakyReLU): mean={out_leaky.mean():.6f}, std={out_leaky.std():.6f}, "
          f"min={out_leaky.min():.6f}, max={out_leaky.max():.6f}")
    print(f"  Absolute diff: mean={diff.mean():.6f}, max={diff.max():.6f}")
    print(f"  Relative diff: mean={rel_diff.mean():.4f}, max={rel_diff.max():.4f}")
    print(f"  Fraction of outputs where diff > 1%: {(rel_diff > 0.01).float().mean():.4f}")
    print(f"  Fraction of outputs where diff > 10%: {(rel_diff > 0.1).float().mean():.4f}")

    # Check where ReLU gives 0 but LeakyReLU gives nonzero (the leak)
    relu_zero_mask = out_relu.abs() < 1e-6
    leaky_at_relu_zero = out_leaky[relu_zero_mask]
    if leaky_at_relu_zero.numel() > 0:
        print(f"\n  Where ReLU output ≈ 0 ({relu_zero_mask.sum()} values):")
        print(f"    LeakyReLU gives: mean={leaky_at_relu_zero.mean():.6f}, "
              f"min={leaky_at_relu_zero.min():.6f}, max={leaky_at_relu_zero.max():.6f}")

    # Check negative outputs
    print(f"\n  Negative output fraction (ReLU): {(out_relu < 0).float().mean():.4f}")
    print(f"  Negative output fraction (LeakyReLU): {(out_leaky < 0).float().mean():.4f}")

    return out_relu, out_leaky


# =====================================================================
# TEST 2: Decoder output distribution (positive/negative range)
# =====================================================================
def test_decoder_output_distribution():
    """
    Probe the pretrained decoder across a dense grid of directions and
    latent codes to see what output range it can produce, and whether
    negative BRDF values are physically meaningful.
    """
    print("\n" + "=" * 72)
    print("TEST 2: Decoder output distribution analysis")
    print("=" * 72)

    decoder_sd = load_decoder_state(DECODER_ONLY_CKPT)
    inferred_hidden = []
    i = 0
    while f"mlp.{i}.weight" in decoder_sd:
        inferred_hidden.append(decoder_sd[f"mlp.{i}.weight"].shape[0])
        i += 1
    inferred_hidden.pop()

    # Build with ReLU (original training activation)
    decoder = build_decoder(inferred_hidden, "leakyrelu")
    decoder.load_state_dict(decoder_sd, strict=False)
    decoder.activation = nn.ReLU()  # match merl branch
    decoder.eval()

    # Also build with LeakyReLU (current code)
    decoder_leaky = build_decoder(inferred_hidden, "leakyrelu")
    decoder_leaky.load_state_dict(decoder_sd, strict=False)
    decoder_leaky.eval()

    torch.manual_seed(0)
    dirs = make_directions_grid(32, 32)  # 1024 directions
    N_dir = dirs.shape[0]

    # Sample latents from different distributions
    latent_stds = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    normal = torch.zeros(N_dir, 3)
    normal[:, 2] = 1.0

    print(f"\n  {'std':>5s} | {'mean_relu':>10s} {'std_relu':>10s} {'min_relu':>10s} {'max_relu':>10s} "
          f"{'neg%_relu':>9s} | {'mean_lrelu':>10s} {'neg%_lrelu':>10s}")
    print("  " + "-" * 100)

    for std in latent_stds:
        latent = torch.randn(N_dir, 16) * std
        with torch.no_grad():
            enc = decoder.encode_directions(dirs, dirs, normal)
            out_relu = decoder(enc, latent)
            out_leaky = decoder_leaky(enc, latent)

        neg_pct_relu = (out_relu < 0).float().mean().item() * 100
        neg_pct_leaky = (out_leaky < 0).float().mean().item() * 100
        print(f"  {std:5.2f} | {out_relu.mean():10.4f} {out_relu.std():10.4f} "
              f"{out_relu.min():10.4f} {out_relu.max():10.4f} {neg_pct_relu:8.2f}% | "
              f"{out_leaky.mean():10.4f} {neg_pct_leaky:8.2f}%")


# =====================================================================
# TEST 3: Angular coverage – training data vs Bonn data
# =====================================================================
def test_angular_coverage():
    """
    Our training data has wi and wo mostly on opposite half-spaces
    (azimuth angle > 90°), while Bonn data has many near-specular
    configurations (azimuth < 90°). This tests the decoder's response
    across these different angular regimes.
    """
    print("\n" + "=" * 72)
    print("TEST 3: Angular coverage — same-half-space vs opposite-half-space")
    print("=" * 72)

    decoder_sd = load_decoder_state(DECODER_ONLY_CKPT)
    inferred_hidden = []
    i = 0
    while f"mlp.{i}.weight" in decoder_sd:
        inferred_hidden.append(decoder_sd[f"mlp.{i}.weight"].shape[0])
        i += 1
    inferred_hidden.pop()

    decoder = build_decoder(inferred_hidden, "leakyrelu")
    decoder.load_state_dict(decoder_sd, strict=False)
    decoder.activation = nn.ReLU()
    decoder.eval()

    torch.manual_seed(42)
    normal = torch.tensor([[0.0, 0.0, 1.0]])

    # Fixed wo (viewer looking down at 30 degrees)
    wo_theta = math.radians(30)
    wo = torch.tensor([[math.sin(wo_theta), 0.0, math.cos(wo_theta)]])

    # Sweep wi across azimuth angles from 0° to 180° in the incident plane
    n_angles = 180
    theta_i_vals = np.linspace(5, 85, 20)  # elevation angles
    phi_i_vals = np.linspace(0, 180, n_angles)  # azimuth angles

    latent = torch.randn(1, 16) * 0.1  # fixed latent

    results = {}
    for theta_i_deg in theta_i_vals:
        theta_i = math.radians(theta_i_deg)
        brdfs_for_theta = []
        for phi_i_deg in phi_i_vals:
            phi_i = math.radians(phi_i_deg)
            wi = torch.tensor([[
                math.sin(theta_i) * math.cos(phi_i),
                math.sin(theta_i) * math.sin(phi_i),
                math.cos(theta_i),
            ]])
            with torch.no_grad():
                enc = decoder.encode_directions(wi, wo.expand(1, -1), normal.expand(1, -1))
                brdf = decoder(enc, latent)
            brdfs_for_theta.append(brdf.squeeze().numpy())
        results[theta_i_deg] = np.array(brdfs_for_theta)

    # Compute the azimuth angle between projected wi and wo in tangent plane
    # phi_diff < 90° = same half-space; phi_diff > 90° = opposite half-space
    print(f"\n  BRDF response at different azimuth angles (phi_diff between wi and wo projections)")
    print(f"  phi_diff < 90° → same half-space (near-specular/retro-reflection)")
    print(f"  phi_diff > 90° → opposite half-space (typical training data regime)")
    print()
    print(f"  {'theta_i':>7s} | {'same-half mean':>14s} {'same-half max':>13s} | "
          f"{'opp-half mean':>13s} {'opp-half max':>13s} | {'ratio':>6s}")
    print("  " + "-" * 80)

    for theta_i_deg in theta_i_vals:
        arr = results[theta_i_deg]
        mean_rgb = arr.mean(axis=-1)  # average over RGB

        same_half = mean_rgb[:n_angles // 2]   # phi 0-90°
        opp_half = mean_rgb[n_angles // 2:]    # phi 90-180°

        same_mean = same_half.mean()
        same_max = same_half.max()
        opp_mean = opp_half.mean()
        opp_max = opp_half.max()
        ratio = same_mean / (opp_mean + 1e-8)

        print(f"  {theta_i_deg:7.1f} | {same_mean:14.4f} {same_max:13.4f} | "
              f"{opp_mean:13.4f} {opp_max:13.4f} | {ratio:6.2f}x")

    # Also check the specular peak direction (mirror reflection)
    print(f"\n  Specular peak test (mirror reflection of wo):")
    wo_vec = wo.squeeze()
    wi_spec = torch.tensor([-wo_vec[0], -wo_vec[1], wo_vec[2]]).unsqueeze(0)
    with torch.no_grad():
        enc = decoder.encode_directions(wi_spec, wo, normal)
        brdf_spec = decoder(enc, latent)
    print(f"    wo = {wo.squeeze().tolist()}")
    print(f"    wi_specular = {wi_spec.squeeze().tolist()}")
    print(f"    BRDF at specular = {brdf_spec.squeeze().tolist()}")

    # And a nearby-specular direction (5° off)
    eps = math.radians(5)
    wi_near_spec = NF.normalize(wi_spec + torch.tensor([[eps, 0, 0]]), dim=-1)
    with torch.no_grad():
        enc = decoder.encode_directions(wi_near_spec, wo, normal)
        brdf_near = decoder(enc, latent)
    print(f"    BRDF near specular (5° off) = {brdf_near.squeeze().tolist()}")


# =====================================================================
# TEST 4: Compare two checkpoints — latent and decoder distributions
# =====================================================================
def test_compare_checkpoints():
    """
    Compare the checkpoint trained with Real-pretrained decoder vs
    the checkpoint trained with Bonn-pretrained decoder.
    Analyze latent distributions, decoder weight differences, and
    BRDF output distributions.
    """
    print("\n" + "=" * 72)
    print("TEST 4: Checkpoint comparison (Real-pretrained vs Bonn-pretrained)")
    print("=" * 72)

    # --- Load latent banks ---
    latent_real = load_latent_bank(BONN_FROM_REAL_CKPT)
    latent_bonn = load_latent_bank(BONN_FROM_BONN_CKPT)
    factor_real = load_factor(BONN_FROM_REAL_CKPT)
    factor_bonn = load_factor(BONN_FROM_BONN_CKPT)

    print(f"\n  Latent bank (from-Real): shape={latent_real.shape}")
    print(f"  Latent bank (from-Bonn): shape={latent_bonn.shape}")

    if factor_real is not None:
        print(f"  Learnable factor (from-Real): {factor_real.tolist()}")
    if factor_bonn is not None:
        print(f"  Learnable factor (from-Bonn): {factor_bonn.tolist()}")

    # Latent distribution stats (BRDF part only, first N-6 dims)
    for name, lat in [("from-Real", latent_real), ("from-Bonn", latent_bonn)]:
        brdf_lat = lat[:, :-6]
        normal_lat = lat[:, -6:-3]
        tangent_lat = lat[:, -3:]
        print(f"\n  [{name}] BRDF latent (first {brdf_lat.shape[1]} dims):")
        print(f"    mean={brdf_lat.mean():.6f}, std={brdf_lat.std():.6f}, "
              f"min={brdf_lat.min():.6f}, max={brdf_lat.max():.6f}")
        print(f"    Per-dim std: {brdf_lat.std(dim=0).tolist()}")
        print(f"    Normal slot: mean_z={normal_lat[:, 2].mean():.4f}, "
              f"std_z={normal_lat[:, 2].std():.4f}")

    # --- Load decoder weights ---
    dec_sd_real = load_decoder_state(BONN_FROM_REAL_CKPT)
    dec_sd_bonn = load_decoder_state(BONN_FROM_BONN_CKPT)

    print(f"\n  Decoder keys (from-Real): {len(dec_sd_real)}")
    print(f"  Decoder keys (from-Bonn): {len(dec_sd_bonn)}")

    # Print shapes of all layers for both
    print(f"\n  Decoder architecture (from-Real):")
    for k, v in dec_sd_real.items():
        if "weight" in k:
            print(f"    {k}: {list(v.shape)}")
    print(f"  Decoder architecture (from-Bonn):")
    for k, v in dec_sd_bonn.items():
        if "weight" in k:
            print(f"    {k}: {list(v.shape)}")

    # --- Build decoders and evaluate ---
    # Real decoder: [256, 128, 128, 64, 32], leakyrelu, trained with ReLU intermediates
    h_real = []
    i = 0
    while f"mlp.{i}.weight" in dec_sd_real:
        h_real.append(dec_sd_real[f"mlp.{i}.weight"].shape[0])
        i += 1
    h_real.pop()

    decoder_real = build_decoder(h_real, "leakyrelu")
    decoder_real.load_state_dict(dec_sd_real, strict=False)
    decoder_real.activation = nn.ReLU()
    decoder_real.eval()

    # Bonn decoder: [256, 256, 256, 256], softplus, latent_dim=24
    h_bonn = []
    i = 0
    while f"mlp.{i}.weight" in dec_sd_bonn:
        h_bonn.append(dec_sd_bonn[f"mlp.{i}.weight"].shape[0])
        i += 1
    h_bonn.pop()

    # Infer latent dim from the first layer: input_dim = encoded_dim + latent_dim
    # SH degree 3 → sh_dim = num_sh_bases(3) = 16, encoded_dim = 16*3 = 48
    bonn_input_dim = dec_sd_bonn["mlp.0.weight"].shape[1]
    bonn_latent_dim = bonn_input_dim - 48  # 72 - 48 = 24
    print(f"  Bonn decoder inferred latent_dim={bonn_latent_dim} (input_dim={bonn_input_dim})")

    decoder_bonn = build_decoder(h_bonn, "softplus", latent_dim=bonn_latent_dim)
    decoder_bonn.load_state_dict(dec_sd_bonn, strict=False)
    decoder_bonn.eval()

    print(f"\n  Real decoder hidden: {h_real}, activation intermediates=ReLU, output=LeakyReLU")
    print(f"  Bonn decoder hidden: {h_bonn}, activation intermediates=LeakyReLU, output=Softplus")

    # Evaluate both decoders on a sweep of specular-like directions
    N = 2048
    torch.manual_seed(42)
    normal = torch.zeros(N, 3); normal[:, 2] = 1.0

    # Near-specular directions: wi close to mirror of wo
    theta_o = torch.rand(N) * math.pi / 3
    wo = torch.stack([torch.sin(theta_o), torch.zeros(N), torch.cos(theta_o)], dim=-1)
    wi_spec = torch.stack([-torch.sin(theta_o), torch.zeros(N), torch.cos(theta_o)], dim=-1)
    wi_perturb = NF.normalize(wi_spec + torch.randn(N, 3) * 0.05, dim=-1)
    wi_perturb[:, 2] = wi_perturb[:, 2].abs()
    wi_perturb = NF.normalize(wi_perturb, dim=-1)

    # Use ACTUAL latents from each checkpoint
    idx_real = torch.randint(0, latent_real.shape[0], (N,))
    brdf_lat_real = latent_real[idx_real, :-6]
    idx_bonn = torch.randint(0, latent_bonn.shape[0], (N,))
    brdf_lat_bonn = latent_bonn[idx_bonn, :decoder_bonn.latent_dim]

    with torch.no_grad():
        enc_r = decoder_real.encode_directions(wi_perturb, wo, normal)
        out_real = decoder_real(enc_r, brdf_lat_real)
        enc_b = decoder_bonn.encode_directions(wi_perturb, wo, normal)
        out_bonn = decoder_bonn(enc_b, brdf_lat_bonn)

    # Apply learnable factors if present
    if factor_real is not None:
        out_real_factored = out_real * factor_real
    else:
        out_real_factored = out_real

    if factor_bonn is not None:
        out_bonn_factored = out_bonn * factor_bonn
    else:
        out_bonn_factored = out_bonn

    print(f"\n  Near-specular BRDF output (decoder only, no factor):")
    for name, out in [("Real-decoder", out_real), ("Bonn-decoder", out_bonn)]:
        print(f"    [{name}] mean={out.mean():.4f}, std={out.std():.4f}, "
              f"min={out.min():.4f}, max={out.max():.4f}, "
              f"neg%={((out < 0).float().mean() * 100):.1f}%")

    print(f"\n  Near-specular BRDF output (with learnable factor applied):")
    for name, out in [("Real-decoder+factor", out_real_factored), ("Bonn-decoder+factor", out_bonn_factored)]:
        print(f"    [{name}] mean={out.mean():.4f}, std={out.std():.4f}, "
              f"min={out.min():.4f}, max={out.max():.4f}, "
              f"neg%={((out < 0).float().mean() * 100):.1f}%")

    # Now test with OFF-specular (typical training data) directions
    phi_offset = torch.rand(N) * math.pi + math.pi / 2
    wi_off = torch.stack([
        torch.sin(theta_o) * torch.cos(phi_offset),
        torch.sin(theta_o) * torch.sin(phi_offset),
        torch.cos(theta_o),
    ], dim=-1)
    wi_off = NF.normalize(wi_off, dim=-1)
    wi_off[:, 2] = wi_off[:, 2].abs()
    wi_off = NF.normalize(wi_off, dim=-1)

    with torch.no_grad():
        enc_r2 = decoder_real.encode_directions(wi_off, wo, normal)
        out_real_off = decoder_real(enc_r2, brdf_lat_real)
        enc_b2 = decoder_bonn.encode_directions(wi_off, wo, normal)
        out_bonn_off = decoder_bonn(enc_b2, brdf_lat_bonn)

    if factor_real is not None:
        out_real_off_f = out_real_off * factor_real
    else:
        out_real_off_f = out_real_off

    print(f"\n  Off-specular (training-data-like) BRDF output (with factor):")
    for name, out in [("Real-decoder+factor", out_real_off_f), ("Bonn-decoder", out_bonn_off)]:
        print(f"    [{name}] mean={out.mean():.4f}, std={out.std():.4f}, "
              f"min={out.min():.4f}, max={out.max():.4f}")

    # Ratio: how much bigger is specular vs off-specular?
    spec_ratio_real = out_real_factored.abs().mean() / (out_real_off_f.abs().mean() + 1e-8)
    spec_ratio_bonn = out_bonn_factored.abs().mean() / (out_bonn_off.abs().mean() + 1e-8)
    print(f"\n  Specular / off-specular ratio:")
    print(f"    Real decoder: {spec_ratio_real:.2f}x")
    print(f"    Bonn decoder: {spec_ratio_bonn:.2f}x")


# =====================================================================
# TEST 5: Specular highlight capability — BRDF peak analysis
# =====================================================================
def test_specular_capability():
    """
    For each decoder, sweep wi around the specular peak for various
    wo directions and measure the maximum BRDF value (highlight intensity).
    This directly tests whether the decoder can represent sharp highlights.
    """
    print("\n" + "=" * 72)
    print("TEST 5: Specular highlight capability — peak BRDF analysis")
    print("=" * 72)

    # Load both decoders
    dec_sd_real = load_decoder_state(BONN_FROM_REAL_CKPT)
    dec_sd_bonn = load_decoder_state(BONN_FROM_BONN_CKPT)
    latent_real = load_latent_bank(BONN_FROM_REAL_CKPT)
    latent_bonn = load_latent_bank(BONN_FROM_BONN_CKPT)
    factor_real = load_factor(BONN_FROM_REAL_CKPT)

    h_real = []
    i = 0
    while f"mlp.{i}.weight" in dec_sd_real:
        h_real.append(dec_sd_real[f"mlp.{i}.weight"].shape[0])
        i += 1
    h_real.pop()
    decoder_real = build_decoder(h_real, "leakyrelu")
    decoder_real.load_state_dict(dec_sd_real, strict=False)
    decoder_real.activation = nn.ReLU()
    decoder_real.eval()

    h_bonn = []
    i = 0
    while f"mlp.{i}.weight" in dec_sd_bonn:
        h_bonn.append(dec_sd_bonn[f"mlp.{i}.weight"].shape[0])
        i += 1
    h_bonn.pop()
    bonn_input_dim = dec_sd_bonn["mlp.0.weight"].shape[1]
    bonn_latent_dim = bonn_input_dim - 48
    decoder_bonn = build_decoder(h_bonn, "softplus", latent_dim=bonn_latent_dim)
    decoder_bonn.load_state_dict(dec_sd_bonn, strict=False)
    decoder_bonn.eval()

    normal = torch.tensor([[0.0, 0.0, 1.0]])

    theta_o_vals = [15, 30, 45, 60]
    n_latents = 50
    torch.manual_seed(0)

    print(f"\n  Peak BRDF value at specular reflection direction (averaged over {n_latents} latents):")
    print(f"  {'theta_o':>7s} | {'Real peak':>10s} {'Real(w/factor)':>14s} {'Real off-spec':>13s} "
          f"| {'Bonn peak':>10s} {'Bonn off-spec':>13s} | {'Dynamic range':>13s}")
    print("  " + "-" * 100)

    for theta_o_deg in theta_o_vals:
        theta_o = math.radians(theta_o_deg)
        wo = torch.tensor([[math.sin(theta_o), 0.0, math.cos(theta_o)]])
        wi_spec = torch.tensor([[-math.sin(theta_o), 0.0, math.cos(theta_o)]])
        wi_off = torch.tensor([[0.0, math.sin(theta_o), math.cos(theta_o)]])

        idx_r = torch.randint(0, latent_real.shape[0], (n_latents,))
        lat_r = latent_real[idx_r, :-6]
        idx_b = torch.randint(0, latent_bonn.shape[0], (n_latents,))
        lat_b = latent_bonn[idx_b, :bonn_latent_dim]

        wo_exp = wo.expand(n_latents, -1)
        wi_s_exp = wi_spec.expand(n_latents, -1)
        wi_o_exp = wi_off.expand(n_latents, -1)
        n_exp = normal.expand(n_latents, -1)

        with torch.no_grad():
            # Real decoder
            enc_s = decoder_real.encode_directions(wi_s_exp, wo_exp, n_exp)
            enc_o = decoder_real.encode_directions(wi_o_exp, wo_exp, n_exp)
            peak_r = decoder_real(enc_s, lat_r).mean(dim=-1).mean().item()
            off_r = decoder_real(enc_o, lat_r).mean(dim=-1).mean().item()

            # Bonn decoder
            enc_s_b = decoder_bonn.encode_directions(wi_s_exp, wo_exp, n_exp)
            enc_o_b = decoder_bonn.encode_directions(wi_o_exp, wo_exp, n_exp)
            peak_b = decoder_bonn(enc_s_b, lat_b).mean(dim=-1).mean().item()
            off_b = decoder_bonn(enc_o_b, lat_b).mean(dim=-1).mean().item()

        peak_r_f = peak_r * (factor_real.mean().item() if factor_real is not None else 1.0)
        dyn_range_real = abs(peak_r_f) / (abs(off_r * (factor_real.mean().item() if factor_real is not None else 1.0)) + 1e-8)
        dyn_range_bonn = abs(peak_b) / (abs(off_b) + 1e-8)

        print(f"  {theta_o_deg:7d} | {peak_r:10.4f} {peak_r_f:14.4f} {off_r:13.4f} "
              f"| {peak_b:10.4f} {off_b:13.4f} | R:{dyn_range_real:.2f} B:{dyn_range_bonn:.2f}")


# =====================================================================
# TEST 6: Same-half-space angular distribution test
# =====================================================================
def test_same_halfspace_brdf_sweep():
    """
    Sweep the angle between wi_projected and wo_projected in the tangent
    plane (azimuth difference) from 0° to 180° and measure the BRDF response.
    0° = retro-reflection, 90° = side, 180° = forward scattering / specular.
    Training data is concentrated around 180° (opposite half-space).
    Bonn data has significant coverage near 0°-90° (same half-space).
    """
    print("\n" + "=" * 72)
    print("TEST 6: BRDF response sweep — azimuth difference 0°-180°")
    print("=" * 72)

    dec_sd = load_decoder_state(DECODER_ONLY_CKPT)
    h = []
    i = 0
    while f"mlp.{i}.weight" in dec_sd:
        h.append(dec_sd[f"mlp.{i}.weight"].shape[0])
        i += 1
    h.pop()

    decoder = build_decoder(h, "leakyrelu")
    decoder.load_state_dict(dec_sd, strict=False)
    decoder.activation = nn.ReLU()
    decoder.eval()

    normal = torch.tensor([[0.0, 0.0, 1.0]])
    theta_o = math.radians(45)
    theta_i = math.radians(45)
    wo = torch.tensor([[math.sin(theta_o), 0.0, math.cos(theta_o)]])

    torch.manual_seed(42)
    n_latents = 100
    latent_bank = load_latent_bank(BONN_FROM_REAL_CKPT)
    idx = torch.randint(0, latent_bank.shape[0], (n_latents,))
    latents = latent_bank[idx, :-6]

    phi_diffs = np.linspace(0, 180, 37)

    print(f"\n  Azimuth difference sweep (theta_i=theta_o=45°, average over {n_latents} latents):")
    print(f"  {'phi_diff':>8s} | {'mean_R':>8s} {'mean_G':>8s} {'mean_B':>8s} | {'mean_gray':>10s}")
    print("  " + "-" * 55)

    for phi_deg in phi_diffs:
        phi = math.radians(phi_deg)
        wi = torch.tensor([[
            math.sin(theta_i) * math.cos(math.pi - phi),
            math.sin(theta_i) * math.sin(math.pi - phi),
            math.cos(theta_i),
        ]])
        wi_exp = wi.expand(n_latents, -1)
        wo_exp = wo.expand(n_latents, -1)
        n_exp = normal.expand(n_latents, -1)

        with torch.no_grad():
            enc = decoder.encode_directions(wi_exp, wo_exp, n_exp)
            brdf = decoder(enc, latents)

        mean_rgb = brdf.mean(dim=0)
        mean_gray = mean_rgb.mean().item()
        print(f"  {phi_deg:8.1f} | {mean_rgb[0]:8.4f} {mean_rgb[1]:8.4f} {mean_rgb[2]:8.4f} | {mean_gray:10.4f}")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("BRDF Decoder Diagnostic Tests")
    print("=" * 72)

    test_relu_vs_leakyrelu_mismatch()
    test_decoder_output_distribution()
    test_angular_coverage()
    test_compare_checkpoints()
    test_specular_capability()
    test_same_halfspace_brdf_sweep()

    print("\n" + "=" * 72)
    print("All tests complete. Output saved to:", OUTPUT_DIR)
    print("=" * 72)
