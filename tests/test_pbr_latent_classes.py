"""Unit tests for BonnPBRLatentBRDF and UBOPBRLatentBRDF classes."""

import sys
import os
import torch
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_cfg(overrides: dict):
    """Build a minimal OmegaConf-like namespace for testing."""
    from types import SimpleNamespace

    defaults = {
        'anisotropic': True,
        'disney': True,
        'soft_constraint': True,
        'predict_frame': True,
        'learnable_factor': False,
        'init_std': 0.1,
        'data_folder': None,
        'single_material_id': None,
        'debug': False,
        'debug_num': 1,
        'debug_rotate': False,
        'debug_swap_channels': False,
        'init_normal_from_gt': False,
        'optimizer': SimpleNamespace(name='Adam'),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_ubo_cfg(overrides: dict):
    """Build minimal config for UBOPBRLatentBRDF."""
    from types import SimpleNamespace

    defaults = {
        'anisotropic': True,
        'disney': True,
        'soft_constraint': True,
        'predict_frame': False,
        'learnable_factor': False,
        'init_std': 0.1,
        'btf_path': None,
        'img_height': 10,
        'img_width': 10,
        'optimizer': SimpleNamespace(name='Adam'),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ======================================================================
# Test 1: PBRDecoder standalone forward pass
# ======================================================================
def test_pbr_decoder_forward():
    """Verify PBRDecoder produces valid output for all model types."""
    from model.neural_brdf_refactored import PBRDecoder
    from types import SimpleNamespace

    B = 64
    wi = torch.randn(B, 3)
    wo = torch.randn(B, 3)
    # Make sure directions have positive z (above hemisphere)
    wi[:, 2] = wi[:, 2].abs() + 0.1
    wo[:, 2] = wo[:, 2].abs() + 0.1
    wi = torch.nn.functional.normalize(wi, dim=-1)
    wo = torch.nn.functional.normalize(wo, dim=-1)

    for mode, lat_dim in [('isotropic', 6), ('anisotropic', 9), ('disney', 12)]:
        aniso = mode in ('anisotropic', 'disney')
        disney = mode == 'disney'
        cfg = SimpleNamespace(anisotropic=aniso, disney=disney)
        decoder = PBRDecoder(cfg=cfg, soft_constraint=True)

        latent = torch.randn(B, lat_dim)
        brdf, pdf = decoder(wi, wo, latent)

        assert brdf.shape == (B, 3), f"{mode}: brdf shape {brdf.shape}"
        assert pdf.shape == (B, 1), f"{mode}: pdf shape {pdf.shape}"
        assert not torch.isnan(brdf).any(), f"{mode}: brdf has NaN"
        assert not torch.isnan(pdf).any(), f"{mode}: pdf has NaN"
        assert (brdf >= 0).all(), f"{mode}: brdf has negative values"
        print(f"  PBRDecoder {mode:12s} OK  brdf range [{brdf.min():.4f}, {brdf.max():.4f}]")

    print("PASS: test_pbr_decoder_forward")


# ======================================================================
# Test 2: BonnPBRLatentBRDF initialization (single-material, no data_folder)
# ======================================================================
def test_bonn_pbr_init_single_material():
    """Test BonnPBRLatentBRDF can be constructed in single-material mode
    without needing the real data folder (bypass metadata loading)."""
    from model.neural_brdf_refactored import BonnPBRLatentBRDF

    num_points = 256
    H, W = 16, 16

    # Monkey-patch the metadata loader to avoid needing real data
    import model.neural_brdf_refactored as mod
    original_loader = mod.BonnLatentBRDF._load_single_material_metadata

    @staticmethod
    def fake_loader(data_folder, mat_id):
        return num_points, H, W

    mod.BonnLatentBRDF._load_single_material_metadata = fake_loader
    try:
        cfg = make_cfg({'single_material_id': 3, 'data_folder': '/tmp/fake'})
        model = BonnPBRLatentBRDF(cfg)

        # Check latent bank size
        assert model.point_latent_bank.weight.shape == (num_points, model.total_latent_dim), \
            f"Bank shape {model.point_latent_bank.weight.shape}"

        # Disney → brdf_latent_dim=12, total=12+6=18
        assert model.brdf_latent_dim == 12, f"brdf_latent_dim={model.brdf_latent_dim}"
        assert model.total_latent_dim == 18, f"total_latent_dim={model.total_latent_dim}"

        # Frame init: last 6 dims should be [0,0,1, 0,1,0]
        normals = model.point_latent_bank.weight[:, -6:-3]
        tangents = model.point_latent_bank.weight[:, -3:]
        assert torch.allclose(normals, torch.tensor([0.0, 0.0, 1.0]).expand_as(normals)), \
            "Normal init wrong"
        assert torch.allclose(tangents, torch.tensor([0.0, 1.0, 0.0]).expand_as(tangents)), \
            "Tangent init wrong"

        print(f"  BonnPBRLatentBRDF init OK — bank {model.point_latent_bank.weight.shape}")
    finally:
        mod.BonnLatentBRDF._load_single_material_metadata = original_loader

    print("PASS: test_bonn_pbr_init_single_material")


# ======================================================================
# Test 3: BonnPBRLatentBRDF eval_brdf forward + gradient
# ======================================================================
def test_bonn_pbr_eval_brdf():
    """Test eval_brdf produces valid output and latent bank gets gradients."""
    from model.neural_brdf_refactored import BonnPBRLatentBRDF
    import model.neural_brdf_refactored as mod

    num_points = 64

    @staticmethod
    def fake_loader(data_folder, mat_id):
        return num_points, 8, 8

    original_loader = mod.BonnLatentBRDF._load_single_material_metadata
    mod.BonnLatentBRDF._load_single_material_metadata = fake_loader
    try:
        cfg = make_cfg({'single_material_id': 0, 'data_folder': '/tmp/fake'})
        model = BonnPBRLatentBRDF(cfg)

        B = 32
        # Random directions in upper hemisphere
        wi = torch.randn(B, 3)
        wo = torch.randn(B, 3)
        wi[:, 2] = wi[:, 2].abs() + 0.1
        wo[:, 2] = wo[:, 2].abs() + 0.1
        wi = torch.nn.functional.normalize(wi, dim=-1)
        wo = torch.nn.functional.normalize(wo, dim=-1)

        pos = torch.randn(B, 3)
        normal = torch.zeros(B, 3)
        normal[:, 2] = 1.0
        point_ids = torch.randint(0, num_points, (B,))
        material_ids = torch.zeros(B, dtype=torch.long)

        brdf, pred_normal, pdf, smooth_loss = model.eval_brdf(
            pos, wi, wo, normal,
            point_ids=point_ids, material_ids=material_ids)

        assert brdf.shape == (B, 3), f"brdf shape {brdf.shape}"
        assert pred_normal.shape == (B, 3), f"normal shape {pred_normal.shape}"
        assert pdf.shape == (B, 1), f"pdf shape {pdf.shape}"
        assert not torch.isnan(brdf).any(), "brdf has NaN"
        assert not torch.isnan(pred_normal).any(), "normal has NaN"

        # Gradient test: loss → backward → check latent bank has grad
        loss = brdf.sum()
        loss.backward()
        assert model.point_latent_bank.weight.grad is not None, "No gradient on latent bank"
        grad_norm = model.point_latent_bank.weight.grad.norm().item()
        assert grad_norm > 0, "Gradient is zero"
        print(f"  eval_brdf OK  brdf [{brdf.min():.4f}, {brdf.max():.4f}]  grad_norm={grad_norm:.4f}")
    finally:
        mod.BonnLatentBRDF._load_single_material_metadata = original_loader

    print("PASS: test_bonn_pbr_eval_brdf")


# ======================================================================
# Test 4: UBOPBRLatentBRDF initialization
# ======================================================================
def test_ubo_pbr_init():
    """Test UBOPBRLatentBRDF can be constructed without real BTF data."""
    from model.neural_brdf_refactored import UBOPBRLatentBRDF

    cfg = make_ubo_cfg({'img_height': 10, 'img_width': 10})
    model = UBOPBRLatentBRDF(cfg)

    total_points = 10 * 10
    # Disney → brdf_latent_dim=12, predict_frame=False → total=12
    assert model.brdf_latent_dim == 12, f"brdf_latent_dim={model.brdf_latent_dim}"
    assert model.total_latent_dim == 12, f"total_latent_dim={model.total_latent_dim}"
    assert model.point_latent_bank.weight.shape == (total_points, 12), \
        f"Bank shape {model.point_latent_bank.weight.shape}"

    print(f"  UBOPBRLatentBRDF init OK — bank {model.point_latent_bank.weight.shape}")
    print("PASS: test_ubo_pbr_init")


# ======================================================================
# Test 5: UBOPBRLatentBRDF eval_brdf forward + gradient
# ======================================================================
def test_ubo_pbr_eval_brdf():
    """Test eval_brdf produces valid output and latent bank gets gradients."""
    from model.neural_brdf_refactored import UBOPBRLatentBRDF

    cfg = make_ubo_cfg({'img_height': 10, 'img_width': 10})
    model = UBOPBRLatentBRDF(cfg)

    total_points = 100
    B = 32

    wi = torch.randn(B, 3)
    wo = torch.randn(B, 3)
    wi[:, 2] = wi[:, 2].abs() + 0.1
    wo[:, 2] = wo[:, 2].abs() + 0.1
    wi = torch.nn.functional.normalize(wi, dim=-1)
    wo = torch.nn.functional.normalize(wo, dim=-1)

    point_ids = torch.randint(0, total_points, (B,))

    brdf, smooth_loss = model.eval_brdf(wi, wo, point_ids=point_ids)

    assert brdf.shape == (B, 3), f"brdf shape {brdf.shape}"
    assert smooth_loss.item() == 0.0, "PBR smooth_loss should be 0"
    assert not torch.isnan(brdf).any(), "brdf has NaN"

    loss = brdf.sum()
    loss.backward()
    assert model.point_latent_bank.weight.grad is not None, "No gradient on latent bank"
    grad_norm = model.point_latent_bank.weight.grad.norm().item()
    assert grad_norm > 0, "Gradient is zero"
    print(f"  eval_brdf OK  brdf [{brdf.min():.4f}, {brdf.max():.4f}]  grad_norm={grad_norm:.4f}")

    print("PASS: test_ubo_pbr_eval_brdf")


# ======================================================================
# Test 6: UBOPBRLatentBRDF with predict_frame=True
# ======================================================================
def test_ubo_pbr_with_frame():
    """Test UBOPBRLatentBRDF works with predict_frame=True."""
    from model.neural_brdf_refactored import UBOPBRLatentBRDF

    cfg = make_ubo_cfg({'img_height': 8, 'img_width': 8, 'predict_frame': True})
    model = UBOPBRLatentBRDF(cfg)

    # Disney + frame: 12 + 6 = 18
    assert model.total_latent_dim == 18
    assert model.point_latent_bank.weight.shape == (64, 18)

    # Frame init check
    normals = model.point_latent_bank.weight[:, -6:-3]
    tangents = model.point_latent_bank.weight[:, -3:]
    assert torch.allclose(normals, torch.tensor([0.0, 0.0, 1.0]).expand_as(normals))
    assert torch.allclose(tangents, torch.tensor([0.0, 1.0, 0.0]).expand_as(tangents))

    B = 16
    wi = torch.randn(B, 3)
    wo = torch.randn(B, 3)
    wi[:, 2] = wi[:, 2].abs() + 0.1
    wo[:, 2] = wo[:, 2].abs() + 0.1
    wi = torch.nn.functional.normalize(wi, dim=-1)
    wo = torch.nn.functional.normalize(wo, dim=-1)
    point_ids = torch.randint(0, 64, (B,))

    brdf, smooth_loss = model.eval_brdf(wi, wo, point_ids=point_ids)
    assert brdf.shape == (B, 3)
    assert not torch.isnan(brdf).any()

    print("PASS: test_ubo_pbr_with_frame")


# ======================================================================
# Test 7: Latent optimization step (simulated training)
# ======================================================================
def test_optimization_step():
    """Simulate one training step: verify loss decreases with gradient descent."""
    from model.neural_brdf_refactored import UBOPBRLatentBRDF

    cfg = make_ubo_cfg({'img_height': 8, 'img_width': 8, 'learnable_factor': True})
    model = UBOPBRLatentBRDF(cfg)

    B = 64
    wi = torch.randn(B, 3)
    wo = torch.randn(B, 3)
    wi[:, 2] = wi[:, 2].abs() + 0.1
    wo[:, 2] = wo[:, 2].abs() + 0.1
    wi = torch.nn.functional.normalize(wi, dim=-1)
    wo = torch.nn.functional.normalize(wo, dim=-1)
    point_ids = torch.randint(0, 64, (B,))

    # Target: uniform gray
    target = torch.ones(B, 3) * 0.5

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for step in range(20):
        optimizer.zero_grad()
        brdf, _ = model.eval_brdf(wi, wo, point_ids=point_ids)
        loss = (brdf - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
    print(f"  Optimization OK: loss {losses[0]:.6f} -> {losses[-1]:.6f} "
          f"({(1 - losses[-1]/losses[0])*100:.1f}% reduction)")

    print("PASS: test_optimization_step")


# ======================================================================
# Test 8: Isotropic and anisotropic (non-Disney) modes
# ======================================================================
def test_model_type_variants():
    """Test isotropic and anisotropic-only modes for both classes."""
    from model.neural_brdf_refactored import UBOPBRLatentBRDF, BonnPBRLatentBRDF
    import model.neural_brdf_refactored as mod

    @staticmethod
    def fake_loader(data_folder, mat_id):
        return 64, 8, 8

    original_loader = mod.BonnLatentBRDF._load_single_material_metadata
    mod.BonnLatentBRDF._load_single_material_metadata = fake_loader

    try:
        for aniso, disney, expected_dim in [(False, False, 6), (True, False, 9), (True, True, 12)]:
            label = 'disney' if disney else ('anisotropic' if aniso else 'isotropic')

            # UBO
            cfg_ubo = make_ubo_cfg({
                'img_height': 8, 'img_width': 8,
                'anisotropic': aniso, 'disney': disney
            })
            ubo = UBOPBRLatentBRDF(cfg_ubo)
            assert ubo.brdf_latent_dim == expected_dim, \
                f"UBO {label}: brdf_latent_dim={ubo.brdf_latent_dim}, expected {expected_dim}"

            B = 16
            wi = torch.nn.functional.normalize(torch.randn(B, 3).abs(), dim=-1)
            wo = torch.nn.functional.normalize(torch.randn(B, 3).abs(), dim=-1)
            point_ids = torch.randint(0, 64, (B,))
            brdf, _ = ubo.eval_brdf(wi, wo, point_ids=point_ids)
            assert brdf.shape == (B, 3) and not torch.isnan(brdf).any()

            # Bonn
            cfg_bonn = make_cfg({
                'single_material_id': 0, 'data_folder': '/tmp/fake',
                'anisotropic': aniso, 'disney': disney
            })
            bonn = BonnPBRLatentBRDF(cfg_bonn)
            assert bonn.brdf_latent_dim == expected_dim

            normal = torch.zeros(B, 3)
            normal[:, 2] = 1.0
            material_ids = torch.zeros(B, dtype=torch.long)
            brdf, _, _, _ = bonn.eval_brdf(
                torch.randn(B, 3), wi, wo, normal,
                point_ids=point_ids, material_ids=material_ids)
            assert brdf.shape == (B, 3) and not torch.isnan(brdf).any()

            print(f"  {label:12s} OK  UBO dim={ubo.total_latent_dim}  Bonn dim={bonn.total_latent_dim}")
    finally:
        mod.BonnLatentBRDF._load_single_material_metadata = original_loader

    print("PASS: test_model_type_variants")


if __name__ == '__main__':
    print("=" * 60)
    print("Running PBR Latent Class Unit Tests")
    print("=" * 60)

    test_pbr_decoder_forward()
    test_bonn_pbr_init_single_material()
    test_bonn_pbr_eval_brdf()
    test_ubo_pbr_init()
    test_ubo_pbr_eval_brdf()
    test_ubo_pbr_with_frame()
    test_optimization_step()
    test_model_type_variants()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
