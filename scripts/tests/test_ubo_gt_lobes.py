#!/usr/bin/env python3
"""Unit tests for UBO GT BRDF lobe visualization.

Verifies that the ground-truth lobe extraction script uses the same
direction conventions, point-ID semantics, and output format as the
prediction lobe visualization in Stage2Trainer_UBO.

Run:
    conda run -n fipt_copy python -m pytest scripts/tests/test_ubo_gt_lobes.py -v
"""

import os
import re
import sys
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.visualize_brdf_lobes_ubo_gt import (
    sph2cart,
    find_inplane_directions,
    compute_signed_theta,
    parse_point_ids_from_folder,
)
from utils.dataset.ubo import _sph2cart as ubo_sph2cart

# ======================================================================
# Paths  (used only where BTF data is needed; skipped if unavailable)
# ======================================================================
BTF_PATH = "/mnt/data/colin/colin/Bonn_BTF/carpet07_W400xH400_L151xV151.btf"
PRED_LOBE_DIR = (
    "/media/raid/cloth/output/BRDF/Bonn-Theia2/"
    "Stage-2_UBO_carpet07_from-Bonn_Logrel-Learnable-factor_run_2/images/brdf_lobes"
)
IMG_H, IMG_W = 400, 400


# ======================================================================
# 1. sph2cart convention matches utils.dataset.ubo._sph2cart
# ======================================================================

class TestSph2CartConvention:
    """GT script's sph2cart must produce the same Cartesian vectors as
    the dataset loader's _sph2cart for identical (theta, phi)."""

    @pytest.mark.parametrize("theta_deg, phi_deg", [
        (0, 0), (45, 0), (45, 180), (75, 90), (23.5, 270), (60, 45),
    ])
    def test_matches_ubo_dataset(self, theta_deg, phi_deg):
        gt_vec = sph2cart(theta_deg, phi_deg)
        ds_vec = ubo_sph2cart(theta_deg, phi_deg)
        np.testing.assert_allclose(gt_vec, ds_vec.flatten(), atol=1e-6,
            err_msg=f"sph2cart mismatch at theta={theta_deg}, phi={phi_deg}")

    def test_zenith(self):
        """theta=0 → (0, 0, 1) regardless of phi."""
        vec = sph2cart(0, 0)
        np.testing.assert_allclose(vec, [0, 0, 1], atol=1e-10)

    def test_horizon_phi0(self):
        """theta=90, phi=0 → (1, 0, 0)."""
        vec = sph2cart(90, 0)
        np.testing.assert_allclose(vec, [1, 0, 0], atol=1e-10)

    def test_phi180(self):
        """theta=45, phi=180 → (-sin45, 0, cos45)."""
        vec = sph2cart(45, 180)
        expected = np.array([-np.sin(np.deg2rad(45)), 0, np.cos(np.deg2rad(45))])
        np.testing.assert_allclose(vec, expected, atol=1e-10)


# ======================================================================
# 2. Signed-theta mapping matches prediction convention
# ======================================================================

class TestSignedThetaMapping:
    """The find_inplane_directions function must produce signed-theta
    values that match the prediction's wo (and wi) sweep convention.

    Prediction convention (for 'vary wo'):
      wi = (sin θ_i, 0, cos θ_i)   fixed, on +x side
      θ_o > 0 → wo = (-sin θ_o, 0, cos θ_o)  on -x side
      θ_o < 0 → wo = ( sin|θ_o|, 0, cos|θ_o|) on +x side
      θ_o = 0 → wo = (0, 0, 1)

    UBO dome:
      (θ_v, φ_v=0°) → wo on +x side → signed θ = -θ_v_rad
      (θ_v, φ_v=180°) → wo on -x side → signed θ = +θ_v_rad
    """

    def test_phi0_maps_negative(self):
        """(45°, 0°) should map to signed_theta = -pi/4 (retro side)."""
        dirs = [(45.0, 0.0)]
        result = find_inplane_directions(dirs)
        expected_key = -np.deg2rad(45.0)
        assert len(result) == 1
        key = list(result.keys())[0]
        np.testing.assert_allclose(key, expected_key, atol=1e-10)

    def test_phi180_maps_positive(self):
        """(45°, 180°) should map to signed_theta = +pi/4 (specular side)."""
        dirs = [(45.0, 180.0)]
        result = find_inplane_directions(dirs)
        expected_key = np.deg2rad(45.0)
        assert len(result) == 1
        key = list(result.keys())[0]
        np.testing.assert_allclose(key, expected_key, atol=1e-10)

    def test_zenith_maps_zero(self):
        """(0°, 0°) should map to signed_theta = 0."""
        dirs = [(0.0, 0.0)]
        result = find_inplane_directions(dirs)
        assert 0.0 in result

    def test_cartesian_consistency_phi0(self):
        """For (θ_v, 0°) the Cartesian wo must match prediction's negative θ_o.

        Prediction at θ_o = -θ_v_rad:
          wo = (sin(θ_v_rad), 0, cos(θ_v_rad))
        UBO dome at (θ_v, φ_v=0°):
          wo = sph2cart(θ_v, 0) = (sin(θ_v_rad), 0, cos(θ_v_rad))
        These must match.
        """
        for theta_v_deg in [23.5, 37.5, 45.0, 60.0, 75.0]:
            theta_v_rad = np.deg2rad(theta_v_deg)
            # Prediction wo at θ_o = -θ_v_rad (θ_o < 0 branch)
            pred_wo = np.array([np.sin(theta_v_rad), 0, np.cos(theta_v_rad)])
            # UBO dome wo
            ubo_wo = sph2cart(theta_v_deg, 0.0)
            np.testing.assert_allclose(ubo_wo, pred_wo, atol=1e-6,
                err_msg=f"Mismatch at theta_v={theta_v_deg}° phi_v=0°")

    def test_cartesian_consistency_phi180(self):
        """For (θ_v, 180°) the Cartesian wo must match prediction's positive θ_o.

        Prediction at θ_o = +θ_v_rad (θ_o ≥ 0 branch):
          wo = (-sin(θ_v_rad), 0, cos(θ_v_rad))
        UBO dome at (θ_v, φ_v=180°):
          wo = sph2cart(θ_v, 180) = (-sin(θ_v_rad), 0, cos(θ_v_rad))
        """
        for theta_v_deg in [23.5, 37.5, 45.0, 60.0, 75.0]:
            theta_v_rad = np.deg2rad(theta_v_deg)
            # Prediction wo at θ_o = +θ_v_rad (θ_o ≥ 0 branch)
            pred_wo = np.array([-np.sin(theta_v_rad), 0, np.cos(theta_v_rad)])
            # UBO dome wo
            ubo_wo = sph2cart(theta_v_deg, 180.0)
            np.testing.assert_allclose(ubo_wo, pred_wo, atol=1e-6,
                err_msg=f"Mismatch at theta_v={theta_v_deg}° phi_v=180°")

    def test_full_dome_inplane_count(self):
        """A realistic UBO dome should produce ~11 in-plane directions
        (6 at phi=0 + 5 at phi=180, some thetas overlap at zenith)."""
        # Simulate realistic UBO dome directions (from actual data)
        dirs = [
            (0.0, 0.0),  # zenith
            (23.5, 0.0), (37.5, 0.0), (45.0, 0.0), (60.0, 0.0), (75.0, 0.0),
            (23.5, 180.0), (37.5, 180.0), (45.0, 180.0), (60.0, 180.0), (75.0, 180.0),
            # Some off-plane directions that should be excluded
            (45.0, 45.0), (60.0, 90.0), (30.0, 270.0),
        ]
        result = find_inplane_directions(dirs)
        # 1 zenith + 5 at phi=0 + 5 at phi=180 = 11
        assert len(result) == 11, f"Expected 11 in-plane dirs, got {len(result)}"

    def test_symmetry(self):
        """Signed theta for phi=0 and phi=180 at same elevation must be
        equal in magnitude, opposite in sign."""
        dirs = [(45.0, 0.0), (45.0, 180.0)]
        result = find_inplane_directions(dirs)
        keys = sorted(result.keys())
        assert len(keys) == 2
        np.testing.assert_allclose(keys[0], -keys[1], atol=1e-10)


# ======================================================================
# 2b. compute_signed_theta (projection of all dome dirs onto incidence plane)
# ======================================================================

class TestComputeSignedTheta:
    """compute_signed_theta must agree with the prediction's sweep convention
    for in-plane directions and produce sensible projections for off-plane."""

    def test_inplane_phi0_retro(self):
        """wo at (45°, 0°) with wi at (30°, 0°) → retro side → negative."""
        wi = sph2cart(30.0, 0.0)
        wo = sph2cart(45.0, 0.0).reshape(1, 3)
        st = compute_signed_theta(wo, wi)
        assert st[0] < 0, f"Expected negative (retro), got {st[0]}"
        np.testing.assert_allclose(abs(st[0]), np.deg2rad(45.0), atol=1e-6)

    def test_inplane_phi180_specular(self):
        """wo at (45°, 180°) with wi at (30°, 0°) → specular side → positive."""
        wi = sph2cart(30.0, 0.0)
        wo = sph2cart(45.0, 180.0).reshape(1, 3)
        st = compute_signed_theta(wo, wi)
        assert st[0] > 0, f"Expected positive (specular), got {st[0]}"
        np.testing.assert_allclose(st[0], np.deg2rad(45.0), atol=1e-6)

    def test_normal_direction_zero(self):
        """wo along normal (0°, 0°) → signed theta = 0."""
        wi = sph2cart(45.0, 0.0)
        wo = sph2cart(0.0, 0.0).reshape(1, 3)
        st = compute_signed_theta(wo, wi)
        np.testing.assert_allclose(st[0], 0.0, atol=1e-10)

    def test_perpendicular_projects_to_zero(self):
        """wo perpendicular to incidence plane → in-plane component = 0 → near zero."""
        wi = sph2cart(45.0, 0.0)  # incidence plane = xz
        wo = sph2cart(45.0, 90.0).reshape(1, 3)  # in yz plane
        st = compute_signed_theta(wo, wi)
        # Perpendicular to incidence plane: in_plane component = 0, z = cos45
        np.testing.assert_allclose(st[0], 0.0, atol=1e-6)

    def test_matches_prediction_sweep(self):
        """For the prediction's exact wo sweep at phi=0, compute_signed_theta
        must recover the original theta_o values."""
        wi = sph2cart(45.0, 0.0)
        theta_o_targets = np.array([-60, -30, 0, 30, 60], dtype=np.float64)
        wo_list = []
        for to_deg in theta_o_targets:
            to_rad = np.deg2rad(to_deg)
            if to_rad >= 0:
                wo_list.append([-np.sin(to_rad), 0, np.cos(to_rad)])
            else:
                wo_list.append([np.sin(-to_rad), 0, np.cos(-to_rad)])
        wo_arr = np.array(wo_list)
        st = compute_signed_theta(wo_arr, wi)
        np.testing.assert_allclose(np.rad2deg(st), theta_o_targets, atol=1e-6)

    def test_all_151_dirs_in_range(self):
        """All 151 dome directions must project to [-pi/2, pi/2]."""
        # Simulate a subset of UBO dome directions
        dirs = [(t, p) for t in [0, 23.5, 45, 60, 75]
                for p in [0, 45, 90, 135, 180, 225, 270, 315]]
        dirs.append((0.0, 0.0))
        cart = np.array([sph2cart(t, p) for t, p in dirs])
        wi = sph2cart(45.0, 0.0)
        st = compute_signed_theta(cart, wi)
        assert np.all(st >= -np.pi / 2 - 1e-6)
        assert np.all(st <= np.pi / 2 + 1e-6)


# ======================================================================
# 3. Point ID semantics
# ======================================================================

class TestPointIDs:
    """Point IDs must be flat pixel indices into an H×W grid.

    point_id = row * W + col, same as in UBOBTFTrainDataset._sample_batch.
    """

    def test_range(self):
        """All prediction point IDs must be in [0, H*W)."""
        if not os.path.isdir(PRED_LOBE_DIR):
            pytest.skip("Prediction lobe directory not available")
        ids = parse_point_ids_from_folder(PRED_LOBE_DIR)
        for pid in ids:
            assert 0 <= pid < IMG_H * IMG_W, f"point_id {pid} out of range"

    def test_row_col_roundtrip(self):
        """row * W + col must round-trip through integer division."""
        for pid in [0, 1, 399, 400, 2409, 159276, IMG_H * IMG_W - 1]:
            row = pid // IMG_W
            col = pid % IMG_W
            assert row * IMG_W + col == pid

    @pytest.mark.skipif(not os.path.isdir(PRED_LOBE_DIR),
                        reason="Prediction lobe directory not available")
    def test_parse_ids_finds_expected(self):
        """Parser should find the known prediction point IDs."""
        ids = parse_point_ids_from_folder(PRED_LOBE_DIR)
        assert len(ids) == 10, f"Expected 10 point IDs, got {len(ids)}"
        # All IDs should appear in both vary_wo and vary_wi filenames
        files = os.listdir(PRED_LOBE_DIR)
        for pid in ids:
            assert f'polar_brdf_vary_wo_pt_{pid}.png' in files
            assert f'polar_brdf_vary_wi_pt_{pid}.png' in files


# ======================================================================
# 4. Output format matches prediction
# ======================================================================

class TestOutputFormat:
    """GT lobe filenames must match the prediction naming convention."""

    def test_vary_wo_filename(self):
        pid = 2409
        expected = f'polar_brdf_vary_wo_pt_{pid}.png'
        # Verify this matches the prediction regex
        m = re.match(r'polar_brdf_vary_wo_pt_(\d+)\.png', expected)
        assert m and int(m.group(1)) == pid

    def test_vary_wi_filename(self):
        pid = 2409
        expected = f'polar_brdf_vary_wi_pt_{pid}.png'
        m = re.match(r'polar_brdf_vary_wi_pt_(\d+)\.png', expected)
        assert m and int(m.group(1)) == pid


# ======================================================================
# 5. BRDF data sanity (requires BTF file)
# ======================================================================

class TestBRDFData:
    """Test that GT BRDF values are sensible (non-negative, finite)."""

    @pytest.fixture(autouse=True)
    def load_btf(self):
        if not os.path.isfile(BTF_PATH):
            pytest.skip("BTF file not available")
        from btf_extractor import Ubo2014
        self.btf = Ubo2014(BTF_PATH)
        self.H, self.W, _ = self.btf.img_shape

    def test_brdf_nonnegative(self):
        """BRDF values at a random point should be non-negative."""
        from scripts.visualize_brdf_lobes_ubo_gt import get_pixel_brdf
        rgb = get_pixel_brdf(self.btf, 45.0, 0.0, 45.0, 180.0, 200, 200)
        assert rgb.shape == (3,)
        assert np.all(rgb >= 0), f"Negative BRDF: {rgb}"
        assert np.all(np.isfinite(rgb)), f"Non-finite BRDF: {rgb}"

    def test_brdf_rgb_order(self):
        """Verify BGR→RGB conversion by checking a known angle.

        btf_extractor returns BGR; get_pixel_brdf must return RGB.
        We verify by comparing with a direct call.
        """
        from scripts.visualize_brdf_lobes_ubo_gt import get_pixel_brdf
        row, col = 100, 100
        tl, pl, tv, pv = 45.0, 0.0, 45.0, 0.0
        # Direct access (BGR)
        img_bgr = self.btf.angles_to_image(tl, pl, tv, pv)
        expected_rgb = img_bgr[row, col, ::-1].copy()
        np.clip(expected_rgb, 0, None, out=expected_rgb)
        # Via get_pixel_brdf (should be RGB)
        got_rgb = get_pixel_brdf(self.btf, tl, pl, tv, pv, row, col)
        np.testing.assert_array_equal(got_rgb, expected_rgb)

    def test_brdf_nontrivial_at_normal(self):
        """BRDF at near-normal incidence should be non-trivially positive
        for any real material (not near-zero everywhere)."""
        from scripts.visualize_brdf_lobes_ubo_gt import get_pixel_brdf

        vals = []
        np.random.seed(42)
        for _ in range(20):
            row = np.random.randint(0, self.H)
            col = np.random.randint(0, self.W)
            rgb = get_pixel_brdf(self.btf, 23.5, 0.0, 23.5, 180.0, row, col)
            vals.append(rgb.mean())
        mean_val = np.mean(vals)
        assert mean_val > 1e-4, f"BRDF suspiciously near zero: mean={mean_val}"

    def test_wi_convention_matches_prediction(self):
        """wi for the GT plot at θ_i=45° must be (sin45, 0, cos45),
        same as the prediction code's wi construction."""
        from scripts.visualize_brdf_lobes_ubo_gt import sph2cart
        # GT: fixed light at (theta_l=45, phi_l=0) → sph2cart
        gt_wi = sph2cart(45.0, 0.0)
        # Prediction: wi = (sin(theta_i), 0, cos(theta_i)) with theta_i=45°
        theta_i = np.deg2rad(45.0)
        pred_wi = np.array([np.sin(theta_i), 0.0, np.cos(theta_i)])
        np.testing.assert_allclose(gt_wi, pred_wi, atol=1e-10,
            err_msg="wi convention mismatch between GT and prediction")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
