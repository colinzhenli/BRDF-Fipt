"""MLP-free basis-function BRDF decoders for the UBO2014 BTF dataset.

This file implements three drop-in alternatives to ``UBOLatentBRDF`` whose
decoder is a basis-function expansion plus an optional linear coefficient
projection ``D_psi``.

Compared with the earlier half-vector-only version, this file supports a
compact multi-projection BRDF basis:

    q_source = "brdf12"

which evaluates 12 different 2D spherical projections of the 4D angular BRDF
input (wi, wo). This is still not a full tensor-product 4D basis, but it is much
more expressive than using only one projection such as half-vector.

Layout:
    BasisBRDFDecoder        -- abstract base, D_psi + diffuse_proj wiring
    SHBasisDecoder          -- analytical spherical-harmonic basis
    SGBasisDecoder          -- learned spherical-Gaussian lobes
    SVBasisDecoder          -- soft spherical-Voronoi anchors

    UBOBasisBRDF            -- LightningModule parent, mirrors UBOLatentBRDF
    UBOSHBRDF / UBOSGBRDF / UBOSVBRDF  -- thin per-basis wrappers

The model classes expose the same ``self.decoder`` attribute and
``eval_brdf(wi, wo, point_ids, return_wi_local=...)`` signature as
``UBOLatentBRDF`` so that ``Stage2Trainer_UBO`` and its ``freeze_decoder`` logic
can work without modification.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as NF
from pytorch_lightning import LightningModule

from utils.ops import components_from_spherical_harmonics, num_sh_bases


# ============================================================================
# Helpers
# ============================================================================

def _fibonacci_sphere(K: int, hemisphere: bool = True) -> torch.Tensor:
    """Return [K, 3] unit vectors evenly spread on the sphere or hemisphere.

    When ``hemisphere=True`` the points lie in z >= 0.
    """
    indices = torch.arange(K, dtype=torch.float64) + 0.5

    if hemisphere:
        # z in [0, 1]
        z = 1.0 - indices / K
    else:
        # z in [-1, 1]
        z = 1.0 - 2.0 * indices / K

    radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden_angle * indices

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    pts = torch.stack([x, y, z], dim=-1).to(torch.float32)
    return NF.normalize(pts, dim=-1)


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return NF.normalize(v, dim=-1, eps=eps)


def _normalize_with_fallback(
    v: torch.Tensor,
    fallback: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize v, but use fallback if the vector is nearly zero."""
    norm = v.norm(dim=-1, keepdim=True)
    v_norm = v / norm.clamp_min(eps)
    return torch.where(norm > eps, v_norm, fallback)


def _local_normal_like(v: torch.Tensor) -> torch.Tensor:
    n = torch.zeros_like(v)
    n[..., 2] = 1.0
    return n


def _mirror_about_normal(v: torch.Tensor) -> torch.Tensor:
    """Mirror a local direction around the local normal.

    For n=(0,0,1), reflection about the normal maps:
        (x, y, z) -> (-x, -y, z)
    """
    out = v.clone()
    out[..., 0:2] = -out[..., 0:2]
    return out


def _safe_cross_feature(
    a: torch.Tensor,
    b: torch.Tensor,
    fallback: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    c = torch.cross(a, b, dim=-1)
    norm = c.norm(dim=-1, keepdim=True)
    c_norm = c / norm.clamp_min(eps)
    return torch.where(norm > eps, c_norm, fallback)


def _single_q(
    wi_local: torch.Tensor,
    wo_local: torch.Tensor,
    q_source: str,
) -> torch.Tensor:
    """Return one [B, 3] spherical projection.

    This keeps backward compatibility with the previous implementation.
    """
    wi = _safe_normalize(wi_local)
    wo = _safe_normalize(wo_local)
    n = _local_normal_like(wi)

    if q_source == "halfvec":
        return _normalize_with_fallback(wi + wo, n)

    if q_source == "wi":
        return wi

    if q_source == "wo":
        return wo

    raise ValueError(f"Unknown single q_source: {q_source!r}")


def _brdf12_q_bank(
    wi_local: torch.Tensor,
    wo_local: torch.Tensor,
) -> torch.Tensor:
    """Return [B, 12, 3] spherical projections of the 4D BRDF input.

    Each branch is still a 2D spherical function, but the 12 branches together
    cover multiple projections of (wi, wo), instead of collapsing everything
    into only the half-vector.

    The 12 projections are:

        0. wi
        1. wo
        2. h = normalize(wi + wo)
        3. d = normalize(wi - wo)
        4. mirror(wi)
        5. mirror(wo)
        6. normalize(wi + n)
        7. normalize(wo + n)
        8. normalize(h + n)
        9. normalize(wo - mirror(wi))
       10. normalize(wi - mirror(wo))
       11. normalize(cross(wi, wo))

    This is a compact multi-projection approximation to a 4D BRDF basis.
    It is not a full tensor-product basis, but it is much stronger than using
    only one projection such as half-vector.
    """
    wi = _safe_normalize(wi_local)
    wo = _safe_normalize(wo_local)
    n = _local_normal_like(wi)

    h = _normalize_with_fallback(wi + wo, n)
    d_io = _normalize_with_fallback(wi - wo, h)

    mi = _mirror_about_normal(wi)
    mo = _mirror_about_normal(wo)

    wi_n = _normalize_with_fallback(wi + n, n)
    wo_n = _normalize_with_fallback(wo + n, n)
    h_n = _normalize_with_fallback(h + n, n)

    spec_o = _normalize_with_fallback(wo - mi, h)
    spec_i = _normalize_with_fallback(wi - mo, h)

    plane = _safe_cross_feature(wi, wo, fallback=h)

    q_bank = torch.stack(
        [
            wi,
            wo,
            h,
            d_io,
            mi,
            mo,
            wi_n,
            wo_n,
            h_n,
            spec_o,
            spec_i,
            plane,
        ],
        dim=1,
    )

    return q_bank


def _q_bank(
    wi_local: torch.Tensor,
    wo_local: torch.Tensor,
    q_source: str,
) -> torch.Tensor:
    """Return [B, M, 3].

    Supported q_source:
        halfvec: M = 1
        wi:      M = 1
        wo:      M = 1
        brdf12:  M = 12
    """
    if q_source == "brdf12":
        return _brdf12_q_bank(wi_local, wo_local)

    q = _single_q(wi_local, wo_local, q_source)
    return q.unsqueeze(1)


def _num_q_branches(q_source: str) -> int:
    if q_source == "brdf12":
        return 12
    if q_source in ["halfvec", "wi", "wo"]:
        return 1
    raise ValueError(f"Unknown q_source: {q_source!r}")


# ============================================================================
# Basis decoder base + concrete classes
# ============================================================================

class BasisBRDFDecoder(nn.Module):
    """Abstract base: ``forward(wi, wo, latent) -> [B, 3]``.

    Subclasses override only:

        basis_values(wi, wo) -> [B, num_basis]

    For q_source='brdf12', basis_values returns the concatenation of basis
    values from 12 spherical projection branches.
    """

    def __init__(
        self,
        cfg_decoder,
        latent_dim: int,
        num_basis: int,
        direct_coefficients: bool,
        use_diffuse: bool,
    ):
        super().__init__()

        self.num_basis = num_basis
        self.K = num_basis  # backward-compatible name

        self.latent_dim = latent_dim
        self.direct_coefficients = direct_coefficients
        self.use_diffuse = use_diffuse

        self.q_source = getattr(cfg_decoder, "q_source", "halfvec")
        self.num_q_branches = _num_q_branches(self.q_source)

        if not self.direct_coefficients:
            self.D_psi = nn.Linear(latent_dim, self.num_basis * 3, bias=False)

            if self.use_diffuse:
                self.diffuse_proj = nn.Linear(latent_dim, 3, bias=False)

    def basis_values(
        self,
        wi_local: torch.Tensor,
        wo_local: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        wi_local: torch.Tensor,
        wo_local: torch.Tensor,
        brdf_latent: torch.Tensor,
    ) -> torch.Tensor:
        phi = self.basis_values(wi_local, wo_local)  # [B, num_basis]
        B = phi.shape[0]

        if self.direct_coefficients:
            coeffs = brdf_latent[:, : self.num_basis * 3].view(B, self.num_basis, 3)

            if self.use_diffuse:
                diffuse_rgb = brdf_latent[
                    :, self.num_basis * 3 : self.num_basis * 3 + 3
                ]
        else:
            coeffs = self.D_psi(brdf_latent).view(B, self.num_basis, 3)

            if self.use_diffuse:
                diffuse_rgb = self.diffuse_proj(brdf_latent)

        brdf = (phi.unsqueeze(-1) * coeffs).sum(dim=-2)  # [B, 3]

        if self.use_diffuse:
            brdf = brdf + diffuse_rgb

        return brdf


class SHBasisDecoder(BasisBRDFDecoder):
    """Spherical harmonic basis over one or more spherical projections.

    For q_source='brdf12':

        num_basis = 12 * num_sh_bases(degree)

    Recommended:
        degree=1 -> 12 * 4 = 48 bases
        degree=2 -> 12 * 9 = 108 bases
    """

    def __init__(
        self,
        cfg_decoder,
        latent_dim: int,
        num_basis: int,
        direct_coefficients: bool,
        use_diffuse: bool,
    ):
        q_source = getattr(cfg_decoder, "q_source", "halfvec")
        default_degree = 1 if q_source == "brdf12" else 3

        self.degree = int(getattr(cfg_decoder, "degree", default_degree))
        self.q_source = q_source
        self.num_q_branches = _num_q_branches(self.q_source)

        expected = self.num_q_branches * num_sh_bases(self.degree)
        assert num_basis == expected, (
            f"SH num_basis mismatch: got {num_basis}, expected {expected} "
            f"for q_source={self.q_source}, degree={self.degree}"
        )

        super().__init__(
            cfg_decoder=cfg_decoder,
            latent_dim=latent_dim,
            num_basis=num_basis,
            direct_coefficients=direct_coefficients,
            use_diffuse=use_diffuse,
        )

    def basis_values(
        self,
        wi_local: torch.Tensor,
        wo_local: torch.Tensor,
    ) -> torch.Tensor:
        q = _q_bank(wi_local, wo_local, self.q_source)  # [B, M, 3]
        B, M, _ = q.shape

        q_flat = q.reshape(B * M, 3)
        sh_flat = components_from_spherical_harmonics(self.degree, q_flat)  # [B*M, S]
        sh = sh_flat.reshape(B, M, -1)

        return sh.reshape(B, -1)  # [B, M*S]


class SGBasisDecoder(BasisBRDFDecoder):
    """Multi-projection spherical Gaussian basis.

    Each branch has K_per_branch SG lobes:

        phi_{m,k}(q_m) = exp(lambda_{m,k} * (dot(q_m, mu_{m,k}) - 1))

    For q_source='brdf12':
        K=36 -> 12 branches * 3 lobes each
        K=48 -> 12 branches * 4 lobes each
    """

    def __init__(
        self,
        cfg_decoder,
        latent_dim: int,
        num_basis: int,
        direct_coefficients: bool,
        use_diffuse: bool,
    ):
        super().__init__(
            cfg_decoder=cfg_decoder,
            latent_dim=latent_dim,
            num_basis=num_basis,
            direct_coefficients=direct_coefficients,
            use_diffuse=use_diffuse,
        )

        self.K_per_branch = int(math.ceil(self.num_basis / self.num_q_branches))
        expected = self.K_per_branch * self.num_q_branches
        assert expected == self.num_basis

        init_lambda = float(getattr(cfg_decoder, "init_lambda", 5.0))

        # Single-direction mode is usually upper hemisphere.
        # brdf12 has branches such as difference and cross product, so use full sphere.
        hemisphere = self.q_source in ["halfvec", "wi", "wo"]

        init_mu = []
        for _ in range(self.num_q_branches):
            init_mu.append(_fibonacci_sphere(self.K_per_branch, hemisphere=hemisphere))
        init_mu = torch.stack(init_mu, dim=0)  # [M, Kb, 3]

        self.mu = nn.Parameter(init_mu)
        self.log_lambda = nn.Parameter(
            torch.full(
                (self.num_q_branches, self.K_per_branch),
                math.log(init_lambda),
            )
        )

    def basis_values(
        self,
        wi_local: torch.Tensor,
        wo_local: torch.Tensor,
    ) -> torch.Tensor:
        q = _q_bank(wi_local, wo_local, self.q_source)  # [B, M, 3]

        mu = NF.normalize(self.mu, dim=-1)              # [M, Kb, 3]
        lam = torch.exp(self.log_lambda)                # [M, Kb]

        # cos[b, m, k] = dot(q[b,m], mu[m,k])
        cos = torch.einsum("bmc,mkc->bmk", q, mu)

        phi = torch.exp(lam.unsqueeze(0) * (cos - 1.0))  # [B, M, Kb]

        # Keep activation magnitude somewhat stable as branch count increases.
        if self.num_q_branches > 1:
            phi = phi / math.sqrt(float(self.num_q_branches))

        return phi.reshape(phi.shape[0], -1)             # [B, M*Kb]


class SVBasisDecoder(BasisBRDFDecoder):
    """Multi-projection soft spherical-Voronoi basis.

    Each branch has its own softmax over K_per_branch anchors:

        phi_{m,k}(q_m) = softmax_k(tau_m * dot(q_m, mu_{m,k}))

    For q_source='brdf12':
        K=36 -> 12 branches * 3 anchors each
        K=48 -> 12 branches * 4 anchors each

    The softmax is computed per branch, not across all branches.
    """

    def __init__(
        self,
        cfg_decoder,
        latent_dim: int,
        num_basis: int,
        direct_coefficients: bool,
        use_diffuse: bool,
    ):
        super().__init__(
            cfg_decoder=cfg_decoder,
            latent_dim=latent_dim,
            num_basis=num_basis,
            direct_coefficients=direct_coefficients,
            use_diffuse=use_diffuse,
        )

        self.K_per_branch = int(math.ceil(self.num_basis / self.num_q_branches))
        expected = self.K_per_branch * self.num_q_branches
        assert expected == self.num_basis

        init_tau = float(getattr(cfg_decoder, "init_tau", 10.0))

        # Single-direction mode is usually upper hemisphere.
        # brdf12 has full-sphere branches.
        hemisphere = self.q_source in ["halfvec", "wi", "wo"]

        init_mu = []
        for _ in range(self.num_q_branches):
            init_mu.append(_fibonacci_sphere(self.K_per_branch, hemisphere=hemisphere))
        init_mu = torch.stack(init_mu, dim=0)  # [M, Kb, 3]

        self.mu = nn.Parameter(init_mu)
        self.log_tau = nn.Parameter(
            torch.full((self.num_q_branches,), math.log(init_tau))
        )

    def basis_values(
        self,
        wi_local: torch.Tensor,
        wo_local: torch.Tensor,
    ) -> torch.Tensor:
        q = _q_bank(wi_local, wo_local, self.q_source)  # [B, M, 3]

        mu = NF.normalize(self.mu, dim=-1)              # [M, Kb, 3]
        tau = torch.exp(self.log_tau)                   # [M]

        logits = torch.einsum("bmc,mkc->bmk", q, mu)    # [B, M, Kb]
        logits = logits * tau.view(1, -1, 1)

        # Per-branch soft Voronoi.
        phi = torch.softmax(logits, dim=-1)

        # Keep activation magnitude somewhat stable as branch count increases.
        if self.num_q_branches > 1:
            phi = phi / math.sqrt(float(self.num_q_branches))

        return phi.reshape(phi.shape[0], -1)            # [B, M*Kb]


# ============================================================================
# LightningModule parent + per-basis wrappers
# ============================================================================

class UBOBasisBRDF(LightningModule):
    """LightningModule parent shared by the SH/SG/SV variants.

    Mirrors UBOLatentBRDF's outer skeleton:
        - point_latent_bank
        - optional predict_frame
        - learnable_factor
        - smooth_reg

    but uses a basis-function decoder instead of an MLP.
    """

    def __init__(self, cfg, decoder_cls):
        super().__init__()
        self.cfg = cfg

        # ---- Top-level flags ------------------------------------------
        self.latent_dim = cfg.latent_dim
        self.predict_frame = getattr(cfg, "predict_frame", False)
        self.use_pos_enc = False  # basis decoders take raw local directions
        self.different_decoder = False

        self.learnable_factor = getattr(cfg, "learnable_factor", False)
        if self.learnable_factor:
            self.factor = nn.Parameter(torch.ones(3))

        # ---- Decoder block --------------------------------------------
        cfg_decoder = cfg.decoder
        self.direct_coefficients = bool(getattr(cfg_decoder, "direct_coefficients", False))
        self.use_diffuse = bool(getattr(cfg_decoder, "use_diffuse", True))

        q_source = getattr(cfg_decoder, "q_source", "halfvec")
        num_q_branches = _num_q_branches(q_source)

        if decoder_cls is SHBasisDecoder:
            default_degree = 1 if q_source == "brdf12" else 3
            degree = int(getattr(cfg_decoder, "degree", default_degree))
            K = num_q_branches * num_sh_bases(degree)
        else:
            # Treat material.decoder.K as a total basis budget.
            # For brdf12, the budget is rounded up so every branch has the same
            # number of bases.
            default_K = 36 if q_source == "brdf12" else 32
            K_budget = int(getattr(cfg_decoder, "K", default_K))
            K_per_branch = int(math.ceil(K_budget / num_q_branches))
            K = K_per_branch * num_q_branches

        print(
            f"{type(self).__name__}: q_source={q_source}, "
            f"num_q_branches={num_q_branches}, total_basis={K}"
        )

        # ---- Per-texel latent layout ---------------------------------
        if self.direct_coefficients:
            self.brdf_latent_dim = K * 3 + (3 if self.use_diffuse else 0)
        else:
            self.brdf_latent_dim = self.latent_dim

        if self.predict_frame:
            self.total_latent_dim = self.brdf_latent_dim + 6
        else:
            self.total_latent_dim = self.brdf_latent_dim

        # ---- BTF size lookup -----------------------------------------
        btf_path = getattr(cfg, "btf_path", None)

        if btf_path is not None:
            from btf_extractor import Ubo2014

            btf = Ubo2014(btf_path)
            H, W, _ = btf.img_shape
            total_points = H * W
            del btf

            print(f"{type(self).__name__}: BTF {H}x{W} = {total_points:,} texels")
        else:
            H = getattr(cfg, "img_height", 400)
            W = getattr(cfg, "img_width", 400)
            total_points = H * W

            print(f"{type(self).__name__}: using config {H}x{W} = {total_points:,} texels")

        self._H = H
        self._W = W

        # ---- Per-texel latent bank -----------------------------------
        self.point_latent_bank = nn.Embedding(
            num_embeddings=total_points,
            embedding_dim=self.total_latent_dim,
            sparse=False,
        )

        nn.init.normal_(self.point_latent_bank.weight, mean=0.0, std=cfg.init_std)

        if self.predict_frame:
            with torch.no_grad():
                self.point_latent_bank.weight[:, -6:-3] = torch.tensor([0.0, 0.0, 1.0])
                self.point_latent_bank.weight[:, -3:] = torch.tensor([0.0, 1.0, 0.0])

        # ---- Decoder --------------------------------------------------
        self.decoder = decoder_cls(
            cfg_decoder=cfg_decoder,
            latent_dim=self.latent_dim,
            num_basis=K,
            direct_coefficients=self.direct_coefficients,
            use_diffuse=self.use_diffuse,
        )

        self.smooth_reg = bool(getattr(cfg_decoder, "smooth_reg", False))
        self.smooth_reg_eps = float(getattr(cfg_decoder, "smooth_reg_eps", 0.01))

        print(f"{type(self).__name__} initialisation complete!")

    # ------------------------------------------------------------------
    # Frame helpers, only used when predict_frame=True.
    # ------------------------------------------------------------------
    def extract_frame_from_latent(self, latent: torch.Tensor):
        predicted_normal = NF.normalize(latent[..., -6:-3], dim=-1)
        predicted_tangent = NF.normalize(latent[..., -3:], dim=-1)

        predicted_tangent = predicted_tangent - (
            torch.sum(
                predicted_tangent * predicted_normal,
                dim=-1,
                keepdim=True,
            )
            * predicted_normal
        )

        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)

        return predicted_normal, predicted_tangent

    def world_to_local(self, v, normal, tangent):
        bitangent = torch.cross(normal, tangent, dim=-1)

        return torch.stack(
            [
                (v * tangent).sum(dim=-1),
                (v * bitangent).sum(dim=-1),
                (v * normal).sum(dim=-1),
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # BRDF evaluation, matches UBOLatentBRDF.eval_brdf signature.
    # ------------------------------------------------------------------
    def eval_brdf(self, wi, wo, point_ids=None, return_wi_local=False):
        if point_ids is None:
            raise ValueError("point_ids must be provided")

        latent = self.point_latent_bank(point_ids)  # [B, total_latent_dim]

        if self.predict_frame:
            predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)

            wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
            wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        else:
            wi_local = wi
            wo_local = wo

        brdf_lat = latent[:, : self.brdf_latent_dim]
        brdf = self.decoder(wi_local, wo_local, brdf_lat)

        if self.smooth_reg:
            eps = self.smooth_reg_eps

            rand_vec = torch.randn_like(wi_local)
            rand_vec = rand_vec - (
                (rand_vec * wi_local).sum(-1, keepdim=True) * wi_local
            )

            axis = NF.normalize(rand_vec, dim=-1)

            wi_perturbed = (
                wi_local * math.cos(eps)
                + torch.cross(axis, wi_local, dim=-1) * math.sin(eps)
            )

            brdf_pert = self.decoder(wi_perturbed, wo_local, brdf_lat)
            smooth_loss = ((brdf_pert - brdf) / eps).pow(2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=wi.device)

        if self.learnable_factor:
            brdf = brdf * self.factor

        if return_wi_local:
            return brdf, smooth_loss, wi_local

        return brdf, smooth_loss


# ============================================================================
# Per-basis wrappers
# ============================================================================

class UBOSHBRDF(UBOBasisBRDF):
    def __init__(self, cfg):
        super().__init__(cfg, SHBasisDecoder)


class UBOSGBRDF(UBOBasisBRDF):
    def __init__(self, cfg):
        super().__init__(cfg, SGBasisDecoder)


class UBOSVBRDF(UBOBasisBRDF):
    def __init__(self, cfg):
        super().__init__(cfg, SVBasisDecoder)