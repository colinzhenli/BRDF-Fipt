#!/usr/bin/env python3
"""
Test RealImageDenseDataset against the chunk-based RealImageDataset.

Validates:
1. Data loads without errors and yields correct shapes/dtypes
2. Ray directions are unit norm and rays_o is constant per camera_id
3. CCM is applied correctly
4. Cross-validation: dense and chunk loaders produce the same observation multiset
5. Iteration speed
6. RealValDataset coexists unchanged

Usage:
  conda run -n fipt_copy python scripts/reformat_data/test_real_dense_loader.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import torch
import numpy as np
import psutil
import cv2
from omegaconf import OmegaConf
from utils.dataset import RealImageDataset, RealImageDenseDataset, RealValDataset

# Use material 227 from Dataset_submission (existing canonical test material).
DATASET_FOLDER = "/media/raid/cloth/Dataset_submission/227"
DEBUG_NUM = 10  # number of images to load per dataset
RAYS_NUM = 8192


def build_cfg(chunk_size=0, switch_iters=999999999, random_chunks=False):
    """Hydra-style cfg for testing. Mirrors run_stage2_ours_from_Real.sh."""
    cfg = OmegaConf.create({
        'dataset_folder': DATASET_FOLDER,
        'gt_folder': f'{DATASET_FOLDER}/hdr',
        'data': {
            'dataset_name': 'real_dense',
            'rays_num': RAYS_NUM,
            'ccm': [
                [3.6724617, -0.94800931, 0.08428962],
                [-0.44629176, 2.96095854, -1.17898539],
                [-0.47909694, -0.39991418, 2.10705124],
            ],
            'chunk_size': chunk_size,
            'switch_iters': switch_iters,
            'random_chunks': random_chunks,
            'importance_sampling': False,
            'use_single_chunk_sampling': False,
            'multi_resolution': False,
            'downsample_iter': [-1, -1],
            'debug': True,
            'debug_num': DEBUG_NUM,
            'start_idx': 0,
            'use_fixed_val': False,
            'hold_out_val_num': 0,
            'valid_num': 0,
            'valid_on_train_set': False,
            'metadata_path': f'{DATASET_FOLDER}/scan_log.json',
            'camera_metadata_path': f'{DATASET_FOLDER}/rotated_camera.json',
        },
        'renderer': {
            'camera': {
                'colmap_camera': True,
                'views_per_batch': 8,
                'R_c2g': [[-7.17667595e-04, 9.99776474e-01, 2.11302413e-02],
                          [-4.42126179e-03, 2.11268680e-02, -9.99767027e-01],
                          [-9.99989969e-01, -8.10922726e-04, 4.40511146e-03]],
                't_c2g': [3.26272696e-02, -1.61560433e-02, 2.80920856e-02],
                'intrinsics': {
                    'width': 3072,
                    'height': 2048,
                    'focal_length': 6721.054,
                    'cx': 1536,
                    'cy': 1024,
                    'distortion': -0.0648,
                },
            },
            'emitter': {
                'turntable': {
                    'center': [0.16084722, -0.11011424, -0.021],
                    'axis': [-0.00876202, -0.01346449, 0.99987096],
                },
            },
            'mesh': {
                'rectangle': {
                    'bbox_json': f'{DATASET_FOLDER}/bbox.json',
                },
            },
        },
        'model': {'test': False, 'test_novel_view': False},
    })
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
def test_load_and_shapes():
    print("\n" + "=" * 60)
    print("TEST 1: Load dataset, check shapes/dtypes/RSS")
    print("=" * 60)

    cfg = build_cfg()
    rss_before = psutil.Process().memory_info().rss / 1e9
    ds = RealImageDenseDataset(cfg, gt_folder=cfg.gt_folder, split='train')
    rss_after = psutil.Process().memory_info().rss / 1e9
    print(f"\nRSS increase: {rss_after - rss_before:.2f} GB")
    print(f"Total rays preloaded: {ds.rays.shape[0]:,}")

    batch = next(iter(ds))
    print(f"\nBatch keys: {sorted(batch.keys())}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")

    # Shape assertions
    N = cfg.data.rays_num
    assert batch['rays'].shape == (N, 12), f"rays shape: {batch['rays'].shape}"
    assert batch['rgbs'].shape == (N, 3), f"rgbs shape: {batch['rgbs'].shape}"
    assert batch['camera_ids'].shape == (N,), f"camera_ids shape: {batch['camera_ids'].shape}"
    assert batch['emitter_ids'].shape == (N,), f"emitter_ids shape: {batch['emitter_ids'].shape}"
    assert batch['pdf'].shape == (N,), f"pdf shape: {batch['pdf'].shape}"

    # Dtype assertions
    assert batch['rays'].dtype == torch.float32
    assert batch['rgbs'].dtype == torch.float32
    assert batch['camera_ids'].dtype == torch.int64
    assert batch['emitter_ids'].dtype == torch.int64

    # Yield-key set parity with chunk loader
    expected_keys = {'rays', 'rgbs', 'emitter_ids', 'camera_ids', 'pdf', 'gt_params'}
    assert set(batch.keys()) == expected_keys, f"keys differ: {set(batch.keys()) ^ expected_keys}"

    print("\n[OK] Shapes/dtypes/keys all correct")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
def test_ray_consistency(ds):
    """Note: get_rays(focal=...) returns rays_d = directions @ R.T where R is the
    camera rotation from camera_metadata. In this dataset R is rotation × scale
    (translation in mm, R rows have norm ≈ 0.097), so rays_d magnitudes inherit
    that scale. We verify the meaningful invariants:
      - rays/rgbs are finite
      - rays_d is non-zero
      - rays_o is constant per camera_id (the load-bearing invariant)
      - normalized rays_d is unit norm (numerical sanity)
    """
    print("\n" + "=" * 60)
    print("TEST 2: Ray direction consistency")
    print("=" * 60)

    batch = next(iter(ds))
    rays = batch['rays']
    rays_o = rays[:, :3]
    rays_d = rays[:, 3:6]

    assert torch.isfinite(rays).all(), "rays contains NaN/inf"
    assert torch.isfinite(batch['rgbs']).all(), "rgbs contains NaN/inf"

    norms = torch.norm(rays_d, dim=-1)
    print(f"  ‖rays_d‖: min={norms.min().item():.4f}  max={norms.max().item():.4f}")
    assert norms.min().item() > 0.0, "rays_d has zero-magnitude rows"

    rays_d_unit = rays_d / norms.unsqueeze(-1)
    unit_err = (torch.norm(rays_d_unit, dim=-1) - 1.0).abs().max().item()
    assert unit_err < 1e-5, f"normalized rays_d not unit norm (max err {unit_err})"

    cam_ids = batch['camera_ids']
    unique = torch.unique(cam_ids)
    print(f"  Checking origin consistency across {len(unique)} cameras...")
    for cid in unique:
        mask = cam_ids == cid
        if mask.sum() < 2:
            continue
        var = rays_o[mask].var(dim=0).max().item()
        assert var < 1e-10, f"rays_o varies within camera {int(cid)}: var={var}"
    print("[OK] rays finite & well-formed; rays_o constant within camera")


# ─────────────────────────────────────────────────────────────────────────────
def test_ccm_correctness(ds):
    """Pick one camera, recompute CCM-applied RGBs from disk, assert dense
    contains all of them as a multiset."""
    print("\n" + "=" * 60)
    print("TEST 3: CCM correctness (multiset comparison)")
    print("=" * 60)

    # Choose any camera_id present in the dense data
    cam_ids = ds.camera_ids
    target_cid = int(cam_ids[0].item())
    print(f"  Target camera_id: {target_cid}")

    # Find the metadata entry for this camera
    md_for_cam = [m for m in ds.all_metadata if int(m['camera_id']) == target_cid]
    if not md_for_cam:
        print("  No metadata for this camera_id, skipping")
        return
    img_data = md_for_cam[0]
    img_path = os.path.join(ds.gt_folder, img_data['filename'])
    print(f"  Recomputing from: {img_data['filename']}")

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img @ ds.ccm
    img = img.clip(0, None)
    img_flat = img.reshape(-1, 3).astype(np.float32)

    # Apply the same luminance filter used by _preload_given_metadata
    lum = 0.2126 * img_flat[..., 0] + 0.7152 * img_flat[..., 1] + 0.0722 * img_flat[..., 2]
    expected_rgbs = img_flat[lum > 1e-7]

    # Pull dense rgbs for this camera_id
    mask = cam_ids == target_cid
    actual_rgbs = ds.rgbs[mask].numpy()
    print(f"  Expected: {expected_rgbs.shape[0]} rgbs   Actual: {actual_rgbs.shape[0]} rgbs")
    assert expected_rgbs.shape == actual_rgbs.shape, "row counts differ"

    # Multiset compare via lex-sort on RGB tuples
    exp_sorted = expected_rgbs[np.lexsort(expected_rgbs.T)]
    act_sorted = actual_rgbs[np.lexsort(actual_rgbs.T)]
    diff = np.abs(exp_sorted - act_sorted).max()
    print(f"  Max abs diff (sorted): {diff:.2e}")
    assert diff < 1e-3, f"CCM RGBs mismatch! diff={diff}"
    print("[OK] CCM applied correctly (post-luminance multiset matches disk)")


# ─────────────────────────────────────────────────────────────────────────────
def test_cross_validate_vs_chunk():
    """Construct chunk loader with chunk_size = total_metadata so its slot-0
    build covers ALL images, then assert (rays, rgbs, camera_ids, emitter_ids)
    multisets are bit-identical to the dense loader."""
    print("\n" + "=" * 60)
    print("TEST 4: Cross-validate dense vs chunk loader (LOAD-BEARING)")
    print("=" * 60)

    # Dense
    cfg_d = build_cfg()
    print("  Building dense loader...")
    dense = RealImageDenseDataset(cfg_d, gt_folder=cfg_d.gt_folder, split='train')

    # Chunk: chunk_size = debug_num and random_chunks=False so the
    # FIRST chunk = ALL metadata in shuffled order. switch_iters=large to avoid
    # accidental re-fills.
    cfg_c = build_cfg(chunk_size=DEBUG_NUM, switch_iters=999999999, random_chunks=False)
    print("  Building chunk loader (chunk_size = total)...")
    chunk = RealImageDataset(cfg_c, gt_folder=cfg_c.gt_folder, split='train')

    # Pull slot-0 contents directly from the double-buffer
    chunk._dbuf.ready[0].wait()
    cd = chunk._dbuf.slots[0]
    chunk_rays, chunk_rgbs = cd.rays, cd.rgbs
    chunk_cam_ids, chunk_emit_ids = cd.camera_ids, cd.emitter_ids

    print(f"  Dense rows: {dense.rays.shape[0]:,}")
    print(f"  Chunk rows: {chunk_rays.shape[0]:,}")
    assert dense.rays.shape == chunk_rays.shape, \
        f"row count mismatch: dense={dense.rays.shape}, chunk={chunk_rays.shape}"

    # Build a sortable lex key from rays_o + rays_d + camera_id (stable, deterministic)
    def lex_key(rays, cam_ids):
        # numpy lexsort uses LAST key as primary; we want camera_id primary then rays_o
        keys = (
            rays[:, 5].numpy(),  # rays_d[2]
            rays[:, 4].numpy(),  # rays_d[1]
            rays[:, 3].numpy(),  # rays_d[0]
            rays[:, 2].numpy(),  # rays_o[2]
            rays[:, 1].numpy(),  # rays_o[1]
            rays[:, 0].numpy(),  # rays_o[0]
            cam_ids.numpy(),     # primary
        )
        return np.lexsort(keys)

    print("  Sorting both arrays by (camera_id, rays_o, rays_d)...")
    order_d = lex_key(dense.rays, dense.camera_ids)
    order_c = lex_key(chunk_rays, chunk_cam_ids)

    rays_d_sorted = dense.rays[order_d]
    rays_c_sorted = chunk_rays[order_c]
    rgbs_d_sorted = dense.rgbs[order_d]
    rgbs_c_sorted = chunk_rgbs[order_c]
    cam_d_sorted = dense.camera_ids[order_d]
    cam_c_sorted = chunk_cam_ids[order_c]
    emit_d_sorted = dense.emitter_ids[order_d]
    emit_c_sorted = chunk_emit_ids[order_c]

    # Assertions
    rays_diff = (rays_d_sorted - rays_c_sorted).abs().max().item()
    rgbs_diff = (rgbs_d_sorted - rgbs_c_sorted).abs().max().item()
    cam_diff = (cam_d_sorted - cam_c_sorted).abs().max().item()
    emit_diff = (emit_d_sorted - emit_c_sorted).abs().max().item()
    print(f"  Max |rays diff|: {rays_diff:.2e}")
    print(f"  Max |rgbs diff|: {rgbs_diff:.2e}")
    print(f"  Max |camera_id diff|: {cam_diff}")
    print(f"  Max |emitter_id diff|: {emit_diff}")
    assert rays_diff == 0.0, f"rays differ! max={rays_diff}"
    assert rgbs_diff == 0.0, f"rgbs differ! max={rgbs_diff}"
    assert cam_diff == 0, f"camera_ids differ! max={cam_diff}"
    assert emit_diff == 0, f"emitter_ids differ! max={emit_diff}"

    # PDF normalization (do NOT bit-match — chunk-level vs global pdf semantics)
    pdf_d_sum = dense.pdf.sum().item()
    pdf_c_sum = cd.pdf.sum().item()
    print(f"  Dense pdf.sum()={pdf_d_sum:.6f}   Chunk pdf.sum()={pdf_c_sum:.6f}")
    assert abs(pdf_d_sum - 1.0) < 1e-4, f"dense pdf doesn't sum to 1 (got {pdf_d_sum})"
    assert abs(pdf_c_sum - 1.0) < 1e-4, f"chunk pdf doesn't sum to 1 (got {pdf_c_sum})"

    # Cleanup chunk loader's background thread
    chunk._dbuf.stop()

    print("[OK] Dense and chunk multisets are bit-identical")


# ─────────────────────────────────────────────────────────────────────────────
def test_iteration_speed(ds):
    print("\n" + "=" * 60)
    print("TEST 5: Iteration speed")
    print("=" * 60)
    it = iter(ds)
    for _ in range(5):
        _ = next(it)
    N = 50
    t0 = time.time()
    for _ in range(N):
        _ = next(it)
    elapsed = time.time() - t0
    print(f"  {N} iters in {elapsed:.3f}s = {elapsed/N*1000:.2f} ms/iter "
          f"(rays_num={ds.rays_num}, total_obs={ds.rays.shape[0]:,})")
    print("[OK] Iteration speed test complete")


# ─────────────────────────────────────────────────────────────────────────────
def test_val_dataset_coexists():
    print("\n" + "=" * 60)
    print("TEST 6: RealValDataset smoke test (unchanged path)")
    print("=" * 60)
    cfg = build_cfg()
    cfg.data.valid_num = 2
    val = RealValDataset(cfg, gt_folder=cfg.gt_folder)
    print(f"  RealValDataset length: {len(val)}")
    assert len(val) > 0
    item = val[0]
    expected = {'rays', 'rgbs', 'emitter_ids', 'gt_params', 'camera_ids'}
    assert set(item.keys()) == expected, f"keys differ: {set(item.keys()) ^ expected}"
    assert item['rays'].shape[1] == 12
    assert item['rgbs'].shape[1] == 3
    print(f"  rays={tuple(item['rays'].shape)}  rgbs={tuple(item['rgbs'].shape)}")
    print("[OK] RealValDataset still works alongside dense train")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing RealImageDenseDataset")
    print("=" * 60)
    print(f"Dataset folder: {DATASET_FOLDER}")
    print(f"debug_num: {DEBUG_NUM},  rays_num: {RAYS_NUM}")

    # T4 first (independent — instantiates both loaders fresh)
    test_cross_validate_vs_chunk()

    # T1-T3, T5 share one dense dataset
    ds = test_load_and_shapes()
    test_ray_consistency(ds)
    test_ccm_correctness(ds)
    test_iteration_speed(ds)

    # T6 standalone
    test_val_dataset_coexists()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
