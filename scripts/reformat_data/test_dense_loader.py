#!/usr/bin/env python3
"""
Test the MultiMaterialDenseDataset against the existing chunk-based loader.

Validates:
1. Data loads without errors
2. Batch shapes and dtypes are correct
3. Ray directions match normalize(xyz - cam_pos) from c2w
4. RGB CCM is applied correctly
5. Memory usage estimate

Usage:
  conda run -n fipt_copy python scripts/reformat_data/test_dense_loader.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import json
import psutil
from pathlib import Path
from omegaconf import OmegaConf
from utils.dataset.points import (
    MultiMaterialDenseDataset,
    MultiMaterialPointDataset,
    build_4x4,
    get_ray_directions_for_pixels,
    get_rays,
    load_camera_turntable_light_metadata,
    load_camera_metadata,
)

DATASET_ROOT = "/media/raid/cloth/capture_data/Dataset_Nov11"
TRAINING_LIST = os.path.join(os.path.dirname(__file__), "debug_training_list.txt")


def build_cfg():
    """Build a minimal config matching run_stage1.sh."""
    cfg = OmegaConf.create({
        'data': {
            'rays_num': 8192,  # small for testing
            'ccm': [
                [3.6724617, -0.94800931, 0.08428962],
                [-0.44629176, 2.96095854, -1.17898539],
                [-0.47909694, -0.39991418, 2.10705124],
            ],
            'filter_observations': False,
            'val_ratio': 0.01,
            'training_list_path': TRAINING_LIST,
            'switch_iters': 5000,
            'chunk_size': 200,
            'dataset_name': 'points_dense',
        },
        'renderer': {
            'camera': {
                'intrinsics': {
                    'focal_length': 6494.0,
                    'cx': 1024.0,
                    'cy': 768.0,
                    'distortion': 0.0,
                    'height': 1536,
                    'width': 2048,
                }
            },
            'mesh': {
                'rectangle': {
                    'center': [0.16, -0.11, -0.067],
                    'width': 0.28,
                    'length': 0.28,
                }
            },
        },
        'dataset_folder': DATASET_ROOT,
    })
    return cfg


def test_load_and_shapes():
    """Test that the dataset loads and produces correct shapes."""
    print("\n" + "=" * 60)
    print("TEST 1: Load dataset and check batch shapes")
    print("=" * 60)

    cfg = build_cfg()
    mem_before = psutil.Process().memory_info().rss / 1e9

    ds = MultiMaterialDenseDataset(cfg, root_folder=DATASET_ROOT, split='train')

    mem_after = psutil.Process().memory_info().rss / 1e9
    print(f"\nMemory usage (RSS increase): {mem_after - mem_before:.2f} GB")
    print(f"Total valid observations: {sum(ds.mat_num_valid_obs):,}")

    # Get one batch
    batch = next(iter(ds))
    print(f"\nBatch keys: {list(batch.keys())}")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

    # Verify shapes
    N = cfg.data.rays_num
    assert batch['rays'].shape == (N, 6), f"rays shape mismatch: {batch['rays'].shape}"
    assert batch['rgbs'].shape == (N, 3), f"rgbs shape mismatch: {batch['rgbs'].shape}"
    assert batch['xyz'].shape == (N, 3), f"xyz shape mismatch: {batch['xyz'].shape}"
    assert batch['emitter_ids'].shape == (N,), f"emitter_ids shape mismatch"
    assert batch['camera_ids'].shape == (N,), f"camera_ids shape mismatch"
    assert batch['material_ids'].shape == (N,), f"material_ids shape mismatch"
    assert batch['point_ids'].shape == (N,), f"point_ids shape mismatch"

    # Verify dtypes
    assert batch['rays'].dtype == torch.float32
    assert batch['rgbs'].dtype == torch.float32
    assert batch['xyz'].dtype == torch.float32
    assert batch['emitter_ids'].dtype == torch.int64
    assert batch['camera_ids'].dtype == torch.int64
    assert batch['material_ids'].dtype == torch.int64
    assert batch['point_ids'].dtype == torch.int64

    print("\n✓ All shapes and dtypes correct!")
    return ds


def test_ray_direction_consistency(ds):
    """Verify ray directions = normalize(xyz - cam_pos)."""
    print("\n" + "=" * 60)
    print("TEST 2: Ray direction consistency")
    print("=" * 60)

    batch = next(iter(ds))
    rays = batch['rays']
    xyz = batch['xyz']
    rays_o = rays[:, :3]
    rays_d = rays[:, 3:6]

    # Recompute rays_d from xyz and rays_o
    expected_d = xyz - rays_o
    expected_d = expected_d / torch.norm(expected_d, dim=-1, keepdim=True)

    diff = (rays_d - expected_d).abs().max().item()
    print(f"Max ray direction difference: {diff:.2e}")
    assert diff < 1e-5, f"Ray directions don't match! max diff = {diff}"

    # Verify rays_o comes from c2w (camera position)
    # Just check that rays_o is consistent across observations with the same camera_id
    cam_ids = batch['camera_ids']
    mat_ids = batch['material_ids']
    unique_combos = torch.unique(torch.stack([mat_ids, cam_ids], dim=1), dim=0)
    for combo in unique_combos[:5]:
        mat_id, cam_id = combo[0].item(), combo[1].item()
        mask = (mat_ids == mat_id) & (cam_ids == cam_id)
        if mask.sum() < 2:
            continue
        origins = rays_o[mask]
        # All rays from same camera should have same origin
        origin_var = origins.var(dim=0).max().item()
        assert origin_var < 1e-10, f"Camera origins inconsistent for mat={mat_id}, cam={cam_id}"

    print("✓ Ray directions are consistent with normalize(xyz - cam_pos)")
    print("✓ Camera origins are consistent within same (material, camera) pair")


def test_rgb_ccm_correctness(ds):
    """Verify CCM is applied correctly by comparing against manual computation."""
    print("\n" + "=" * 60)
    print("TEST 3: RGB CCM correctness")
    print("=" * 60)

    # Load raw structured data for material 0
    data = np.load(os.path.join(DATASET_ROOT, "0", "observations_structured.npz"))
    rgbs_dense = data['rgbs']  # (K, V, 3) uint16
    ccm = np.array(ds.ccm.numpy())

    # Pick a few known valid observations from the dataset
    batch = next(iter(ds))
    mat_mask = batch['material_ids'] == 0
    if mat_mask.sum() == 0:
        print("  No material 0 observations in batch, skipping detailed CCM check")
        return

    # Get the first material-0 observation
    idx = torch.where(mat_mask)[0][0].item()
    cam_id = batch['camera_ids'][idx].item()
    point_id = batch['point_ids'][idx].item()

    # Find this point in the structured NPZ
    point_ids_np = data['point_ids']
    v_idx = np.where(point_ids_np == point_id)[0]
    if len(v_idx) == 0:
        print(f"  point_id {point_id} not found in structured NPZ, skipping")
        return
    v_idx = v_idx[0]

    # Manual CCM computation
    raw_rgb = rgbs_dense[cam_id, v_idx].astype(np.float64)
    expected_rgb = np.clip(raw_rgb @ ccm, 0, None).astype(np.float32)
    actual_rgb = batch['rgbs'][idx].numpy()

    diff = np.abs(expected_rgb - actual_rgb).max()
    print(f"  Raw RGB: {raw_rgb}")
    print(f"  Expected (CCM): {expected_rgb}")
    print(f"  Actual:         {actual_rgb}")
    print(f"  Max diff: {diff:.2e}")
    assert diff < 1e-3, f"CCM mismatch! diff = {diff}"
    print("✓ CCM applied correctly")


def test_cross_validate_with_chunks():
    """
    Cross-validate: for material 0, compare a sample of observations from the
    dense loader against the same data loaded from the original chunks.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Cross-validate dense vs chunk data (material 0)")
    print("=" * 60)

    material_dir = Path(DATASET_ROOT) / "0"
    ccm = np.array([
        [3.6724617, -0.94800931, 0.08428962],
        [-0.44629176, 2.96095854, -1.17898539],
        [-0.47909694, -0.39991418, 2.10705124],
    ])

    # Load structured data
    struct_data = np.load(material_dir / "observations_structured.npz")
    struct_xyz = struct_data['xyz']
    struct_point_ids = struct_data['point_ids']
    struct_rgbs = struct_data['rgbs']  # (K, V, 3) uint16
    K, V, _ = struct_rgbs.shape

    # Load one original chunk
    chunk_path = sorted((material_dir / "observations").glob("observations_chunk_*.npz"))[0]
    chunk_obs = np.load(chunk_path)['observations']  # (N, 10) float64

    # Sample 1000 observations from chunk
    np.random.seed(42)
    sample_indices = np.random.choice(len(chunk_obs), min(1000, len(chunk_obs)), replace=False)
    mismatches_rgb = 0
    mismatches_xyz = 0

    for i in sample_indices:
        obs = chunk_obs[i]
        chunk_xyz = obs[:3].astype(np.float32)
        chunk_img_id = int(obs[3]) - 1  # 1-based -> 0-based
        chunk_rgb = obs[6:9]  # raw float64 (originally uint16)
        chunk_pid = int(obs[9])

        # Find in structured data
        v_idx = np.where(struct_point_ids == chunk_pid)[0]
        if len(v_idx) == 0:
            continue
        v_idx = v_idx[0]

        # Compare xyz
        struct_xyz_val = struct_xyz[v_idx]
        if not np.allclose(chunk_xyz, struct_xyz_val, atol=1e-5):
            mismatches_xyz += 1

        # Compare raw RGB
        struct_rgb = struct_rgbs[chunk_img_id, v_idx].astype(np.float64)
        if not np.allclose(chunk_rgb, struct_rgb, atol=1.0):
            mismatches_rgb += 1
            if mismatches_rgb <= 3:
                print(f"  RGB mismatch at img={chunk_img_id}, pid={chunk_pid}: "
                      f"chunk={chunk_rgb}, struct={struct_rgb}")

    print(f"  Checked {len(sample_indices)} observations from chunk 0")
    print(f"  XYZ mismatches: {mismatches_xyz}")
    print(f"  RGB mismatches: {mismatches_rgb}")
    assert mismatches_xyz == 0, "XYZ values don't match!"
    assert mismatches_rgb == 0, "RGB values don't match!"
    print("✓ Dense data matches original chunks exactly")


def test_iteration_speed(ds):
    """Measure iteration speed (ray computation + CCM at sample time)."""
    print("\n" + "=" * 60)
    print("TEST 5: Iteration speed (including on-the-fly ray + CCM computation)")
    print("=" * 60)

    import time
    it = iter(ds)
    # Warmup
    for _ in range(5):
        _ = next(it)

    # Timed iterations
    N_iters = 50
    t0 = time.time()
    for _ in range(N_iters):
        batch = next(it)
    elapsed = time.time() - t0

    print(f"  {N_iters} iterations in {elapsed:.3f}s = {elapsed/N_iters*1000:.1f} ms/iter")
    print(f"  rays_num = {ds.rays_num}, total_obs = {sum(ds.mat_num_valid_obs):,}")
    print("✓ Iteration speed test complete")


def test_val_split():
    """Test that validation split loads and iterates finitely."""
    print("\n" + "=" * 60)
    print("TEST 6: Validation split")
    print("=" * 60)

    cfg = build_cfg()
    val_ds = MultiMaterialDenseDataset(cfg, root_folder=DATASET_ROOT, split='val')

    # Should iterate finitely
    num_batches = 0
    for batch in val_ds:
        num_batches += 1
        if num_batches > 5:
            break

    print(f"  Val dataset length: {len(val_ds)}")
    print(f"  Iterated {num_batches} batches")
    assert num_batches > 0, "No validation batches!"
    print("✓ Validation split works correctly")


if __name__ == '__main__':
    print("Testing MultiMaterialDenseDataset")
    print("=" * 60)

    # Test 4 is independent (no dataset needed)
    test_cross_validate_with_chunks()

    # Tests 1-3, 5 use the same dataset
    ds = test_load_and_shapes()
    test_ray_direction_consistency(ds)
    test_rgb_ccm_correctness(ds)
    test_iteration_speed(ds)

    # Test 6: validation split
    test_val_split()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
