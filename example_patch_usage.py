"""
Example demonstrating how to use the GreyPatchBRDF class with spatial patch indexing.

The GreyPatchBRDF class now supports:
1. Automatic patch identification from 3D positions (using only x,y coordinates)
2. Filtering to only return BRDF for the top 5 grayscale patches
3. Returning zero BRDF for non-grayscale patches
"""

import torch
from model.mipmap_brdf import GreyPatchBRDF

# ─────────────────────────────────────────────────────────────────────
# Example 1: Initialize the BRDF model with patch layout parameters
# ─────────────────────────────────────────────────────────────────────
brdf_model = GreyPatchBRDF(
    inc_thresh_deg=2.0,      # Incident angle threshold (degrees)
    view_thresh_deg=2.0,     # View angle threshold (degrees)
    device="cuda",
    # Patch layout parameters:
    center_x=0.0,            # x-coordinate of middle-top patch center
    center_y=0.0,            # y-coordinate of middle-top patch center
    patch_width=0.04,        # Width of each square patch (meters)
    patch_distance=0.05,     # Distance between patch centers (meters)
    grid_rows=4,             # Number of rows in the color checker
    grid_cols=6,             # Number of columns in the color checker
    grayscale_patch_ids=[0, 1, 2, 3, 4]  # First 5 patches are grayscale
)

# ─────────────────────────────────────────────────────────────────────
# Example 2: Get grayscale patch centers for initialization
# ─────────────────────────────────────────────────────────────────────
gray_centers, gray_ids = brdf_model.get_grayscale_patch_centers_and_ids()
print("Grayscale patch centers:")
print(gray_centers)
print("\nGrayscale patch IDs:", gray_ids)

# ─────────────────────────────────────────────────────────────────────
# Example 3: Evaluate BRDF using positions (automatic patch detection)
# ─────────────────────────────────────────────────────────────────────
# Simulate some ray intersections at different 3D positions
N = 100
positions = torch.randn(N, 3).cuda()  # Random positions
wi = torch.randn(N, 3).cuda()         # Light directions
wo = torch.randn(N, 3).cuda()         # View directions
normal = torch.tensor([[0., 0., 1.]]).expand(N, 3).cuda()  # Surface normals

# Option A: Let the model determine patch IDs from positions
brdf_values = brdf_model.eval_brdf(
    wi=wi,
    wo=wo,
    normal=normal,
    positions=positions  # Pass positions, patch_index will be computed automatically
)
print(f"\nBRDF values shape: {brdf_values.shape}")
print(f"Non-zero BRDF count: {(brdf_values > 0).sum().item()} out of {N}")

# ─────────────────────────────────────────────────────────────────────
# Example 4: Get patch IDs directly for debugging
# ─────────────────────────────────────────────────────────────────────
patch_ids, is_grayscale = brdf_model.get_patch_id_from_position(positions)
print(f"\nPatch IDs range: [{patch_ids.min().item()}, {patch_ids.max().item()}]")
print(f"Grayscale patches: {is_grayscale.sum().item()} out of {N}")
print(f"Outside patches: {(patch_ids == -1).sum().item()} out of {N}")

# ─────────────────────────────────────────────────────────────────────
# Example 5: Integration with path tracing
# ─────────────────────────────────────────────────────────────────────
def path_tracing_example(brdf_model, positions, wi, wo, normal):
    """
    Example showing how to use the BRDF model in path tracing context.
    
    Args:
        brdf_model: GreyPatchBRDF instance
        positions: (N, 3) intersection positions from ray tracing
        wi: (N, 3) incident light directions
        wo: (N, 3) outgoing view directions
        normal: (N, 3) surface normals
    
    Returns:
        brdf_values: (N, 1) BRDF values (0 for non-grayscale patches)
    """
    # Evaluate BRDF - automatically identifies patches from positions
    # Only grayscale patches (top 5) will have non-zero BRDF
    brdf = brdf_model.eval_brdf(
        wi=wi,
        wo=wo,
        normal=normal,
        positions=positions  # Uses x,y coordinates only
    )
    
    # brdf will be:
    # - Non-zero for rays hitting the 5 grayscale patches (if angles are valid)
    # - Zero for rays hitting non-grayscale patches
    # - Zero for rays outside all patches
    
    return brdf

# Run the example
brdf_result = path_tracing_example(brdf_model, positions, wi, wo, normal)
print(f"\nPath tracing BRDF result shape: {brdf_result.shape}")
print(f"Non-zero results: {(brdf_result > 0).sum().item()}")

# ─────────────────────────────────────────────────────────────────────
# Example 6: Visualize patch layout
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PATCH LAYOUT")
print("="*70)
all_centers = brdf_model.patch_centers
print(f"Total patches: {all_centers.shape[0]}")
print(f"Grid layout: {brdf_model.grid_rows} rows x {brdf_model.grid_cols} cols")
print(f"\nGrayscale patch centers (x, y):")
for i, (center, pid) in enumerate(zip(gray_centers, gray_ids)):
    print(f"  Patch {pid}: ({center[0].item():.4f}, {center[1].item():.4f})")

