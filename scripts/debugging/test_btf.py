"""Quick test script to check btf_extractor image output for carpet07."""
import sys
import numpy as np

from btf_extractor import Ubo2014

btf_path = "/mnt/data/colin/colin/Bonn_BTF/carpet07_W400xH400_L151xV151.btf"
print(f"Loading: {btf_path}")
btf = Ubo2014(btf_path)

print(f"img_shape: {btf.img_shape}")
print(f"n_angles:  {len(btf.angles_set)}")

angles = sorted(btf.angles_set)
a = angles[len(angles) // 2]
print(f"Test angle: {a}")

img = btf.angles_to_image(*a)
print(f"img dtype={img.dtype}, shape={img.shape}")
print(f"img min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}")
print(f"Channel means: R={img[:,:,0].mean():.4f}, G={img[:,:,1].mean():.4f}, B={img[:,:,2].mean():.4f}")

# Save as PNG to visually check
try:
    import cv2
    img_clip = np.clip(img, 0.0, 1.0)
    img_u8 = (img_clip * 255).astype(np.uint8)
    # Save as-is (RGB order) via cv2 (which expects BGR)
    cv2.imwrite("/home/zla247/projects/BRDF-Fipt/scripts/debugging/carpet07_rgb_direct.png",
                img_u8)  # deliberate: no color conversion
    cv2.imwrite("/home/zla247/projects/BRDF-Fipt/scripts/debugging/carpet07_rgb2bgr.png",
                cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite("/home/zla247/projects/BRDF-Fipt/scripts/debugging/carpet07_bgr2rgb.png",
                cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB))
    print("Saved test images to scripts/debugging/")
except Exception as e:
    print(f"Could not save images: {e}")
