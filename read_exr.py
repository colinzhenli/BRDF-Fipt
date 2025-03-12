import imageio
imageio.plugins.freeimage.download()
import numpy as np
envmap_path = './envmap.exr'
envmap = imageio.imread(envmap_path).astype('float32')  # Shape: (H, W, 3)
env_map = np.clip(envmap, 0, 1)
envmap_np = env_map ** (1 / 2.2)

envmap_8bit = (envmap_np * 255).astype(np.uint8)

# Save the image
imageio.imwrite('envmap.png', envmap_8bit)
print(f"Saved environment map as envmap.png")