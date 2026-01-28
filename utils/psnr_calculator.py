import numpy as np
import torch
from typing import Union
import os


def calculate_psnr(gt_image: Union[np.ndarray, torch.Tensor, str], 
                   result_image: Union[np.ndarray, torch.Tensor, str],
                   max_value: float = None) -> float:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between ground truth and result images.
    
    Args:
        gt_image: Ground truth image. Can be:
                  - numpy array of shape (H, W) or (H, W, C)
                  - torch tensor of shape (H, W) or (H, W, C)
                  - file path to an image
        result_image: Result image in the same format as gt_image
        max_value: Maximum possible pixel value. If None, automatically determined:
                   - For uint8 images: 255
                   - For float images in [0, 1]: 1.0
                   - For other float images: max of gt_image
    
    Returns:
        PSNR value in dB
    
    Example:
        >>> psnr = calculate_psnr('ground_truth.png', 'result.png')
        >>> print(f"PSNR: {psnr:.2f} dB")
    """
    # Load images if they are file paths
    if isinstance(gt_image, str):
        gt_image = load_image(gt_image)
    if isinstance(result_image, str):
        result_image = load_image(result_image)
    
    # Convert to numpy arrays
    if isinstance(gt_image, torch.Tensor):
        gt_image = gt_image.detach().cpu().numpy()
    if isinstance(result_image, torch.Tensor):
        result_image = result_image.detach().cpu().numpy()
    

    # Ensure same shape
    if gt_image.shape != result_image.shape:
        raise ValueError(f"Image shapes must match. Got gt: {gt_image.shape}, result: {result_image.shape}")
    
    # Determine max_value if not provided
    if max_value is None:
        if gt_image.dtype == np.uint8:
            max_value = 255.0
        elif gt_image.max() <= 1.0 and gt_image.min() >= 0.0:
            max_value = 1.0
        else:
            max_value = gt_image.max()
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((gt_image.astype(np.float64) - result_image.astype(np.float64)) ** 2)
    
    # Handle edge case where images are identical
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 10 * np.log10((max_value ** 2) / mse)
    
    return psnr


def calculate_psnr_batch(gt_images: Union[np.ndarray, torch.Tensor],
                         result_images: Union[np.ndarray, torch.Tensor],
                         max_value: float = None) -> np.ndarray:
    """
    Calculate PSNR for a batch of images.
    
    Args:
        gt_images: Ground truth images of shape (B, H, W) or (B, H, W, C)
        result_images: Result images of shape (B, H, W) or (B, H, W, C)
        max_value: Maximum possible pixel value
    
    Returns:
        Array of PSNR values for each image pair
    """
    # Convert to numpy if needed
    if isinstance(gt_images, torch.Tensor):
        gt_images = gt_images.detach().cpu().numpy()
    if isinstance(result_images, torch.Tensor):
        result_images = result_images.detach().cpu().numpy()
    
    batch_size = gt_images.shape[0]
    psnr_values = np.zeros(batch_size)
    
    for i in range(batch_size):
        psnr_values[i] = calculate_psnr(gt_images[i], result_images[i], max_value)
    
    return psnr_values


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path. Supports various formats including HDR/EXR.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Image as numpy array
    """
    import cv2
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext in ['.exr', '.hdr']:
        # Load HDR/EXR images
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        # Convert BGR to RGB for color images
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Load standard images (PNG, JPG, etc.)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        # Convert BGR to RGB for color images
        if len(img.shape) == 3 and img.shape[2] >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img.astype(np.float32)


def main():
    """Example usage of PSNR calculator."""
    
    gt_image="/home/featurize/work/SGHyperMaterials/output/Hard-material_Dec_4/GT.png"
    result_image="/home/featurize/work/SGHyperMaterials/output/Hard-material_Dec_4/overfitted3.png"
    max_value=255
    psnr = calculate_psnr(gt_image, result_image, max_value)
    print(f"PSNR: {psnr:.2f} dB")


if __name__ == '__main__':
    exit(main())

