import numpy as np
import os

def generate_roughness_metallic_pairs(roughness_range=(0.2, 0.9), metallic_range=(0.2, 0.9), 
                                      num_roughness=8, num_metallic=8):
    """
    Generate evenly spaced roughness and metallic pairs within the specified ranges.
    
    Args:
        roughness_range: Tuple of (min, max) roughness values
        metallic_range: Tuple of (min, max) metallic values
        num_roughness: Number of roughness values to generate
        num_metallic: Number of metallic values to generate
        
    Returns:
        List of (roughness, metallic) pairs
    """
    roughness_values = np.linspace(roughness_range[0], roughness_range[1], num_roughness)
    metallic_values = np.linspace(metallic_range[0], metallic_range[1], num_metallic)
    
    pairs = []
    for r in roughness_values:
        for m in metallic_values:
            pairs.append((round(r, 2), round(m, 2)))
    
    return pairs

def split_and_save_pairs(pairs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, output_dir="data"):
    """
    Split the pairs into train, validation, and test sets and save them to text files.
    
    Args:
        pairs: List of (roughness, metallic) pairs
        train_ratio: Ratio of pairs to use for training
        val_ratio: Ratio of pairs to use for validation
        test_ratio: Ratio of pairs to use for testing
        output_dir: Directory to save the output files
    """
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Shuffle the pairs
    np.random.shuffle(pairs)
    
    # Calculate split indices
    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # Split the pairs
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train+n_val]
    test_pairs = pairs[n_train+n_val:]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to text files
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        for r, m in train_pairs:
            f.write(f"{r} {m}\n")
    
    with open(os.path.join(output_dir, "val.txt"), "w") as f:
        for r, m in val_pairs:
            f.write(f"{r} {m}\n")
    
    with open(os.path.join(output_dir, "test.txt"), "w") as f:
        for r, m in test_pairs:
            f.write(f"{r} {m}\n")
    
    print(f"Generated {len(train_pairs)} training pairs, {len(val_pairs)} validation pairs, and {len(test_pairs)} test pairs")
    print(f"Files saved to {output_dir}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate roughness-metallic pairs
    pairs = generate_roughness_metallic_pairs(
        roughness_range=(0.1, 0.9),
        metallic_range=(0.1, 0.9),
        num_roughness=8,
        num_metallic=8
    )
    
    # Split and save pairs
    split_and_save_pairs(
        pairs,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        output_dir="metadata"
    )

    
