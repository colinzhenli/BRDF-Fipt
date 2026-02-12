#!/usr/bin/env python3
"""
Visualize the light direction distribution of a dataset.

This script reads:
1. bbox.json to get the center position
2. scan_log.json to get all light positions (with proper transforms applied)

Then computes incident directions from lights to center, transforms to local (theta, phi) space,
and plots two figures:
- Validation set: first 240 images (by scan_id)
- Training set: remaining images

Usage:
    python scripts/visualize_light_distribution.py /path/to/dataset_folder
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.transform import build_rot_about_point, rodrigues_axis_angle
from utils.io import rotation_position_to_light2world


# Default calibration parameters (from config/renderer/multiarea_emitter.yaml)
DEFAULT_TURNTABLE_CENTER = [0.16084722, -0.11011424, -0.021]
DEFAULT_TURNTABLE_AXIS = [-0.00876202, -0.01346449, 0.99987096]
DEFAULT_BASE2_TO_BASE1 = [
    [9.99999959e-01, 2.33227594e-04, -1.69051819e-04, 7.22042468e-03],
    [-2.36740680e-04, 9.99777598e-01, -2.10878871e-02, 8.26339062e-03],
    [1.64095944e-04, 2.10879262e-02, 9.99777611e-01, 3.93500345e-03],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
]


def load_bbox_center(dataset_folder):
    """Load the center position from bbox.json."""
    bbox_path = os.path.join(dataset_folder, "bbox.json")
    if not os.path.exists(bbox_path):
        raise FileNotFoundError(f"bbox.json not found at {bbox_path}")
    
    with open(bbox_path, 'r') as f:
        bbox_data = json.load(f)
    
    center = np.array(bbox_data['bbox_center'])
    print(f"Loaded center from bbox.json: {center}")
    return center


def read_light_transforms_numpy(json_path, turntable_center, turntable_axis, base2_to_base1):
    """
    Read light data from JSON file and return light-to-world transformation matrices.
    This follows the same logic as utils/io.py read_light_transforms but returns per-scan data.
    
    Args:
        json_path: Path to scan_log.json
        turntable_center: (3,) turntable center position
        turntable_axis: (3,) turntable rotation axis
        base2_to_base1: (4, 4) transformation from robot base2 to base1
        
    Returns:
        list of tuples: [(scan_id, light_position_world), ...]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    turntable_center = np.array(turntable_center)
    turntable_axis = np.array(turntable_axis)
    base2_to_base1 = np.array(base2_to_base1)
    
    light_data = []
    
    for item in data:
        scan_id = item['scan_id']
        turn_angle = item.get('turn_angle', 0.0)
        
        rotation_matrix = np.array(item['rotation_matrix_light'])
        position = np.array(item['position_light'])  # in mm
        
        # Convert to light-to-world transformation matrix (position converted to meters)
        light2world = np.eye(4)
        light2world[:3, :3] = rotation_matrix
        light2world[:3, 3] = position / 1000.0  # mm to meters
        
        # Apply base2_to_base1 transformation
        light2world = base2_to_base1 @ light2world
        
        # Transform world back to 0-angle world (undo turntable rotation)
        R_turn = rodrigues_axis_angle(turntable_axis, -turn_angle)
        Tw2w0 = build_rot_about_point(R_turn, turntable_center)
        light2world = Tw2w0 @ light2world
        
        # Extract position from transformation matrix
        light_position = light2world[:3, 3]
        light_data.append((scan_id, light_position))
    
    print(f"Loaded {len(light_data)} light positions from scan_log.json (with transforms)")
    return light_data


def compute_incident_direction(light_pos, center):
    """
    Compute incident direction from surface center to light (normalized).
    
    In BRDF convention, incident direction (wi) points FROM surface TOWARD light.
    
    Args:
        light_pos: (3,) light position
        center: (3,) surface center position
    
    Returns:
        (3,) normalized incident direction (pointing from surface to light)
    """
    direction = light_pos - center  # FROM surface TO light
    direction = direction / np.linalg.norm(direction)
    return direction


def direction_to_theta_phi(direction):
    """
    Convert a 3D direction to spherical coordinates (theta, phi).
    
    Assumes z-up coordinate system with surface normal (0, 0, 1):
    - theta: angle between incident direction and normal (0° = from above, 90° = grazing)
    - phi: azimuthal angle in xy-plane from x-axis (0° to 360°)
    
    Args:
        direction: (3,) normalized direction vector (pointing from light to surface)
    
    Returns:
        theta: angle from normal in radians [0, pi/2]
        phi: azimuthal angle in radians [0, 2*pi]
    """
    x, y, z = direction
    
    # direction points from surface to light (BRDF incident direction wi)
    # theta = angle between wi and normal (0,0,1)
    # cos(theta) = dot(direction, normal) = z
    cos_theta = z
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    # phi: azimuthal angle in xy-plane from x-axis [0, 2*pi]
    phi = np.arctan2(y, x)
    # Convert from [-pi, pi] to [0, 2*pi]
    if phi < 0:
        phi += 2 * np.pi
    
    return theta, phi


def plot_theta_phi_distribution(theta_phi_list, title, output_path):
    """
    Plot the distribution of directions in theta-phi space.
    
    Args:
        theta_phi_list: list of (theta, phi) tuples in radians
        title: plot title
        output_path: path to save the figure
    """
    thetas = np.array([tp[0] for tp in theta_phi_list])
    phis = np.array([tp[1] for tp in theta_phi_list])
    
    # Convert to degrees for better readability
    thetas_deg = np.degrees(thetas)
    phis_deg = np.degrees(phis)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(phis_deg, thetas_deg, c=range(len(thetas_deg)), 
                         cmap='viridis', alpha=0.7, s=20)
    
    ax.set_xlabel('Phi (azimuthal angle) [degrees]', fontsize=12)
    ax.set_ylabel('Theta (angle from normal) [degrees]', fontsize=12)
    ax.set_title(f'{title}\n({len(theta_phi_list)} samples)', fontsize=14)
    
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 90)
    
    ax.grid(True, alpha=0.3)
    ax.axhline(y=45, color='r', linestyle='--', alpha=0.5, label='45° from normal')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sample index (scan order)', fontsize=10)
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize light direction distribution of a dataset')
    parser.add_argument('dataset_folder', type=str,
                        help='Path to the dataset folder containing bbox.json and scan_log.json')
    parser.add_argument('--val_split', type=int, default=120,
                        help='Number of samples for validation set (default: 240)')
    parser.add_argument('--turntable_center', type=float, nargs=3, default=None,
                        help='Turntable center [x, y, z] (default: from config)')
    parser.add_argument('--turntable_axis', type=float, nargs=3, default=None,
                        help='Turntable axis [x, y, z] (default: from config)')
    args = parser.parse_args()
    
    dataset_folder = args.dataset_folder
    val_split = args.val_split
    
    # Use default or provided calibration parameters
    turntable_center = args.turntable_center if args.turntable_center else DEFAULT_TURNTABLE_CENTER
    turntable_axis = args.turntable_axis if args.turntable_axis else DEFAULT_TURNTABLE_AXIS
    base2_to_base1 = DEFAULT_BASE2_TO_BASE1
    
    # Validate folder exists
    if not os.path.isdir(dataset_folder):
        raise ValueError(f"Dataset folder does not exist: {dataset_folder}")
    
    print(f"Processing dataset: {dataset_folder}")
    print(f"Validation split: first {val_split} samples")
    print(f"Turntable center: {turntable_center}")
    print(f"Turntable axis: {turntable_axis}")
    
    # Load data
    center = load_bbox_center(dataset_folder)
    scan_log_path = os.path.join(dataset_folder, "scan_log.json")
    light_data = read_light_transforms_numpy(scan_log_path, turntable_center, turntable_axis, base2_to_base1)
    
    # Sort by scan_id to ensure correct ordering
    light_data_sorted = sorted(light_data, key=lambda x: x[0])
    
    # Compute incident directions and convert to theta-phi
    theta_phi_all = []
    for scan_id, light_pos in light_data_sorted:
        direction = compute_incident_direction(light_pos, center)
        theta, phi = direction_to_theta_phi(direction)
        theta_phi_all.append((theta, phi))
    
    # Split into validation and training sets
    theta_phi_val = theta_phi_all[:val_split]
    theta_phi_train = theta_phi_all[val_split:]
    
    print(f"Validation set: {len(theta_phi_val)} samples")
    print(f"Training set: {len(theta_phi_train)} samples")
    
    # Generate output paths
    val_output = os.path.join(dataset_folder, "light_distribution_validation.png")
    train_output = os.path.join(dataset_folder, "light_distribution_training.png")
    
    # Plot figures
    plot_theta_phi_distribution(theta_phi_val, 
                                "Light Direction Distribution - Validation Set",
                                val_output)
    
    plot_theta_phi_distribution(theta_phi_train,
                                "Light Direction Distribution - Training Set", 
                                train_output)
    
    print("\nDone!")
    print(f"Validation figure: {val_output}")
    print(f"Training figure: {train_output}")


if __name__ == "__main__":
    main()
