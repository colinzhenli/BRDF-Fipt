import json
import os
import glob
import torch

def load_camera_light_metadata(json_path):
    """
    Load camera and light relation from JSON file and create metadata for images.
    
    Args:
        json_path (str): Path to the JSON file containing camera-light relations
        image_folder (str): Path to folder containing images with naming pattern 
                           "scan-{light_id}-{camera_id}-phi{phi}_theta{theta}.png"
    
    Returns:
        dict: Metadata containing:
            - metadata: list of dicts with keys ['id', 'camera_id', 'light_id', 'filename']
            - camera_metadata: dict mapping camera_id to camera info
            - emitter_metadata: dict mapping light_id to light info
    """
    # Load JSON file
    with open(json_path, 'r') as f:
        camera_light_data = json.load(f)
    
    # Create camera and emitter metadata dictionaries
    camera_metadata = {}
    emitter_metadata = {}
    
    # Create metadata list
    metadata = []
    overall_id = 0
    
    for item in camera_light_data:
        camera_id = item['id']
        light_id = item['light_id']
        phi = item['phi']
        theta = item['theta']
        
        # Store camera metadata only for non-appeared camera id
        if str(camera_id) not in camera_metadata:
            camera_metadata[str(camera_id)] = {
                'position': [pos / 1000.0 for pos in item['position']],
                'rotation_matrix': item['rotation_matrix'],
                'euler': item['euler']
            }
        
        # Store emitter metadata (light info)
        if str(light_id) not in emitter_metadata:
            emitter_metadata[str(light_id)] = {
                'position': [pos / 1000.0 for pos in item['position_light']],
                'rotation_matrix': item['rotation_matrix_light']
            }
        
        # Generate filename based on the pattern: scan-{light_id}-{camera_id}-phi{phi}_theta{theta}.png
        # Round phi and theta to 3 decimal places
        phi_rounded = f"{phi:.3f}"
        theta_rounded = f"{theta:.3f}"
        filename = f"scan-{light_id}-{camera_id}-phi{phi_rounded}_theta{theta_rounded}.png"
        
        # Create metadata entry
        metadata.append({
            'id': overall_id,
            'camera_id': str(camera_id),
            'emitter_id': light_id,  # Using emitter_id to match SphereImageDataset
            'filename': filename
        })
        overall_id += 1
    
    return metadata, camera_metadata, emitter_metadata

def load_camera_metadata(json_path):
    """
    Load only camera metadata from JSON file.
    
    Args:
        json_path (str): Path to the JSON file containing camera and light data
        
    Returns:
        dict: Dictionary mapping camera_id (str) to camera metadata
    """
    import json
    
    # Load JSON file
    with open(json_path, 'r') as f:
        camera_light_data = json.load(f)
    
    # Create camera metadata dictionary
    camera_metadata = {}
    
    for item in camera_light_data:
        camera_id = item['id']
        
        # Store camera metadata only for non-appeared camera id
        if str(camera_id) not in camera_metadata:
            camera_metadata[str(camera_id)] = {
                'position': [pos / 1000.0 for pos in item['position']],
                'rotation_matrix': item['rotation_matrix'],
                'euler': item['euler']
            }
    
    return camera_metadata

def rotation_position_to_light2world(rotation_matrix, position):
    """
    Convert rotation matrix and position to light-to-world transformation matrix.
    
    Args:
        rotation_matrix (list or tensor): 3x3 rotation matrix
        position (list or tensor): 3D position vector
        
    Returns:
        torch.Tensor: 4x4 light-to-world transformation matrix
    """

    
    # Convert to tensors if needed
    if not isinstance(rotation_matrix, torch.Tensor):
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)
    if not isinstance(position, torch.Tensor):
        position = torch.tensor(position, dtype=torch.float32)
    
    # Ensure proper shapes
    rotation_matrix = rotation_matrix.view(3, 3)
    position = position.view(3)
    
    # Create 4x4 transformation matrix
    light2world = torch.eye(4, dtype=torch.float32)
    light2world[:3, :3] = rotation_matrix
    light2world[:3, 3] = position / 1000.0 # convert to meter
    
    return light2world

def read_light_transforms(json_path):
    """
    Read light data from JSON file and return light-to-world transformation matrices.
    
    Args:
        json_path (str): Path to the JSON file containing light data
        
    Returns:
        torch.Tensor: (N, 4, 4) tensor where N is number of lights,
                     each 4x4 matrix is a light-to-world transformation
    """
    import json
    import torch
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    light_transforms = []
    
    # Read the first position for each unique light ID
    seen_light_ids = set()
    
    for item in data:
        light_id = item.get('light_id', 0)
        
        # Only process if we haven't seen this light ID before
        if light_id not in seen_light_ids:
            seen_light_ids.add(light_id)
            
            rotation_matrix = item['rotation_matrix_light']
            position = item['position_light'] # convert to meter
            
            # Convert to light-to-world transformation matrix
            light2world = rotation_position_to_light2world(rotation_matrix, position)
            light_transforms.append(light2world)
    
    return torch.stack(light_transforms).cuda()  # (N, 4, 4)
