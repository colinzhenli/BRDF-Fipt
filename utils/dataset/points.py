import torch
import torch.nn.functional as NF
from torch.utils.data import Dataset
import json
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import math
from pathlib import Path
from torch.utils.data import IterableDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.io import load_camera_turntable_light_metadata, load_camera_metadata, load_camera_metadata_from_robotic_log
import threading, queue, time
from dataclasses import dataclass
import random

def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T

def _cv_to_gl(cv):
    # convert to GL convention used in iNGP
    gl = cv * torch.tensor([1, -1, -1, 1])
    return gl

def get_ray_directions(H, W, focal, cx, cy, distortion):
    """ get camera ray direction with radial distortion correction, using opengl convention
    Args:
        H,W: height and width
        focal: focal length
        cx, cy: principal point coordinates
        distortion: radial distortion coefficient k1
    """
    x_coords = torch.linspace(0.5, W - 0.5, W)
    y_coords = torch.linspace(0.5, H - 0.5, H)
    j, i = torch.meshgrid([y_coords, x_coords])
    
    # Convert to normalized coordinates relative to principal point
    x_norm = (i - cx) / focal
    y_norm = (j - cy) / focal
    
    # Apply radial distortion correction
    r_squared = x_norm**2 + y_norm**2
    distortion_factor = 1 + distortion * r_squared
    
    x_corrected = x_norm * distortion_factor
    y_corrected = y_norm * distortion_factor
    
    directions = torch.stack([x_corrected, -y_corrected, -torch.ones_like(i)], -1)

    return directions

def get_rays(directions, c2w, focal=None):
    """ world space camera ray
    Args:
        directions: camera ray direction (local)
        c2w: 3x4 camera to world matrix
        focal: if not None, return ray differentials as well
    """
    R = c2w[:,:3]
    rays_d = directions @ R.T
    
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    if focal is not None:
        dxdu = torch.tensor([1.0/focal,0,0])[None,None].expand_as(directions)@R.T
        dydv = torch.tensor([0,1.0/focal,0])[None,None].expand_as(directions)@R.T
        dxdu = dxdu.view(-1,3)
        dydv = dydv.view(-1,3)
        return rays_o, rays_d, dxdu, dydv
    else:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        return rays_o, rays_d

def read_image(path, img_hw):
    img = plt.imread(path)[...,:3]
    assert img.shape[0] == img_hw[0]
    assert img.shape[1] == img_hw[1]
    return torch.from_numpy(img.astype(np.float32))

def open_exr(file,img_hw):
    """ open image exr file """
    img = cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
    assert img.shape[0] == img_hw[0]
    assert img.shape[1] == img_hw[1]
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = img[...,[2,1,0]]
    img = torch.from_numpy(img.astype(np.float32))
    return img

def get_c2w(camera):
    position = torch.tensor(camera['position'], dtype=torch.float32)
    target = torch.tensor(camera['look_at'], dtype=torch.float32)
    up = torch.tensor(camera.get('up', [0,1,0]), dtype=torch.float32)
     
    forward = target - position
    forward = forward / torch.norm(forward)
    # Ensure `up` is not parallel to `forward`
    if torch.abs(torch.dot(forward, up)) > 0.99:  # Too parallel, adjust up
        up = torch.tensor([1.0, 0.0, 0.0]) if torch.abs(forward[0]) < 0.99 else torch.tensor([0.0, 1.0, 0.0])
    """ right hand coordinate system """
    right = torch.cross(up, forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)
    
    c2w = torch.eye(4)
    c2w[:3,:3] = torch.stack([right, up, forward], dim=1)
    c2w[:3,3] = position
    c2w = _cv_to_gl(c2w)
    c2w = c2w[:3,:4]
    return c2w

def get_c2w_from_robot_pose(camera_info, R_c2g, t_c2g):
    """ get camera to world matrix from robot pose """
    g2w = build_4x4(camera_info["rotation_matrix"], camera_info["position"])
    c2g = build_4x4(R_c2g, t_c2g)
    c2w = g2w @ c2g
    return torch.from_numpy(c2w[:3, :4]).float()

def get_ray_directions_for_pixels(pixel_coords, focal, cx, cy, distortion):
    """
    Get camera ray directions for specific pixel coordinates with radial distortion correction.
    
    Args:
        pixel_coords: (N, 2) tensor of [u, v] pixel coordinates
        focal: focal length
        cx, cy: principal point coordinates
        distortion: radial distortion coefficient k1
        
    Returns:
        directions: (N, 3) tensor of ray directions in camera space
    """
    u = pixel_coords[:, 0]  # (N,)
    v = pixel_coords[:, 1]  # (N,)
    
    # Convert to normalized coordinates relative to principal point
    x_norm = (u - cx) / focal
    y_norm = (v - cy) / focal
    
    # Apply radial distortion correction
    r_squared = x_norm**2 + y_norm**2
    distortion_factor = 1 + distortion * r_squared
    
    x_corrected = x_norm * distortion_factor
    y_corrected = y_norm * distortion_factor
    
    # Stack into direction vectors (OpenGL convention: -y, -z)
    directions = torch.stack([x_corrected, -y_corrected, -torch.ones_like(u)], dim=-1)  # (N, 3)
    
    return directions

class MultiMaterialPointDataset(IterableDataset if True else Dataset):
    """
    Dataset for multiple materials loaded from point observations.
    Each material has its own folder containing:
    - hdr/ (images, not loaded)
    - scan_log.json (emitter metadata)
    - rotated_camera.json (camera poses)
    - point_metadata.json (contains num_points for this material)
    - observations/ (chunked observation files: observations_chunk_00.npz, observations_chunk_01.npz, ...)
      OR sparse/observations.npz (legacy single file format)
    
    Each observation chunk contains: [x, y, z, image_id, pixel_x, pixel_y, r, g, b, point_id]
    where point_id is the LOCAL point index (0 to num_points-1) within this material.
    
    This class uses double buffering to randomly load chunks per material in the background,
    keeping memory usage constant while providing fresh data every N iterations.
    Supports both training and validation splits from the same data.
    """
    
    # ------------------------------
    # Embedded helper classes
    # ------------------------------
    @dataclass
    class ChunkData:
        """Container for loaded chunk data."""
        rays: torch.Tensor
        rgbs: torch.Tensor
        xyz: torch.Tensor
        camera_ids: torch.Tensor
        emitter_ids: torch.Tensor
        material_ids: torch.Tensor
        point_ids: torch.Tensor  # LOCAL point IDs (per-material)
    
    class _DoubleBuffer:
        """Two RAM slots with a background thread that fills the inactive slot."""
        def __init__(self, build_chunk_fn):
            self.build_chunk_fn = build_chunk_fn           # fn()->ChunkData
            self.slots = [None, None]
            self.ready = [threading.Event(), threading.Event()]
            self.active = 0
            self._stop = False
            self._q = queue.Queue(maxsize=2)
            self._t = threading.Thread(target=self._worker, daemon=True)
            self._t.start()
        
        def _worker(self):
            while not self._stop:
                try:
                    slot_id = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                data = self.build_chunk_fn()  # Build new chunk
                self.slots[slot_id] = data
                self.ready[slot_id].set()
        
        def request_fill(self, slot_id):
            self.ready[slot_id].clear()
            try:
                self._q.put_nowait(slot_id)
            except queue.Full:
                # Drop oldest request to keep moving
                try:
                    _ = self._q.get_nowait()
                except queue.Empty:
                    pass
                self._q.put_nowait(slot_id)
        
        def wait_initial(self):
            self.ready[self.active].wait()
        
        def try_swap(self):
            nxt = 1 - self.active
            if self.ready[nxt].is_set():
                old = self.active
                self.active = nxt
                self.ready[old].clear()  # Clear old slot's ready flag
                return True
            return False
        
        def current(self):
            self.ready[self.active].wait()
            return self.slots[self.active]
        
        def stop(self):
            self._stop = True
            self._t.join(timeout=1.0)
    
    def __init__(self, cfg, root_folder, split='train'):
        """
        Args:
            cfg: configuration object
            root_folder: path to folder containing material subfolders (0, 1, 2, ...)
            split: 'train' or 'val'
        """
        self.cfg = cfg
        self.root_folder = root_folder
        self.split = split
        self.rays_num = cfg.data.rays_num
        
        # Camera intrinsics
        self.intrinsics = cfg.renderer.camera.intrinsics
        self.focal = self.intrinsics['focal_length']
        self.cx = self.intrinsics['cx']
        self.cy = self.intrinsics['cy']
        self.distortion = self.intrinsics['distortion']
        self.img_hw = (self.intrinsics['height'], self.intrinsics['width'])
        
        # Color correction matrix
        self.ccm = np.array(cfg.data.ccm)
        
        # Train/val split ratio
        self.val_ratio = getattr(cfg.data, 'val_ratio', 0.1)
        
        # Double buffer settings
        self.switch_iters = getattr(cfg.data, 'switch_iters', 1000)  # How often to reload chunks
        self.step = 0
        
        # Debug mode settings
        self.debug = getattr(cfg.data, 'debug', False)
        self.debug_num_materials = getattr(cfg.data, 'debug_num_materials', 1)
        
        # Discover material folders
        self.material_folders = sorted([
            d for d in Path(root_folder).iterdir() 
            if d.is_dir() and d.name.isdigit()
        ], key=lambda x: int(x.name))
        
        # In debug mode, limit to first N materials (sorted by folder name)
        if self.debug:
            original_count = len(self.material_folders)
            self.material_folders = self.material_folders[:self.debug_num_materials]
            print(f"\n[DEBUG MODE] Limiting materials from {original_count} to {len(self.material_folders)}")
        
        print(f"\n{'='*60}")
        print(f"Loading MultiMaterial Dataset ({split})")
        print(f"{'='*60}")
        print(f"Found {len(self.material_folders)} material folders: {[int(f.name) for f in self.material_folders]}")
        
        # Discover chunks and load metadata (lightweight)
        self._discover_chunks_and_metadata()
        
        # Initialize based on split
        if split == 'train':
            print(f"\nInitializing double buffer (chunk reload every {self.switch_iters} iters)...")
            self._dbuf = MultiMaterialPointDataset._DoubleBuffer(lambda: self._load_chunks(split='train')   )
            # Prefill two | 91745/335183748ctive) and slot 1 (next)
            self._dbuf.request_fill(0)
            self._dbuf.request_fill(1)
            self._dbuf.wait_initial()  # Ensure first active chunk exists
            print("Double buffer initialized!")
        else:
            # For validation, load all validation observations directly
            print(f"\nLoading validation data...")
            chunk_data = self._load_chunks(split='val', load_all=True)
            self.all_rays = chunk_data.rays
            self.all_rgbs = chunk_data.rgbs
            self.all_xyz = chunk_data.xyz
            self.all_camera_ids = chunk_data.camera_ids
            self.all_emitter_ids = chunk_data.emitter_ids
            self.all_material_ids = chunk_data.material_ids
            self.all_point_ids = chunk_data.point_ids
            print(f"Validation data loaded: {len(self.all_rays):,} observations")
        
        print(f"\nDataset ready!")
        print(f"{'='*60}\n")
    
    def _discover_chunks_and_metadata(self):
        """Discover available chunks and load metadata (camera/emitter lookups) for all materials."""
        self.material_chunks = {}  # material_id -> list of chunk file paths
        self.camera_lookups = {}   # material_id -> {camera_id (int) -> c2w matrix}
        self.emitter_lookups = {}  # material_id -> tensor of emitter_ids indexed by overall_id
        
        print("\nDiscovering chunks and loading metadata...")
        for material_folder in tqdm(self.material_folders, desc="Scanning materials"):
            material_id = int(material_folder.name)
            
            # Discover chunk files in observations/ folder
            obs_folder = material_folder / "observations"
            chunk_files = sorted(obs_folder.glob("observations_chunk_*.npz"))
            self.material_chunks[material_id] = chunk_files
            print(f"  Material {material_id}: Found {len(chunk_files)} chunks in {obs_folder}")
            
            # Load metadata using existing utility functions (like real.py does)
            scan_log_path = str(material_folder / "scan_log.json")
            camera_json_path = str(material_folder / "rotated_camera.json")
            
            # load_camera_turntable_light_metadata returns:
            #   metadata: list of dicts with ['overall_id', 'camera_id', 'emitter_id', 'filename', 'turn_angle']
            #   camera_metadata: not used here (we use rotated_camera.json instead)
            #   (position already in meters)
            metadata_list, _, _ = load_camera_turntable_light_metadata(scan_log_path)
            
            # load_camera_metadata returns dict {str(camera_id) -> {'position': [...], 'rotation_matrix': [...]}}
            # (position already in meters)
            camera_metadata = load_camera_metadata(camera_json_path)
            
            # Build camera lookup: camera_id (int) -> c2w (4x4 torch tensor)
            camera_lookup = {}
            for cam_id_str, cam_info in camera_metadata.items():
                position = np.array(cam_info['position'])  # already in meters
                rotation_matrix = np.array(cam_info['rotation_matrix'])
                c2w = build_4x4(rotation_matrix, position)
                camera_lookup[int(cam_id_str)] = torch.from_numpy(c2w).float()
            self.camera_lookups[material_id] = camera_lookup
            
            # Build emitter lookup as tensor: index by overall_id to get emitter_id
            # Sort by overall_id to ensure correct indexing
            sorted_metadata = sorted(metadata_list, key=lambda x: int(x['overall_id']))
            emitter_ids = np.array([int(entry['emitter_id']) for entry in sorted_metadata])
            self.emitter_lookups[material_id] = torch.from_numpy(emitter_ids).long()
        
        print(f"Metadata loaded for {len(self.material_chunks)} materials")
    
    def set_step(self, step: int):
        """Called by training loop to track current step for chunk reloading."""
        self.step = step
        # Request next chunk build at boundaries; swap happens lazily in __iter__
        if hasattr(self, '_dbuf') and step > 0 and step % self.switch_iters == 0:
            next_slot = 1 - self._dbuf.active
            print(f"[Step {step}] Requesting new random chunks to load into slot {next_slot}")
            self._dbuf.request_fill(next_slot)
    
    def _load_chunks(self, split='train', load_all=False) -> "MultiMaterialPointDataset.ChunkData":
        """
        Load chunks and filter by split at CHUNK level.
        
        Args:
            split: 'train' or 'val' - determines which chunks to use
                   First (1-val_ratio) chunks for training, last val_ratio chunks for validation
            load_all: if True, load ALL chunks in the split (for validation); 
                      if False, load one random chunk from the split (for training)
        
        Returns:
            ChunkData with observations from the selected chunks
        """
        all_rays = []
        all_rgbs = []
        all_xyz = []
        all_emitter_ids = []
        all_camera_ids = []
        all_material_ids = []
        all_point_ids = []
        
        is_val = (split == 'val')
        desc = f"Loading {'all' if load_all else 'random'} chunks for {split}"
        print(f"\n[{'Main' if is_val else 'Background'}] {desc}...")
        
        for material_folder in (tqdm(self.material_folders, desc=desc) if load_all else self.material_folders):
            material_id = int(material_folder.name)
            
            # Get metadata (already loaded, thread-safe to read)
            camera_lookup = self.camera_lookups[material_id]
            emitter_lookup = self.emitter_lookups[material_id]
            
            # Split chunks: first (1-val_ratio) for training, last val_ratio for validation
            all_chunks = self.material_chunks[material_id]  # Already sorted
            n_chunks = len(all_chunks)
            split_idx = int(n_chunks * (1 - self.val_ratio))
            
            if is_val:
                split_chunks = all_chunks[split_idx:]  # Last val_ratio chunks for validation
            else:
                split_chunks = all_chunks[:split_idx]  # First (1-val_ratio) chunks for training
            
            if len(split_chunks) == 0:
                print(f"  Warning: Material {material_id} has no {split} chunks!")
                continue
            
            # Determine which chunks to actually load
            if load_all:
                chunks_to_load = split_chunks  # Load all chunks in this split
            else:
                chunks_to_load = [random.choice(split_chunks)]  # Random single chunk from this split
            
            for chunk_path in chunks_to_load:
                # Load observations from chunk
                obs_data = np.load(chunk_path)
                observations = obs_data['observations']  # (N, 10): [x, y, z, image_id, pixel_x, pixel_y, r, g, b, point_id]
                
                if len(observations) == 0:
                    continue
                
                if not load_all:
                    print(f"  [Background] Material {material_id}: Loaded {chunk_path.name} ({len(observations):,} obs)")
                
                # Process observations - vectorized
                xyz = torch.from_numpy(observations[:, :3]).float()
                image_ids = observations[:, 3].astype(np.int32) - 1 # Colmap starts at 1
                image_ids_t = torch.from_numpy(image_ids).long()
                pixel_coords = torch.from_numpy(observations[:, 4:6]).float()
                # Apply color correction matrix to RGB values
                rgbs_np = observations[:, 6:9].astype(np.float64) @ self.ccm
                rgbs_np = rgbs_np.clip(0, None)
                rgbs = torch.from_numpy(rgbs_np).float()
                point_ids = torch.from_numpy(observations[:, 9].astype(np.int64)).long()
                
                # Vectorized: emitter_ids and camera_ids via direct tensor indexing
                chunk_emitter_ids = emitter_lookup[image_ids_t]  # (N,)
                chunk_camera_ids = image_ids_t.clone()  # (N,)
                
                # Generate rays - requires loop over unique images (get_rays needs single c2w)
                rays_list = []
                valid_mask_list = []
                
                # Group by image_id for ray generation
                unique_image_ids = np.unique(image_ids)
                for img_id in unique_image_ids:
                    if img_id not in camera_lookup:
                        continue
                    
                    c2w_full = camera_lookup[img_id]
                    c2w = c2w_full[:3, :4]
                    
                    # Get mask for this image
                    mask = image_ids_t == img_id
                    pixels = pixel_coords[mask]
                    
                    directions = get_ray_directions_for_pixels(
                        pixels, self.focal, self.cx, self.cy, self.distortion
                    )
                    
                    rays_o, rays_d = get_rays(directions, c2w, focal=None)
                    rays = torch.cat([rays_o, rays_d], dim=-1)
                    
                    rays_list.append((mask, rays))
                    valid_mask_list.append(mask)
                
                # Combine rays back into original order
                if len(rays_list) > 0:
                    # Create output tensor and fill in rays at correct positions
                    valid_mask = torch.zeros(len(image_ids_t), dtype=torch.bool)
                    for mask, _ in rays_list:
                        valid_mask |= mask
                    
                    chunk_rays = torch.zeros(valid_mask.sum(), 6)
                    chunk_xyz = xyz[valid_mask]
                    chunk_rgbs = rgbs[valid_mask]
                    chunk_point_ids = point_ids[valid_mask]
                    chunk_emitter_ids = chunk_emitter_ids[valid_mask]
                    chunk_camera_ids = chunk_camera_ids[valid_mask]
                    
                    # Map original indices to valid indices
                    valid_indices = torch.where(valid_mask)[0]
                    idx_map = torch.full((len(image_ids_t),), -1, dtype=torch.long)
                    idx_map[valid_indices] = torch.arange(len(valid_indices))
                    
                    for mask, rays in rays_list:
                        mapped_idx = idx_map[mask]
                        chunk_rays[mapped_idx] = rays
                    
                    chunk_material_ids = torch.full((chunk_rays.shape[0],), material_id, dtype=torch.long)
                    
                    all_rays.append(chunk_rays)
                    all_rgbs.append(chunk_rgbs)
                    all_xyz.append(chunk_xyz)
                    all_emitter_ids.append(chunk_emitter_ids)
                    all_camera_ids.append(chunk_camera_ids)
                    all_material_ids.append(chunk_material_ids)
                    all_point_ids.append(chunk_point_ids)
        
        # Concatenate across all materials
        rays = torch.cat(all_rays, dim=0)
        rgbs = torch.cat(all_rgbs, dim=0)
        xyz = torch.cat(all_xyz, dim=0)
        emitter_ids = torch.cat(all_emitter_ids, dim=0)
        camera_ids = torch.cat(all_camera_ids, dim=0)
        material_ids = torch.cat(all_material_ids, dim=0)
        point_ids = torch.cat(all_point_ids, dim=0)
        
        print(f"[{'Main' if is_val else 'Background'}] Chunk built: {len(rays):,} total {split} observations")
        
        return MultiMaterialPointDataset.ChunkData(
            rays=rays, rgbs=rgbs, xyz=xyz,
            camera_ids=camera_ids, emitter_ids=emitter_ids, material_ids=material_ids,
            point_ids=point_ids
        )
    
    def __len__(self):
        """Return number of iterations (batches) in this split."""
        if hasattr(self, 'all_rays'):
            # For validation: return number of batches to iterate through all data once
            return math.ceil(len(self.all_rays) / self.rays_num)
        else:
            # For training with double buffer, return a large number
            return 1000000
    
    def __iter__(self):
        """Infinite iterator for training (samples random batches)."""
        # Training mode with double buffer
        if hasattr(self, '_dbuf'):
            while True:
                # Non-blocking swap if next slot ready
                if self._dbuf.try_swap():
                    print(f"[Step {self.step}] ✓ Switched to new chunk (slot {self._dbuf.active})")
                
                # Use current active chunk
                chunk = self._dbuf.current()
                if chunk is None or chunk.rays.numel() == 0:
                    time.sleep(0.01)
                    continue
                
                # Random sample rays_num rays from current chunk
                total_rays = chunk.rays.shape[0]
                sample_idx = torch.randint(0, total_rays, (self.rays_num,), dtype=torch.long)
                
                yield {
                    'rays': chunk.rays[sample_idx],
                    'rgbs': chunk.rgbs[sample_idx],
                    'xyz': chunk.xyz[sample_idx],
                    'emitter_ids': chunk.emitter_ids[sample_idx],
                    'camera_ids': chunk.camera_ids[sample_idx],
                    'material_ids': chunk.material_ids[sample_idx],
                    'point_ids': chunk.point_ids[sample_idx],
                    'gt_params': torch.zeros(1),
                }
        
        # Validation mode with static data - finite iterator through all data once
        else:
            total_rays = self.all_rays.shape[0]
            num_batches = math.ceil(total_rays / self.rays_num)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.rays_num
                end_idx = min(start_idx + self.rays_num, total_rays)
                
                yield {
                    'rays': self.all_rays[start_idx:end_idx],
                    'rgbs': self.all_rgbs[start_idx:end_idx],
                    'xyz': self.all_xyz[start_idx:end_idx],
                    'emitter_ids': self.all_emitter_ids[start_idx:end_idx],
                    'camera_ids': self.all_camera_ids[start_idx:end_idx],
                    'material_ids': self.all_material_ids[start_idx:end_idx],
                    'point_ids': self.all_point_ids[start_idx:end_idx],
                    'gt_params': torch.zeros(1),
                }
    