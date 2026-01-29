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
import struct
import glob
from utils.dataset.MERL import MERLInterface
from utils.ops import rotate_to_canonical_frame

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
        
        # XY filter bounds from mesh.rectangle config (filter to half the region)
        self.filter_observations = getattr(cfg.data, 'filter_observations', True)
        rect_cfg = cfg.renderer.mesh.rectangle
        self.filter_center = rect_cfg.center  # [x, y, z]
        # Half of the original width/length gives the new region dimensions
        self.filter_half_width = rect_cfg.width / 4  # half of (width/2)
        self.filter_half_length = rect_cfg.length / 4  # half of (length/2)
        print(f"XY filter enabled: center=({self.filter_center[0]:.4f}, {self.filter_center[1]:.4f}), "
              f"half_width={self.filter_half_width:.4f}, half_length={self.filter_half_length:.4f}")
        
        # Train/val split ratio
        self.val_ratio = getattr(cfg.data, 'val_ratio', 0.1)
        
        # Double buffer settings
        self.switch_iters = getattr(cfg.data, 'switch_iters', 1000)  # How often to reload chunks
        self.chunk_size = getattr(cfg.data, 'chunk_size', 200)  # Number of materials to sample per chunk
        self.step = 0
        
        # Read training list from txt file
        self.training_list_path = cfg.data.training_list_path
        self.training_list = []
        with open(self.training_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    self.training_list.append(int(line))
        
        # Build material folders from training list
        self.material_folders = [Path(root_folder) / str(mid) for mid in self.training_list]
        
        print(f"\n{'='*60}")
        print(f"Loading MultiMaterial Dataset ({split})")
        print(f"{'='*60}")
        print(f"Training list path: {self.training_list_path}")
        print(f"Loaded {len(self.training_list)} materials: {self.training_list}")
        
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
            
            # Debug visualization: show points with material_id == 0
            visualize = False
            if visualize:
                import open3d as o3d
                mask = self.all_material_ids == 102
                xyz = self.all_xyz[mask].cpu().numpy()
                # Convert int16 RGB to float [0, 1] for open3d
                rgbs = np.clip(self.all_rgbs[mask].cpu().numpy().astype(np.float32), 0, 65535) / 65535.0
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(rgbs)
                o3d.io.write_point_cloud("/media/raid/cloth/output/visualizaitons/material_102_points.ply", pcd)
                print(f"Saved debug point cloud to material_102_points.ply ({len(xyz)} points)")
        
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
    
    def _filter_observations_by_xy(self, observations: np.ndarray) -> np.ndarray:
        """
        Filter observations to keep only points within the XY region.
        
        Args:
            observations: (N, 10) array with [x, y, z, image_id, pixel_x, pixel_y, r, g, b, point_id]
            
        Returns:
            Filtered observations array
        """
        x = observations[:, 0]
        y = observations[:, 1]
        
        # Keep points within half of the original width/length from center
        x_mask = np.abs(x - self.filter_center[0]) <= self.filter_half_width
        y_mask = np.abs(y - self.filter_center[1]) <= self.filter_half_length
        
        xy_mask = x_mask & y_mask
        return observations[xy_mask]
    
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
        
        # Select material folders: all for validation, random sample for training
        num_to_sample = min(self.chunk_size, len(self.material_folders))
        selected_folders = random.sample(self.material_folders, num_to_sample)
        
        for material_folder in (tqdm(selected_folders, desc=desc)):
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
                split_chunks = all_chunks  # Use all chunks for training
            
            if len(split_chunks) == 0:
                print(f"  Warning: Material {material_id} has no {split} chunks!")
                continue
            
            # Determine which chunks to actually load
            if load_all:
                chunks_to_load = split_chunks  # Load all chunks in this split
            else:
                chunks_to_load = [random.choice(split_chunks)]  # Random single chunk from this split
            
            for chunk_path in chunks_to_load:
                # Load observations from chunk with error handling
                try:
                    obs_data = np.load(chunk_path)
                    observations = obs_data['observations']  # (N, 10): [x, y, z, image_id, pixel_x, pixel_y, r, g, b, point_id]
                except (EOFError, IOError, ValueError, KeyError) as e:
                    print(f"  [Warning] Skipping corrupted chunk {chunk_path}: {type(e).__name__}: {e}")
                    continue
                
                # Filter by XY region
                original_count = len(observations)
                if self.filter_observations:
                    observations = self._filter_observations_by_xy(observations)
                
                if len(observations) == 0:
                    continue
                
                if not load_all:
                    print(f"  [Background] Material {material_id}: Loaded {chunk_path.name} ({len(observations):,}/{original_count:,} obs after XY filter)")
                
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


class MERLBRDFIterableDataset(IterableDataset):
    """
    Iterable dataset for MERL BRDF data using MERLInterface.
    
    In each iteration, samples material_id, wi, wo and calculates rgb using lookup_wiwo.
    
    Args:
        data_folder: Path to folder containing .binary BRDF files
        batch_size: Number of samples per batch (default: 1024)
        split: 'train' or 'val' (default: 'train')
        material_names: Optional list of specific materials to load (without .binary extension)
        device: Device to store tensors on (default: 'cuda' if available, else 'cpu')
    """
    
    def __init__(self, cfg, data_folder, batch_size=1024, split='train', material_names=None, device=None):
        self.cfg = cfg
        self.data_folder = Path(data_folder)
        self.batch_size = batch_size
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.split = split
        
        print(f"\n{'='*60}")
        print(f"Loading MERL BRDF Dataset")
        print(f"{'='*60}")
        print(f"Data folder: {data_folder}")    
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Split: {split}")
        
        # Initialize MERLInterface
        self.merl_interface = MERLInterface(str(self.data_folder), device=self.device)
        
        # Get number of materials
        self.n_materials = len(self.merl_interface.material_names)
        
        print(f"Loaded {self.n_materials} materials")
        print(f"{'='*60}\n")

    
    
    def _sample_direction(self, batch_size):
        """
        Sample random direction vectors in the upper hemisphere (z > 0).
        
        Args:
            batch_size: Number of directions to sample
            
        Returns:
            directions: [batch_size, 3] normalized direction vectors
        """
        # Sample uniform on sphere, then keep only upper hemisphere
        directions = torch.randn(batch_size, 3, device=self.device)
        directions = NF.normalize(directions, dim=-1)
        # Ensure z > 0 (upper hemisphere)
        directions[..., 2] = torch.abs(directions[..., 2])
        # Renormalize to ensure unit length
        directions = NF.normalize(directions, dim=-1)
        return directions
    
    def __iter__(self):
        """Iterator yielding batches of material_id, wi, wo, and rgb."""
        if self.split == 'train':
            while True:
                # Sample random material IDs
                if self.cfg.model.stage == 2:
                    material_id = torch.full((self.batch_size,), 0, dtype=torch.long, device=self.device)
                else:
                    material_id = torch.randint(0, self.n_materials, (self.batch_size,), device=self.device)
                
                # Sample random incoming (wi) and outgoing (wo) directions
                wi = self._sample_direction(self.batch_size)
                wo = self._sample_direction(self.batch_size)

                wi,wo=rotate_to_canonical_frame(wi,wo)
                
                # Calculate rgb using lookup_wiwo
                rgb,_,_,_ = self.merl_interface.lookup_wiwo(wi, wo, material_id)
                
                yield {
                    'material_id': material_id,
                    'wi': wi,
                    'wo': wo,
                    'rgb': rgb,
                }
        else:
            # For validation, yield a single batch
            if self.cfg.model.stage == 2:
                material_id = torch.full((self.batch_size,), 0, dtype=torch.long, device=self.device)
            else:
                material_id = torch.randint(0, self.n_materials, (self.batch_size,), device=self.device)
                
            wi = self._sample_direction(self.batch_size)
            wo = self._sample_direction(self.batch_size)
            wi,wo=rotate_to_canonical_frame(wi,wo)

            rgb,_,_,_ = self.merl_interface.lookup_wiwo(wi, wo, material_id)
            
            yield {
                'material_id': material_id,
                'wi': wi,
                'wo': wo,
                'rgb': rgb,
            }

class MERLBRDFFixedDataset(IterableDataset):
    """
    Iterable dataset for MERL BRDF data using MERLInterface.
    
    In each iteration, samples material_id, wi, wo and calculates rgb using lookup_wiwo.
    
    Args:
        data_folder: Path to folder containing .binary BRDF files
        batch_size: Number of samples per batch (default: 1024)
        split: 'train' or 'val' (default: 'train')
        material_names: Optional list of specific materials to load (without .binary extension)
        device: Device to store tensors on (default: 'cuda' if available, else 'cpu')
    """
    
    def __init__(self, cfg, data_folder, batch_size=1024, split='train', material_names=None, device=None):
        self.cfg = cfg
        self.data_folder = Path(data_folder)
        self.batch_size = batch_size
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.split = split
        
        print(f"\n{'='*60}")
        print(f"Loading MERL BRDF Dataset")
        print(f"{'='*60}")
        print(f"Data folder: {data_folder}")    
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Split: {split}")
        
        # Initialize MERLInterface
        self.merl_interface = MERLInterface(str(self.data_folder), device=self.device)
        
        # Get number of materials
        self.n_materials = len(self.merl_interface.material_names)
        
        print(f"Loaded {self.n_materials} materials")
        print(f"{'='*60}\n")
        self._generate_samples()
    
    def _sample_direction(self, batch_size):
        """
        Sample random direction vectors in the upper hemisphere (z > 0).
        
        Args:
            batch_size: Number of directions to sample
            
        Returns:
            directions: [batch_size, 3] normalized direction vectors
        """
        # Sample uniform on sphere, then keep only upper hemisphere
        directions = torch.randn(batch_size, 3, device=self.device)
        directions = NF.normalize(directions, dim=-1)
        # Ensure z > 0 (upper hemisphere)
        directions[..., 2] = torch.abs(directions[..., 2])
        # Renormalize to ensure unit length
        directions = NF.normalize(directions, dim=-1)
        return directions

    def _generate_samples(self):
        self.wi = self._sample_direction(self.batch_size)
        self.wo = self._sample_direction(self.batch_size)
        self.wi,self.wo=rotate_to_canonical_frame(self.wi,self.wo)
        
        # Calculate rgb using lookup_wiwo
        self.material_id = torch.full((self.batch_size,), 0, dtype=torch.long, device=self.device)
        self.rgb,_,_,_ = self.merl_interface.lookup_wiwo(self.wi, self.wo, self.material_id)
        
    
    def __iter__(self):
        """Iterator yielding batches of material_id, wi, wo, and rgb."""
        if self.split == 'train':
            while True:
                yield {
                    'material_id': self.material_id,
                    'wi': self.wi,
                    'wo': self.wo,
                    'rgb': self.rgb,
                }
        else:
            
            yield {
                'material_id': self.material_id,
                'wi': self.wi,
                'wo': self.wo,
                'rgb': self.rgb,
            }
    
class MERLBRDFFixedDataset_hd(IterableDataset):
    """
    Fixed dataset for MERL BRDF data using Rusinkiewicz parameterization.
    
    Randomly generates n (theta_h, theta_d, phi_d) angle sets and queries MERL BRDF to get RGB values.
    All data is stored in CUDA memory for fast training.
    Returns half vector, difference vector, BRDF value, and material_id.
    
    Args:
        cfg: Configuration object
        data_folder: Path to folder containing .binary BRDF files
        n_samples: Number of samples to generate and store (default: 1000000)
        material_id: Which material to use (default: 3)
        split: 'train' or 'val' (default: 'train')
        device: Device to store tensors on (default: 'cuda' if available, else 'cpu')
    """
    
    def __init__(self, cfg, data_folder, n_samples=1000000, material_id=0, split='train', device=None):
        self.cfg = cfg
        self.data_folder = Path(data_folder)
        self.n_samples = n_samples
        self.batch_size = n_samples
        self.material_id = material_id
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.split = split
        
        print(f"\n{'='*60}")
        print(f"Loading Fixed MERL BRDF Dataset (Half-Diff Parameterization) ({split})")
        print(f"{'='*60}")
        print(f"Data folder: {data_folder}")
        print(f"Device: {self.device}")
        print(f"Material ID: {material_id}")
        print(f"Number of samples: {n_samples:,}")
        print(f"Split: {split}")
        
        # Initialize MERLInterface
        self.merl_interface = MERLInterface(str(self.data_folder), device=self.device)
        
        # Generate and store fixed samples
        print(f"Generating {n_samples:,} random BRDF samples using Rusinkiewicz angles...")
        self._generate_samples()
        
    
    def _sample_direction(self, batch_size):
        """
        Sample random direction vectors in the upper hemisphere (z > 0).
        
        Args:
            batch_size: Number of directions to sample
            
        Returns:
            directions: [batch_size, 3] normalized direction vectors
        """
        # Sample uniform on sphere, then keep only upper hemisphere
        directions = torch.randn(batch_size, 3, device=self.device)
        directions = NF.normalize(directions, dim=-1)
        # Ensure z > 0 (upper hemisphere)
        directions[..., 2] = torch.abs(directions[..., 2])
        # Renormalize to ensure unit length
        directions = NF.normalize(directions, dim=-1)
        return directions
    
        # Calculate rotation angle: negative of wi's azimuthal angle
        # phi_wi = atan2(wi.y, wi.x)
        # We want to rotate by -phi_wi to make wi.x = 0
        phi_wi = torch.atan2(wi[:, 1], wi[:, 0])  # [B]
        
        # Rotation matrix around z-axis by angle -phi_wi:
        # [cos(-phi)  -sin(-phi)  0]   [cos(phi)   sin(phi)  0]
        # [sin(-phi)   cos(-phi)  0] = [-sin(phi)  cos(phi)  0]
        # [   0           0       1]   [   0          0      1]
        
        cos_phi = torch.cos(phi_wi)  # [B]
        sin_phi = torch.sin(phi_wi)  # [B]
        
        # Apply rotation to wi
        wi_rotated = torch.zeros_like(wi)
        wi_rotated[:, 0] = cos_phi * wi[:, 0] + sin_phi * wi[:, 1]
        wi_rotated[:, 1] = -sin_phi * wi[:, 0] + cos_phi * wi[:, 1]
        wi_rotated[:, 2] = wi[:, 2]
        
        # Apply same rotation to wo
        wo_rotated = torch.zeros_like(wo)
        wo_rotated[:, 0] = cos_phi * wo[:, 0] + sin_phi * wo[:, 1]
        wo_rotated[:, 1] = -sin_phi * wo[:, 0] + cos_phi * wo[:, 1]
        wo_rotated[:, 2] = wo[:, 2]
        
        return wi_rotated, wo_rotated
    
    def sample_rusinkiewicz_angles(self, batch_size):
        """
        Directly sample Rusinkiewicz half-difference angles.
        
        This samples the MERL parameterization directly instead of converting from wi/wo.
        Provides more direct control over the half-difference parameter space.
        
        Args:
            batch_size: Number of angle sets to sample
        
        Returns:
            theta_h: [batch_size] half vector elevation angle [0, π/2]
            theta_d: [batch_size] difference vector elevation angle [0, π/2]
            phi_d: [batch_size] difference vector azimuthal angle [0, 2π]
        """
        import math
        
        # Sample theta_h uniformly in [0, π/2]
        theta_h = torch.rand(batch_size, device=self.device) * (math.pi / 2)
        
        # Sample theta_d uniformly in [0, π/2]
        theta_d = torch.rand(batch_size, device=self.device) * (math.pi / 2)
        
        # Sample phi_d uniformly in [0, 2π]
        phi_d = torch.rand(batch_size, device=self.device) * (math.pi * 2.0)
        
        return theta_h, theta_d, phi_d

    def rusinkiewicz_to_vectors(self,theta_h, theta_d, phi_d):
        """
        使用 PyTorch 将 Rusinkiewicz 坐标转换为 wi 和 wo 矢量。
        支持批处理 (Batch processing)。
        
        参数:
            theta_h (Tensor): 半矢量与法线夹角，形状为 (...)
            theta_d (Tensor): 入射光与半矢量夹角，形状为 (...)
            phi_d   (Tensor): 入射光绕半矢量的方位角，形状为 (...)
            
        返回:
            wi, wo (Tensor): 形状为 (..., 3) 的单位矢量
        """
        # 1. 计算三角函数
        sh, ch = torch.sin(theta_h), torch.cos(theta_h)
        sd, cd = torch.sin(theta_d), torch.cos(theta_d)
        spd, cpd = torch.sin(phi_d), torch.cos(phi_d)

        # 2. 在半矢量局部空间 (H-frame) 构建 wi 和 wo
        # 此时 H = [0, 0, 1]
        # wi_local = [sin(td)cos(pd), sin(td)sin(pd), cos(td)]
        wi_x_local = sd * cpd
        wi_y_local = sd * spd
        wi_z_local = cd

        # wo 是 wi 绕 H(Z轴) 旋转180度，即 x,y 取反
        wo_x_local = -wi_x_local
        wo_y_local = -wi_y_local
        wo_z_local = wi_z_local

        # 3. 将局部坐标旋转到法线坐标系 (绕 Y 轴旋转 theta_h)
        # 旋转公式:
        # x' = x*cos(th) + z*sin(th)
        # y' = y
        # z' = -x*sin(th) + z*cos(th)
        
        wi_x = wi_x_local * ch + wi_z_local * sh
        wi_y = wi_y_local
        wi_z = -wi_x_local * sh + wi_z_local * ch

        wo_x = wo_x_local * ch + wo_z_local * sh
        wo_y = wo_y_local
        wo_z = -wo_x_local * sh + wo_z_local * ch

        # 4. 拼接成 (..., 3) 形状的张量
        wi = torch.stack([wi_x, wi_y, wi_z], dim=-1)
        wo = torch.stack([wo_x, wo_y, wo_z], dim=-1)

        return wi, wo
    
    '''
    def _generate_samples_wiwo(self):
        """Generate all samples at once and store in memory."""
        # Sample random Rusinkiewicz angles
        wi=self._sample_direction(self.batch_size)
        wo=self._sample_direction(self.batch_size)
        
        self.wi=wi
        self.wo=wo
        wi_rotated, wo_rotated = rotate_to_canonical_frame(wi, wo)
        # Calculate BRDF values using lookup_angle
        
        # Create material_id tensor
        material_ids = torch.full((self.n_samples,), self.material_id, dtype=torch.long, device=self.device)
        
        self.rgb,phi_d,theta_d,theta_h = self.merl_interface.lookup_wiwo(wi, wo, material_ids)
        self.rgb_rotated,phi_d_rotated,theta_d_rotated,theta_h_rotated = self.merl_interface.lookup_wiwo(wi_rotated, wo_rotated, material_ids)

        print("rgb_diff",torch.mean(self.rgb - self.rgb_rotated))
        print("phi_d_diff",torch.mean(phi_d - phi_d_rotated))
        print("theta_d_diff",torch.mean(theta_d - theta_d_rotated))
        print("theta_h_diff",torch.mean(theta_h - theta_h_rotated))
        
        print("phi_d",torch.max(phi_d),torch.min(phi_d))
        print("theta_d",torch.max(theta_d),torch.min(theta_d))
        print("theta_h",torch.max(theta_h),torch.min(theta_h))
        
        # Convert angles to vectors using rangles_to_rvectors
        # Returns [hx, hy, hz, dx, dy, dz] with shape [batch_size, 6]
        vectors = self.merl_interface.rangles_to_rvectors(theta_h, theta_d, phi_d)
        
        # Split into h_vec and d_vec
        self.h_vec = vectors[:, :3]  # [batch_size, 3]
        self.d_vec = vectors[:, 3:]  # [batch_size, 3]
        
        #h_norms = torch.linalg.norm(self.h_vec, dim=1)
        #print(f"Norms of h_vec: {h_norms}")
        
        # Store material_ids for consistency
        self.material_ids = material_ids
        self.material_ids_train = torch.full((self.n_samples,), 0, dtype=torch.long, device=self.device)
    '''
    
    def _generate_samples(self):
        """
        Generate samples by directly sampling Rusinkiewicz angles.
        
        Alternative to sampling wi/wo directions - samples the half-difference
        parameterization directly for more uniform coverage of BRDF space.
        """
        # Directly sample Rusinkiewicz angles
        theta_h, theta_d, phi_d = self.sample_rusinkiewicz_angles(self.batch_size)
        
        # Create material_id tensor
        material_ids = torch.full((self.n_samples,), self.material_id, dtype=torch.long, device=self.device)
        
        # Calculate BRDF values directly using lookup_angle
        self.rgb = self.merl_interface.lookup_angle(phi_d, theta_d, theta_h, material_ids)
        self.rgb=torch.clamp(self.rgb, min=0.0)
        print("Direct angle sampling:")
        print("  phi_d:", torch.max(phi_d).item(), torch.min(phi_d).item())
        print("  theta_d:", torch.max(theta_d).item(), torch.min(theta_d).item())
        print("  theta_h:", torch.max(theta_h).item(), torch.min(theta_h).item())
        
        self.wi,self.wo=self.rusinkiewicz_to_vectors(theta_h, theta_d, phi_d)
        self.wi,self.wo=rotate_to_canonical_frame(self.wi,self.wo)
        
        
        # Store material_ids for consistency
        self.material_ids = material_ids
        self.material_ids_train = torch.full((self.n_samples,), 0, dtype=torch.long, device=self.device)
    
    def __iter__(self):
        """Iterator yielding batches of material_id, wi, wo, and rgb."""
        if self.split == 'train':
            while True:
                yield {
                    'material_id': self.material_ids,
                    'wi': self.wi,
                    'wo': self.wo,
                    'rgb': self.rgb,
                }
        else:
            
            yield {
                'material_id': self.material_ids,
                'wi': self.wi,
                'wo': self.wo,
                'rgb': self.rgb,
            }


class MERLBRDFIterableDataset_hd(IterableDataset):
    """
    Iterable dataset for MERL BRDF data using Rusinkiewicz parameterization.
    
    In each iteration, samples theta_h, theta_d, phi_d and calculates BRDF values using lookup_angle.
    Returns half vector, difference vector, BRDF value, and material_id.
    
    Args:
        data_folder: Path to folder containing .binary BRDF files
        batch_size: Number of samples per batch (default: 1024)
        split: 'train' or 'val' (default: 'train')
        material_names: Optional list of specific materials to load (without .binary extension)
        device: Device to store tensors on (default: 'cuda' if available, else 'cpu')
    """
    
    def __init__(self, cfg, data_folder, batch_size=1024, split='train', material_names=None, device=None):
        self.cfg = cfg
        self.data_folder = Path(data_folder)
        self.batch_size = batch_size
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.split = split
        
        print(f"\n{'='*60}")
        print(f"Loading MERL BRDF Dataset (Half-Diff Parameterization)")
        print(f"{'='*60}")
        print(f"Data folder: {data_folder}")    
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Split: {split}")
        
        # Initialize MERLInterface
        self.merl_interface = MERLInterface(str(self.data_folder), device=self.device)
        
        # Get number of materials
        self.n_materials = len(self.merl_interface.material_names)
        
        print(f"Loaded {self.n_materials} materials")
        print(f"{'='*60}\n")
    
    def sample_rusinkiewicz_angles(self, batch_size):
        """
        Directly sample Rusinkiewicz half-difference angles.
        
        This samples the MERL parameterization directly instead of converting from wi/wo.
        Provides more direct control over the half-difference parameter space.
        
        Args:
            batch_size: Number of angle sets to sample
        
        Returns:
            theta_h: [batch_size] half vector elevation angle [0, π/2]
            theta_d: [batch_size] difference vector elevation angle [0, π/2]
            phi_d: [batch_size] difference vector azimuthal angle [0, 2π]
        """
        import math
        
        # Sample theta_h uniformly in [0, π/2]
        theta_h = torch.rand(batch_size, device=self.device) * (math.pi / 2)
        
        # Sample theta_d uniformly in [0, π/2]
        theta_d = torch.rand(batch_size, device=self.device) * (math.pi / 2)
        
        # Sample phi_d uniformly in [0, 2π]
        phi_d = torch.rand(batch_size, device=self.device) * (math.pi * 2.0)
        
        return theta_h, theta_d, phi_d


    def _sample_direction(self, batch_size):
        """
        Sample random direction vectors in the upper hemisphere (z > 0).
        
        Args:
            batch_size: Number of directions to sample
            
        Returns:
            directions: [batch_size, 3] normalized direction vectors
        """
        # Sample uniform on sphere, then keep only upper hemisphere
        directions = torch.randn(batch_size, 3, device=self.device)
        directions = NF.normalize(directions, dim=-1)
        # Ensure z > 0 (upper hemisphere)
        directions[..., 2] = torch.abs(directions[..., 2])
        # Renormalize to ensure unit length
        directions = NF.normalize(directions, dim=-1)
        return directions

    def rusinkiewicz_to_vectors(self,theta_h, theta_d, phi_d):
        """
        使用 PyTorch 将 Rusinkiewicz 坐标转换为 wi 和 wo 矢量。
        支持批处理 (Batch processing)。
        
        参数:
            theta_h (Tensor): 半矢量与法线夹角，形状为 (...)
            theta_d (Tensor): 入射光与半矢量夹角，形状为 (...)
            phi_d   (Tensor): 入射光绕半矢量的方位角，形状为 (...)
            
        返回:
            wi, wo (Tensor): 形状为 (..., 3) 的单位矢量
        """
        # 1. 计算三角函数
        sh, ch = torch.sin(theta_h), torch.cos(theta_h)
        sd, cd = torch.sin(theta_d), torch.cos(theta_d)
        spd, cpd = torch.sin(phi_d), torch.cos(phi_d)

        # 2. 在半矢量局部空间 (H-frame) 构建 wi 和 wo
        # 此时 H = [0, 0, 1]
        # wi_local = [sin(td)cos(pd), sin(td)sin(pd), cos(td)]
        wi_x_local = sd * cpd
        wi_y_local = sd * spd
        wi_z_local = cd

        # wo 是 wi 绕 H(Z轴) 旋转180度，即 x,y 取反
        wo_x_local = -wi_x_local
        wo_y_local = -wi_y_local
        wo_z_local = wi_z_local

        # 3. 将局部坐标旋转到法线坐标系 (绕 Y 轴旋转 theta_h)
        # 旋转公式:
        # x' = x*cos(th) + z*sin(th)
        # y' = y
        # z' = -x*sin(th) + z*cos(th)
        
        wi_x = wi_x_local * ch + wi_z_local * sh
        wi_y = wi_y_local
        wi_z = -wi_x_local * sh + wi_z_local * ch

        wo_x = wo_x_local * ch + wo_z_local * sh
        wo_y = wo_y_local
        wo_z = -wo_x_local * sh + wo_z_local * ch

        # 4. 拼接成 (..., 3) 形状的张量
        wi = torch.stack([wi_x, wi_y, wi_z], dim=-1)
        wo = torch.stack([wo_x, wo_y, wo_z], dim=-1)

        return wi, wo
    
    def __iter__(self):
        if self.split == 'train':
            while True:
                # Sample random material IDs
                if self.cfg.model.stage == 2:
                    material_id = torch.full((self.batch_size,), 0, dtype=torch.long, device=self.device)
                else:
                    material_id = torch.randint(0, self.n_materials, (self.batch_size,), device=self.device)
                
                # Sample random Rusinkiewicz angles
                theta_h, theta_d, phi_d = self.sample_rusinkiewicz_angles(self.batch_size)
                rgb = self.merl_interface.lookup_angle(phi_d, theta_d, theta_h, material_id)
                rgb=torch.clamp(rgb, min=0.0)

                wi,wo=self.rusinkiewicz_to_vectors(theta_h, theta_d, phi_d)
                wi,wo=rotate_to_canonical_frame(wi,wo)
                
                yield {
                    'material_id': material_id,
                    'wi': wi,
                    'wo': wo,
                    'rgb': rgb,
                }
        else:
            if self.cfg.model.stage == 2:
                material_id = torch.full((self.batch_size,), 0, dtype=torch.long, device=self.device)
            else:
                material_id = torch.randint(0, self.n_materials, (self.batch_size,), device=self.device)
            
            # Sample random Rusinkiewicz angles
            theta_h, theta_d, phi_d = self.sample_rusinkiewicz_angles(self.batch_size)
            rgb = self.merl_interface.lookup_angle(phi_d, theta_d, theta_h, material_id)
            rgb=torch.clamp(rgb, min=0.0)
            wi,wo=self.rusinkiewicz_to_vectors(theta_h, theta_d, phi_d)
            wi,wo=rotate_to_canonical_frame(wi,wo)
            yield {
                'material_id': material_id,
                'wi': wi,
                'wo': wo,
                'rgb': rgb,
            }
    '''
    def __iter__wiwo(self):
        """Iterator yielding batches of h_vec, d_vec, rgb, and material_id."""
        if self.split == 'train':
            while True:
                # Sample random material IDs
                if self.cfg.model.stage == 2:
                    material_id = torch.full((self.batch_size,), 0, dtype=torch.long, device=self.device)
                else:
                    material_id = torch.randint(0, self.n_materials, (self.batch_size,), device=self.device)
                
                # Sample random Rusinkiewicz angles
                wi=self._sample_direction(self.batch_size)
                wo=self._sample_direction(self.batch_size)
                
                # Calculate BRDF values using lookup_angle
                rgb,phi_d,theta_d,theta_h = self.merl_interface.lookup_wiwo(wi, wo, material_id)
                
                # Convert angles to vectors using rangles_to_rvectors
                # Returns [hx, hy, hz, dx, dy, dz] with shape [batch_size, 6]
                vectors = self.merl_interface.rangles_to_rvectors(theta_h, theta_d, phi_d)
                
                # Split into h_vec and d_vec
                h_vec = vectors[:, :3]  # [batch_size, 3]
                d_vec = vectors[:, 3:]  # [batch_size, 3]
                
                yield {
                    'material_id': material_id,
                    'wi': h_vec,
                    'wo': d_vec,
                    'rgb': rgb,
                }
        else:
            # For validation, yield a single batch
            if self.cfg.model.stage == 2:
                material_id = torch.full((self.batch_size,), 0, dtype=torch.long, device=self.device)
            else:
                material_id = torch.randint(0, self.n_materials, (self.batch_size,), device=self.device)
            
            # Sample random Rusinkiewicz angles
            wi=self._sample_direction(self.batch_size)
            wo=self._sample_direction(self.batch_size)
            
            # Calculate BRDF values using lookup_angle
            rgb,phi_d,theta_d,theta_h = self.merl_interface.lookup_wiwo(wi, wo, material_id)
            
            # Convert angles to vectors
            vectors = self.merl_interface.rangles_to_rvectors(theta_h, theta_d, phi_d)
            h_vec = vectors[:, :3]
            d_vec = vectors[:, 3:]
            
            yield {
                'material_id': material_id,
                'wi': h_vec,
                'wo': d_vec,
                'rgb': rgb,
            }
    '''