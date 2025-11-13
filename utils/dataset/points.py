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

def load_metadata(colmap_camera, metadata_path, camera_metadata_path, gt_folder, cfg, debug, debug_num, split, turntable_center, turntable_axis, R_c2g, t_c2g, start_idx):
    metadata, _, _= load_camera_turntable_light_metadata(metadata_path)
    
    if colmap_camera:
        camera_metadata = load_camera_metadata(camera_metadata_path)
    else:
        camera_metadata = load_camera_metadata_from_robotic_log(metadata_path, turntable_center, turntable_axis, R_c2g, t_c2g)
    
    # Filter out metadata entries with non-existent image files and filtered_puple_ids
    valid_metadata = []
    filtered_purple_ids = getattr(cfg.data, 'filtered_purple_ids', [])
    
    for item in metadata:
        # Add "masked_" prefix to filename
        file_name = item["filename"]
        # if not file_name.startswith("masked_"):
        #     file_name = "masked_" + file_name
        #     item["filename"] = file_name
        # file_name = item["filename"]
        img_path = os.path.join(gt_folder, file_name)
        overall_id = item.get("overall_id")
        
        # Skip if image file doesn't exist or is 0 bytes
        if not os.path.exists(img_path):
            print(f"Warning: Image file {img_path} does not exist, skipping from metadata...")
            continue
        
        # Skip if image file is 0 bytes
        if os.path.getsize(img_path) == 0:
            print(f"Warning: Image file {img_path} is 0 bytes, skipping from metadata...")
            continue
            
        # Skip if overall_id is in filtered_puple_ids
        
        if int(overall_id) in filtered_purple_ids:
            print(f"Warning: overall_id {overall_id} is in filtered_puple_ids, skipping from metadata...")
            continue
            
        valid_metadata.append(item)
    
    metadata = valid_metadata
    # # Filter out images with overall_id >= 1000
    # metadata = [item for item in metadata if int(item.get("overall_id", 0)) < 1000]
    # print(f"After filtering overall_id >= 1000: {len(metadata)} images remaining")
    # print(f"Loaded {len(metadata)} valid images out of {len(metadata) + len([item for item in metadata if not os.path.exists(os.path.join(gt_folder, item['filename']))])} total metadata entries")
    
    total_images = len(metadata)
    
    # Split metadata into training and validation sets with fixed random seed
    torch.manual_seed(42)  # Fixed seed for reproducible splits
    indices = torch.randperm(total_images)
    
    # Use 80% for training
    if debug:
        selected_metadata = metadata[start_idx:start_idx+debug_num]
    else:
        split_idx = int(0.8 * total_images)
        if split == 'train':
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]
            
        selected_metadata = [metadata[i] for i in selected_indices] 
        
    return selected_metadata, camera_metadata

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
    - observations/ (chunked observation files: observations_chunk_00.npz, observations_chunk_01.npz, ...)
      OR sparse/observations.npz (legacy single file format)
    
    Each observation chunk contains: [x, y, z, image_id, pixel_x, pixel_y, r, g, b]
    
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
        
        # Double buffer settings
        self.switch_iters = getattr(cfg.data, 'switch_iters', 1000)  # How often to reload chunks
        self.step = 0
        
        # Discover material folders
        self.material_folders = sorted([
            d for d in Path(root_folder).iterdir() 
            if d.is_dir() and d.name.isdigit()
        ], key=lambda x: int(x.name))
        
        print(f"\n{'='*60}")
        print(f"Loading MultiMaterial Dataset ({split})")
        print(f"{'='*60}")
        print(f"Found {len(self.material_folders)} material folders: {[int(f.name) for f in self.material_folders]}")
        
        # Discover chunks and load metadata (lightweight)
        self._discover_chunks_and_metadata()
        
        # Initialize double buffer for training split
        if split == 'train':
            print(f"\nInitializing double buffer (chunk reload every {self.switch_iters} iters)...")
            self._dbuf = MultiMaterialPointDataset._DoubleBuffer(self._build_chunk_fn)
            # Prefill two chunks: slot 0 (active) and slot 1 (next)
            self._dbuf.request_fill(0)
            self._dbuf.request_fill(1)
            self._dbuf.wait_initial()  # Ensure first active chunk exists
            print("Double buffer initialized!")
        else:
            # For validation, use simple single load
            self._load_all_materials()
            self._create_split()
        
        print(f"\nDataset ready!")
        print(f"{'='*60}\n")
    
    def _discover_chunks_and_metadata(self):
        """Discover available chunks and load metadata (camera/emitter lookups) for all materials."""
        self.material_chunks = {}  # material_id -> list of chunk file paths
        self.camera_lookups = {}   # material_id -> {camera_id -> c2w matrix}
        self.emitter_lookups = {}  # material_id -> {scan_id -> emitter_id}
        
        print("\nDiscovering chunks and loading metadata...")
        for material_folder in tqdm(self.material_folders, desc="Scanning materials"):
            material_id = int(material_folder.name)
            
            # Discover chunk files in observations/ folder
            obs_folder = material_folder / "observations"
            chunk_files = sorted(obs_folder.glob("observations_chunk_*.npz"))
            self.material_chunks[material_id] = chunk_files
            print(f"  Material {material_id}: Found {len(chunk_files)} chunks in {obs_folder}")
            
            # Load camera metadata (c2w matrices) - small, keep in memory
            camera_json_path = material_folder / "rotated_camera.json"
            with open(camera_json_path, 'r') as f:
                camera_list = json.load(f)
            
            camera_lookup = {}
            for cam_entry in camera_list:
                cam_id = cam_entry['camera_id']
                position = np.array(cam_entry['position']) / 1000.0  # mm to m
                rotation_matrix = np.array(cam_entry['rotation_matrix'])
                c2w = build_4x4(rotation_matrix, position)
                camera_lookup[cam_id] = torch.from_numpy(c2w).float()
            self.camera_lookups[material_id] = camera_lookup
            
            # Load scan log for emitter IDs
            scan_log_path = material_folder / "scan_log.json"
            with open(scan_log_path, 'r') as f:
                scan_log = json.load(f)
            
            emitter_lookup = {}
            for scan_entry in scan_log:
                scan_id = scan_entry['scan_id']
                emitter_id = scan_entry.get('emitter_id', scan_id)
                emitter_lookup[scan_id] = emitter_id
            self.emitter_lookups[material_id] = emitter_lookup
        
        print(f"Metadata loaded for {len(self.material_chunks)} materials")
    
    def set_step(self, step: int):
        """Called by training loop to track current step for chunk reloading."""
        self.step = step
        # Request next chunk build at boundaries; swap happens lazily in __iter__
        if hasattr(self, '_dbuf') and step > 0 and step % self.switch_iters == 0:
            next_slot = 1 - self._dbuf.active
            print(f"[Step {step}] Requesting new random chunks to load into slot {next_slot}")
            self._dbuf.request_fill(next_slot)
    
    def _build_chunk_fn(self) -> "MultiMaterialPointDataset.ChunkData":
        """Build a new chunk by randomly loading one chunk per material (thread-safe)."""
        all_rays = []
        all_rgbs = []
        all_xyz = []
        all_emitter_ids = []
        all_camera_ids = []
        all_material_ids = []
        
        print(f"\n[Background] Loading new random chunks for all materials...")
        for material_folder in self.material_folders:
            material_id = int(material_folder.name)
            
            # Get metadata (already loaded, thread-safe to read)
            camera_lookup = self.camera_lookups[material_id]
            emitter_lookup = self.emitter_lookups[material_id]
            
            # Randomly select one chunk file
            available_chunks = self.material_chunks[material_id]
            chunk_path = random.choice(available_chunks)
            
            # Load observations from selected chunk
            obs_data = np.load(chunk_path)
            observations = obs_data['observations']  # (N, 9): [x, y, z, image_id, pixel_x, pixel_y, r, g, b]
            
            print(f"  [Background] Material {material_id}: Loaded chunk {chunk_path.name} ({len(observations):,} obs)")
            
            # Process observations (same logic as _load_all_materials)
            xyz = torch.from_numpy(observations[:, :3]).float()
            image_ids = observations[:, 3].astype(np.int32)
            pixel_coords = torch.from_numpy(observations[:, 4:6]).float()
            rgbs = torch.from_numpy(observations[:, 6:9]).float()
            
            # Generate rays for each observation
            rays_list = []
            xyz_list = []
            rgbs_list = []
            emitter_ids_list = []
            camera_ids_list = []
            
            # Group by image_id for efficient processing
            from collections import defaultdict
            obs_by_image = defaultdict(list)
            for obs_idx, img_id in enumerate(image_ids):
                obs_by_image[img_id].append(obs_idx)
            
            for img_id, obs_indices in obs_by_image.items():
                if img_id not in camera_lookup:
                    continue
                
                c2w_full = camera_lookup[img_id]
                c2w = c2w_full[:3, :4]
                
                obs_indices_t = torch.tensor(obs_indices, dtype=torch.long)
                pixels = pixel_coords[obs_indices_t]
                
                directions = get_ray_directions_for_pixels(
                    pixels, self.focal, self.cx, self.cy, self.distortion
                )
                
                rays_o, rays_d = get_rays(directions, c2w, focal=None)
                rays = torch.cat([rays_o, rays_d], dim=-1)
                
                rays_list.append(rays)
                xyz_list.append(xyz[obs_indices_t])
                rgbs_list.append(rgbs[obs_indices_t])
                
                emitter_id = emitter_lookup.get(img_id, 0)
                emitter_ids_batch = torch.full((len(obs_indices),), emitter_id, dtype=torch.long)
                emitter_ids_list.append(emitter_ids_batch)
                
                camera_ids_batch = torch.full((len(obs_indices),), img_id, dtype=torch.long)
                camera_ids_list.append(camera_ids_batch)
            
            # Concatenate for this material
            if len(rays_list) > 0:
                material_rays = torch.cat(rays_list, dim=0)
                material_xyz = torch.cat(xyz_list, dim=0)
                material_rgbs = torch.cat(rgbs_list, dim=0)
                material_emitter_ids = torch.cat(emitter_ids_list, dim=0)
                material_camera_ids = torch.cat(camera_ids_list, dim=0)
                material_material_ids = torch.full((material_rays.shape[0],), material_id, dtype=torch.long)
                
                all_rays.append(material_rays)
                all_rgbs.append(material_rgbs)
                all_xyz.append(material_xyz)
                all_emitter_ids.append(material_emitter_ids)
                all_camera_ids.append(material_camera_ids)
                all_material_ids.append(material_material_ids)
        
        # Concatenate across all materials
        rays = torch.cat(all_rays, dim=0)
        rgbs = torch.cat(all_rgbs, dim=0)
        xyz = torch.cat(all_xyz, dim=0)
        emitter_ids = torch.cat(all_emitter_ids, dim=0)
        camera_ids = torch.cat(all_camera_ids, dim=0)
        material_ids = torch.cat(all_material_ids, dim=0)
        
        print(f"[Background] Chunk built: {len(rays):,} total observations")
        
        return MultiMaterialPointDataset.ChunkData(
            rays=rays, rgbs=rgbs, xyz=xyz,
            camera_ids=camera_ids, emitter_ids=emitter_ids, material_ids=material_ids
        )
    
    def _load_all_materials(self):
        """Load one random chunk per material and process into rays."""
        all_rays = []
        all_rgbs = []
        all_xyz = []
        all_emitter_ids = []
        all_camera_ids = []
        all_material_ids = []
        
        print("\nLoading random chunk for each material...")
        for material_folder in tqdm(self.material_folders, desc="Loading materials"):
            material_id = int(material_folder.name)
            
            # Get metadata (already loaded)
            camera_lookup = self.camera_lookups[material_id]
            emitter_lookup = self.emitter_lookups[material_id]
            
            # Randomly select one chunk file
            available_chunks = self.material_chunks[material_id]
            chunk_path = random.choice(available_chunks)
            
            # Load observations from selected chunk
            obs_data = np.load(chunk_path)
            observations = obs_data['observations']  # (N, 9): [x, y, z, image_id, pixel_x, pixel_y, r, g, b]
            
            print(f"  Material {material_id}: Loaded chunk {chunk_path.name} with {len(observations)} observations")
            
            # Process observations
            xyz = torch.from_numpy(observations[:, :3]).float()  # (N, 3)
            image_ids = observations[:, 3].astype(np.int32)  # (N,)
            pixel_coords = torch.from_numpy(observations[:, 4:6]).float()  # (N, 2)
            rgbs = torch.from_numpy(observations[:, 6:9]).float()  # (N, 3)
            
            # Generate rays for each observation
            rays_list = []
            xyz_list = []
            rgbs_list = []
            emitter_ids_list = []
            camera_ids_list = []
            
            # Group by image_id for efficient processing
            from collections import defaultdict
            obs_by_image = defaultdict(list)
            for obs_idx, img_id in enumerate(image_ids):
                obs_by_image[img_id].append(obs_idx)
            
            for img_id, obs_indices in obs_by_image.items():
                # Get camera c2w
                if img_id not in camera_lookup:
                    print(f"    Warning: image_id {img_id} not found in camera_lookup, skipping...")
                    continue
                
                c2w_full = camera_lookup[img_id]  # (4, 4)
                c2w = c2w_full[:3, :4]  # (3, 4)
                
                # Get pixel coordinates for this image's observations
                obs_indices_t = torch.tensor(obs_indices, dtype=torch.long)
                pixels = pixel_coords[obs_indices_t]  # (M, 2)
                
                # Generate ray directions for these specific pixels (with distortion)
                directions = get_ray_directions_for_pixels(
                    pixels, self.focal, self.cx, self.cy, self.distortion
                )  # (M, 3)
                
                # Transform to world space using get_rays (without modification)
                rays_o, rays_d = get_rays(directions, c2w, focal=None)  # (M, 3), (M, 3)
                rays = torch.cat([rays_o, rays_d], dim=-1)  # (M, 6)
                
                rays_list.append(rays)
                
                # Keep xyz and rgbs aligned with rays
                xyz_list.append(xyz[obs_indices_t])
                rgbs_list.append(rgbs[obs_indices_t])
                
                # Get emitter IDs
                emitter_id = emitter_lookup.get(img_id, 0)  # default to 0 if not found
                emitter_ids_batch = torch.full((len(obs_indices),), emitter_id, dtype=torch.long)
                emitter_ids_list.append(emitter_ids_batch)
                
                # Camera IDs
                camera_ids_batch = torch.full((len(obs_indices),), img_id, dtype=torch.long)
                camera_ids_list.append(camera_ids_batch)
            
            # Concatenate for this material
            if len(rays_list) > 0:
                material_rays = torch.cat(rays_list, dim=0)
                material_xyz = torch.cat(xyz_list, dim=0)
                material_rgbs = torch.cat(rgbs_list, dim=0)
                material_emitter_ids = torch.cat(emitter_ids_list, dim=0)
                material_camera_ids = torch.cat(camera_ids_list, dim=0)
                
                # Create material IDs
                material_material_ids = torch.full((material_rays.shape[0],), material_id, dtype=torch.long)
                
                all_rays.append(material_rays)
                all_rgbs.append(material_rgbs)
                all_xyz.append(material_xyz)
                all_emitter_ids.append(material_emitter_ids)
                all_camera_ids.append(material_camera_ids)
                all_material_ids.append(material_material_ids)
        
        # Concatenate across all materials
        self.all_rays = torch.cat(all_rays, dim=0)  # (N_total, 6)
        self.all_rgbs = torch.cat(all_rgbs, dim=0)  # (N_total, 3)
        self.all_xyz = torch.cat(all_xyz, dim=0)  # (N_total, 3)
        self.all_emitter_ids = torch.cat(all_emitter_ids, dim=0)  # (N_total,)
        self.all_camera_ids = torch.cat(all_camera_ids, dim=0)  # (N_total,)
        self.all_material_ids = torch.cat(all_material_ids, dim=0)  # (N_total,)
        
        print(f"\nTotal observations loaded: {len(self.all_rays):,}")
        print(f"  Materials: {torch.unique(self.all_material_ids).tolist()}")
        print(f"  Cameras: {len(torch.unique(self.all_camera_ids))}")
        print(f"  Emitters: {len(torch.unique(self.all_emitter_ids))}")
    
    def _create_split(self):
        """Split data into train/val with fixed random seed."""
        total = len(self.all_rays)
        
        # Shuffle with fixed seed for reproducibility
        torch.manual_seed(42)
        all_indices = torch.randperm(total)
        
        # 80/20 split
        split_idx = int(0.8 * total)
        if self.split == 'train':
            self.indices = all_indices[:split_idx]
        else:  # 'val'
            self.indices = all_indices[split_idx:]
        
        print(f"\nSplit: {self.split}")
        print(f"  Total rays: {total:,}")
        print(f"  Split rays: {len(self.indices):,} ({100*len(self.indices)/total:.1f}%)")
    
    def __len__(self):
        """Return number of observations in this split."""
        if hasattr(self, 'indices'):
            return len(self.indices)
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
                    'gt_params': torch.zeros(1),
                }
        
        # Validation mode with static data
        else:
            while True:
                # Random sample rays_num rays from split indices
                sample_idx = torch.randint(0, len(self.indices), (self.rays_num,), dtype=torch.long)
                actual_idx = self.indices[sample_idx]
                
                yield {
                    'rays': self.all_rays[actual_idx],  # (N, 6)
                    'rgbs': self.all_rgbs[actual_idx],  # (N, 3)
                    'xyz': self.all_xyz[actual_idx],  # (N, 3)
                    'emitter_ids': self.all_emitter_ids[actual_idx],  # (N,)
                    'camera_ids': self.all_camera_ids[actual_idx],  # (N,)
                    'material_ids': self.all_material_ids[actual_idx],  # (N,)
                    'gt_params': torch.zeros(1),
                }
    
    def __getitem__(self, idx):
        """For validation: return a single batch of rays."""
        # For validation, we can return a batch at index `idx`
        # This allows for validation with DataLoader
        batch_size = min(self.rays_num, len(self.indices) - idx * self.rays_num)
        start_idx = idx * self.rays_num
        end_idx = min(start_idx + self.rays_num, len(self.indices))
        
        actual_indices = self.indices[start_idx:end_idx]
        
        return {
            'rays': self.all_rays[actual_indices],
            'rgbs': self.all_rgbs[actual_indices],
            'xyz': self.all_xyz[actual_indices],
            'emitter_ids': self.all_emitter_ids[actual_indices],
            'camera_ids': self.all_camera_ids[actual_indices],
            'material_ids': self.all_material_ids[actual_indices],
            'gt_params': torch.zeros(1),
        }
        