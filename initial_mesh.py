import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nvdiffrast.torch as dr
import os
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torchvision
import math
import json
from tqdm import tqdm
import torchviz
import cv2
import logging
import bpy
import bmesh
import pl_bolts
from torch.utils.data import Dataset
from pytorch3d.structures import Meshes
# from pytorch3d.ops import mesh_laplacian_smoothing
from pytorch3d.loss import mesh_laplacian_smoothing
from torchviz import make_dot
import trimesh
import open3d as o3d
from model.brdf import SvPBRBRDF
from model.emitter import PresetPointEmitter
from utils.dataset import get_c2w, get_rays, get_ray_directions


class MeshRasterizationDataset(Dataset):
    """ validation dataset that loads images from metadata, returns complete images """
    def __init__(self, cfg, gt_folder, split):
        self.cfg = cfg
        self.pixel = False
        self.gt_folder = gt_folder
        self.img_hw = cfg.renderer.resolution
        h, w = self.img_hw
        self.camera_angle_x = cfg.renderer.camera.camera_angle_x
        self.focal = (0.5 * w / np.tan(0.5 * self.camera_angle_x)).item()
        self.directions = get_ray_directions(h, w, self.focal)
        
        # Load metadata from JSON file
        metadata_path = os.path.join(gt_folder, "metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load camera metadata from separate JSON file
        camera_metadata_path = os.path.join(gt_folder, "camera_metadata.json")
        with open(camera_metadata_path, 'r') as f:
            self.camera_metadata = json.load(f)
        
        # Load emitter metadata from separate JSON file
        emitter_metadata_path = os.path.join(gt_folder, "emitter_metadata.json")
        with open(emitter_metadata_path, 'r') as f:
            self.emitter_metadata = json.load(f)
        
        self.total_images = len(self.metadata)
        
        # Split metadata into training and validation sets with fixed random seed
        torch.manual_seed(42)  # Fixed seed for reproducible splits
        indices = torch.randperm(self.total_images)
        
        # Check if debug mode is enabled
        debug_mode = False
        
        if debug_mode:
            # Use all data for validation in debug mode
            selected_indices = indices
        else:
            # Use 20% for validation
            # Use 80% for training, 20% for validation
            split_idx = int(0.8 * self.total_images)
            if split == 'train':
                selected_indices = indices[:split_idx]
            else:  # validation
                selected_indices = indices[split_idx:]
        
        # Filter metadata based on split
        self.metadata = [self.metadata[i] for i in selected_indices]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_data = self.metadata[idx]
        
        # Get camera info using camera_id
        camera_id = img_data["camera_id"]
        camera_info = self.camera_metadata[camera_id]
        camera_dict = {
            "position": camera_info["position"],
            "look_at": camera_info["look_at"],
            "up": camera_info["up"]
        }
        
        # Generate rays for this camera
        c2w = get_c2w(camera_dict)
        
        # Load original RGB image (without gamma correction)
        original_filename = img_data["filename"].replace(".png", "_original.png")
        img_path = os.path.join(self.gt_folder, original_filename)
        
        if img_path.endswith('.png'):
            # Load PNG image (original linear RGB)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float() / 255.0
        else:
            raise ValueError(f"Unsupported image format: {img_path}")
        
        img_flat = img.reshape(-1, 3)
        
        # Get emitter ID directly from metadata
        emitter_id = img_data["emitter_id"]
        return {
            'c2w': c2w,
            'rgbs': img_flat,
            'emitter_ids': emitter_id,
        }
        
class MeshTextureOptimizer(pl.LightningModule):
    def __init__(self, cfg, gt_material_cfg):
        super().__init__()
        self.cfg = cfg
        self.soft_constraint = False
        self.downsample = True
        self.texture_res = 1024
        self.save_hyperparameters(cfg)
        # self.mesh_resolution = 512
        
        # Initialize mesh vertices (sphere-like geometry)
        # self.init_mesh()
        self.load_mesh("/media/raid/zla247/BRDF-Fipt/cube_with_uv_transformed.obj")
        
        # Initialize learnable PBR texture parameters instead of loading from file
        # Based on isotropic version from load_pbr_texture function
        if self.soft_constraint:
            init_std = 3.0
        else:
            init_std = 1.0
        self.pbr_texture = nn.Parameter(torch.randn(1, self.texture_res, self.texture_res, 6)*init_std)
        self.normal_map = nn.Parameter(torch.zeros(1, self.texture_res, self.texture_res, 3))
        self.normal_map.data[:, :, :, 2] = 1.0  # Set z-component to 1 for (0, 0, 1)
        # Initialize color channels (first 3 channels) to all ones for white base color
        # with torch.no_grad():
        # self.pbr_texture.data[:, :, :, :3] = 1.0
        # Initialize normal map to (0, 0, 1) - pointing up in tangent space
        self.brdf = SvPBRBRDF(gt_material_cfg, pbr_texture=self.pbr_texture)
        
        # Initialize emitter from metadata
        self.emitter = PresetPointEmitter(
            read_from_metadata=True,
            metadata_path=os.path.join(cfg.dataset_folder, 'emitter_metadata.json')
        )
        
        # Setup rasterizer context
        self.glctx = dr.RasterizeGLContext()
        
        # Image dimensions
        self.img_hw = cfg.renderer.resolution
        # Gamma correction
        self.gamma = lambda x: torch.pow(x.clamp(min=0.0), 1.0/2.2)     
        self.max_vertex_displacement = 1e-6
        self.lap_loss_weight = 0.1

    def compute_tangent_space(self, vertices, faces, uvs, normals=None, eps=1e-8):
        """
        vertices: (V,3) float
        faces:    (F,3) int
        uvs:      (V,2) float  (per-vertex; if your mesh has per-corner UVs, expand vertices first)
        normals:  (V,3) float, if None, must be computed beforehand
        returns:
        T:   (V,3) unit tangent, orthonormal to N
        B:   (V,3) unit bitangent (right-handed with N using sign)
        N:   (V,3) unit normal (same as input normals but re-normalized)
        sign:(V,)  +1/-1 handedness; bitangent = sign * cross(N, T)
        """
        V = vertices.shape[0]
        if normals is None:
            raise ValueError("normals required")
        # ensure float64 for stability
        v = vertices.astype(np.float64)
        n = normals.astype(np.float64)
        uv = uvs.astype(np.float64)
        f = faces.astype(np.int64)

        p0, p1, p2 = v[f[:,0]], v[f[:,1]], v[f[:,2]]
        uv0, uv1, uv2 = uv[f[:,0]], uv[f[:,1]], uv[f[:,2]]

        e1 = p1 - p0
        e2 = p2 - p0
        duv1 = uv1 - uv0
        duv2 = uv2 - uv0

        det = duv1[:,0]*duv2[:,1] - duv1[:,1]*duv2[:,0]
        valid = np.abs(det) > eps
        r = np.zeros_like(det)
        r[valid] = 1.0 / det[valid]

        # Face tangents/bitangents (area-weighted)
        t_face = np.zeros_like(p0)
        b_face = np.zeros_like(p0)
        t_face[valid] = (e1[valid] * duv2[valid,1:2] - e2[valid] * duv1[valid,1:2]) * r[valid,None]
        b_face[valid] = (e2[valid] * duv1[valid,0:1] - e1[valid] * duv2[valid,0:1]) * r[valid,None]

        # Optional: weight by triangle area to improve stability
        area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
        t_face *= area2[:,None]
        b_face *= area2[:,None]

        # Accumulate to vertices
        T = np.zeros((V,3), dtype=np.float64)
        B = np.zeros((V,3), dtype=np.float64)
        for corner in range(3):
            np.add.at(T, f[:,corner], t_face)
            np.add.at(B, f[:,corner], b_face)

        # Orthonormalize T to N and normalize
        n_norm = np.linalg.norm(n, axis=1, keepdims=True) + eps
        N = n / n_norm

        T = T - N * np.sum(N*T, axis=1, keepdims=True)
        T_norm = np.linalg.norm(T, axis=1, keepdims=True)
        bad = (T_norm[:,0] < eps)

        # Fallback tangent for degenerate UVs: build any basis around N
        if np.any(bad):
            up = np.tile(np.array([[0.0,1.0,0.0]]), (V,1))
            collinear = (np.abs((N*up).sum(axis=1)) > 0.999)
            up[collinear] = np.array([1.0,0.0,0.0])
            T[bad] = np.cross(up[bad], N[bad])
            T_norm = np.linalg.norm(T, axis=1, keepdims=True)

        T = T / (T_norm + eps)

        # Handedness sign using the accumulated bitangent
        # Final B is sign * cross(N, T)
        sign = np.sign(np.sum(np.cross(N, T) * B, axis=1))
        sign[sign == 0] = 1.0  # default
        B_final = (sign[:,None]) * np.cross(N, T)

        # Normalize B
        B_norm = np.linalg.norm(B_final, axis=1, keepdims=True) + eps
        B_final = B_final / B_norm

        return T.astype(np.float32), B_final.astype(np.float32), N.astype(np.float32), sign.astype(np.float32)

    # def load_mesh(self, mesh_path):
    #     # Clear existing mesh
    #     bpy.ops.wm.read_factory_settings(use_empty=True)

    #     # Import mesh
    #     bpy.ops.import_scene.obj(filepath=mesh_path)

    #     obj = bpy.context.selected_objects[0]
    #     bpy.context.view_layer.objects.active = obj

    #     # Triangulate mesh
    #     bpy.ops.object.mode_set(mode='EDIT')
    #     bpy.ops.mesh.select_all(action='SELECT')
    #     bpy.ops.mesh.quads_convert_to_tris()
    #     bpy.ops.object.mode_set(mode='OBJECT')

    #     mesh = obj.data

    #     # Ensure UVs and tangents exist
    #     bpy.ops.object.mode_set(mode='OBJECT')
    #     mesh.calc_tangents()

    #     # Vertices and faces
    #     vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
    #     faces = np.array([p.vertices[:] for p in mesh.polygons], dtype=np.int32)

    #     # UVs and UV indices
    #     uv_layer = mesh.uv_layers.active.data
    #     loops = mesh.loops
    #     uvs = []
    #     uv_indices = []

    #     for poly in mesh.polygons:
    #         poly_uv_idx = []
    #         tbn = []
    #         for loop_idx in poly.loop_indices:
    #             loop = loops[loop_idx]
    #             uv = uv_layer[loop_idx].uv
    #             uvs.append(uv[:])

    #             t = loop.tangent[:]
    #             n = loop.normal[:]
    #             b = np.cross(n, t)

    #             tbn.append(np.stack([t, b, n], axis=0))
    #             poly_uv_idx.append(loop.vertex_index)
    #         uv_indices.append(poly_uv_idx)

    #     uvs = np.array(uvs, dtype=np.float32)
    #     uv_indices = np.array(uv_indices, dtype=np.int32)
    #     TBN = np.array(tbn, dtype=np.float32).reshape(-1, 3, 3)
    #     T = TBN[:, 0, :]
    #     B = TBN[:, 1, :]
    #     N = TBN[:, 2, :]
        
    #     self.register_buffer('T', torch.tensor(T, dtype=torch.float32))
    #     self.register_buffer('B', torch.tensor(B, dtype=torch.float32))
    #     self.register_buffer('N', torch.tensor(N, dtype=torch.float32))
    #     self.register_buffer('vertices', torch.tensor(vertices, dtype=torch.float32))
    #     self.register_buffer('faces', torch.tensor(faces, dtype=torch.int32))
    #     self.register_buffer('uvs', torch.tensor(uvs, dtype=torch.float32))
    #     self.register_buffer('uv_indices', torch.tensor(uv_indices, dtype=torch.int32))

  
    def load_mesh(self, mesh_path):
        # Load the mesh from the specified path
        mesh = trimesh.load(mesh_path)
        
        # Extract vertices, faces, and UV coordinates
        vertices = mesh.vertices.astype(np.float64)
        faces = mesh.faces.astype(np.int32)
        # Vertex normals (computed lazily if not present in file)
        vertex_normals = mesh.vertex_normals.astype(np.float32)

        # Get UV coordinates from visual attributes
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uvs = mesh.visual.uv.astype(np.float32)
            
        else:
            # If no UVs, generate simple planar mapping
            print("Warning: No UV coordinates found, generating planar mapping")
            uvs = np.zeros((len(vertices), 2), dtype=np.float32)
            # Simple planar projection onto XZ plane
            uvs[:, 0] = (vertices[:, 0] - vertices[:, 0].min()) / (vertices[:, 0].max() - vertices[:, 0].min())
            uvs[:, 1] = (vertices[:, 2] - vertices[:, 2].min()) / (vertices[:, 2].max() - vertices[:, 2].min())
        T, B, N, sign = self.compute_tangent_space(vertices, faces, uvs, vertex_normals)
        # Convert to tensors and store
        vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
        faces_tensor = torch.tensor(faces, dtype=torch.int32)
        uvs_tensor = torch.tensor(uvs, dtype=torch.float32)
        self.register_buffer('T', torch.tensor(T, dtype=torch.float32))
        self.register_buffer('B', torch.tensor(B, dtype=torch.float32))
        self.register_buffer('N', torch.tensor(N, dtype=torch.float32))
        self.register_buffer('vertices', vertices_tensor)
        self.register_buffer('faces', faces_tensor)
        self.register_buffer('uvs', uvs_tensor)
        self.register_buffer('uv_indices', faces_tensor)
        
    def init_mesh(self):
        """Initialize a rectangle mesh with learnable vertices using Open3D for proper UV mapping"""
        import open3d as o3d
        
        # Create rectangle mesh using Open3D
        width_res = self.mesh_resolution
        height_res = self.mesh_resolution
        def rect_grid(width_res, height_res, size=0.4):
            xs = np.linspace(-size/2, size/2, width_res)
            zs = np.linspace(-size/2, size/2, height_res)
            xv, zv = np.meshgrid(xs, zs)
            vertices = np.column_stack([xv.ravel(), np.zeros_like(xv).ravel(), zv.ravel()])

            # Build faces with consistent CCW winding seen from +Y
            faces = []
            for i in range(height_res - 1):
                for j in range(width_res - 1):
                    v0 = i * width_res + j
                    v1 = v0 + 1
                    v2 = v0 + width_res
                    v3 = v2 + 1
                    faces.append([v0, v2, v1])  # CCW
                    faces.append([v1, v2, v3])  # CCW
            return np.asarray(vertices, np.float64), np.asarray(faces, np.int32)

        mesh = o3d.geometry.TriangleMesh()
        verts, tris = rect_grid(width_res, height_res)
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)

        # Create UV coordinates and UV indices
        uvs = []
        uv_indices = []
        # Create UV coordinates for each vertex
        for i in range(height_res):
            for j in range(width_res):
                u = j / (width_res - 1)  # Normalize to [0, 1]
                v = i / (height_res - 1)  # Normalize to [0, 1]
                uvs.append([u, v])
        
        # Create UV indices that match the triangle faces
        for face_idx, face in enumerate(tris):
            # Each triangle face uses the same vertex indices for UV coordinates
            v0_idx, v1_idx, v2_idx = face
            uv_indices.append([v0_idx, v1_idx, v2_idx])
        
        # Convert to tensors and store
        uvs_tensor = torch.tensor(uvs, dtype=torch.float32)
        uv_indices_tensor = torch.tensor(uv_indices, dtype=torch.int32)
        self.register_buffer('uvs', uvs_tensor)
        self.register_buffer('uv_indices', uv_indices_tensor)
        
        # For each triangle, we need to create vertex colors
        vertex_colors = []
        col_indices = []
        
        for face_idx, face in enumerate(tris):
            # Get the 3 vertices of this triangle
            v0_idx, v1_idx, v2_idx = face
            
            # Calculate vertex colors for each vertex of this triangle
            # Based on the vertex positions in the grid (using UV-like mapping for colors)
            for vertex_idx in [v0_idx, v1_idx, v2_idx]:
                i = vertex_idx // width_res
                j = vertex_idx % width_res
                u = j / (width_res - 1)
                v = i / (height_res - 1)
                # Create RGB color based on UV coordinates
                vertex_colors.append([u, v, 0.5])  # R=u, G=v, B=0.5
            
            # Color indices for this triangle (3 consecutive color coordinates)
            base_col_idx = face_idx * 3
            col_indices.append([base_col_idx, base_col_idx + 1, base_col_idx + 2])
        # Convert to tensors
        vertices_tensor = torch.tensor(verts, dtype=torch.float32)
        faces_tensor = torch.tensor(tris, dtype=torch.int32)
        vertex_colors_tensor = torch.tensor(vertex_colors, dtype=torch.float32)
        col_indices_tensor = torch.tensor(col_indices, dtype=torch.int32)
        
        self.vertices = nn.Parameter(vertices_tensor)
        self.vertex_colors = nn.Parameter(vertex_colors_tensor)
        self.register_buffer('faces', faces_tensor)
        self.register_buffer('col_indices', col_indices_tensor)
        
    def transform_normal_map_to_world(self, T_flat, B_flat, N_flat, normal_map_flat):
        """Transform normal map from tangent space to world space"""
        normal_map_tangent = F.normalize(normal_map_flat, dim=-1)
        
        # Apply TBN transformation
        world_normal = (normal_map_tangent[..., 0:1] * T_flat + 
                       normal_map_tangent[..., 1:2] * B_flat + 
                       normal_map_tangent[..., 2:3] * N_flat)
        # Normalize the result
        world_normal = F.normalize(world_normal, dim=-1)
        return world_normal
    
    def laplacian_loss(self, method="cot"):
        """
        Compute Laplacian–smooth regulariser for the current learnable mesh.
        Args:
            method : "uniform", "cot" or "cotcurv"  (cot = most common)
            weight : scaling factor λ
        """
        # PyTorch3D wants int64 faces and B×V×3 verts.  We have a single mesh.
        verts = self.vertices.unsqueeze(0)          # (1, V, 3)
        faces = self.faces.to(torch.int64).unsqueeze(0)  # (1, F, 3)

        mesh = Meshes(verts=verts, faces=faces)
        lap = mesh_laplacian_smoothing(mesh, method=method)
        return self.lap_loss_weight * lap

    # def clip_vertex_gradients(self, max_displacement=0.01):
    #     """
    #     Clip vertex gradients to limit movement per iteration.
    #     Args:
    #         max_displacement: maximum allowed displacement per iteration
    #     """
    #     if self.vertices.grad is not None:
    #         # Compute gradient norms for each vertex
    #         grad_norms = torch.norm(self.vertices.grad, dim=-1, keepdim=True)  # [num_vertices, 1]
            
    #         # Clip gradients where norm exceeds threshold
    #         max_grad_norm = max_displacement
    #         clip_mask = grad_norms > max_grad_norm
            
    #         if clip_mask.any():
    #             # Scale down gradients that exceed the threshold
    #             scale_factor = max_grad_norm / (grad_norms + 1e-8)
    #             scale_factor = torch.where(clip_mask, scale_factor, torch.ones_like(scale_factor))
    #             self.vertices.grad = self.vertices.grad * scale_factor
                
    # def on_after_backward(self):
    #     """Called after backward pass to clip vertex gradients"""
    #     self.clip_vertex_gradients(self.max_vertex_displacement)

    # def on_after_backward(self):
    #     """Hook after backward pass to check gradient health."""
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
    #                 print(f"⚠️ Gradient NaN/Inf detected in: {name}")
    #                 import pdb; pdb.set_trace()

    def render_view(self, c2w, emitter_idx, downsample_factor=1):
        """Render a single view with specific camera and emitter"""
                
        # Get ray directions
        h, w = self.img_hw
        if downsample_factor > 1:
            h, w = h // downsample_factor, w // downsample_factor
        focal = (0.5 * w / np.tan(0.5 * self.cfg.renderer.camera.camera_angle_x)).item()
        directions = get_ray_directions(h, w, focal)
        
        # Get rays
        rays_o, rays_d, _, _ = get_rays(directions, c2w.cpu(), focal=focal)
        rays_o = rays_o.to(self.device)
        
        # Transform vertices to clip space
        mvp = self.get_mvp_matrix(c2w)
        
        # Convert vertices to homogeneous coordinates
        vertices_homo = torch.cat([self.vertices, torch.ones(self.vertices.shape[0], 1, device=self.vertices.device)], dim=1)
        pos_clip = torch.matmul(vertices_homo.unsqueeze(0), mvp.T)
        # Rasterize
        rast_out, rast_out_db = dr.rasterize(
            self.glctx, pos_clip, self.faces, (h, w)
        )
        
        # Interpolate attributes (positions and UVs)
        pos_world, _ = dr.interpolate(
            self.vertices[..., :3].unsqueeze(0), rast_out, self.faces
        )
        uv, _ = dr.interpolate(
            self.uvs.unsqueeze(0), rast_out, self.uv_indices
        )
        tex = dr.texture(self.pbr_texture, uv, filter_mode='linear')
        
        normal_map = dr.texture(
            self.normal_map, uv, filter_mode='linear'
        )
        T, _ = dr.interpolate(
            self.T.unsqueeze(0), rast_out, self.faces
        )
        B, _ = dr.interpolate(
            self.B.unsqueeze(0), rast_out, self.faces
        )
        N, _ = dr.interpolate(
            self.N.unsqueeze(0), rast_out, self.faces
        )
        
        # Get valid pixel mask
        # mask = rast_out[..., 3:4] > 0
        mask = torch.full((h * w,), True, device=self.device)
        
        # Flatten for processing
        pos_flat = pos_world.reshape(-1, 3)[mask.reshape(-1)]
        tex_flat = tex.reshape(-1, 6)[mask.reshape(-1)]
        normal_map_flat = normal_map.reshape(-1, 3)[mask.reshape(-1)]
        
        T_flat = T.reshape(-1, 3)[mask.reshape(-1)]
        B_flat = B.reshape(-1, 3)[mask.reshape(-1)]
        N_flat = N.reshape(-1, 3)[mask.reshape(-1)]
        normal_flat = self.transform_normal_map_to_world(T_flat, B_flat, N_flat, normal_map_flat)
        
        if pos_flat.shape[0] == 0:
            # No valid pixels
            return torch.zeros(h, w, 3, device=self.device)
        
        # Compute view direction
        view_dir = F.normalize(rays_o.reshape(-1, 3)[mask.reshape(-1)] - pos_flat, dim=-1)
        
        # Sample light direction
        light_dir, _, light_pos, _ = self.emitter.sample_emitter(
            pos_flat, 
            torch.full((pos_flat.shape[0],), emitter_idx, device=self.device, dtype=torch.long)
        )
        
        
        # Evaluate emitter
        light_radiance, _, _ = self.emitter.eval_emitter(
            pos_flat,
            torch.full((pos_flat.shape[0],), emitter_idx, device=self.device, dtype=torch.long)
        )
        
        # Compute lighting
        cos_theta = torch.clamp(torch.sum(light_dir * normal_flat, dim=-1, keepdim=True), 0.0, 1.0)
        
        # Distance attenuation
        light_dist = torch.norm(light_pos - pos_flat, dim=-1, keepdim=True)
        attenuation = 1.0 / (light_dist ** 2 + 1e-6)
        
        # Check if light and view are on the same side of the surface
        light_dot_normal = torch.sum(light_dir * normal_flat, dim=-1, keepdim=True)
        view_dot_normal = torch.sum(view_dir * normal_flat, dim=-1, keepdim=True)
        same_side_mask = (light_dot_normal > 0.0) & (view_dot_normal > 0.0)
        same_side_mask = same_side_mask.reshape(-1)
        brdf = torch.zeros(light_radiance.shape[0], 3, device=self.device)
        
        # Evaluate BRDF
        if self.soft_constraint:
            albedo = torch.sigmoid(tex_flat[..., :3])
            roughness = torch.sigmoid(tex_flat[..., 3:4])
            metallic = torch.sigmoid(tex_flat[..., 4:5])
            brdf[same_side_mask], _ = self.brdf.compute_svbrdf_pdf(albedo[same_side_mask], roughness[same_side_mask], metallic[same_side_mask], light_dir[same_side_mask], view_dir[same_side_mask], normal_flat[same_side_mask])
        else: 
            albedo = torch.clamp(tex_flat[..., :3], 0.01, float('inf'))  # [eps, inf]
            roughness = torch.clamp(tex_flat[..., 3:4], 0.01, 0.99)  # [eps, 1-eps]
            metallic = torch.clamp(tex_flat[..., 4:5], 0.01, 0.99)  # [eps, 1-eps]
            brdf[same_side_mask], _ = self.brdf.compute_svbrdf_pdf(albedo[same_side_mask], roughness[same_side_mask], metallic[same_side_mask], light_dir[same_side_mask], view_dir[same_side_mask], normal_flat[same_side_mask])
        
        # Final color
        color = brdf * light_radiance * cos_theta * attenuation
        # Debug hook: stop if NaN is present
        if torch.isnan(color).any() or torch.isinf(color).any():
            import pdb; pdb.set_trace()  # <-- VSCode will pause here

        # Set NaN values to 0 if there are any
        # color = torch.where(torch.isnan(color), torch.zeros_like(color), color)
        
        # Reconstruct image
        img = torch.zeros(h * w, 3, device=self.device)
        img[mask.reshape(-1)] = color
        img = img.reshape(h, w, 3)
        
        return img
        
    def get_mvp_matrix(self, c2w):
        """Compute model-view-projection matrix"""
        # View matrix (inverse of c2w)
        # Convert 3x4 c2w to 4x4 homogeneous matrix
        c2w_4x4 = torch.eye(4, device=c2w.device, dtype=c2w.dtype)
        c2w_4x4[:3, :4] = c2w
        fix_LH_to_RH = torch.diag(torch.tensor([-1., -1., -1., 1.],
                                       dtype=c2w.dtype,
                                       device=c2w.device))
        c2w_4x4 = c2w_4x4 @ fix_LH_to_RH
        view_matrix = torch.inverse(c2w_4x4)
        
        # Projection matrix
        fov = self.cfg.renderer.camera.camera_angle_x
        aspect = 1.0
        near, far = 0.1, 1000.0
        
        f = 1.0 / math.tan(fov / 2.0)
        proj_matrix = torch.tensor([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0]
        ], device=self.device, dtype=torch.float32)
        
        # MVP = Projection * View
        mvp = torch.matmul(proj_matrix, view_matrix)
        return mvp
        
    def compute_vertex_normals(self):
        """Compute vertex normals from mesh geometry"""
        vertices = self.vertices[..., :3]  # Remove homogeneous coordinate
        faces = self.faces
        
        # Get vertices for each face
        v0 = vertices[faces[:, 0]]  # [num_faces, 3]
        v1 = vertices[faces[:, 1]]  # [num_faces, 3]
        v2 = vertices[faces[:, 2]]  # [num_faces, 3]
        
        # Compute face normals using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = F.normalize(torch.cross(edge1, edge2, dim=-1), dim=-1)
        
        # Compute vertex normals by averaging adjacent face normals
        vertex_normals = torch.zeros_like(vertices)
        vertex_counts = torch.zeros(vertices.shape[0], device=vertices.device)
        
        # Accumulate face normals at vertices
        for i in range(3):
            vertex_indices = faces[:, i]
            vertex_normals.index_add_(0, vertex_indices, face_normals)
            vertex_counts.index_add_(0, vertex_indices, torch.ones(faces.shape[0], device=vertices.device))
        
        # Average the normals (avoid division by zero)
        vertex_counts = vertex_counts.clamp_min(1.0)
        vertex_normals = vertex_normals / vertex_counts.unsqueeze(-1)
        vertex_normals = F.normalize(vertex_normals, dim=-1)
        
        return vertex_normals
        
    def eval_brdf(self, material_code, pos, light_dir, view_dir, normal, uv):
        """Evaluate BRDF at given positions"""
        return self.brdf.eval_brdf(material_code, pos, light_dir, view_dir, normal, uv)
        
    def training_step(self, batch, batch_idx):
        """Training step with multiple views and lights"""
        rgbs_gt, c2w, emitter_ids = batch['rgbs'], batch['c2w'], batch['emitter_ids']

        # Determine downsampling factor based on training progress
        current_step = self.global_step
        if current_step < 400:
            downsample_factor = 4  # 1/4 resolution for first 1000 steps
        elif current_step < 800:
            downsample_factor = 2  # 1/2 resolution for steps 1000-2000
        else:
            downsample_factor = 1  # Full resolution after 800 steps
            
        if not self.downsample:
            downsample_factor = 1
        
        total_loss = 0.0
        total_psnr = 0.0
        num_samples = len(c2w)

        rendered_images = []
        gt_images = []

        for i, (c2w, emitter_id) in enumerate(zip(c2w, emitter_ids)):
            rendered_img = self.render_view(c2w, emitter_id.item(), downsample_factor)
            h, w = self.img_hw
            gt_img = rgbs_gt[i].reshape(h, w, 3)

            # Apply downsampling if needed
            if downsample_factor > 1:
                new_h, new_w = h // downsample_factor, w // downsample_factor                  
                # Downsample ground truth image
                gt_img = torch.nn.functional.interpolate(
                    gt_img.permute(2, 0, 1).unsqueeze(0),  # (1, C, H, W)
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)  # Back to (H, W, C)
            # Compute loss
            if self.cfg.model.loss.recon_loss.name == "l1":
                recon_loss = F.l1_loss(rendered_img, gt_img)
            else:
                recon_loss = F.mse_loss(rendered_img, gt_img)

            total_loss += recon_loss

            # PSNR
            psnr_loss = F.mse_loss(self.gamma(rendered_img), self.gamma(gt_img))
            psnr = 10.0 * torch.log10((1.0 ** 2) / psnr_loss.clamp_min(1e-5))
            total_psnr += psnr

            rendered_images.append(rendered_img)
            gt_images.append(gt_img)

        # Combine
        avg_loss = total_loss / num_samples
        avg_psnr = total_psnr / num_samples

        lap_loss = 0.0
        loss = avg_loss + lap_loss

        # Check for NaNs in loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            import pdb; pdb.set_trace()

        # VSCode-compatible logging
        self.log_dict({
            'train/recon': avg_loss,
            'train/lap'  : lap_loss,
            'train/total': loss,
            'train/psnr' : avg_psnr,
        }, prog_bar=True, batch_size=num_samples)

        return loss

        
    def validation_step(self, batch, batch_idx):
        """Validation step with image saving"""
        rgbs_gt, c2w, emitter_ids = batch['rgbs'], batch['c2w'], batch['emitter_ids']
        
        total_loss = 0.0
        total_psnr = 0.0
        num_samples = len(c2w)
        
        # Create output directory
        output_dir = os.path.join(
            self.cfg.exp_output_root_path,
            f'mesh_texture_optimization'
        )
        os.makedirs(output_dir, exist_ok=True)
        # Determine downsampling factor based on training progress
        current_step = self.global_step
        if current_step < 200:
            downsample_factor = 4  # 1/4 resolution for first 1000 steps
        elif current_step < 400:
            downsample_factor = 2  # 1/2 resolution for steps 1000-2000
        else:
            downsample_factor = 1  # Full resolution after 800 steps
            
        # if not self.downsample:
        downsample_factor = 1
        # Render and save each view
        for i, (c2w, emitter_id) in enumerate(zip(c2w, emitter_ids)):
            # Render view
            rendered_img = self.render_view(c2w, emitter_id.item(), downsample_factor)
            
            # Get corresponding ground truth
            h, w = self.img_hw
            gt_img = rgbs_gt[i].reshape(h, w, 3)
            # Apply downsampling if needed
            if downsample_factor > 1:
                new_h, new_w = h // downsample_factor, w // downsample_factor                  
                # Downsample ground truth image
                gt_img = torch.nn.functional.interpolate(
                    gt_img.permute(2, 0, 1).unsqueeze(0),  # (1, C, H, W)
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)  # Back to (H, W, C)
            # Compute loss
            if self.cfg.model.loss.recon_loss.name == "l1":
                recon_loss = F.l1_loss(rendered_img, gt_img)
            else:
                recon_loss = F.mse_loss(rendered_img, gt_img)
            
            total_loss += recon_loss
            
            # Compute PSNR
            psnr_loss = F.mse_loss(self.gamma(rendered_img), self.gamma(gt_img))
            psnr = 10.0 * torch.log10((1.0 ** 2) / psnr_loss.clamp_min(1e-5))
            total_psnr += psnr
            
            # Save images
            torchvision.utils.save_image(
                self.gamma(rendered_img.permute(2, 0, 1)),
                os.path.join(output_dir, f'rendered_view_{batch_idx}_{i}.png')
            )
            torchvision.utils.save_image(
                self.gamma(gt_img.permute(2, 0, 1)),
                os.path.join(output_dir, f'gt_view_{batch_idx}_{i}.png')
            )
        
        # Average losses
        avg_loss = total_loss / num_samples
        avg_psnr = total_psnr / num_samples
        
        # Compute laplacian loss
        # lap_loss = self.laplacian_loss(method="cot")
        lap_loss = 0.0
        loss = avg_loss + lap_loss  # + 0.01 * self.brdf.pbr_texture.norm()
        
        # Log validation metrics
        self.log_dict({
            'val/recon': avg_loss,
            'val/total': loss,
            'val/psnr': avg_psnr,
            'val/lap': lap_loss,
        }, prog_bar=True, batch_size=num_samples)
        
        return loss
   
    def configure_optimizers(self):
        params_to_optimize = self.parameters()
        
        if self.hparams.model.optimizer.name == "SGD":
            optimizer = torch.optim.SGD(
                params_to_optimize,
                lr=self.hparams.model.optimizer.lr,
                momentum=0.9,
                weight_decay=1e-4,
            )
            scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=int(self.hparams.model.optimizer.warmup_steps_ratio * self.hparams.model.trainer.max_steps),
                max_epochs=self.hparams.model.trainer.max_steps,
                eta_min=0,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }

        elif self.hparams.model.optimizer.name == 'Adam':
            optimizer = torch.optim.Adam(
                params_to_optimize,
                lr=self.hparams.model.optimizer.lr,
                betas=(0.9, 0.999),
                weight_decay=self.hparams.model.optimizer.weight_decay,
            )
            return optimizer

        else:
            logging.error('Optimizer type not supported')

    # def on_validation_end(self):
    #     """Called at the end of validation - save mesh periodically"""
    #     # Save mesh every few epochs to track optimization progress
    #     epoch_dir = f"epoch_{self.current_epoch:04d}"
    #     save_dir = os.path.join(self.cfg.exp_output_root_path, "optimized_assets", epoch_dir)
    #     self.save_optimized_assets(save_dir)

    def save_optimized_assets(self, save_dir):
        """Save optimized mesh and texture"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save mesh vertices
        torch.save(self.vertices.detach().cpu(), os.path.join(save_dir, 'optimized_vertices.pt'))
        torch.save(self.faces.cpu(), os.path.join(save_dir, 'mesh_faces.pt'))
        torch.save(self.uvs.cpu(), os.path.join(save_dir, 'mesh_uvs.pt'))
        
        # Save BRDF texture
        if hasattr(self.brdf, 'pbr_texture'):
            torch.save(self.brdf.pbr_texture.detach().cpu(), os.path.join(save_dir, 'optimized_texture.pt'))
        
        # Save Open3D mesh
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        
        # Convert tensors to numpy arrays
        vertices_np = self.vertices.detach().cpu().numpy()
        faces_np = self.faces.cpu().numpy()
        uvs_np = self.uvs.cpu().numpy()
        
        # Set mesh data
        mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
        mesh.triangles = o3d.utility.Vector3iVector(faces_np)
        
        # Set UV coordinates if available
        if hasattr(self, 'uv_indices'):
            mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs_np)
            mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(faces_np))
        
        # Compute normals
        mesh.compute_vertex_normals()
        
        # Save mesh in multiple formats
        o3d.io.write_triangle_mesh(os.path.join(save_dir, 'optimized_mesh.obj'), mesh)
        o3d.io.write_triangle_mesh(os.path.join(save_dir, 'optimized_mesh.ply'), mesh)
        
        print(f"Optimized assets saved to {save_dir}")

    def save_initial_assets(self, save_dir):
        """Save initial mesh and texture before optimization"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save initial mesh vertices
        torch.save(self.vertices.detach().cpu(), os.path.join(save_dir, 'initial_vertices.pt'))
        torch.save(self.faces.cpu(), os.path.join(save_dir, 'mesh_faces.pt'))
        torch.save(self.vertex_colors.cpu(), os.path.join(save_dir, 'mesh_vertex_colors.pt'))
        
        # Save initial BRDF texture
        if hasattr(self.brdf, 'pbr_texture'):
            torch.save(self.brdf.pbr_texture.detach().cpu(), os.path.join(save_dir, 'initial_texture.pt'))
        
        # Save Open3D mesh
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        
        # Convert tensors to numpy arrays
        vertices_np = self.vertices.detach().cpu().numpy()
        faces_np = self.faces.cpu().numpy()
        vertex_colors_np = self.vertex_colors.detach().cpu().numpy()
                    
        # Set mesh data
        mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
        mesh.triangles = o3d.utility.Vector3iVector(faces_np)
        
        # # Set UV coordinates if available
        # if hasattr(self, 'uv_indices'):
        #     mesh.triangle_uvs = o3d.utility.Vector2dVector(vertex_colors_np)
        #     mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(faces_np))
        
        # Compute normals
        mesh.compute_vertex_normals()
        
        # Save mesh in multiple formats
        o3d.io.write_triangle_mesh(os.path.join(save_dir, 'initial_mesh.obj'), mesh)
        o3d.io.write_triangle_mesh(os.path.join(save_dir, 'initial_mesh.ply'), mesh)
        
        print(f"Initial assets saved to {save_dir}")
        
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Setup dataset
    # Create a dataset that returns camera and emitter info for nvdiffrast rasterization
    train_dataset = MeshRasterizationDataset(cfg, cfg.dataset_folder, split='train')
    val_dataset = MeshRasterizationDataset(cfg, cfg.dataset_folder, split='val')

    print("==> initializing data loader ...")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=8,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
    )
    
    print("==> initializing model ...")
    gt_material_cfg = hydra.compose(config_name="config", overrides=["material=svpbr"]).material
    model = MeshTextureOptimizer(cfg, gt_material_cfg)
    
    print("==> initializing logger ...")
    logger = hydra.utils.instantiate(cfg.model.logger, save_dir=cfg.exp_output_root_path)

    print("==> initializing monitor ...")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.model.checkpoint_monitor.dirpath, f'mesh_texture_optimization'),
        filename=cfg.model.checkpoint_monitor.filename,
        save_top_k=cfg.model.checkpoint_monitor.save_top_k, 
        every_n_epochs=cfg.model.checkpoint_monitor.every_n_epochs,
        monitor='val/total',
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    print("==> initializing trainer ...")
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor], logger=logger, **cfg.model.trainer
    )
    # Train model
    save_dir = os.path.join(cfg.exp_output_root_path, "initial_assets")
    # model.save_initial_assets(save_dir)
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Save optimized assets
    save_dir = os.path.join(cfg.exp_output_root_path, "optimized_assets")
    # model.save_optimized_assets(save_dir)


if __name__ == "__main__":
    main()
