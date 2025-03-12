import torch
import torch.nn.functional as NF
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

import os
import sys
sys.path.append('..')
from pathlib import Path
from utils.dataset import RealDataset, SyntheticDataset, SphereDataset
from utils.ops import *
from utils.path_tracing import ray_intersect, path_tracing, path_tracing_fix_emitter, path_tracing_envmap_emitter
from model.brdf import NGPBRDF, BaseBRDF, PBRBRDF
from model.emitter import SLFEmitter, PointEmitter, EnvMapEmitter
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import cv2
import numpy as np
# File paths
brdf_path = './checkpoints/indoor_synthetic_kitchen_run_1/last.ckpt'
emitter_path = '/localhome/zla247/theia2_data/output/BRDF-Fipt/fipt_indoor_synthetic/kitchen/'
dataset_path = '/localhome/zla247/theia2_data/fipt_indoor_synthetic/sphere'
dataset_type = 'sphere'
device = torch.device('cuda:0')
emitter_type = 'envmap'


# Load geometry
mesh_file = os.path.join(dataset_path, 'scene.obj')
assert Path(mesh_file).exists(), 'mesh file does not exist: ' + mesh_file
# scene = mitsuba.load_dict({
#     'type': 'scene',
#     'shape_id': {
#         'type': 'obj',
#         'filename': mesh_file,
#     }
# })
scene = mitsuba.load_dict({
    "type" : "scene",
    "shape_id" : {
        "type" : "sphere",
        "center" : [0.0, 0.0, 0.0],
        "radius" : 0.2,
        "flip_normals" : False,
    }
})
if dataset_type == 'synthetic':
    dataset = SyntheticDataset(dataset_path, split='train', pixel=False, ray_diff=True)
elif dataset_type == 'real':
    dataset = RealDataset(dataset_path, split='train', pixel=False, ray_diff=True)
elif dataset_type == 'sphere':
    dataset = SphereDataset(dataset_path, None, None, split='train', pixel=False, ray_diff=True)
img_hw = dataset.img_hw

# Load BRDF and emitters
mask = torch.load(os.path.join(emitter_path, 'vslf.npz'), map_location='cpu')
state_dict = torch.load(brdf_path, map_location='cpu')['state_dict']
weight = {}
for k,v in state_dict.items():
    if 'material.' in k:
        weight[k.replace('material.', '')] = v
# material_net = NGPBRDF(mask['voxel_min'], mask['voxel_max'])
material_net = PBRBRDF()
material_net.to(device)
for p in material_net.parameters():
    p.requires_grad = False

if emitter_type == 'envmap':
    emitter_net = EnvMapEmitter('envmap.exr')
elif emitter_type == 'point':
    emitter_net = PointEmitter(position=torch.tensor([-2, 2.0, 0.0]),
                        intensity=torch.tensor([50.0, 50.0, 50.0]),
                        radius=0.1)
else:
    emitter_net = SLFEmitter(os.path.join(emitter_path, 'emitter.pth'),
                        os.path.join(emitter_path, 'vslf.npz'))
emitter_net.to(device)
for p in emitter_net.parameters():
    p.requires_grad = False

# Rendering parameters
spp = 2048
SPP = 16  # batch size
indir_depth = 7
img_id = 6

# Get rays for rendering
batch = dataset[img_id]
rays = batch['rays'].to(device)
rays_x = rays[...,:3]
rays_d = rays[...,3:6]
dxdu, dydv = rays[...,6:9], rays[...,9:12]

# Render image
L = torch.zeros_like(rays_x)
for _ in tqdm(range(spp//SPP)):
    if emitter_type == 'envmap':
        L += path_tracing_envmap_emitter(scene, emitter_net, material_net, rays_x, rays_d, dxdu, dydv, SPP, indir_depth=0)
    elif emitter_type == 'point':
        L += path_tracing_fix_emitter(scene, emitter_net, material_net, rays_x, rays_d, dxdu, dydv, SPP, indir_depth=0)
    else:
        L += path_tracing(scene, emitter_net, material_net, rays_x, rays_d, dxdu, dydv, SPP, indir_depth=7)
L = L.reshape(*img_hw,-1).cpu()/(spp//SPP)

# # Save result using OpenCV
# output_path = f'./images/{emitter_type}_sphere.png'
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
# output_img = (L.pow(1/2.2).numpy() * 255).astype(np.uint8)

# cv2.imwrite(output_path, output_img[...,::-1]) # Convert RGB to BGR for OpenCV
# # Display result
# plt.figure()
# plt.imshow(L.pow(1/2.2))
# plt.axis('off')
# plt.savefig('./images/{}_sphere.png'.format(emitter_type))
# plt.show()
# plt.imsave('./Mar11_results/{}_sphere_3.png'.format(emitter_type), L.pow(1/2.2).numpy().clip(0,1))  # Ensures correct normalization

plt.figure()
plt.imshow(L.pow(1/2.2))
plt.axis('off')
plt.savefig('./Mar11_results/brdf-emission.png')
plt.show()