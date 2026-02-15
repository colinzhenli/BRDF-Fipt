"""
Bonn BRDF Interface - PyTorch-based loader for Bonn BRDF database.

This implementation provides efficient BRDF lookups from the Bonn BRDF dataset,
which uses EXR images and MATLAB calibration files.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import scipy.io as spio
import warnings
import pyexr
import OpenEXR
import Imath

class BonnInterface:
    """
    PyTorch-based interface for Bonn BRDF database.
    
    The Bonn BRDF dataset consists of:
    - Calibration data (.mat files) containing camera and light positions
    - XYZ position maps (.exr files) for geometric information
    - Polychromatic and panchromatic images (.exr files)
    
    Args:
        data_dir: Path to directory containing Bonn BRDF data
        material_name: Name of the material (e.g., 'mat0003')
        device: Device to store tensors ('cuda' or 'cpu')
    """
    
    def __init__(self, data_dir, material_name='mat0003', device='cuda'):
        """
        Initialize Bonn BRDF interface and load material data.
        
        Args:
            data_dir: Path to directory containing Bonn BRDF data files
            material_name: Material identifier (default: 'mat0003')
            device: Device to store tensors ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.data_dir = Path(data_dir)
        self.material_name = material_name
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        print(f"\n{'='*60}")
        print(f"Loading Bonn BRDF Dataset: {material_name}")
        print(f"{'='*60}")
        print(f"Directory: {self.data_dir}")
        print(f"Device: {self.device}")
        
        # Load calibration data
        self._load_calibration()
        
        # Load geometric data (XYZ positions)
        self._load_geometry()
        
        # Load image data
        self._load_images()

        self.L_tensor = None
        self.V_tensor = None
        self.poly=None
        self.material_ids=None
        
        print(f"{'='*60}\n")
    
    def _loadmat(self, filename):
        """wrapper around scipy.io.loadmat that avoids conversion of nested matlab structs to np.arrays"""
        mat = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        for key in mat:
            if isinstance(mat[key], spio.matlab.mio5_params.mat_struct):
                mat[key] = self._to_dict(mat[key])
        return mat
    
    def _to_dict(self, matobj):
        """
        Convert MATLAB struct to Python dictionary.
        
        Args:
            matobj: MATLAB struct object
        
        Returns:
            Python dictionary
        """
        new_dict = {}
        for fn in matobj._fieldnames:
            val = matobj.__dict__[fn]
            if isinstance(val, spio.matlab.mio5_params.mat_struct):
                new_dict[fn] = self._to_dict(val)
            else:
                new_dict[fn] = val
        return new_dict
    
    def _load_calibration(self):
        """Load calibration data from .mat file."""
        # Look for calibration file
        calib_files = list(self.data_dir.glob(f"*_calibration.mat"))
        if len(calib_files) == 0:
            raise FileNotFoundError(f"No calibration file found in {self.data_dir}")
        
        calib_file = calib_files[0]
        print(f"Loading calibration from: {calib_file.name}")
        
        self.calib = self._loadmat(str(calib_file))
        
        # Extract available rotations
        self.rotations = [key for key in self.calib.keys() if key.startswith('rot')]
        print(f"  Available rotations: {self.rotations}")

        self.light_ids = [key for key in self.calib['rot000'].keys() if key.startswith('il')]
        self.camera_ids = [key for key in self.calib.keys() if key.startswith('cv')]
        print(f"  Available light ids: {self.light_ids}")
        
        # Get LLS angles if available
        if 'llsAnglesDegrees' in self.calib:
            self.lls_angles = self.calib['llsAnglesDegrees']
            print(f"  LLS angles: {self.lls_angles}")
    
    def _read_exr(self, filename):
        """
        Read EXR file and return as numpy array.
        
        Args:
            filename: Path to EXR file
        
        Returns:
            Numpy array with image data
        """
        exr_file = OpenEXR.InputFile(str(filename))
        header = exr_file.header()
        
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Get channel names
        channels = header['channels'].keys()
        
        # Read all channels
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channel_data = {}
        
        for channel in channels:
            channel_str = exr_file.channel(channel, FLOAT)
            channel_array = np.frombuffer(channel_str, dtype=np.float32)
            channel_array = channel_array.reshape(height, width)
            channel_data[channel] = channel_array
        
        # Stack RGB channels if available
        if 'R' in channels and 'G' in channels and 'B' in channels:
            img = np.stack([channel_data['R'], channel_data['G'], channel_data['B']], axis=-1)
        elif 'X' in channels and 'Y' in channels and 'Z' in channels:
            img = np.stack([channel_data['X'], channel_data['Y'], channel_data['Z']], axis=-1)
        else:
            # Return first channel
            img = channel_data[list(channels)[0]]
        
        return img
    
    def _load_geometry(self):
        """Load XYZ position maps from EXR files."""
        # Look for XYZ file
        xyz_files = list(self.data_dir.glob(f"{self.material_name}_xyz_*.exr"))
        if len(xyz_files) == 0:
            print(f"  Warning: No XYZ geometry file found")
            self.xyz = None
            return
        
        xyz_file = xyz_files[0]
        print(f"Loading geometry from: {xyz_file.name}")
        
        filename = f"{self.material_name}_xyz_rot000.exr"
        filepath = self.data_dir / filename
        self.xyz = pyexr.read(filepath)
        self.height, self.width = self.xyz.shape[:2]
        
        print(f"  Geometry shape: {self.xyz.shape}")
        
        # Convert to torch tensor
        self.xyz_tensor = torch.from_numpy(self.xyz).to(self.device).float()
    
    def _load_images(self):
        """Load polychromatic and panchromatic images."""
        # Look for poly2pan conversion data
        poly2pan_files = list(self.data_dir.glob(f"{self.material_name}_poly2pan.mat"))
        if len(poly2pan_files) > 0:
            print(f"Loading poly2pan conversion from: {poly2pan_files[0].name}")
            self.poly2pan = self._loadmat(str(poly2pan_files[0]))
        else:
            print("  Warning: No poly2pan conversion file found")
            self.poly2pan = None
        
        # Note: Actual image loading would be done on-demand to save memory
        print("  Image loading: on-demand (not preloaded)")
    
    def compute_directions(self, rotation='rot000', camera_id='cv01', light_id='il09'):
        """
        Compute light and view directions for all pixels.
        
        Args:
            rotation: Rotation key (e.g., 'rot000', 'rot045')
            camera_id: Camera identifier (e.g., 'cv01')
            light_id: Light identifier (e.g., 'il09')
        
        Returns:
            L: [H, W, 3] normalized light direction vectors
            V: [H, W, 3] normalized view direction vectors
        """
        camera_pos = self.calib[rotation][camera_id]
        
        light_pos = self.calib[rotation][light_id]
        
        print("before convert to torch tensors")
        # Convert to torch tensors
        camera_pos = torch.from_numpy(np.array(camera_pos)).to(self.device).float()
        light_pos = torch.from_numpy(np.array(light_pos)).to(self.device).float()
        
        print("after convert to torch tensors")
        print("camera_pos",camera_pos.shape)
        print("light_pos",light_pos.shape)
        print("self.xyz_tensor",self.xyz_tensor.shape)
        # Compute directions: direction = normalize(position - xyz)
        L = light_pos[None, None, :] - self.xyz_tensor  # [H, W, 3]
        V = camera_pos[None, None, :] - self.xyz_tensor  # [H, W, 3]
        

        # Normalize
        L = F.normalize(L, dim=-1)
        V = F.normalize(V, dim=-1)
        
        return L, V
    
    def load_image(self, image_type='poly'):
        """
        Load a specific image.
        
        Args:
            image_type: 'poly' for polychromatic or 'pan' for panchromatic
            rotation: Rotation identifier
            camera_id: Camera identifier
            light_id: Light identifier
        
        Returns:
            Image tensor
        """
        # Construct poly
        filename = f"{self.material_name}_poly.exr"
        filepath = self.data_dir / filename
        poly = pyexr.open(filepath)
        polyChannels = poly.channel_map['all']
        poly = poly.get(group='all', precision=pyexr.HALF)
        print("poly_channels",polyChannels[:3])
        poly_cv01_il026_rot000 = poly[:, :, :3]
        '''
        # Construct pan
        filename = f"{self.material_name}_pan.exr"
        filepath = self.data_dir / filename
        pan = pyexr.open(filepath)
        panChannels = pan.channel_map['all']
        pan = pan.get(group='all', precision=pyexr.HALF)
        print("pan_channels",panChannels[:3])
        pan_cv01_il026_rot000 = pan[:, :, :3]

        poly2pan_file = f"{self.material_name}_calibration.mat"
        poly2pan_filepath = self.data_dir / poly2pan_file
        poly2pan = self._loadmat(poly2pan_filepath)
        poly2panIndices = poly2pan['poly2panIndices']
        poly2panWeights = poly2pan['poly2panWeights']
        '''

        
        # INSERT_YOUR_CODE
        # Extract the camera ids from polyChannels as cam_ids
        # Assuming polyChannels is a list of strings like "poly_cv01_il026_rot000"
        cam_ids = [ch.split('_')[1] for ch in polyChannels]
        light_ids = [ch.split('_')[2].replace('il0', 'il') if ch.split('_')[2].startswith('il0') and len(ch.split('_')[2]) == 5 else ch.split('_')[2] for ch in polyChannels]
        rotations = [ch.split('_')[3] for ch in polyChannels]
        L_list = []
        V_list = []
        for i in range(len(cam_ids)):
            L,V=self.compute_directions(rotations[i], cam_ids[i], light_ids[i])
            L_list.append(L)
            V_list.append(V)
        L_tensor = torch.stack(L_list, dim=0)
        V_tensor = torch.stack(V_list, dim=0)
        self.L_tensor = L_tensor[::3].permute(1,2,0,3).reshape(-1,100,3)
        self.V_tensor = V_tensor[::3].permute(1,2,0,3).reshape(-1,100,3)
        self.poly = torch.tensor(poly).reshape(512, 512, 100, 3).reshape(-1,100,3)
        print("poly",self.poly.shape)
        print("L_tensor",self.L_tensor.shape)
        print("V_tensor",self.V_tensor.shape)
        return poly

    def get_brdf(self):
        poly=self.load_image(image_type="poly")
        
        self.material_ids = torch.arange(self.poly.shape[0]).repeat_interleave(self.poly.shape[1])
        self.poly = self.poly.reshape(-1,3)
        self.L_tensor = self.L_tensor.reshape(-1,3)
        self.V_tensor = self.V_tensor.reshape(-1,3)
        print("poly_info",torch.max(self.poly),torch.min(self.poly))
        print("L_x_info",torch.max(self.L_tensor[:,0]),torch.min(self.L_tensor[:,0]))
        print("L_y_info",torch.max(self.L_tensor[:,1]),torch.min(self.L_tensor[:,1]))
        print("L_z_info",torch.max(self.L_tensor[:,2]),torch.min(self.L_tensor[:,2]))
        print("V_x_info",torch.max(self.V_tensor[:,0]),torch.min(self.V_tensor[:,0]))
        print("V_y_info",torch.max(self.V_tensor[:,1]),torch.min(self.V_tensor[:,1]))
        print("V_z_info",torch.max(self.V_tensor[:,2]),torch.min(self.V_tensor[:,2]))
        return self.material_ids, self.poly, self.L_tensor, self.V_tensor
    
    def lookup_wiwo(self, wi, wo, pixel_coords=None):
        """
        Lookup BRDF values for given incoming/outgoing directions.
        
        Note: Bonn dataset stores images, not analytic BRDFs.
        This method finds the closest measured configuration.
        
        Args:
            wi: [B, 3] incoming direction vectors
            wo: [B, 3] outgoing direction vectors
            pixel_coords: Optional [B, 2] pixel coordinates (u, v)
        
        Returns:
            rgb: [B, 3] BRDF RGB values
        """
        # This is a placeholder - actual implementation would:
        # 1. Find the closest camera/light configuration
        # 2. Look up the corresponding image
        # 3. Return the pixel value at the specified location
        
        raise NotImplementedError(
            "Bonn BRDF lookup requires finding closest measured configuration. "
            "Use compute_directions() and load_image() to access measured data."
        )
    
    def get_available_configs(self):
        """
        Get list of available camera/light configurations.
        
        Returns:
            List of configuration dictionaries
        """
        configs = []
        for rotation in self.rotations:
            rot_data = self.calib[rotation]
            # Extract camera and light IDs
            camera_ids = [k for k in rot_data.keys() if k.startswith('cv')]
            light_ids = [k for k in rot_data.keys() if k.startswith('il')]
            
            for cam_id in camera_ids:
                for light_id in light_ids:
                    configs.append({
                        'rotation': rotation,
                        'camera': cam_id,
                        'light': light_id
                    })
        
        return configs
    
    def __repr__(self):
        return (f"BonnInterface(material='{self.material_name}', "
                f"device='{self.device}', "
                f"rotations={len(self.rotations)}, "
                f"geometry={'loaded' if self.xyz is not None else 'not loaded'})")


def main():
    """Example usage of BonnInterface."""
    data_dir = "/home/featurize/work/BRDF-Fipt_merl/utils/dataset/Bonn_test"
    material_name='mat0001'
    
    # Load dataset
    bonn = BonnInterface(data_dir, material_name=material_name, device='cuda')
    
    print(f"\n{bonn}")
    
    # Get available configurations
    configs = bonn.get_available_configs()
    print(f"\nAvailable configurations: {len(configs)}")
    print(f"Example configs: {configs[:3]}")
    
    # Compute directions for a specific config
    if len(bonn.rotations) > 0:
        rotation = bonn.rotations[0]
        print(f"\nComputing directions for {rotation}...")
        try:
            L, V = bonn.compute_directions(rotation=rotation)
            print(f"  Light directions (L): {L.shape}")
            print(f"  View directions (V): {V.shape}")
            print(f"  L range: [{L.min():.3f}, {L.max():.3f}]")
            print(f"  V range: [{V.min():.3f}, {V.max():.3f}]")
        except Exception as e:
            print(f"  Error: {e}")

    material_ids, poly, L_tensor, V_tensor = bonn.get_brdf()


if __name__ == '__main__':
    main()
