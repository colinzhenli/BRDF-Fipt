# from pysmtb.iv import iv
"""
This dataset loads AxF SVBRDFs and corresponding OpenEXR HDR images to memory and returns fixed size patches centered
on the pixels of the materials from the __getitem__() method.
"""
import json
from datetime import timedelta
from glob import glob

import axf_class
from axf_utils import read_sRGB_to_XYZ_matrix
import hashlib
import numpy as np
import os
import re
import torch
from copy import deepcopy
from pysmtb.utils import clamp, sortrows, strparse
from tqdm import tqdm
from typing import List, Tuple, Union
from warnings import warn

from pysmtb.image import read_openexr as read_exr
from pysmtb.image import split_patches
from pysmtb.utils import Dct

from src.utils import io_read_bin, io_write_bin, normalize, parse_meas_image_filename, safe_divide, str2bool


def make_torch_batch(batch, device: torch.device = None, input_channel_dim: int = 0, dtype: torch.dtype = torch.float32):
    """prepare batch with numpy ndarrays for torch model, optionally with GPU upload"""
    if device is None:
        device = torch.device('cpu')

    def convert(inp: torch.Tensor) -> torch.Tensor:
        if inp.dtype == torch.float16:
            return inp.type(dtype)
        else:
            return inp

    for k in batch.keys():
        if isinstance(batch[k], np.ndarray):
            m = batch[k]
            if m.ndim == 3:
                if input_channel_dim == 2:
                    m = m.transpose((2, 0, 1))
                else:
                    assert input_channel_dim == 0
            elif m.ndim == 4:
                assert input_channel_dim == 1, 'input_channel_dim must be 1 if ndim == 4'
            if m.ndim != 4:
                m = m[None, ...]
            batch[k] = convert(torch.from_numpy(m)).to(device)
        elif k == 'light_positions':
            batch[k] = [convert(torch.from_numpy(li[None, ...])).to(device) for li in batch[k]]
        elif isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].type(dtype).to(device)
    return batch


###############################################################################
'''
DATA GENERATOR
'''


class Dataset(torch.utils.data.Dataset):
    """Custom Dataset class for TAC7 images and AxFs"""

    def __init__(self,
                 split_filename: str,
                 input_dir: str,
                 suffix: str = '',
                 input_names: str = 'poly,pan,lls,unref_xyz,unref_normal',
                 label_names: Union[List[str], Tuple[str]] = ('diffuse', 'specular', 'refined_normal', 'roughness', 'anisotropy', 'fresnel', 'refined_heightmap'),  # , 'alpha'
                 load_confidences: bool = False,
                 first_mat_ind: int = 0,
                 last_mat_ind: int = 0,
                 ignore_cached: bool = False,
                 dtype: 'type' = np.float16,
                 dtype_out: 'type' = np.float32,
                 cast: bool = False,
                 normalize_inputs: Union[bool, str] = False,  # one of True, False, 'per_channel'
                 normalize_outputs: Union[bool, str] = False,  # one of True, False, 'per_channel'
                 normalization_dataset: 'Dataset' = None,
                 concatenate_maps: bool = True,
                 albedos_xyz: bool = False,
                 normals_only_xy: bool = False,
                 roughness_mapping: str = 'log',
                 aniso_mapping: str = 'cossin',
                 extract_label_center: bool = True,
                 samples_per_epoch: int = -1,
                 shuffle: bool = True,
                 reconstructed_axf_path: str = None,
                 crop_coords: dict = None,
                 padding: bool = False,
                 with_replacement: bool = False,
                 dataset_debug: bool = False,
                 verbosity: int = 0,
                 logger=None,
                 normalization_from_default_file: bool = False,
                 pattern: str = '.*',  # regular expression for material names
                 filter: str = '.*',  # regular expression for measurement image filenames ("cv(01,03)_il0(25,26).*")
                 cvs: str = '.*',  # regular expression for cameras
                 ils: str = '.*',  # regular expression for point lights
                 las: str = '.*',  # regular expression for linear light source inclination angles
                 rots: str = '.*',  # regular expression for turntable rotations
                 use_inverse_tac7_transforms: bool = False,
                 filter_roi_angles: bool = False,
                 max_abs_roi_angle: float = 5,
                 keep_axfs: bool = False,
                 patch_size: int = 15,
                 separate_poly: bool = False,
                 mode: str = 'regression',  # alternative: classification
                 classification_stats_source: str = 'file',
                 label_num_classes: str = '{'
                                          '"diffuse": 20, '
                                          '"specular": 20, '
                                          '"refined_normal": 20, '
                                          '"roughness": 20, '
                                          '"anisotropy": 20, '
                                          '"fresnel": 20,'
                                          '"refined_heightmap": 20,'
                                          '"alpha": 20}',
                 return_coordinates: bool = False,
                 input_rendered_meas: bool = True,
                 device: 'torch.device' = 'cpu',
                 min_timestamp: int = None,
                 max_timestamp: int = None,
                 ):
        """Create dataset from split file and data input directory."""

        # store all parameters in object
        self.kwargs = vars()
        # remove large objects from stored kwargs, as we do NOT want to serialize them!
        self.kwargs.pop('self')
        self.kwargs.pop('normalization_dataset', None)

        # txt file containing one material name per line for a training / validation / testing split
        self.split_filename = split_filename

        # index of last material, set to 0 to load all
        self.first_mat_ind = first_mat_ind
        self.last_mat_ind = last_mat_ind

        # regular expression for matching material names
        self.pattern = pattern

        # FIXME: setting this True is the preferable way of transforming pixels / 3D positions, however, it is broken right now
        self.use_inverse_tac7_transforms = use_inverse_tac7_transforms

        # FIXME: remove this hack to counteract the above problem
        self.filter_roi_angles = filter_roi_angles
        self.max_abs_roi_angle = max_abs_roi_angle

        # folder containing AxFs, measurement XML files and ..._rect folders
        self.input_dir = input_dir

        # suffix to material names (e.g. _refit16)
        self.suffix = suffix

        # keep loaded AxF objects or free up their memory
        self.keep_axfs = keep_axfs

        # active input modalities
        self.input_names = input_names
        self.input_names = self.input_names.split(',')
        self.input_names_meas = [name for name in self.input_names if name in ['lls', 'pan', 'poly']]

        # instead of using measurement images as inputs, use rerenderings of those
        self.input_rendered_meas = input_rendered_meas

        # confidences are not considered normal input modalities and are handled separately
        self.load_confidences = load_confidences

        # store polychromatic measurement images in separate array
        self.separate_poly = bool(separate_poly)

        # return pixel x and y coordinates for each patch
        self.return_coordinates = return_coordinates

        # active modalities used as labels
        self.label_names = label_names
        self.label_names = self.label_names.split(',')

        # lookup for default channel counts for each map type
        self.default_num_chans_maps = dict(diffuse=3,
                                           specular=3,
                                           refined_normal=3,
                                           roughness=2,
                                           anisotropy=1,
                                           fresnel=1,
                                           refined_heightmap=1,
                                           alpha=1)

        unknown_label_names = [l for l in self.label_names if l not in self.default_num_chans_maps.keys()]
        if len(unknown_label_names):
            raise Exception('the following selected labels are unknown: ' + str(unknown_label_names)
                            + ', supported label names: ' + str(list(self.default_num_chans_maps.keys())))

        # when classification is used instead of regression, we need to specify how many classes we want to afford per
        # label type
        self.classification = mode == 'classification'
        self.classification_stats_source = classification_stats_source
        assert self.classification_stats_source in ['file', 'compute'], 'unexpected value for classification_stats_source: ' + self.classification_stats_source
        self.label_num_classes = json.loads(label_num_classes)
        self.label_intervals = Dct()
        self.hash_classification_settings = hashlib.md5(str(self.label_num_classes).encode()).hexdigest()[:8]

        if self.load_confidences and not len(self.input_names_meas):
            raise Exception('confidence maps (reliability channels) can only be loaded when one of "poly", "pan", "lls" is in input_names')

        # --filter-like argument to select specific measurement images only
        self.filter = filter
        self.filter_cvs = cvs
        self.filter_ils = ils
        self.filter_las = las
        self.filter_rots = rots
        self.filter_cvs = self.filter_cvs.split(',') if self.filter_cvs is not None else '.*'
        self.filter_ils = self.filter_ils.split(',') if self.filter_ils is not None else '.*'
        self.filter_las = self.filter_las.split(',') if self.filter_las is not None else '.*'
        self.filter_rots = self.filter_rots.split(',') if self.filter_rots is not None else '.*'

        # force rewriting of cached files
        self.ignore_cached = ignore_cached

        # minimum and maximum occuring timestamps (use those for normalizing timestamps consistently over different splits)
        # if omitted, the earlies and latest timestamp in this split are used to normalize all timestamps to [0, 1]
        # see axf.AxF.get_meas_timestamp()
        min_timestamp = min_timestamp
        max_timestamp = max_timestamp

        # (square) size of patches to sample from the materials
        self.patch_size = patch_size

        # data type of labels and measurements, TODO: make use of these parameters
        self.dtype = dtype
        self.dtype_out = dtype_out
        self.cast = cast  # call .astype() on output arrays

        # apply normalization by subtracting mean and dividing by standard deviation to the measurement images
        self.normalize_inputs = normalize_inputs  # one of True, False, 'per_channel'
        if isinstance(self.normalize_inputs, str):
            self.normalize_inputs = self.normalize_inputs.lower()
        if self.normalize_inputs != 'per_channel':
            self.normalize_inputs = str2bool(self.normalize_inputs)

        # apply normalization by subtracting mean and dividing by standard deviation, individually for each output map type
        self.normalize_outputs = normalize_outputs  # one of True, False, 'per_channel'
        self.normalize_outputs = self.normalize_outputs.lower()
        if self.normalize_outputs != 'per_channel':
            self.normalize_outputs = str2bool(self.normalize_outputs)

        # set this argument to load normalization data from another dataset (e.g. to initialize the validation split with the same mean & std)
        normalization_dataset = normalization_dataset

        # allow fallback to default (i.e. splitTrain_INPNORMHASH_....npz) normalization files
        self.normalization_from_default_file = normalization_from_default_file

        # concatenate the selected AxF maps as one big tensor under batch['labels']
        self.concatenate_maps = concatenate_maps

        # dictionary that registers all kinds of label transformations
        self.label_mappings = Dct()

        # store alebdo maps in CIE XYZ instead of sRGB (this makes them bounded by [0,1])
        self.albedos_xyz = albedos_xyz
        if self.albedos_xyz:
            self.label_mappings['diffuse'] = self.map_albedo
            self.label_mappings['specular'] = self.map_albedo

        # TODO: self.normals_only_xy, self.roughness_mapping, self.aniso_mapping all belong into one common dictionary
        #  that maps map names to list of transformations, e.g. self.transformations['refined_normal'] = ['xy_only'];
        #  this will enable easier hash computations for more robust file caching and in general ease implementations
        # store only X- and Y-coordinates of normals (Z can be reconstructed due to normalization)
        self.normals_only_xy = normals_only_xy
        if self.normals_only_xy:
            self.label_mappings['refined_normal'] = self.map_normal

        # type of mapping to apply to the roughness maps (none, invsqrt, log)
        self.roughness_mapping = roughness_mapping
        if self.roughness_mapping != 'none':
            self.label_mappings['roughness'] = self.map_roughness

        # type of mapping to apply to the anisotropy maps (none, cossin)
        self.aniso_mapping = aniso_mapping
        if self.aniso_mapping != 'none':
            self.label_mappings['anisotropy'] = self.map_anisotropy

        # return only the center pixel from the label patches
        self.extract_label_center = extract_label_center

        # number of samples to produce in one epoch, set to -1 to use all available samples, can be used to select a
        # fraction from (0,1]
        self.samples_per_epoch = samples_per_epoch

        # optionally disable shuffling
        self.shuffle = shuffle

        # determine margin resulting from network reconstruction without padding by loading a reconstructed AxF and
        # comparing its size to the corresponding original AxF from the database
        # this margin is necessary to update the ROI when loading measurement images so that they match the ROI of
        # reconstructed AxFs
        self.reconstructed_axf_path = reconstructed_axf_path
        self.reconstructed_axf = None

        # size to which to crop materials after loading (use this to equalize their size to avoid bias during training)
        self.crop_coords = crop_coords
        if self.crop_coords == 'none' or self.crop_coords == 'full':
            self.crop_coords = None
        if self.crop_coords is not None:
            x0, x1, y0, y1 = (int(coord) for coord in self.crop_coords.split(','))
            self.crop_coords = dict(x0=x0, x1=x1, y0=y0, y1=y1)

        # add padding of half the patch size around the material boundaries
        self.padding = padding

        # allow repeated drawing of samples (can happen if samples_per_epoch is larger than the actual number of samples
        # in all materials, usually in toy examples)
        self.with_replacement = with_replacement

        # enable debug output
        self.debug = dataset_debug

        # use logging class to enable additional output to log files
        self.verbosity = verbosity

        if logger:
            def log(*args):
                if verbosity > 1:
                    logger.print(*args)
        else:
            def log(*args):
                if verbosity > 1:
                    print(*args)

        # some parameter bounds, those are hard-coded for now for AxF's Geisler-Moroder Ward BRDF with Fresnel term and
        # need to be adapted for other models like GGX
        self.min_roughness = 0.003  # empirically obtained limit, in practice the lowest occurring alpha is around 0.01 for fabrics
        self.max_roughness = 0.6  # hard upper limit used in Pantora
        self.min_fresnel = 0.015  # probably also empirically obtained, repeatedly occurring limit in many AxFs
        self.max_fresnel = 1  # physical upper bound

        # input and output normalization std lower bound
        self.division_threshold = 1e-7

        # read split file
        if not isinstance(self.split_filename, list):
            with open(self.split_filename, 'r') as f:
                self.material_names = f.readlines()
        else:
            self.material_names = self.split_filename
        # strip out empty or commented (#) lines
        self.material_names = [matname.strip() for matname in self.material_names]
        self.material_names = [s for s in self.material_names if len(s) and not s[0] == '#']
        # we should not sort here not to risk breaking the material order from the split files
        # self.material_names.sort()

        # store full split for reference
        self.material_names_full_split = self.material_names.copy()

        # match material names against regular expression, store indices unique to split file
        if self.pattern is not None:
            self.split_mat_inds, self.material_names = zip(*[(mat_ind, name) for mat_ind, name in enumerate(self.material_names) if re.match(self.pattern, name)])
            key = os.path.split(self.split_filename)[1] + '_' + self.pattern
        else:
            self.split_mat_inds = np.array(range(len(self.material_names))) + self.first_mat_ind
            key = os.path.split(self.split_filename)[1]

        # store filtered split for reference
        self.material_names_matching = {mat_ind: mat_name for mat_ind, mat_name in zip(self.split_mat_inds, self.material_names)}

        if self.filter_roi_angles:
            from axf_utils import read_roi
            rois = [read_roi(os.path.join(self.input_dir, mat_name + self.suffix + '.axf')) for mat_name in self.material_names]
            roi_angles = np.array([roi['angle'] for roi in rois])
            roi_filter = np.abs(roi_angles) <= self.max_abs_roi_angle
            self.material_names = [mat_name for mat_ind, mat_name in enumerate(self.material_names) if roi_filter[mat_ind]]
            self.split_mat_inds = [mat_ind for mi, mat_ind in enumerate(self.split_mat_inds) if roi_filter[mi]]

        # now subselect
        if self.first_mat_ind or self.last_mat_ind:
            if self.last_mat_ind == 0:
                self.last_mat_ind = len(self.material_names)
            self.material_names = self.material_names[self.first_mat_ind:self.last_mat_ind]
            self.split_mat_inds = self.split_mat_inds[self.first_mat_ind:self.last_mat_ind]
        self.num_mats = len(self.material_names)
        self.mat_inds = np.array(range(self.num_mats))
        self.split_mat_inds = np.array(self.split_mat_inds)

        if self.num_mats == 0:
            raise Exception('no matching materials found!')

        # generate hash of all parameters to automatically identify reusability of cached files
        props = dict(filter=self.filter,
                     filter_cvs=self.filter_cvs,
                     filter_ils=self.filter_ils,
                     filter_las=self.filter_las,
                     filter_rots=self.filter_rots)
        self.hash = hashlib.md5(str(props).encode()).hexdigest()[:8]

        # compute hash of all parameters that affect the inputs, used for loading cached normalization from disk
        props_input = dict(input_names=self.input_names,
                           channels_hash=self.hash)
        self.hash_inputs = hashlib.md5(str(props_input).encode()).hexdigest()[:8]
        if not os.path.exists(os.path.join(self.input_dir, 'INPNORMHASH_' + self.hash_inputs + '.txt')):
            with open(os.path.join(self.input_dir, 'INPNORMHASH_' + self.hash_inputs + '.txt'), 'w') as fp:
                json.dump(dict(props=props_input, kwargs=self.kwargs), fp, indent=True)
        
        if not os.path.exists(os.path.join(self.input_dir, 'HASH_' + self.hash + '.txt')):
            with open(os.path.join(self.input_dir, 'HASH_' + self.hash + '.txt'), 'w') as fp:
                json.dump(dict(props=props, kwargs=self.kwargs), fp, indent=True)

        props_crop = props.copy()
        props_crop['crop'] = self.crop_coords
        self.crop_hash = hashlib.md5(str(props_crop).encode()).hexdigest()[:8]

        if not os.path.exists(os.path.join(self.input_dir, 'HASH_' + self.hash + '_CROPHASH_' + self.crop_hash + '.txt')):
            with open(os.path.join(self.input_dir, 'HASH_' + self.hash + '_CROPHASH_' + self.crop_hash + '.txt'), 'w') as fp:
                json.dump(dict(props=props_crop, kwargs=self.kwargs), fp, indent=True)

        # optionally filter by camera, light and rotation
        cvs = '|'.join(['%02d' % int(cam) for cam in self.filter_cvs]) if self.filter_cvs != '.*' else '.*'
        rots = '|'.join(['%03d' % int(rot) for rot in self.filter_rots]) if self.filter_rots != '.*' else '.*'
        ils = '|'.join(['%03d' % int(led) for led in self.filter_ils]) if self.filter_ils != '.*' else '.*'
        las = '|'.join(['%05.2f' % int(la) for la in self.filter_las]) if self.filter_las != '.*' else '.*'

        # some TAC7 constants
        self.cvs_inclinations = np.array([5, 45, 67.5, 22.5])
        self.cvs_perm = (0, 3, 1, 2)  # sort cameras by their inclination angle
        self.poly_ils = np.r_[25:35]  # polychromatic leds
        self.poly_cvs = np.r_[1:5]  # cameras capturing polychromatic images
        self.poly_rots = [0, 45, 90, 135, 180]  # rotations of polychromatic images

        # apply margin resulting from network reconstructions to cropping ROI
        if self.reconstructed_axf_path is not None:
            self.reconstructed_axf = self.get_axf(mat_ind=None, mat_path=self.reconstructed_axf_path, crop=False)

        # loading
        self.axfs = {}
        self.xmls = {}
        self.heights = {}
        self.widths = {}
        self.crops = {}
        self.maps = {}
        self.confidences = {}
        self.inputs = {k: {} for k in self.input_names}
        self.meta = Dct()
        self.calib = Dct()
        self.cam_positions = {}
        self.point_light_positions = {}
        self.lls_positions = {}
        self.light_lookup = {}
        progress = tqdm(total=len(self.material_names),
                        desc='loading',
                        disable=self.verbosity < 1,
                        position=2,
                        ncols=75,
                        ascii=True,
                        bar_format='{desc}: {percentage:3.3f}%|{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                        )
        for mat_ind, mat_name in enumerate(self.material_names):
            log('loading material %d / %d: %s' % (mat_ind + 1, self.num_mats, mat_name))

            mat_path = os.path.join(self.input_dir, mat_name) + self.suffix

            # AxF & XML loading

            # read the AxF SVBRDF file
            self.axfs[mat_ind] = self.get_axf(mat_ind=mat_ind, crop=False)

            # assemble measurement image filenames for parsing meta data
            if os.path.exists(os.path.join(mat_path + '_rect')):
                # always prefer parsing actual EXR files
                image_fpaths = sorted(glob(os.path.join(mat_path + '_rect', '*.exr')))
                image_fnames = [os.path.splitext(os.path.split(fn)[1])[0] for fn in image_fpaths]
                with open(mat_path + '_image_filenames.json', 'w') as fp:
                    json.dump(image_fnames, fp, indent=4)
            else:
                # in case of limited storage, EXRs might be missing in favor of cached npz files; since we still need
                # filenames for metadata parsing, we load those from a json dump
                with open(mat_path + '_image_filenames.json', 'r') as fp:
                    image_fnames = json.load(fp, indent=4)
                image_fpaths = [os.path.join(mat_path + '_rect', fn) for fn in image_fnames]

            image_fnames_filtered = []
            for fn in image_fpaths:
                filename = os.path.splitext(os.path.split(fn)[1])[0]
                if not bool(re.match(self.filter, filename)) or not bool(
                        re.match(f'cv({cvs})_(il({ils})|lls01_la({las}))_rot({rots})', filename)):
                    continue
                image_fnames_filtered.append(fn)

            # parse meta data
            if mat_ind == 0:
                self.load_meta_data(image_fnames_filtered)

            # load TAC7 calibration from measurement XML file
            self.xmls[mat_ind] = self.axfs[mat_ind].meas_xml
            self.load_calib(mat_ind)

            # sanity check: some XML files were messed up in the past due to wrong copying / repeated processing
            roi_xml = deepcopy(self.xmls[mat_ind].roi)
            roi_axf = deepcopy(self.axfs[mat_ind].roi)
            if roi_xml['height'] != roi_axf['height'] or roi_xml['width'] != roi_axf['width']:
                # construct updated <rect ...> tag for xml file
                rect_fixed = f'<rect angle="%.2f" blendarea="%d" bottom="%d" down="%d" left="%d" right="%d" top="%d" totalhor="%d" totalver="%d"/>' % (
                    roi_axf['angle'], roi_xml['blendarea'], roi_axf['top'] + roi_axf['height'], roi_xml['down'],
                    roi_axf['left'], roi_axf['left'] + roi_axf['width'], roi_axf['top'],
                    roi_axf['totalhor'], roi_axf['totalver']
                )
                roi_axf.pop('cropXs')
                roi_axf.pop('cropYs')
                roi_xml.pop('cropXs')
                roi_xml.pop('cropYs')
                raise Exception('<rect .../> tag in XML file %s differs from AxF ROI, it should be %s' % (mat_path + '.xml', rect_fixed))

            # material dimensions before cropping
            h_orig = self.axfs[mat_ind].height()
            w_orig = self.axfs[mat_ind].width()

            # set bounds for crop ROI of this material
            self.update_crop(mat_ind, self.crop_coords, h_orig, w_orig)
            self.axfs[mat_ind] = self.crop_axf(mat_ind, copy=False)

            # store dimensions for easier access
            self.widths[mat_ind] = self.axfs[mat_ind].w()
            self.heights[mat_ind] = self.axfs[mat_ind].h()

            # store SVBRDF maps as dictionaries (need to deep-copy here because we might change some of the maps)
            self.maps[mat_ind] = self.get_maps(axf=self.axfs[mat_ind])

            # the sRGB to XYZ conversion matrix is constant for all materials, hence we can store it exactly once
            # FIXME: is this a universal property or does it depend on the TAC7 device used for measuring the AxFs?
            # FIXME: annotate whether this is supposed to be left or right multiplied!
            self.rgb_to_xyz_matrix_left_mul = read_sRGB_to_XYZ_matrix(os.path.join(self.input_dir, self.material_names[0]) + self.suffix + '.axf')
            self.xyz_to_rgb_matrix_left_mul = np.linalg.inv(self.rgb_to_xyz_matrix_left_mul)

            diffuse = self.axfs[mat_ind].diffuse
            specular = self.axfs[mat_ind].specular * self.axfs[mat_ind].fresnel
            diffuse_XYZ = np.einsum('ij,hwj->hwi', self.rgb_to_xyz_matrix_left_mul, diffuse).reshape((-1, 3))
            specular_XYZ = np.einsum('ij,hwj->hwi', self.rgb_to_xyz_matrix_left_mul, specular).reshape((-1, 3))
            if verbosity > 2 and (diffuse_XYZ.min() < -1e-2 or specular_XYZ.min() < -1e-2):
                warn('XYZ conversion of albedos gave values < 0: diffuse_XYZ.min(): %f, specular_XYZ.min(): %f'
                     % (diffuse_XYZ.min(), specular_XYZ.min()))

            # MEASUREMENT IMAGES
            for name in self.input_names_meas:
                self.inputs[name][mat_ind] = []
            self.confidences[mat_ind] = []

            # check if images have been cached for faster loading
            cached_cropped = True
            cached = True
            for modality in self.input_names_meas:
                cached = cached and os.path.exists(mat_path + '_HASH_' + self.hash + '_' + modality + '.bin')
                cached_cropped = cached_cropped and os.path.exists(mat_path + '_HASH_' + self.hash + '_CROPHASH_' + self.crop_hash + '_' + modality + '.bin')
            if self.load_confidences:
                cached = cached and os.path.exists(mat_path + '_HASH_' + self.hash + '_confidences.bin')
                cached_cropped = cached_cropped and os.path.exists(mat_path + '_HASH_' + self.hash + '_CROPHASH_' + self.crop_hash + '_confidences.bin')
            if self.reconstructed_axf is not None:
                # we have to set this False here because a reconstructed AxF was specified on dataset construction,
                # i.e. the effective crop ROI has changed potentially
                cached_cropped = False
            if self.debug:
                print('cached: ' + str(cached) + ', cached_cropped: ' + str(cached_cropped) + ', ignore_cached: ' + str(self.ignore_cached) + ', HASH_' + self.hash + '_CROPHASH_' + self.crop_hash)

            if self.input_rendered_meas:
                # RE-RENDER MEASUREMENTS: instead of loading measurement images, render them from AxF
                for modality in self.input_names_meas:
                    progress.update(n=1 / len(self.input_names_meas))
                    rerendering_cache_filename = mat_path + '_HASH_' + self.hash + '_CROPHASH_' + self.crop_hash\
                                                 + '_RERENDERING_' + modality + '.bin'
                    if os.path.exists(rerendering_cache_filename):
                        self.inputs[modality][mat_ind] = io_read_bin(rerendering_cache_filename)[0]
                    else:
                        renderings = self.render_meas(mat_ind=mat_ind, device=device)
                        r = renderings.pop(modality)  # h x w x c x n
                        h, w, c, n = r.shape
                        self.inputs[modality][mat_ind] = r.transpose((3, 2, 0, 1)).reshape((c * n, h, w))  # c * n x h x w

                        # cache binary file to disk
                        io_write_bin(self.inputs[modality][mat_ind], rerendering_cache_filename)

            elif not self.ignore_cached and (cached or cached_cropped):
                # READ CACHED MEASUREMENTS: cached binary format is much faster to read than individual OpenEXR files
                for modality in self.input_names_meas:
                    progress.update(n=1 / len(self.input_names_meas))
                    if cached_cropped:
                        # ideally read the readily cropped images
                        self.inputs[modality][mat_ind] = io_read_bin(mat_path + '_HASH_' + self.hash + '_CROPHASH_' + self.crop_hash + '_' + modality + '.bin')[0]
                    else:
                        # read uncropped images
                        self.inputs[modality][mat_ind] = io_read_bin(mat_path + '_HASH_' + self.hash + '_' + modality + '.bin')[0]

                    if self.meta['chs_' + modality] != self.inputs[modality][mat_ind].shape[0]:
                        # for now we rely on all loaded images having the same channel sampling
                        # changes would be necessary at least in all variables starting with self.channels_...
                        raise Exception('unexpected channel number, got %d instead of %d'
                                        % (self.inputs[modality][mat_ind].shape[0], self.meta['chs_' + modality]))

                    if self.load_confidences:
                        if cached_cropped:
                            self.confidences[mat_ind] = io_read_bin(mat_path + '_HASH_' + self.hash + '_CROPHASH_' + self.crop_hash + '_confidences.bin')[0]
                        else:
                            self.confidences[mat_ind] = io_read_bin(mat_path + '_HASH_' + self.hash + '_confidences.bin')[0]
            else:
                # READ MEASUREMENT IMAGES: OpenEXR post-processed measurement images
                for fn in image_fnames_filtered:
                    progress.update(n=1 / len(image_fnames_filtered))
                    filename = os.path.splitext(os.path.split(fn)[1])[0]
                    meta = parse_meas_image_filename(filename)
                    if meta['la'] is not None and 'lls' not in self.input_names_meas:
                        # skip LLS images if we don't want to load them at all
                        continue
                    if meta['il'] is not None and 'poly' not in self.input_names_meas and 'pan' not in self.input_names_meas:
                        # for the rare but possible case that only LLS images are to be used, we skip the point lit ones
                        continue

                    image = self.read_meas_image(fn)

                    # LLS
                    if meta['la'] is not None and 'lls' in self.input_names_meas:
                        self.inputs['lls'][mat_ind].append(image['pan'].transpose((2, 0, 1)))

                    # POLY
                    if meta['il'] in self.poly_ils and 'poly' in self.input_names_meas:
                        self.inputs['poly'][mat_ind].append(image['poly'].transpose((2, 0, 1)))

                    # PAN
                    if meta['il'] is not None and meta['il'] not in self.poly_ils:
                        self.inputs['pan'][mat_ind].append(image['pan'].transpose((2, 0, 1)))

                    # CONFIDENCE
                    if self.load_confidences:
                        self.confidences[mat_ind].append(image['confidence'].transpose((2, 0, 1)))

                # concatenate each modality into one big 3D array, count channels, write cache files
                for modality in self.input_names_meas:
                    if not len(self.inputs[modality][mat_ind]):
                        raise Exception('no images loaded for meas_' + modality)
                    self.inputs[modality][mat_ind] = np.concatenate(self.inputs[modality][mat_ind], axis=0)

                    if self.meta['chs_' + modality] != self.inputs[modality][mat_ind].shape[0]:
                        # for now we rely on all loaded images having the same channel sampling
                        # changes would be necessary at least in all variables starting with self.channels_...
                        raise Exception('unexpected channel number for %s, got %d instead of %d'
                                        % (modality, self.inputs[modality][mat_ind].shape[0], self.meta['chs_' + modality]))

                    # cache binary files to disk
                    io_write_bin(self.inputs[modality][mat_ind], mat_path + '_HASH_' + self.hash + '_' + modality + '.bin')

                # CONFIDENCES
                if self.load_confidences:
                    self.confidences[mat_ind] = np.concatenate(self.confidences[mat_ind], axis=0)

                    # cache binary files to disk
                    io_write_bin(self.confidences[mat_ind], mat_path + '_HASH_' + self.hash + '_confidences.bin')

            # CROPPING & CACHING
            if not cached_cropped:
                for modality in self.input_names_meas:
                    self.inputs[modality][mat_ind] = self.crop_image(mat_ind, self.inputs[modality][mat_ind],
                                                                     subtract_reconstruction_margin=self.reconstructed_axf is not None, channel_dim=0)
                    was_cropped = (h_orig, w_orig) != self.inputs[modality][mat_ind].shape[1:3]

                    if was_cropped and self.reconstructed_axf is None:
                        # cache cropped binary files to disk
                        io_write_bin(self.inputs[modality][mat_ind], mat_path + '_HASH_' + self.hash + '_CROPHASH_' + self.crop_hash + '_' + modality + '.bin')

                if self.load_confidences:
                    self.confidences[mat_ind] = self.crop_image(mat_ind, self.confidences[mat_ind],
                                                                subtract_reconstruction_margin=self.reconstructed_axf is not None, channel_dim=0)
                    was_cropped = (h_orig, w_orig) != self.confidences[mat_ind].shape[1:3]

                    # cache cropped binary files to disk
                    if was_cropped and self.reconstructed_axf is None:
                        io_write_bin(self.confidences[mat_ind], mat_path + '_HASH_' + self.hash + '_CROPHASH_' + self.crop_hash + '_confidences.bin')

        if not len(self.maps) and not len(self.inputs[self.input_names_meas[0]]):
            raise Exception('no materials loaded!')

        # record AxF timestamps
        self.timestamps = dict([(mat_ind, self.xmls[mat_ind].get_meas_datetime()) for mat_ind in range(self.num_mats)])
        if min_timestamp is None:
            self.min_timestamp = min(list(self.timestamps.values()))
        if max_timestamp is None:
            self.max_timestamp = max(list(self.timestamps.values()))
        if self.min_timestamp == self.max_timestamp:
            self.max_timestamp = self.min_timestamp + timedelta(seconds=1)

        # free up memory
        if not self.keep_axfs:
            del(self.axfs)
            self.axfs = {}

        # mapping from input modality name to channel indices
        self.chinds_inputs = dict()
        ci = 0
        for k in self.input_names:
            c0 = ci
            if k == 'poly' and self.separate_poly:
                # separate polychrome images are not supposed to be used as normal inputs but in a dedicated network
                # --> exclude them here
                continue
            elif k == 'unref_normal':
                if self.normals_only_xy:
                    nc = 2
                else:
                    nc = 3
            elif k == 'unref_xyz':
                nc = 3
            elif k in ['poly', 'pan', 'lls']:
                nc = self.meta['chs_' + k]
            else:
                raise Exception('please implement lookup of channel number of input modality ' + k)
            c1 = ci + nc
            self.chinds_inputs[k] = torch.from_numpy(np.r_[c0:c1])
            ci += len(self.chinds_inputs[k])

        # record channel counts for each of the SVBRDF parameter maps
        self.num_chans_maps = dict([(k, self.maps[0][k].shape[0]) for k in self.label_names])
        self.num_chans_maps_mapped = self.num_chans_maps.copy()
        if self.aniso_mapping == 'cossin' and 'anisotropy' in self.num_chans_maps:
            # we override this here, as we only want a single output from the network that will be mapped manually using sin() and cos()
            # i.e. the mapping doesn't affect the network's final layer output dimension (hence unmapped == 1)
            self.num_chans_maps['anisotropy'] = 1
            # the mapped channel count IS affected because the network applies the cossin mapping as a nonlinearity,
            # i.e. the actual number of output channels is increased
            self.num_chans_maps_mapped['anisotropy'] = 2
        if self.normals_only_xy and 'refined_normal' in self.num_chans_maps:
            # here we adjust the channel count for the normal map, as the network predicts 2 channels when normals_only_xy == True
            self.num_chans_maps['refined_normal'] = 2
            # the mapped normal channels remain the same, as the normals are not reconstructed for the loss computation
            self.num_chans_maps_mapped['refined_normal'] = 2

        # overall label channel count without mappings, must be 13 when normals_xy_only == 1 and aniso_mapping == cossin
        self.num_chans_labels = np.sum(list(self.num_chans_maps.values()))
        # overall label channel count with mappings applied, must be 14 when normals_xy_only == 1 and aniso_mapping == cossin
        self.num_chans_labels_mapped = np.sum(list(self.num_chans_maps_mapped.values()))

        # mapping from label modality name to channel indices
        self.chinds_labels = {}
        self.chinds_labels_mapped = {}
        ci = 0
        cim = 0
        for k in self.label_names:
            c0 = ci
            c1 = ci + self.num_chans_maps[k]
            self.chinds_labels[k] = torch.from_numpy(np.r_[c0:c1])
            ci += self.num_chans_maps[k]
            c0 = cim
            c1 = cim + self.num_chans_maps_mapped[k]
            self.chinds_labels_mapped[k] = torch.from_numpy(np.r_[c0:c1])
            cim += self.num_chans_maps_mapped[k]

        # number of pixels in each material
        if not self.padding:
            self.pixel_counts = np.array(
                [(self.widths[mat_ind] - self.patch_size + 1) *
                 (self.heights[mat_ind] - self.patch_size + 1) for mat_ind in range(len(self.mat_inds))])
        else:
            self.pixel_counts = np.array([self.widths[mat_ind] *
                                          self.heights[mat_ind] for mat_ind in range(len(self.mat_inds))])

        if self.samples_per_epoch == -1:
            self.samples_per_epoch = np.sum(self.pixel_counts)
        else: #if self.samples_per_epoch < 1:
            self.samples_per_epoch *= np.sum(self.pixel_counts)
        #assert self.samples_per_epoch > 1, 'samples_per_epoch must be either -1, from (0, 1] or an integer'

        # number of samples to draw from each material
        self.num_samples_per_mat = (self.pixel_counts / np.sum(self.pixel_counts) * self.samples_per_epoch).astype('int')

        if self.with_replacement:
            self.num_samples_per_mat = np.maximum(self.pixel_counts, self.num_samples_per_mat)
        else:
            # we can't pick more samples than there are pixels in the material
            self.num_samples_per_mat = np.minimum(self.pixel_counts, self.num_samples_per_mat)

        # we randomly sample pixels from each material
        self.sampled_pixels = []
        self.resample_pixel_ids()

        # for each sampled pixel, we need to store the corresponding material index as well
        self.sampled_mat_inds = []
        for mat_ind in range(self.num_mats):
            self.sampled_mat_inds.append(np.repeat(mat_ind, self.num_samples_per_mat[mat_ind]))
        self.sampled_mat_inds = np.array([item for sublist in self.sampled_mat_inds for item in sublist])

        # initialize per-material mean / std arrays
        self.means_inputs = dict([(k, np.zeros((self.meta['chs_' + k], self.num_mats), dtype=np.float32)) for k in self.input_names])
        self.stds_inputs = dict([(k, np.ones((self.meta['chs_' + k], self.num_mats), dtype=np.float32)) for k in self.input_names])
        self.means_outputs = dict([(k, torch.zeros((self.num_chans_maps[k], self.num_mats), dtype=torch.float32)) for k in self.label_names])
        self.stds_outputs = dict([(k, torch.ones((self.num_chans_maps[k], self.num_mats), dtype=torch.float32)) for k in self.label_names])

        # initialize / load global mean / std arrays
        if normalization_dataset is not None:
            self.mean_inputs = normalization_dataset.mean_inputs
            self.std_inputs = normalization_dataset.std_inputs
            self.mean_outputs = normalization_dataset.mean_outputs
            self.std_outputs = normalization_dataset.std_outputs
        else:
            self.mean_inputs = dict([(k, np.zeros((self.meta['chs_' + k],), dtype=np.float32)) for k in self.input_names])
            self.std_inputs = dict([(k, np.ones((self.meta['chs_' + k], ), dtype=np.float32)) for k in self.input_names])
            self.mean_outputs = dict([(k, torch.zeros((self.num_chans_maps[k]), dtype=torch.float32)) for k in self.label_names])
            self.std_outputs = dict([(k, torch.ones((self.num_chans_maps[k]), dtype=torch.float32)) for k in self.label_names])
            self.compute_mean_std()

        # ensure division by std is safe so we don't have to check this on the fly all the time and can directly divide
        for k in list(self.std_inputs.keys()):
            self.std_inputs[k][self.std_inputs[k] <= self.division_threshold] = 1
            self.stds_inputs[k][self.stds_inputs[k] <= self.division_threshold] = 1
        for k in list(self.std_outputs.keys()):
            self.std_outputs[k][self.std_outputs[k] <= self.division_threshold] = 1
            self.stds_outputs[k][self.stds_outputs[k] <= self.division_threshold] = 1

        # compute color to pan / lls conversion coefficients
        # TODO: use spectra from XML, verify with numeric factors
        # self.poly_to_pan_weights = self.estimate_poly_to_pan_weights()

        # test dataloader
        if self.reconstructed_axf is None:
            # only test dataloader when it is used for training (self.sampled_pixels causes out of bounds errors because
            # we might have stripped the reconstruction margin otherwise)
            batch = self.__getitem__(0)

        poly = self.get_channel_inds('poly', sorting=['rot', 'cv', 'il'])
        pan = self.get_channel_inds('pan', sorting=['rot', 'cv', 'il'])
        lls = self.get_channel_inds('lls', sorting=['rot', 'cv', 'la'])

        # analyze label distributions for adaptive quantization
        class_stats_fname = os.path.join(self.input_dir, 'CLASSSTATSHASH_' + self.hash_classification_settings + '.json')
        if self.classification and self.classification_stats_source == 'file':
            # load classification statistics from previously cashed file
            with open(class_stats_fname, 'r') as fp:
                self.label_intervals = json.load(fp)

            # check that all labels exist
            for label_name in self.label_names:
                if label_name not in self.label_intervals:
                    raise Exception('label ' + label_name + ' not in class statistics: ' + class_stats_fname)
        elif self.classification:
            # gather all labels to compute discretization boundaries per label type
            if self.crop_coords is not None:
                warn(f'classification statistics cache {class_stats_fname} could not be found, computing class '
                     f'statistics over cropped dataset')

            if self.last_mat_ind != 0 or self.first_mat_ind != 0 or self.pattern != '.*':
                raise Exception(f'classification statistics cache {class_stats_fname} could not be found, the entire '
                                f'training split must be loaded for mean / std computation, set --last_mat_ind to 0 and --train_mat_pattern to ".*"!')

            # compute statistics
            self.quantize_labels()

            # cache statistics to file if it doesn't exist yet
            if not os.path.exists(class_stats_fname):
                with open(class_stats_fname, 'w') as fp:
                    json.dump({k: [v for v in vs] for k, vs in self.label_intervals.items()}, fp, indent=True)

    def load_meta_data(self, image_fnames_filtered: List[str]) -> Dct:
        # parse meta data from measurement image filenames

        self.meta = Dct(num_total=0, chs_total=0, cvs=[], rots=[], ils=[], las=[],
                        is_pointlit=[], is_lls=[], is_pan=[], is_poly=[],
                        inds_pointlit=[], inds_lls=[], inds_pan=[], inds_poly=[])
        self.meta.update({'chs_' + name: 0 for name in self.input_names})
        self.meta.update({'num_' + name: 0 for name in self.input_names})
        self.meta.update({'names_' + name: [] for name in self.input_names_meas})
        self.meta.update({'inds_' + name: [] for name in self.input_names_meas})
        self.meta['names_confidence'] = []
        self.meta['inds_confidence'] = []
        self.meta['chs_unref_normal'] = 2 if self.normals_only_xy else 3
        self.meta['num_unref_normal'] = 1
        self.meta['chs_unref_xyz'] = 3
        self.meta['num_unref_xyz'] = 1

        im_ind_local = 0
        for fn in image_fnames_filtered:
            filename = os.path.splitext(os.path.split(fn)[1])[0]

            meta = parse_meas_image_filename(filename)
            if meta['la'] is not None and 'lls' not in self.input_names_meas:
                # skip LLS images if we don't want to load them at all
                continue
            if meta['il'] is not None and 'poly' not in self.input_names_meas and 'pan' not in self.input_names_meas:
                # for the rare but possible case that only LLS images are to be used, we skip the point lit ones
                continue

            # LLS
            if meta['la'] is not None and 'lls' in self.input_names_meas:
                self.meta['names_lls'].append(filename)
                self.meta['inds_lls'].append(im_ind_local)

            # POLY
            if meta['il'] in self.poly_ils and 'poly' in self.input_names_meas:
                self.meta['names_poly'].append(filename)
                self.meta['inds_poly'].append(im_ind_local)

            # PAN
            if meta['il'] is not None and meta['il'] not in self.poly_ils:
                self.meta['names_pan'].append(filename)
                self.meta['inds_pan'].append(im_ind_local)

            # always append confidence names and indices
            self.meta['names_confidence'].append(filename)
            self.meta['inds_confidence'].append(im_ind_local)

            self.meta['cvs'].append(meta['cv'])
            self.meta['rots'].append(meta['rot'])
            if meta['il'] is not None:
                is_poly = meta['il'] in self.poly_ils
                self.meta['ils'].append(meta['il'])
                self.meta['las'].append(-100)
                self.meta['is_pointlit'].append(True)
                self.meta['is_lls'].append(False)
                self.meta['is_poly'].append(is_poly)
                self.meta['is_pan'].append(not is_poly)
                self.meta['inds_pointlit'].append(im_ind_local)
            else:
                self.meta['las'].append(meta['la'])
                self.meta['ils'].append(0)
                self.meta['is_lls'].append(True)
                self.meta['is_pointlit'].append(False)
                self.meta['is_poly'].append(False)
                self.meta['is_pan'].append(False)

            im_ind_local += 1

        # get pan & poly lookups within point lit indices only
        inds_pl = np.array(self.meta['inds_pointlit'])
        inds_pan = np.array(self.meta['inds_pan'])
        inds_poly = np.array(self.meta['inds_poly'])
        self.meta['inds_pan_in_pointlits'] = np.where(np.any(inds_pl[:, None] == inds_pan[None, :], axis=1))[0].tolist()
        self.meta['inds_poly_in_pointlits'] = np.where(np.any(inds_pl[:, None] == inds_poly[None, :], axis=1))[0].tolist()

        self.meta2 = deepcopy(self.meta)

        # convert to numpy arrays
        for k in self.meta.keys():
            if isinstance(self.meta[k], list) and not isinstance(self.meta[k][0], str):
                self.meta[k] = np.array(self.meta[k])

        # store per-modality meta data
        for modality in self.input_names_meas:
            # count number of images and channels per modality in lls, pan & poly
            self.meta['num_' + modality] = len(self.meta['names_' + modality])
            self.meta['chs_' + modality] = self.meta['num_' + modality] * (3 if modality == 'poly' else 1)
            self.meta['num_total'] += self.meta['num_' + modality]
            self.meta['chs_total'] += self.meta['chs_' + modality]

            # parse per modality cvs, ils, las & rots from filenames
            if modality == 'lls':
                tmp = strparse(self.meta['names_' + modality], 'cv(\\d+)_lls01_la(-?\\d+\\.\\d+)_rot(\\d+)')
                self.meta['las_' + modality] = np.array([float(la) for la in tmp[:, 1]])
            else:
                tmp = strparse(self.meta['names_' + modality], 'cv(\\d+)_il(\\d+)_rot(\\d+)')
                self.meta['ils_' + modality] = np.array([int(il) for il in tmp[:, 1]])
            self.meta['cvs_' + modality] = tmp[:, 0].astype(int)
            self.meta['rots_' + modality] = tmp[:, 2].astype(int)
        return self.meta

    def load_calib(self, mat_ind: int):
        # load calibration data from measurement XML file

        # sanity check: ensure all materials share the same measurement turntable rotations
        rot_angles = self.xmls[mat_ind].rot_angles
        if mat_ind == 0:
            self.calib['rot_angles'] = rot_angles
            self.calib['mats_pix_to_world'] = np.zeros((self.num_mats, 3, 4, len(rot_angles)), dtype=self.dtype_out)
            self.calib['mats_normal'] = np.zeros((self.num_mats, 3, 3, len(rot_angles)), dtype=self.dtype_out)
        else:
            assert np.all(rot_angles == self.calib['rot_angles']), 'rotation angles for matieral %s are different, ' \
                                                                   'got %s, expected %s' % (
                                                                       self.material_names[mat_ind], str(rot_angles),
                                                                       str(self.calib['rot_angles']))

        if not self.use_inverse_tac7_transforms:
            # FIXME: make use of inverse geometric transform to save some processing in renderlayer; currently
            #  prevented by missing / incorrect normal transform
            # transformation from pixel to world coordinates, accounting for ROI translation & rotation, as well as
            # the target3d coordinate system (canonical TAC7 coordinates), and finally each of the turntable
            # rotations (hence we store a stack of 3x4 matrices, sorted by ROI rotation angles);
            # matrix performing the ROI & turntable rotation (needs to be applied to normal vectors)
            pix2worlds = self.xmls[mat_ind].mat_pix_to_world_tt
            mat_roi_rot = self.xmls[mat_ind].mat_roi_rot[:3, :3]
            for rot_ind, rot in enumerate(self.calib['rot_angles']):
                tt_rot_mat = self.xmls[mat_ind].getTurntableRotMat(rot)[:3, :3]
                self.calib['mats_normal'][mat_ind, :, :, rot_ind] = tt_rot_mat @ mat_roi_rot
                self.calib['mats_pix_to_world'][mat_ind, :, :, rot_ind] = pix2worlds[rot][:3, :]

        # GEOMETRIC CALIBRATION
        num_images = self.meta['num_total']
        self.cam_positions[mat_ind] = np.zeros((3, num_images), dtype=np.float32)
        self.point_light_positions[mat_ind] = []
        self.lls_positions[mat_ind] = []
        self.light_lookup = Dct(inds_lls=9999 * np.ones(num_images, dtype=np.int32),
                                inds_point=999 * np.ones(num_images, dtype=np.int32),
                                is_lls=np.zeros(num_images, dtype=np.bool8))

        # for forward transformations (use_inverse_tac7_transforms == False), we store camera & light positions in
        # canonical TAC7 coordinates, i.e. 3D points for turntable rotation 0, later, we rotate the 3D pixel
        # coordinates and normals from canonical TAC7 coordinates according to the turntable rotation; for inverse
        # transformations we store camera & light positions rotated by the inverse turntable rotation (and w.r.t.
        # the ROI rotation, which is broken right now (FIXME))
        ind_point = 0
        ind_lls = 0
        for ci in range(num_images):
            cv = self.meta['cvs'][ci]
            rot = self.meta['rots'][ci]
            if self.use_inverse_tac7_transforms:
                self.cam_positions[mat_ind][:, ci] = self.xmls[mat_ind].get_camera_pos(cv, rotAngleDegrees=rot,
                                                                                       inverse=True)[:, 0]
            else:
                self.cam_positions[mat_ind][:, ci] = self.xmls[mat_ind].get_camera_pos(cv, rotAngleDegrees=0,
                                                                                       inverse=False)[:, 0]

            lid = self.meta['ils'][ci]
            la = self.meta['las'][ci]
            if lid == 0:
                # LLS
                if self.use_inverse_tac7_transforms:
                    self.lls_positions[mat_ind].append(
                        self.xmls[mat_ind].get_lls_corners(la, rotAngleDegrees=rot, inverse=True).astype(np.float32))
                else:
                    self.lls_positions[mat_ind].append(
                        self.xmls[mat_ind].get_lls_corners(la, rotAngleDegrees=0, inverse=False).astype(np.float32))
                self.light_lookup['is_lls'][ci] = True
                self.light_lookup['inds_lls'][ci] = ind_lls
                ind_lls += 1
            else:
                # point light
                if self.use_inverse_tac7_transforms:
                    self.point_light_positions[mat_ind].append(
                        self.xmls[mat_ind].get_light_pos(lid, rotAngleDegrees=rot, inverse=True).astype(np.float32))
                else:
                    self.point_light_positions[mat_ind].append(
                        self.xmls[mat_ind].get_light_pos(lid, rotAngleDegrees=0, inverse=False).astype(np.float32))
                self.light_lookup['is_lls'][ci] = False
                self.light_lookup['inds_point'][ci] = ind_point
                ind_point += 1
        if len(self.lls_positions[mat_ind]):
            self.lls_positions[mat_ind] = np.stack(self.lls_positions[mat_ind], axis=1)  # 3 x NLLS x 4
        self.point_light_positions[mat_ind] = np.stack(self.point_light_positions[mat_ind], axis=1)  # 3 x NPL x 1
        self.light_lookup['inds_point'] = np.array(self.light_lookup['inds_point'])
        self.light_lookup['inds_lls'] = np.array(self.light_lookup['inds_lls'])
        self.light_lookup['is_lls'] = np.array(self.light_lookup['is_lls'])

        # RADIOMETRIC CALIBRATION
        # load colorspace conversion coefficients
        # FIXME: coefficients are broken
        coeffs = self.xmls[mat_ind].compute_color_conversion_coefficients(bandwidth=10,
                                                                          colorspace='xyz' if self.albedos_xyz else 'srgb')
        if 'poly2pan' not in self.calib.keys():
            self.calib['poly2pan'] = np.ones((self.num_mats, self.meta['num_total'], 3)) / 3
            self.calib['poly2poly'] = np.ones((self.num_mats, self.meta['num_total'], 3))
            self.calib['pan2pan'] = np.ones((self.num_mats, self.meta['num_total']))
        for ii in range(self.meta['num_total']):
            cv = self.meta['cvs'][ii]
            if self.meta['is_lls'][ii]:
                self.calib['poly2pan'][mat_ind, ii, :] = coeffs['poly2lls'][cv]
                self.calib['pan2pan'][mat_ind, ii] = coeffs['pan2lls'][cv]
            else:
                il = self.meta['ils'][ii]
                self.calib['poly2pan'][mat_ind, ii, :] = coeffs['poly2pan'][cv][il]
                self.calib['poly2poly'][mat_ind, ii, :] = coeffs['poly2poly'][cv][il]
                self.calib['pan2pan'][mat_ind, ii] = coeffs['pan2pan'][cv][il]

    def get_split_preview(self, mat_ind, matching=False):
        from PIL import Image
        if matching:
            preview = Image.open(os.path.join(self.input_dir, self.material_names_matching[mat_ind] + self.suffix + '_preview.jpg'))
        else:
            preview = Image.open(os.path.join(self.input_dir, self.material_names_full_split[mat_ind] + self.suffix + '_preview.jpg'))
        return np.array(preview, dtype=np.float32) / 255

    def get_split_material_inds(self):
        """get the actual material indices as they occur in the split file"""
        return self.split_mat_inds

    def find_mat_ind(self, inp: str):
        """try if input matches any loaded materialname and return its index, returns None on failure
        input can be either directly a material name, or a path, from which the file name (without extension) is is
        interpreted as the material name"""
        mat_name = os.path.split(inp)[1]
        mat_name = os.path.splitext(mat_name)[0]
        # remove potential prefix from names as they are stored in some AxF files
        mat_name = mat_name.replace('Generated from', '')
        # look up all loaded material names as prefix of the stripped filename, in case there are additional suffixes
        # attached after the material name (e.g. _refit16)
        inds = np.where([mn in mat_name for mn in self.material_names])[0]
        if inds is not None and len(inds) > 0:
            return inds[0]
        else:
            return

    def get_axf(self, mat_ind: int = None, mat_path: str = None, crop: bool = False):
        """load an AxF file specified by its material index in the (filtered and sliced!) split file, alternatively an
        arbitrary filename can be provided via mat_name; optionally the AxF object can be cropped, specified as a string
        with 'x0,x1,y0,y1'"""

        if mat_ind is not None:
            # given material index, compose AxF and XML paths
            mat_path = os.path.join(self.input_dir, self.material_names[mat_ind]) + self.suffix
            mat_path_dataset = mat_path
        else:
            # only mat_path given, first check if XML file exists by changing extension to .xml
            assert mat_path is not None, 'if mat_ind is not given, mat_path must be'
            mat_path = os.path.splitext(mat_path)[0]
            mat_folder, mat_name = os.path.split(mat_path)

            mat_path_dataset = os.path.join(mat_folder, mat_name)
            if not os.path.exists(mat_path_dataset + '.xml'):
                # mat_path might point to an external file, while heightmap & XML need to be loaded from the dataset
                mat_ind = self.find_mat_ind(mat_path)
                if mat_ind is not None:
                    # XML ROI most likely needs to be updated to the one in the AxF, this happens automatically on
                    # loading the XML into the axf object via load_calib(), and further below for the unrefined
                    # heightmap as well
                    mat_path_dataset = os.path.join(self.input_dir, self.material_names[mat_ind]) + self.suffix
            if not os.path.exists(mat_path_dataset + '.xml'):
                raise FileNotFoundError('cannot find measurement XML file matching ' + mat_path)

        if not os.path.exists(mat_path + '.axf'):
            raise FileNotFoundError('AxF file does not exist: ' + mat_path + '.axf')
        if not os.path.exists(mat_path_dataset + '.xml'):
            raise FileNotFoundError('XML file does not exist: ' + mat_path_dataset + '.xml')
        if not os.path.exists(mat_path_dataset + '.exr'):
            raise FileNotFoundError('EXR file does not exist: ' + mat_path_dataset + '.exr')

        # load AxF & XML from disk
        axf = axf_class.AxF(fname=mat_path + '.axf', load_calib=False)
        axf.load_calib_xml(mat_path_dataset + '.xml')
        height_axf, width_axf, _ = axf.shape()
        axf.custom_roi = Dct(left=0, top=0, width=width_axf, height=height_axf, mat_ind=mat_ind)

        # read unrefined heightmap for geometry inputs
        unref_heightmap, chans = read_exr(mat_path_dataset + '.exr', pixel_type='float')
        if not chans[0] == 'height':
            raise Exception('could not load unrefined heightmap %s, expected channel "height" but got %s' %
                            (mat_path_dataset + '.exr', chans[0]))

        height_heightmap, width_heightmap, _ = unref_heightmap.shape
        if height_heightmap < height_axf or width_heightmap < width_axf:
            raise Exception('heightmap is smaller than AxF!')
        else:
            # none-dataset AxF might be cropped --> apply ROI to heightmap
            # no need to update the measurement xml inside the axf object, as that is directly cropped on loading
            rx = axf.meas_xml.roi
            axf.custom_roi.left = rx['left'] - rx['uncropped_left']
            axf.custom_roi.top = rx['top'] - rx['uncropped_top']
            r = axf.custom_roi
            unref_heightmap = unref_heightmap[r.top:r.top + r.height, r.left:r.left + r.width]
        unref_heightmap = unref_heightmap.astype(self.dtype)
        if crop:
            assert mat_ind is not None, 'mat_ind must be specified if AxF should be cropped'
            axf = self.crop_axf(mat_ind, copy=False, axf=axf)
            unref_heightmap = self.crop_image(mat_ind, unref_heightmap, subtract_reconstruction_margin=False, channel_dim=2)
        axf.unref_heightmap = unref_heightmap
        return axf

    def get_unref_heightmap(self, mat_ind: int = None, mat_path: str = None, axf: axf_class.AxF = None):
        # TODO: call this from get_axf() / get_maps()?
        """given one of material index, material path or AxF, try and load a corresponding unrefined heightmap"""

        if mat_path is not None:
            mat_ind = self.find_mat_ind(mat_path)
        elif axf is not None:
            mat_ind = self.find_mat_ind(axf.name)

        if mat_ind is not None:
            mat_path = os.path.join(self.input_dir, self.material_names[mat_ind]) + self.suffix

    def get_xml(self, mat_ind: int = None, mat_path: str = None, axf: axf_class.AxF = None) -> axf_class.MeasXML:
        """given material index, material path or AxF object, return the corresponding MeasXML object"""
        if axf is not None and axf.meas_xml is not None:
            # best case: XML is readily loaded in AxF object
            return axf.meas_xml
        elif mat_ind is None and mat_path is not None:
            # attempt to extract material name from some path (base of filename) and to look it up in the loaded
            # materials
            mat_ind = self.find_mat_ind(mat_path)
        elif mat_ind is None and axf is not None:
            # try to find material index from name stored in AxF object
            mat_ind = self.find_mat_ind(axf.name)

        if mat_ind is not None:
            # if we could determine the material index, return the loaded XML object
            return self.xmls[mat_ind]
        elif axf.path is not None:
            # last resort: replace .axf with .xml in path stored in AxF object
            xml_path = os.path.splitext(axf.path)[0] + '.xml'
            if os.path.exists(xml_path):
                return axf_class.MeasXML(xml_path)

        raise Exception('could not determine XML for input to get_xml()')

    def get_maps(self, mat_ind: int = None, axf: axf_class.AxF = None, mat_path: str = None,
                 apply_mappings: bool = False, divide_specular_by_fresnel: bool = False) -> dict:
        """extract SVBRDF maps from AxF object and optionally apply mappings, AxF can be specified directly or via its
        material index"""

        if mat_ind is None and axf is None:
            assert mat_path is not None, 'mat_path must be specified if neither mat_ind nor axf are input to get_maps()'
            axf = self.get_axf(mat_path=mat_path)

        # load AxF & store unrefined heightmap in axf object
        if axf is None:
            assert mat_ind is not None, 'either mat_ind or a loaded AxF must be provided to get_maps()'
            axf = self.get_axf(mat_ind=mat_ind)
            xml = self.get_xml(mat_ind=mat_ind)
        else:
            xml = self.get_xml(axf=axf)

        if axf.unref_heightmap is None:
            if mat_ind is None:
                mat_ind = self.find_mat_ind(axf.name if mat_path is None else mat_path)

        # dictionary with H x W x C maps
        maps = axf.getMapsDict(copy=True,
                               normals_only_xy=False,  # will be applied later
                               expand=True,
                               dtype=self.dtype,
                               sRGB2XYZ=False)  # will be applied on the fly later

        # rename some maps
        maps = self.from_axf_names(maps)

        # compute tangent frames
        # vertex & shading normals are both defined in local space (i.e. with an unrotated ROI), therefore we don't
        # need to rotate the unrefined normals by the ROI angle, even if self.use_inverse_tac7_transforms == False
        (dy, dx) = np.gradient(maps['unref_heightmap'][:, :, 0],
                               1 / xml.pixels_per_mm_height,
                               1 / xml.pixels_per_mm_width)
        maps['unref_normal'] = np.stack((-dx, -dy, np.ones(dx.shape, dtype=dx.dtype)), axis=2)
        maps['unref_normal'] /= np.linalg.norm(maps['unref_normal'], axis=2)[:, :, None]
        maps['unref_tangent'] = np.cross(maps['unref_normal'], np.array([0, -1, 0]), axisa=2, axisb=0)
        maps['unref_tangent'] /= np.linalg.norm(maps['unref_tangent'], axis=2)[:, :, None]
        maps['unref_bitangent'] = np.cross(maps['unref_normal'], maps['unref_tangent'], axis=2)
        maps['unref_bitangent'] /= np.linalg.norm(maps['unref_bitangent'], axis=2)[:, :, None]
        maps['unref_normal'] = maps['unref_normal'].astype(self.dtype)
        maps['unref_tangent'] = maps['unref_tangent'].astype(self.dtype)
        maps['unref_bitangent'] = maps['unref_bitangent'].astype(self.dtype)

        # store precomputed xyz world coordinates for each material
        maps['refined_xyz'] = xml.pixelsToWorld(dtype=self.dtype)['pixels_world']

        # create unrefined XYZ coordinate arrays as inputs
        maps['unref_xyz'] = xml.pixelsToWorld(heightmap=maps['unref_heightmap'][..., 0], dtype=self.dtype)['pixels_world']

        # shift channel dimension to the front to match PyTorch's arrangement
        # this needs to come before any of the mappings!
        for k in maps.keys():
            maps[k] = maps[k].transpose((2, 0, 1))

        # clamp critical parameters to their valid ranges (shouldn't be necessary here)
        maps = self.clamp_maps(maps)

        # multiply Fresnel F0 into specular albedo to make it linear and bounded
        if not divide_specular_by_fresnel:
            maps['specular'] *= maps['fresnel']

        if apply_mappings:
            # transform parameter maps (this has to happen after clamping!)
            maps['normals'] = maps['normals'][:2]

            maps['roughness'] = self.map_roughness(maps['roughness'])
            maps['anisotropy'] = self.map_anisotropy(maps['anisotropy'])

        # check if SVBRDF maps are fine
        if np.any(np.array([np.any(np.isnan(maps[k].ravel())) for k in maps.keys()])):
            raise Exception('NaN(s) found in material %s!' % axf.name)

        return maps

    def get_meas(self, mat_ind: int, crop: bool = False):
        """extract adequately cropped measurement images from dataset"""

        meta_poly = self.get_channel_inds(modality='poly', sorting=['cv', 'il', 'rot'])
        poly_chinds = meta_poly['chinds']

        meta_pan = self.get_channel_inds(modality='pan', sorting=['cv', 'rot', 'il'])
        pan_chinds = meta_pan['chinds'][:, 0]

        meta_lls = self.get_channel_inds(modality='lls', sorting=['cv', 'rot', 'la'])
        lls_chinds = meta_lls['chinds'][:, 0]

        # H x W x C x N arrays
        lls_meas = self.inputs['lls'][mat_ind][lls_chinds, ...]
        lls_meas = self.crop_image(mat_ind, lls_meas, channel_dim=0,
                                   subtract_reconstruction_margin=crop)[..., None].transpose((1, 2, 3, 0))
        pan_meas = self.inputs['pan'][mat_ind][pan_chinds, ...]
        pan_meas = self.crop_image(mat_ind, pan_meas, channel_dim=0,
                                   subtract_reconstruction_margin=crop)[..., None].transpose((1, 2, 3, 0))
        poly_meas = self.inputs['poly'][mat_ind][poly_chinds, ...]
        poly_meas = self.crop_image(mat_ind, poly_meas, channel_dim=1,
                                    subtract_reconstruction_margin=crop).transpose((2, 3, 1, 0))
        return Dct(lls=lls_meas, pan=pan_meas, poly=poly_meas)

    def render_meas(self, mat_ind: int, device: 'torch.device' = 'cpu'):
        """instead of using actual measured images, rerender those"""
        from src.eval_helpers import rerender_meas

        dev = None
        if device is None:
            dev = 'cpu'
        elif isinstance(device, str):
            m = re.match(r'(\d),', device)
            if m is not None:
                dev = 'cuda:' + m[1]
            if ',' not in device:
                dev = 'cuda:0'
            if dev is None:
                raise Exception('device could not be parsed, got ' + str(device))

        with torch.no_grad():
            axf = self.get_axf(mat_ind=mat_ind, crop=True)
            maps = self.get_maps(axf=axf, mat_ind=mat_ind, apply_mappings=False, divide_specular_by_fresnel=True)
            renderings = rerender_meas(self,
                                       mat_ind=mat_ind,
                                       axf=axf,
                                       maps=maps,
                                       device=dev,
                                       dtype=np.float16,
                                       sorting_lls=('cv', 'la', 'rot'),
                                       sorting_pan=('cv', 'il', 'rot'),
                                       sorting_poly=('cv', 'il', 'rot'),
                                       )
        return renderings

    def update_crop(self, mat_ind, crop, h_orig, w_orig):
        if crop is not None:
            # FIXME: stop this mess of adding ones here!
            self.crops[mat_ind] = {}
            self.crops[mat_ind]['x0'] = np.maximum(0, crop['x0'])
            self.crops[mat_ind]['x1'] = np.clip(crop['x1'], crop['x0'] + 1, w_orig + 1)  # +1 for easier indexing
            self.crops[mat_ind]['y0'] = np.maximum(0, crop['y0'])
            self.crops[mat_ind]['y1'] = np.clip(crop['y1'], crop['y0'] + 1, h_orig + 1)  # +1 for easier indexing
        else:
            self.crops[mat_ind] = None

    def get_crop_coords(self, mat_ind):
        """return lower and upper x and y coordinates for cropping a material specified by its index

        returns x0, x1, y0, y1, which should be used for slicing directly, i.e., without adding 1: image[y0:y1, x0:x1]"""
        if self.crops is not None:
            crop = self.crops[mat_ind]
            x0, x1, y0, y1 = crop['x0'], crop['x1'], crop['y0'], crop['y1']
        else:
            x0, x1, y0, y1 = 0, self.widths[mat_ind] + 1, 0, self.heights[mat_ind] + 1
        return x0, x1, y0, y1

    def crop_axf(self, mat_ind, copy=True, axf=None):
        """crop a loaded AxF object and return it (by default a deepcopy)"""
        if axf is None:
            axf = self.axfs[mat_ind]
        if copy:
            axf = deepcopy(axf)
        if self.crops[mat_ind] is not None:
            # FIXME: remove subtraction of 1 here
            axf.crop(x0=self.crops[mat_ind]['x0'],
                     x1=self.crops[mat_ind]['x1'] - 1,
                     y0=self.crops[mat_ind]['y0'],
                     y1=self.crops[mat_ind]['y1'] - 1,
                     adjust=True)
        return axf

    def crop_image(self, mat_ind, image, subtract_reconstruction_margin: bool, channel_dim=0):
        """crop measurement image / heightmap / other images associated to a material"""
        if self.crops[mat_ind] is None:
            return image
        # FIXME: we should add ones here to the upper bounds
        x0 = self.crops[mat_ind]['x0']
        x1 = self.crops[mat_ind]['x1']
        y0 = self.crops[mat_ind]['y0']
        y1 = self.crops[mat_ind]['y1']
        if subtract_reconstruction_margin and self.reconstructed_axf is not None:
            x0 += self.reconstructed_axf.custom_roi['left']
            y0 += self.reconstructed_axf.custom_roi['top']
            x1 = x0 + np.minimum(self.crops[mat_ind]['x1'] - self.crops[mat_ind]['x0'],
                                 self.reconstructed_axf.custom_roi['width'])
            y1 = y0 + np.minimum(self.crops[mat_ind]['y1'] - self.crops[mat_ind]['y0'],
                                 self.reconstructed_axf.custom_roi['height'])
        if channel_dim == 0:
            return image[:, y0:y1, x0:x1]
        elif channel_dim == 1:
            # polychrome images are N x 3 x h x w when RGB is not unrolled
            return image[:, :, y0:y1, x0:x1]
        elif channel_dim == 2:
            return image[y0:y1, x0:x1, :]
        else:
            raise Exception('channel_dim needs to be 0 or 2')

    def get_channel_inds(self, modality='poly', sorting=('cv', 'il', 'rot'), sort_by_cv_angle=False):
        assert len(sorting) == 3, 'sorting must contain permutation of ("cv", "il", "rot") or ("cv", "la", "rot")'
        if modality == 'lls':
            assert 'cv' in sorting and 'la' in sorting and 'rot' in sorting, 'sorting must be permutation of ("cv", "la", "rot")'
            keys = [np.where(np.array(['cv', 'la', 'rot']) == k)[0][0] for k in sorting]
        else:
            assert 'cv' in sorting and 'il' in sorting and 'rot' in sorting, 'sorting must be permutation of ("cv", "il", "rot")'
            keys = [np.where(np.array(['cv', 'il', 'rot']) == k)[0][0] for k in sorting]

        if modality == 'pan_poly':
            # panchromatic images with corresponding polychromatic measurement
            raise NotImplementedError('panchromatic channels for polychromatic images are no longer loaded')

        iminds = self.meta['inds_' + modality]
        chinds = np.r_[:self.meta['chs_' + modality]]
        if modality == 'poly':
            chinds = chinds.reshape((3, -1), order='F').T
        else:
            chinds = chinds[:, None]
        rots = self.meta['rots_' + modality]
        cvs = self.meta['cvs_' + modality]
        if sort_by_cv_angle:
            cam = self.cvs_inclinations[cvs - 1]
        else:
            cam = cvs
        if modality == 'lls':
            las = self.meta['las_' + modality]
            meta = np.r_[cam[None], las[None], rots[None], np.r_[:len(iminds)][None]].T
        else:
            ils = self.meta['ils_' + modality]
            meta = np.r_[cam[None], ils[None], rots[None], np.r_[:len(iminds)][None]].T

        meta = sortrows(meta, order=keys)
        perm = meta[:, 3].astype(np.int64)

        return dict(chinds=chinds[perm, :],
                    iminds=iminds[perm],
                    perm=perm,
                    cvs=cvs[perm],
                    rots=meta[:, 2],
                    ils=meta[:, 1].astype(np.int64) if modality != 'lls' else None,
                    las=meta[:, 1] if modality == 'lls' else None)

    def get_confidence_lookup(self):
        return dict(lls=self.meta['inds_lls'], pan=self.meta['inds_pan'], poly=self.meta['inds_poly'])

    def get_meas_image(self, mat_ind, cv, rot, il=None, la=None, output_channel_dim=2, modalities=('poly', 'pan'), dtype=None):
        # FIXME: adjust
        raise NotImplementedError('get_meas_image() needs to be adapted')
        if dtype is None:
            dtype = self.dtype
        cvs = self.meas_channel_meta['cvs']
        ils = self.meas_channel_meta['ils']
        las = self.meas_channel_meta['las']
        rots = self.meas_channel_meta['rots']
        if la is None:
            # point lit image
            chinds = np.where(np.all(np.r_[cvs[None] == cv, ils[None] == il, rots[None] == rot], axis=0))[0]
        else:
            chinds = np.where(np.all(np.r_[cvs[None] == cv, las[None] == la, rots[None] == rot], axis=0))[0]
        if self.separate_poly and 'poly' in modalities:
            cvs_sep_poly = self.meas_channel_meta['cvs_separate_poly']
            ils_sep_poly = self.meas_channel_meta['ils_separate_poly']
            rots_sep_poly = self.meas_channel_meta['rots_separate_poly']
            chinds_separate_poly = np.where(np.all(np.r_[cvs_sep_poly[None] == cv, ils_sep_poly[None] == il, rots_sep_poly[None] == rot], axis=0))[0]
            is_poly = self.meas_channel_meta['is_poly'][chinds_separate_poly]
        else:
            is_poly = np.ones(len(chinds), dtype=np.bool)
        is_pan = np.logical_or(self.meas_channel_meta['is_pan'][chinds], self.meas_channel_meta['is_lls'][chinds])
        mat_name = self.material_names[mat_ind]
        image_dir = os.path.join(self.input_dir, mat_name) + self.suffix + '_rect'
        if il is None:
            fname = 'cv%02d_lls01_la%05.2f_rot%03d.exr' % (cv, la, rot)
        else:
            fname = 'cv%02d_il%03d_rot%03d.exr' % (cv, il, rot)
        fname = os.path.join(image_dir, fname)
        if 'poly' in modalities and not np.any(is_poly) \
                or 'pan' in modalities and not (np.any(is_pan) or np.any(is_lls))\
                or 'confidence' in modalities and not self.load_confidences:
            # load from disk
            if not os.path.exists(fname):
                raise Exception('file %s does not exist' % fname)
            im = self.read_meas_image(fname)
            for k in im.keys():
                if im[k] is not None:
                    im[k] = self.crop_image(mat_ind, im[k], channel_dim=2)
            input_channel_dim = 2
        else:
            # everything already loaded
            im = {}
            for mod in modalities:
                if mod == 'confidence':
                    im[mod] = self.confidences[mat_ind][chinds[mod], :, :]
                elif mod == 'poly' and self.separate_poly:
                    if self.separate_poly:
                        chinds_poly = chinds_separate_poly
                    else:
                        chinds_poly = chinds[is_poly]
                    im[mod] = self.inputs['poly'][mat_ind][chinds_poly, :, :]
                else:
                    im[mod] = self.inputs[modality][mat_ind][chinds[is_pan], :, :]
            input_channel_dim = 0
        # bring in requested format
        for k in im.keys():
            if im[k] is None:
                continue
            if input_channel_dim == 0:
                if output_channel_dim == 2:
                    im[k] = im[k].transpose((1, 2, 0))
                elif output_channel_dim != input_channel_dim:
                    raise NotImplementedError('output_channel_dim == %d is not supported' % output_channel_dim)
            else:
                if output_channel_dim == 0:
                    im[k] = im[k].transpose((2, 0, 1))
                elif output_channel_dim != input_channel_dim:
                    raise NotImplementedError('output_channel_dim == %d is not supported' % output_channel_dim)
            im[k] = im[k].astype(dtype)
        return im

    def read_meas_image(self, fname):
        im, chans = read_exr(fname, channels=None, pixel_type='half', sort_rgb=True)
        if im.shape[2] == 5:
            im = dict(poly=im[:, :, :3],
                      pan=im[:, :, 3:4],
                      confidence=im[:, :, 4:5])
        elif im.shape[2] == 2:
            im = dict(poly=None,
                      pan=im[:, :, :1],
                      confidence=im[:, :, 1:])
        else:
            raise Exception('unexpected channel count')
        return im
    
    def estimate_poly_to_pan_weights(self):
        """numerically estimate sRGB to panchromatic color space conversion coefficients"""
        # TODO: hash material names that go into the estimation?
        ofname = os.path.join(os.path.split(self.split_filename)[0], 'poly2pan.npy')
        # ofname = os.path.split(self.split_filename)[0]
        # ofname = os.path.join(ofname, 'poly2pan_HASH_' + self.hash + '_CROPHASH_' + self.crop_hash + '.npy')
        if os.path.exists(ofname):
            return np.load(ofname)

        if self.num_mats < 100:
            raise Exception('trying to estimate poly to pan coefficients from %d materials, please load at least 100' % self.num_mats)

        # query a bunch of polychromatic images together with the panchromatic counter parts
        poly_pans = []
        for mat_ind in self.mat_inds:
            poly_pans.append([self.get_meas_image(mat_ind, cv=1, rot=0, il=il, modalities=['poly', 'pan'], dtype=np.float32) for il in [26, 27, 28]])
        polys = [p['poly'] for pp in poly_pans for p in pp]
        pans = [p['pan'] for pp in poly_pans for p in pp]
        all_polys = []
        all_pans = []
        per_mat_coeffs = []
        for i in range(len(poly_pans)):
            pan = pans[i]
            pan = pan.reshape((-1, 1), order='F')
            poly = polys[i]
            poly = poly.reshape((-1, 3), order='F')
            # mean_pan = pan.mean()
            # pan /=  mean_pan
            # poly /=  mean_pan

            all_polys.append(poly.copy())
            all_pans.append(pan.copy())

            # it is also somewhat robust to compute coefficients per material and later average them
            # this is her just as a reference
            per_mat_coeffs.append(np.linalg.lstsq(np.concatenate((poly, np.ones((poly.shape[0], 1), dtype=poly.dtype)), axis=1), pan))
        # solving globally with least squares is extremely sensitive to outliers
        per_mat_res = np.array([c[1] for c in per_mat_coeffs])
        per_mat_svs = np.array([c[-1] for c in per_mat_coeffs])
        per_mat_coeffs = np.array([c[0][:, 0] for c in per_mat_coeffs])
        all_polys = np.concatenate(all_polys, axis=0)
        all_pans = np.concatenate(all_pans, axis=0)
        coeffs = np.linalg.lstsq(np.concatenate((all_polys, np.ones((all_polys.shape[0], 1), dtype=poly.dtype)), axis=1), all_pans)
        res = coeffs[1]
        coeffs = coeffs[0][:, 0]

        # use robust least squares instead
        from scipy.optimize import least_squares

        def fun(x, t, y):
            return (x @ t - y)[0]
        x0 = np.array([1 / 3, 1 / 3, 1 / 3, 0])
        lhs = np.concatenate((all_polys, np.ones((all_polys.shape[0], 1))), axis=1).T
        rhs = all_pans.T
        output = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(lhs, rhs), verbose=2)
        coeffs_robust = output.x[None].astype(np.float32)

        # save outputs to disk to avoid lenghty calculation
        np.save(ofname, coeffs_robust)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(per_mat_coeffs)
        # plt.plot(per_mat_coeffs.mean(axis=0, keepdims=True).repeat(per_mat_coeffs.shape[0], axis=0))
        # plt.plot(coeffs_robust[None].repeat(per_mat_coeffs.shape[0], axis=0))
        # plt.plot(coeffs[None].repeat(per_mat_coeffs.shape[0], axis=0))

        # return coeffs_robust, coeffs, res, per_mat_coeffs, per_mat_res
        return coeffs_robust

    def resample_pixel_ids(self):
        """sample new pixels from all loaded materials, this can optionally be done after each epoch"""
        # TODO: account for network receptive field when selecting patches to avoid redundant training samples due to
        #  overlap between patches around oversampled pixels
        # TODO: add flag --uniform_sampling, which causes self.num_samples_per_mat to have the same number for each
        #  material, avoiding bias when training on differently sized materials
        self.sampled_pixels = []
        if self.shuffle:
            for mat_ind in range(self.num_mats):
                self.sampled_pixels.append(
                    np.random.choice(self.pixel_counts[mat_ind], self.num_samples_per_mat[mat_ind], replace=self.with_replacement))
        else:
            for mat_ind in range(self.num_mats):
                self.sampled_pixels.append(np.r_[0:self.pixel_counts[mat_ind]])
        self.sampled_pixels = np.array([item for sublist in self.sampled_pixels for item in sublist])

    def compute_mean_std(self):
        """compute mean and approximate standard deviation over the loaded AxF parameter maps"""
        # TODO: if applying the means / stds (dynamically in __getitem__() causes too much overhead, precompute once?
        #   this will mess up all stored maps for direct usage in visualizations / AxF objects, though.)
        # TODO: pre-compute this statically on the ENTIRE training set, but over all channels, store in dedicated files
        #  and read those back in during image loading with the same channel sampling -- too error prone, the best idea
        #  is probably just computing the means / stds on the training split and propagating to validation / test splits

        if self.normalize_inputs:
            # try to load dataset input mean / std from cached file
            inp_norm_filename = os.path.splitext(self.split_filename)[0] + '_INPNORMHASH_' + self.hash_inputs + '.npz'
            default_inp_norm_filename = os.path.join(os.path.split(self.split_filename)[0], 'splitTrain_INPNORMHASH_' + self.hash_inputs + '.npz')
            if not os.path.exists(inp_norm_filename) and os.path.exists(default_inp_norm_filename):
                if self.normalization_from_default_file:
                    inp_norm_filename = default_inp_norm_filename
                else:
                    raise Exception(f'input normalization file {inp_norm_filename} cannot be found, but '
                                    f'{default_inp_norm_filename} exists; set --normalization_from_default_file 1 to '
                                    f'enable fallback to this default file')

            if os.path.exists(inp_norm_filename):
                # load entire dataset mean & std
                tmp = np.load(inp_norm_filename, allow_pickle=True)
                self.mean_inputs = tmp['mean_inputs'].item()
                self.means_inputs = tmp['means_inputs'].item()
                self.std_inputs = tmp['std_inputs'].item()
                self.stds_inputs = tmp['stds_inputs'].item()
            else:
                # compute per material mean & std, average into dataset mean & std (this is no longer a proper std but close enough for normalization purposes)
                if self.crop_coords is not None:
                    raise Exception(f'input normalization cache {inp_norm_filename} could not be found, cropping must be disabled when computing dataset mean / std')

                if self.last_mat_ind != 0 or self.first_mat_ind != 0 or self.pattern != '.*':
                    raise Exception(f'input normalization cache {inp_norm_filename} could not be found, the entire '
                                    f'split must be loaded for mean / std computation, set --last_mat_ind to 0 and --train_mat_pattern to ".*"!')

                progress = tqdm(total=len(self.material_names), desc='computing input channel means & stds', ascii=True, disable=self.verbosity < 1)
                for mat_ind, mat_name in enumerate(self.material_names):
                    progress.update(n=1)
                    mat_path = os.path.join(self.input_dir, mat_name) + self.suffix

                    for modality in self.input_names_meas:
                        # load / compute per material mean and std
                        check = os.path.exists(mat_path + '_HASH_' + self.hash + '_meas_mean_' + modality + '.bin')
                        check = check and os.path.exists(mat_path + '_HASH_' + self.hash + '_meas_std_' + modality + '.bin')
                        if check:
                            self.means_inputs[modality][:, mat_ind] = io_read_bin(mat_path + '_HASH_' + self.hash + '_meas_mean_' + modality + '.bin')[0]
                            self.stds_inputs[modality][:, mat_ind] = io_read_bin(mat_path + '_HASH_' + self.hash + '_meas_std_' + modality + '.bin')[0]
                        else:
                            # compute on float32 here to avoid numerical issues
                            self.means_inputs[modality][:, mat_ind] = np.mean(self.inputs[modality][mat_ind].astype(np.float32), axis=(1, 2))
                            # this way of computing the STD is incorrect, as it depends on the global mean, for now the best we can do, though...
                            self.stds_inputs[modality][:, mat_ind] = np.std(self.inputs[modality][mat_ind].astype(np.float32), axis=(1, 2))

                            io_write_bin(self.means_inputs[modality][:, mat_ind], mat_path + '_HASH_' + self.hash + '_meas_mean_' + modality + '.bin')
                            io_write_bin(self.stds_inputs[modality][:, mat_ind], mat_path + '_HASH_' + self.hash + '_meas_std_' + modality + '.bin')

                    # compute mean & std for input modalities different than the measurement images
                    for k in [k for k in self.input_names if k not in self.input_names_meas]:
                        if k == 'unref_normal' and self.normals_only_xy:
                            self.means_inputs[k][:, mat_ind] = np.mean(self.maps[mat_ind][k][:2, :, :].astype(np.float32), axis=(1, 2))
                            self.stds_inputs[k][:, mat_ind] = np.std(self.maps[mat_ind][k][:2, :, :].astype(np.float32), axis=(1, 2))
                        else:
                            self.means_inputs[k][:, mat_ind] = np.mean(self.inputs[k][mat_ind].astype(np.float32), axis=(1, 2))
                            self.stds_inputs[k][:, mat_ind] = np.std(self.inputs[k][mat_ind].astype(np.float32), axis=(1, 2))

                # compute mean and approximated std over entire dataset, both for inputs and outputs, and convert back to desired data type
                for k in self.input_names:
                    self.mean_inputs[k] = np.mean(self.means_inputs[k], axis=1).astype(self.dtype)
                    self.std_inputs[k] = np.mean(self.stds_inputs[k], axis=1).astype(self.dtype)

                # save entire dataset input mean & std
                np.savez(inp_norm_filename,
                         mean_inputs=self.mean_inputs,
                         means_inputs=self.means_inputs,
                         std_inputs=self.std_inputs,
                         stds_inputs=self.stds_inputs)

        if self.normalize_outputs:
            # compute hash of all parameters that affect the labels
            props_output = dict(label_names=self.label_names,
                                channels_hash=self.hash,
                                normals_only_xy=self.normals_only_xy,
                                roughness_mapping=self.roughness_mapping,
                                aniso_mapping=self.aniso_mapping)
            self.hash_norm_out = hashlib.md5(str(props_output).encode()).hexdigest()[:8]

            if not os.path.exists(os.path.join(self.input_dir, 'OUTNORMHASH_' + self.hash_norm_out + '.txt')):
                with open(os.path.join(self.input_dir, 'OUTNORMHASH_' + self.hash_norm_out + '.txt'), 'w') as fp:
                    json.dump(dict(props=props_output, kwargs=self.kwargs), fp, indent=True)

            # try to load dataset labels mean / std from cached file
            out_norm_filename = os.path.splitext(self.split_filename)[0] + '_OUTNORMHASH_' + self.hash_norm_out + '.npz'
            default_out_norm_filename = os.path.join(os.path.split(self.split_filename)[0], 'splitTrain_OUTNORMHASH_' + self.hash_norm_out + '.npz')
            if not os.path.exists(out_norm_filename) and os.path.exists(default_out_norm_filename):
                if self.normalization_from_default_file:
                    out_norm_filename = default_out_norm_filename
                else:
                    raise Exception(f'output normalization file {out_norm_filename} cannot be found, but '
                                    f'{default_out_norm_filename} exists; set --normalization_from_default_file 1 to '
                                    f'enable fallback to this default file')
            if os.path.exists(out_norm_filename):
                tmp = np.load(out_norm_filename, allow_pickle=True)
                self.mean_outputs = torch.tensor(tmp['mean_outputs'].item())
                self.means_outputs = torch.tensor(tmp['means_outputs'].item())
                self.std_outputs = torch.tensor(tmp['std_outputs'].item())
                self.stds_outputs = torch.tensor(tmp['stds_outputs'].item())
            else:
                if self.crop_coords is not None:
                    raise Exception('cropping must be disabled when computing dataset mean / std')
                if self.last_mat_ind != 0 or self.first_mat_ind != 0 or self.pattern != '.*':
                    raise Exception('the entire split must be loaded for mean / std computation, set '
                                    '--last_mat_ind to 0 and --train_mat_pattern to ".*"!')

                progress = tqdm(total=len(self.material_names), desc='computing output channel means & stds', ascii=True, disable=self.verbosity < 1)
                for mat_ind, mat_name in enumerate(self.material_names):
                    progress.update(n=1)
                    # compute mean and std per map
                    for k in self.label_names:
                        # compute on float32 here to avoid numerical issues
                        self.means_outputs[k][:, mat_ind] = torch.tensor(np.mean(self.maps[mat_ind][k].astype(np.float32), axis=(1, 2)))
                        self.stds_outputs[k][:, mat_ind] = torch.tensor(np.std(self.maps[mat_ind][k].astype(np.float32), axis=(1, 2)))

                    for k in self.label_names:
                        self.mean_outputs[k] = torch.tensor(np.mean(self.means_outputs[k], axis=1).astype(self.dtype))
                        self.std_outputs[k] = torch.tensor(np.mean(self.stds_outputs[k], axis=1).astype(self.dtype))

                # save entire dataset input mean & std
                np.savez(out_norm_filename,
                         mean_outputs=self.mean_outputs,
                         means_outputs=self.means_outputs,
                         std_outputs=self.std_outputs,
                         stds_outputs=self.stds_outputs)

    def get_all_light_view_calib(self, center_only: bool = False) -> Dct:
        all_dirs = Dct()
        for mat_ind in self.mat_inds:
            calib = self.get_light_view_calib(mat_ind=mat_ind, center_only=center_only)
            for input_name in self.input_names_meas:
                inds = self.meta['inds_' + input_name]
                if input_name in ['pan', 'poly']:
                    inds_ = self.meta['inds_' + input_name + '_in_pointlits']
                    light_positions = calib.point_light_positions[:, :, :, inds_]
                    light_dirs = calib.point_light_dirs[:, :, :, inds_]
                elif input_name == 'lls':
                    light_positions = calib.lls_light_positions
                    light_dirs = calib.lls_light_dirs
                else:
                    raise Exception('unexpected input name: ' + str(input_name))

                if 'light_positions_' + input_name not in all_dirs:
                    all_dirs['light_positions_' + input_name] = []
                if 'light_dirs_' + input_name not in all_dirs:
                    all_dirs['light_dirs_' + input_name] = []
                if 'view_positions_' + input_name not in all_dirs:
                    all_dirs['view_positions_' + input_name] = []
                if 'view_dirs_' + input_name not in all_dirs:
                    all_dirs['view_dirs_' + input_name] = []

                all_dirs['light_positions_' + input_name].append(light_positions)
                all_dirs['light_dirs_' + input_name].append(light_dirs)
                all_dirs['view_positions_' + input_name].append(calib.view_positions[:, :, :, inds])
                all_dirs['view_dirs_' + input_name].append(calib.view_dirs[:, :, :, inds])
        for input_name in self.input_names_meas:
            all_dirs['light_positions_' + input_name] = np.stack(all_dirs['light_positions_' + input_name], axis=0)
            all_dirs['light_dirs_' + input_name] = np.stack(all_dirs['light_dirs_' + input_name], axis=0)
            all_dirs['view_positions_' + input_name] = np.stack(all_dirs['view_positions_' + input_name], axis=0)
            all_dirs['view_dirs_' + input_name] = np.stack(all_dirs['view_dirs_' + input_name], axis=0)
        return all_dirs

    def get_light_view_calib(self, mat_ind: int, center_only: bool = False) -> Dct:
        """return light and view directions and positions; directions are optionally static for each material by
        computing them only from the material center; LLS corners are merged at their center into a single position"""
        cam_positions = self.cam_positions[mat_ind]  # 3 x n
        point_light_positions = self.point_light_positions[mat_ind][:, :, 0]  # 3 x n_pl
        lls_positions = self.lls_positions[mat_ind]  # 3 x n_lls x 4
        lls_positions = np.mean(lls_positions, axis=-1)  # 3 x n_lls
        xyz = self.maps[mat_ind]['unref_xyz']  # 3 x h x w
        if center_only:
            xyz = np.mean(xyz, axis=(1, 2), keepdims=True)
        view_dirs = normalize(cam_positions[:, None, None, :] - xyz[:, :, :, None], axis=0)
        lls_light_dirs = normalize(lls_positions[:, None, None, :] - xyz[:, :, :, None], axis=0)
        point_light_dirs = normalize(point_light_positions[:, None, None, :] - xyz[:, :, :, None], axis=0)
        return Dct(lls_light_dirs=lls_light_dirs,
                   point_light_dirs=point_light_dirs,
                   view_dirs=view_dirs,
                   lls_light_positions=lls_positions[:, None, None, :],
                   point_light_positions=point_light_positions[:, None, None, :],
                   view_positions=cam_positions[:, None, None, :],
                   )

    def __len__(self):
        # returns the total number of samples in the dataset
        return len(self.sampled_pixels)

    def map_labels(self, labels: dict,
                   invert: bool = False,
                   invert_aniso_mapping: bool = True,
                   reconstruct_normals: bool = False,
                   copy: bool = False,
                   as_tensors: bool = False,
                   logits_to_labels: bool = None) -> dict:
        """Apply or undo label transformations. Assumes label transformations are registered in the dictionary
        self.label_mappings. Only translates class indices back to label values in case self.mode == 'classification'."""

        if logits_to_labels is None:
            logits_to_labels = self.classification

        if logits_to_labels:
            # convert between continuous values and class indices
            if not invert:
                output = self.labels_to_classes(labels, copy=copy)
            else:
                output = self.classes_to_labels(labels, copy=copy, as_tensors=as_tensors)
        else:
            # apply / invert label mappings
            output = labels
            if copy:
                output = deepcopy(labels)

            # ensure all bounded parameters are in the valid ranges
            if invert:
                output = self.clamp_maps(output)

            for key, mapping in self.label_mappings.items():
                if key in output:
                    if invert and key == 'anisotropy' and not invert_aniso_mapping:
                        # in certain scenarios, we want to keep the anisotropy mapping (e.g. when passing predictions
                        # from gray to color branch
                        continue
                    if invert and key == 'refined_normal' and not reconstruct_normals:
                        # 3D normals will be reconstructed from 2D only on final AxF assignment, as we here need to
                        # keep them 2D for per-map loss calculations
                        continue
                    output[key] = mapping(output[key], invert=invert)
        return output

    def map_albedo(self, albedo, invert: bool = False):
        """Transform albedo maps. The only currently implemented mapping is from sRGB to XYZ."""
        if self.classifcation:
            return albedo
        if self.albedos_xyz:
            if albedo.shape[0] == 1:
                # nothing to convert when we're handling a panchromatic albedo (e.g. from the network's gray branch)
                return albedo
            if invert:
                if isinstance(albedo, torch.Tensor):
                    albedo = torch.einsum('ij,jxy->ixy', torch.tensor(self.xyz_to_rgb_matrix_left_mul).to(albedo.device), albedo)
                else:
                    albedo = np.matmul(self.xyz_to_rgb_matrix_left_mul, albedo.reshape((3, -1))).reshape(albedo.shape)
            else:
                if isinstance(albedo, torch.Tensor):
                    albedo = torch.einsum('ij,jxy->ixy', torch.tensor(self.rgb_to_xyz_matrix_left_mul).to(albedo.device), albedo)
                else:
                    albedo = (np.matmul(self.rgb_to_xyz_matrix_left_mul, albedo.reshape((3, -1)))).reshape(albedo.shape)
        return albedo

    def map_roughness(self, roughness, invert: bool = False):
        """Transform roughness to a more perceptually uniform domain. Implemented mappings are log and invsqrt."""
        if self.classification:
            return roughness
        lo = self.min_roughness
        hi = self.max_roughness
        if self.roughness_mapping == 'log':
            if invert:
                if isinstance(roughness, torch.Tensor):
                    return torch.exp(np.log(hi) - torch.clip(roughness, 0, 1) * (-np.log(lo) + np.log(hi)))
                else:
                    return np.exp(np.log(hi) - np.clip(roughness, 0, 1) * (-np.log(lo) + np.log(hi)))
            else:
                if isinstance(roughness, torch.Tensor):
                    return (-torch.log(torch.clip(roughness, lo, hi)) + np.log(hi)) / (-np.log(lo) + np.log(hi))
                else:
                    return (-np.log(np.clip(roughness, lo, hi)) + np.log(hi)) / (-np.log(lo) + np.log(hi))
        elif self.roughness_mapping == 'invsqrt':
            if invert:
                if isinstance(roughness, torch.Tensor):
                    return torch.clip(lo / torch.clip(roughness, 0.07, 1) ** 2, lo, hi)
                else:
                    return np.clip(lo / np.clip(roughness, 0.07, 1) ** 2, lo, hi)
            else:
                if isinstance(roughness, torch.Tensor):
                    return lo ** 0.5 / torch.clip(roughness, lo, hi) ** 0.5
                else:
                    return lo ** 0.5 / np.clip(roughness, lo, hi) ** 0.5
        elif self.roughness_mapping == 'none':
            return roughness
        else:
            raise Exception('roughness mapping ' + self.roughness_mapping + ' not implemented')
    
    def map_anisotropy(self, anisotropy, invert: bool = False):
        """Transform anisotropy map. The only available mapping cossin takes angles a and transforms them to cos(2 * a) and
        sin(2 * a), thus creating two out of one map."""
        if self.classification:
            return anisotropy
        if self.aniso_mapping == 'cossin':
            if invert:
                dim = int(anisotropy.ndim == 4)
                if isinstance(anisotropy, torch.Tensor):
                    return torch.atan2(anisotropy.narrow(dim, 1, 1), anisotropy.narrow(dim, 0, 1)) / 2.
                else:
                    if dim == 0:
                        return np.arctan2(anisotropy[1:2, :, :], anisotropy[0:1, :, :]) / 2.
                    else:
                        return np.arctan2(anisotropy[:, 1:2, :, :], anisotropy[:, 0:1, :, :]) / 2.
            else:
                # channels still in 3rd dimension
                if isinstance(anisotropy, torch.Tensor):
                    return torch.cat([torch.cos(2 * anisotropy), torch.sin(2 * anisotropy)], dim=0)
                else:
                    return np.concatenate([np.cos(2 * anisotropy), np.sin(2 * anisotropy)], axis=0)
        elif self.aniso_mapping.lower() == 'none':
            return anisotropy
        else:
            raise Exception('anisotropy mapping ' + self.aniso_mapping + ' not implemented')

    def map_normal(self, normal, invert: bool = False):
        """Transform normals. The only implemented mapping is to discard the z-coordinate (and reconstruct it from
        incoming 2D vectors under the assumption that they're unit vectors."""
        # TODO: unused (could be called from __getitem__())
        # TODO: should also already work for classification, since we're slicing there in the index maps instead
        # TODO: add check for classification and potentially other mapping schemes
        if self.classification:
            return normal
        if self.normals_only_xy:
            if invert:
                dim = int(normal.ndim == 4)
                if isinstance(normal, torch.Tensor):
                    normal_orig = normal.clone()
                    xy_lengths = torch.sqrt((normal_orig ** 2).sum(dim=dim, keepdim=True)).expand(
                        (-1, 2, -1, -1))
                    # scale too long vectors down to lie within unit circle
                    too_long = xy_lengths > 0.99
                    normal_scaled = 0.99 * normal_orig[too_long] / xy_lengths[too_long]
                    normal[too_long] = normal_scaled

                    nz = torch.sqrt((1. - (normal.narrow(dim, 0, 2) ** 2).sum(dim=dim, keepdim=True)).clip(
                        min=torch.tensor(0.)))
                    return torch.cat((normal.narrow(dim, 0, 2), nz), dim=dim)
                else:
                    xy_lengths = np.sqrt(np.sum(normal ** 2, axis=dim, keepdims=True)).repeat(2, axis=dim)
                    too_long = xy_lengths > 0.99
                    normal[too_long] = 0.99 * normal[too_long] / xy_lengths[too_long]
                    nz = np.sqrt(np.maximum(0., 1. - np.sum(normal ** 2, axis=dim, keepdims=True)))
                    return np.concatenate((normal, nz), axis=dim)
            else:
                dim = int(normal.ndim == 4)
                if isinstance(normal, torch.Tensor):
                    return normal.narrow(dim, 0, 2)
                else:
                    if dim == 0:
                        return normal[:2]
                    else:
                        return normal[:, :2]
        return normal

    def clamp_maps(self, maps):
        """clip out-of-bounds values in dictionary of SVBRDF maps (stored as numpy arrays or torch tensors)"""
        if self.classification:
            return maps

        if 'roughness' in maps.keys():
            maps['roughness'] = clamp(maps['roughness'], self.min_roughness, self.max_roughness)
        if 'fresnel' in maps.keys():
            maps['fresnel'] = clamp(maps['fresnel'], self.min_fresnel, self.max_fresnel)
        if 'refined_normal' in maps.keys():
            if self.normals_only_xy:
                dim = int(maps['refined_normal'].ndim == 4)
                # TODO: ensure norm(xy) < 1
                if isinstance(maps['refined_normal'], torch.Tensor):
                    maps['refined_normal'] = maps['refined_normal'].clip(-1., 1.)
                else:
                    maps['refined_normal'] = np.maximum(-1., np.minimum(1., maps['refined_normal']))
        return maps

    def from_axf_names(self, map_dict):
        """convert axf.AxF map names to ours"""
        map_dict['refined_heightmap'] = map_dict.pop('heightmap')
        map_dict['refined_normal'] = map_dict.pop('normals')
        map_dict['roughness'] = map_dict.pop('lobes')
        map_dict['anisotropy'] = map_dict.pop('rotations')
        return map_dict

    def to_axf_names(self, map_dict):
        """convert our map names to the ones used in axf.AxF"""
        if 'refined_heightmap' in map_dict.keys():
            map_dict['heightmap'] = map_dict.pop('refined_heightmap')
        if 'refined_normal' in map_dict.keys():
            map_dict['normals'] = map_dict.pop('refined_normal')
        if 'roughness' in map_dict.keys():
            map_dict['lobes'] = map_dict.pop('roughness')
        if 'anisotropy' in map_dict.keys():
            map_dict['rotations'] = map_dict.pop('anisotropy')
        return map_dict

    def tensor_to_maps(self, tensor,
                       reconstruct_normals: bool = False,
                       color_albedos: bool = True,
                       invert_aniso_mapping: bool = True,
                       divide_specular_by_fresnel: bool = True,
                       to_numpy: bool = False,
                       fallback_heightmap = None,
                       logits_to_labels: bool = None,
                       ):
        maps = dict()
        ci = 0
        for k in self.label_names:
            if k in ['diffuse', 'specular'] and not color_albedos:
                nc = 1
            elif k == 'anisotropy' and invert_aniso_mapping:
                nc = self.num_chans_maps_mapped[k]
            else:
                nc = self.num_chans_maps[k]
            if tensor.ndim == 4:
                maps[k] = tensor[:, np.r_[ci:ci + nc], :, :]
            elif tensor.ndim == 3:
                maps[k] = tensor[np.r_[ci:ci + nc], :, :]
            else:
                raise Exception('tensor dimension mismatch, expected 3 or 4, got %d' % tensor.ndim)

            # undo output normalization
            maps[k] = self.unnormalize_output(maps[k], k, color_albedos)

            ci += nc

        maps = self.map_labels(maps,
                               invert=True,
                               invert_aniso_mapping=invert_aniso_mapping,
                               reconstruct_normals=reconstruct_normals,
                               copy=False,
                               logits_to_labels=logits_to_labels)

        # check if heightmap was predicted, otherwise replace it with fallback unrefined heightmap
        if 'refined_heightmap' not in maps.keys() and fallback_heightmap is not None:
            maps['refined_heightmap'] = fallback_heightmap

        # divide Fresnel F0 out of specular albedo to make it compatible with Ward-Fresnel model
        if divide_specular_by_fresnel and 'specular' in maps.keys() and 'fresnel' in maps.keys():
            maps['specular'] = safe_divide(maps['specular'], maps['fresnel'])

        if to_numpy:
            maps = {k: v.detach().cpu().numpy()[0].transpose((1, 2, 0)) for k, v in maps.items()}
            maps = self.to_axf_names(maps)
        return maps

    def unnormalize_input(self, input, key, channel_inds=None):
        # this is only used for visualization purposes
        if not self.normalize_inputs:
            return input
        mean = self.mean_inputs[key][:, None, None]
        std = self.std_inputs[key][:, None, None]
        if input.ndim == 4:
            mean = mean[None]
            std = std[None]
        if self.normalize_inputs != 'per_channel':
            mean = np.mean(mean, keepdims=True)
            std = np.mean(std, keepdims=True)
        elif channel_inds is not None:
            if mean.ndim == 3:
                assert len(channel_inds) == input.shape[0], \
                    'input.shape[0] = %d but should be %d = len(channel_inds)' % (input.shape[0], len(channel_inds))
                mean = mean[channel_inds]
                std = std[channel_inds]
            else:
                assert len(channel_inds) == input.shape[1], \
                    'input.shape[1] = %d but should be %d = len(channel_inds)' % (input.shape[1], len(channel_inds))
                mean = mean[:, channel_inds]
                std = std[:, channel_inds]
        return std * input + mean

    def normalize_output(self, input, key):
        if self.classification or not self.normalize_outputs:
            return input
        mean = self.mean_outputs[key][:, None, None]
        std = self.std_outputs[key][:, None, None]
        if isinstance(input, np.ndarray):
            mean = mean.numpy()
            std = std.numpy()
        if self.normalize_outputs != 'per_channel':
            mean = mean.mean(keepdims=True)
            std = std.mean(keepdims=True)
        # return safe_divide(input - mean, std)
        # this division should be safe since we clamp std at self.division_threshold
        return (input - mean) / std

    def unnormalize_output(self, input, key, color_albedos=True):
        if self.classification or not self.normalize_outputs:
            return input
        mean = self.mean_outputs[key][:, None, None]
        std = self.std_outputs[key][:, None, None]
        if input.ndim == 4:
            mean = mean[None]
            std = std[None]
        if isinstance(input, np.ndarray):
            mean = mean.numpy()
            std = std.numpy()
        if self.normalize_outputs != 'per_channel' and not color_albedos:
            mean = mean.mean(keepdims=True)
            std = std.mean(keepdims=True)
        if key in ['diffuse', 'specular'] and not color_albedos:
            # TODO: make use of RGB to pan weights instead of averaging
            mean = mean.mean(keepdims=True)
            std = std.mean(keepdims=True)
        return std * input + mean

    def quantize_labels(self, visualize: bool = False):
        """compute discretization bin boundaries for each type of label over entire loaded dataset"""
        # loop over all label types
        n_tot = 0  # just a helper for visualization
        self.label_intervals = Dct()
        for i, k in enumerate(self.label_names):
            self.label_intervals[k] = Dct()

            # gather samples for current label type from all loaded materials
            all_values = []
            for mat_ind in tqdm(range(self.num_mats), 'computing statistics for ' + k):
                current_map = self.maps[mat_ind][k]
                all_values.append(current_map.reshape((current_map.shape[0], -1)).tolist())
            all_values = np.concatenate(all_values, axis=1)

            # discretize current label's range into bins with equal number of samples
            self.label_intervals[k] = []
            for c in range(all_values.shape[0]):
                # assemble all values and sort them
                tmp = np.sort(all_values[c, :])
                n = len(tmp)

                # first bound is always negative infinity so we ensure to assign out-of-bounds values with a valid class label
                self.label_intervals[k].append([-np.inf])

                # given the vector of n unique (sorted) values, compute exact interval bounds via interpolation at the
                # float positions i * k / n where the  enumerated samples would be split into equal sized chunks
                self.label_intervals[k][c].extend(np.interp(np.linspace(0, n - 2, self.label_num_classes[k]), np.r_[:n], tmp))

                # last bound is always infinity so we ensure to assign out-of-bounds values with a valid class label
                self.label_intervals[k][c].append(np.inf)
                n_tot += 1

        if visualize:
            import matplotlib.pyplot as plt
            plt.switch_backend('qt5agg')
            plt.ion()
            nr = int(np.floor(n_tot ** 0.5))
            nc = int(np.ceil(n_tot / nr))
            fig, axs = plt.subplots(nr, nc)
            axs = axs.ravel()
            i = 0
            for k in self.label_names:
                for c in range(len(self.label_intervals[k])):
                    axs[i].plot(self.label_intervals[k][c], '-x')
                    axs[i].set_title(k + ('_%d' % c))
                    i += 1
            print()

    def get_num_classes(self, label_name: str) -> int:
        """get number of classes for a requested label type"""
        if self.classification:
            if label_name in self.label_num_classes:
                return self.label_num_classes[label_name]
            else:
                raise Exception('no number of classes stored for label ' + label_name)
        else:
            raise Exception('dataset is not in classification mode')

    def get_class_bounds(self, label_name: str, channel: int = 0) -> np.ndarray:
        """return array with class label discretization boundaries for requested label type and channel index"""
        if self.label_intervals is not None:
            if label_name in self.label_intervals and channel < len(self.label_intervals[label_name]):
                return self.label_intervals[label_name][channel]
            else:
                raise Exception('label ' + label_name + ' has no discretization bounds or channel index %d is out of bounds' % channel)
        else:
            raise Exception('dataset is not in classification mode')

    def labels_to_classes(self, labels: dict, copy: bool = False) -> dict:
        """Given a dictionary of continuous-valued labels, look up the class index for each value."""
        def long_like(inp):
            if isinstance(inp, np.ndarray):
                return np.zeros_like(inp, dtype=np.long)
            else:
                return torch.zeros_like(inp, dtype=torch.long)

        output = labels
        if copy:
            output = deepcopy(labels)
        for key in self.label_intervals.keys():
            tmp = long_like(labels[key])
            for channel in range(labels[key].shape[0]):
                lookup = self.get_class_bounds(label_name=key, channel=channel)
                inds = np.r_[:len(lookup)]
                tmp[channel, ...] = np.interp(labels[key][channel, ...], lookup, inds).astype(np.long)
            output[key] = tmp
        return output

    def classes_to_labels(self, predictions: dict, copy: bool = False, as_tensors: bool = False) -> dict:
        """Given a dictionary of arrays with class indices, transform them back to continuous values."""
        def array_like(inp):
            nc = len(inp)
            b, _, h, w = inp[0].shape
            return np.zeros((b, nc, h, w), dtype=np.float32)

        output = predictions
        if copy:
            output = deepcopy(predictions)

        if not isinstance(list(predictions.values())[0], list):
            # in evaluation mode, network predictions have already gone through this mapping...
            return output

        for key in self.label_intervals.keys():
            tmp = array_like(predictions[key])
            for channel in range(len(predictions[key])):
                lookup = self.get_class_bounds(label_name=key, channel=channel)
                inds = np.r_[:len(lookup)]
                classes = predictions[key][channel]
                if isinstance(classes, torch.Tensor):
                    classes = torch.argmax(classes, dim=1)
                    classes = classes.detach().cpu().numpy()
                tmp[:, channel, ...] = np.interp(classes + 0.5, inds, lookup)
            output[key] = tmp
            if as_tensors:
                output[key] = torch.tensor(output[key], dtype=predictions[key][0].dtype, device=predictions[key][0].device)
        return output

    def get_eval_batch(self,
                       mat_ind: int,
                       axf: axf_class.AxF,
                       maps: dict,
                       device: torch.device,
                       dtype: torch.dtype = torch.float32,
                       excluded_inputs: list = ()):
        """return batch ready for input to a model for a specific material, optionally for a patch specified either
        explicitly, or via crop coordinates: crop = dict(x0=0, y0=0, x1=100, y1=100), which will index into the material
        as images[:, y0:y1, x0:x1]; excluded_inputs can be used to disable returning certain inputs, e.g., the
        measurement images, if they're not needed, e.g., for evaluation of the renderlayer"""
        if axf.custom_roi is not None:
            x0 = axf.custom_roi.left
            y0 = axf.custom_roi.top
            w = axf.w()
            h = axf.h()
        elif 'uncropped_left' in axf.roi and 'uncropped_top' in axf.roi:
            x0 = axf.roi['uncropped_left'] - axf.roi['left']
            y0 = axf.roi['uncropped_top'] - axf.roi['top']
            w = axf.roi['width']
            h = axf.roi['height']
        else:
            x0 = 0
            y0 = 0
            w = axf.roi['width']
            h = axf.roi['height']
        xs = np.r_[x0:x0+w]
        ys = np.r_[y0:y0+h]
        # batch = self.__getitem__(index=None, mat_ind=mat_ind, xs=xs, ys=ys)
        batch = dict(mat_ind=np.array(mat_ind), xs=xs, ys=ys)

        #
        if not self.use_inverse_tac7_transforms and 'xys' not in excluded_inputs:
            batch['xys'] = np.stack(np.meshgrid(xs, ys), axis=0)

        # return geometric cues
        if 'unref_heightmap' not in excluded_inputs:
            batch['unref_heightmap'] = maps['unref_heightmap'][:1, :, :]

        assert hasattr(axf, 'custom_roi'), 'custom_roi field not found, axf objet should be loaded by Dataset.get_axf()'
        r = axf.custom_roi

        for k in self.input_names:
            if k in excluded_inputs:
                continue
            if k == 'unref_xyz':
                batch[k] = maps[k][:3, :, :].astype(self.dtype_out)
            elif k == 'unref_normal':
                if self.normals_only_xy:
                    batch[k] = maps[k][:2, :, :].astype(self.dtype_out)
                else:
                    batch[k] = maps[k][:3, :, :].astype(self.dtype_out)
            elif k in ['poly', 'pan', 'lls']:
                batch[k] = self.inputs[k][mat_ind][:, r.top:r.top + r.height, r.left:r.left + r.width]
            else:
                raise Exception('please implement returning input modality ' + k)

        # confidences treated separately
        if self.load_confidences and 'confidences' not in excluded_inputs:
            batch['confidences'] = self.confidences[mat_ind][:, r.top:r.top + r.height, r.left:r.left + r.width]

        return make_torch_batch(batch, device=device, dtype=dtype)

    def __getitem__(self, index, **kwargs):
        # given index from {0, ..., len(dataset)-1}, return image patch and SVBRDF pixel corresponding to this sample
        # index, if index is omitted, the x- and y-coordinates can be specified instead explicitly as xs=[...], ys=[...],
        # the material index as mat_ind=...
        patch_size = kwargs.pop('patch_size', None)
        if patch_size is None:
            patch_size = self.patch_size

        if index is None:
            # evaluation mode where we extract patches manually
            mat_ind = kwargs['mat_ind']
            xs = kwargs['xs']
            ys = kwargs['ys']
            full_width = self.widths[mat_ind]
            full_height = self.heights[mat_ind]
        else:
            # returns a single patch and corresponding label and calibration data
            mat_ind = self.sampled_mat_inds[index]
            pixel_index = self.sampled_pixels[index]

            # Load arbitrary patch and get label
            full_width = self.widths[mat_ind]
            full_height = self.heights[mat_ind]

            if not self.padding:
                width = full_width - patch_size + 1
                height = full_height - patch_size + 1
            else:
                width = full_width
                height = full_height

            x_center = int(min(pixel_index % width, width - 1))
            y_center = min(pixel_index // width, height - 1)

            half_patch_size = patch_size // 2

            if not self.padding:
                x_center += half_patch_size
                y_center += half_patch_size

            xs = np.mgrid[x_center - half_patch_size : x_center + patch_size - half_patch_size]
            ys = np.mgrid[y_center - half_patch_size : y_center + patch_size - half_patch_size]

        xs = np.clip(xs, 0, full_width - 1)
        ys = np.clip(ys, 0, full_height - 1)

        if index is not None and self.extract_label_center:
            xs_labels = np.minimum(full_width - 1, np.r_[x_center])
            ys_labels = np.minimum(full_height - 1, np.r_[y_center])
            label_size = (1, 1)
        else:
            xs_labels = xs
            ys_labels = ys
            label_size = (len(ys), len(xs))

        def _index(inp, nc, ys, xs):
            # if self.patch_size == 1:
            #     return inp[np.r_[0:nc], ys.item(), xs.item()]
            # else:
            return inp[np.ix_(np.r_[0:nc], ys, xs)]

        ret = dict()
        if self.concatenate_maps or self.classification:
            # we still need to return a concatenated labels tensor for evaluation purposes
            # TODO: provide this only during evaluation!
            ret['labels'] = np.zeros((self.num_chans_labels_mapped,) + label_size, dtype=np.float32)
            ci = 0
            for k in self.label_names:
                nc = self.maps[mat_ind][k].shape[0]
                if k == 'refined_normal' and self.normals_only_xy:
                    # we implicitly apply the normal xy mapping here by discarding the z-coordinates
                    nc = 2 if self.normals_only_xy else 3
                    tmp = self.maps[mat_ind][k][np.ix_(np.r_[0:nc], ys_labels, xs_labels)]
                elif k == 'anisotropy':
                    # anisotropy cossin mapping needs to be applied on every access as it is not precomputed
                    nc = 2 if self.aniso_mapping == 'cossin' else 1
                    tmp = self.map_anisotropy(self.maps[mat_ind][k][np.ix_(np.r_[0:1], ys_labels, xs_labels)])
                elif k == 'roughness':
                    tmp = self.map_roughness(self.maps[mat_ind][k][np.ix_(np.r_[0:2], ys_labels, xs_labels)])
                else:
                    tmp = self.maps[mat_ind][k][np.ix_(np.r_[0:nc], ys_labels, xs_labels)]
                ret['labels'][np.mgrid[ci:ci+nc], :, :] = self.normalize_output(tmp, k)
                ci += nc

        if not self.concatenate_maps:
            for k in self.label_names:
                nc = self.maps[mat_ind][k].shape[0]
                if k == 'refined_normal' and self.normals_only_xy:
                    # we implicitly apply the normal xy mapping here by discarding the z-coordinates
                    nc = 2 if self.normals_only_xy else 3
                    tmp = _index(self.maps[mat_ind][k], nc, ys_labels, xs_labels)
                elif k == 'roughness':
                    tmp = self.map_roughness(_index(self.maps[mat_ind][k], 2, ys_labels, xs_labels))
                elif k == 'anisotropy':
                    # anisotropy cossin mapping needs to be applied on every access as it is not precomputed
                    tmp = self.map_anisotropy(_index(self.maps[mat_ind][k], 1, ys_labels, xs_labels))
                else:
                    tmp = _index(self.maps[mat_ind][k], nc, ys_labels, xs_labels)
                if not self.classification:
                    ret[k] = self.normalize_output(tmp, k)
                else:
                    ret[k] = tmp

        if self.classification:
            ret = self.map_labels(ret, invert=False)

        # always return geometric cues
        ret['unref_heightmap'] = _index(self.maps[mat_ind]['unref_heightmap'], 1, ys, xs)
        ret['unref_xyz'] = _index(self.maps[mat_ind]['unref_xyz'], 3, ys, xs).astype(self.dtype_out)
        if self.normals_only_xy:
            ret['unref_normal'] = _index(self.maps[mat_ind]['unref_normal'], 2, ys, xs).astype(self.dtype_out)
        else:
            ret['unref_normal'] = _index(self.maps[mat_ind]['unref_normal'], 3, ys, xs).astype(self.dtype_out)

        for k in self.input_names:
            if k in ['poly', 'pan', 'lls']:
                ret[k] = _index(self.inputs[k][mat_ind], self.meta['chs_' + k], ys, xs)
            elif k not in ['unref_xyz', 'unref_normal', 'unref_heightmap']:
                raise Exception('please implement returning input modality ' + k)

        # confidences treated separately
        if self.load_confidences:
            ret['confidences'] = _index(self.confidences[mat_ind], self.meta['num_total'], ys, xs)

        # store normalized timestamp
        # ret['timestamp'] = (self.timestamps[mat_ind] - self.min_timestamp).total_seconds() \
        #                    / (self.max_timestamp - self.min_timestamp).total_seconds()

        # # DEBUG
        # ret['xs'] = xs
        # ret['ys'] = ys

        ret['mat_ind'] = mat_ind

        if self.return_coordinates:
            ret['xs'] = xs
            ret['ys'] = ys

        if not self.use_inverse_tac7_transforms:
            ret['xys'] = np.stack(np.meshgrid(xs_labels, ys_labels), axis=0)

        if self.cast:
            for k in ret.keys():
                if isinstance(ret[k], np.ndarray) and ret[k].dtype != self.dtype_out:
                    ret[k] = ret[k].astype(self.dtype_out)

        return ret


# ###############################################################################
# # infinitely repeating dataloader to avoid overhead from recreation in each epoch
# see https://github.com/pytorch/pytorch/issues/15849
# and https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
# ###############################################################################

class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class DataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class PatchDataset(torch.utils.data.Dataset):
    """wrapper dataset that extracts larger, overlapping patches from a single material from a base dataset, to be used
    for reconstructing full materials during inference"""
    def __init__(self, dataset, mat_ind=0, patch_size=128, min_patch_size=15, padding='none'):
        # underlying base dataset
        self.dataset = dataset

        # we work on a single from potentially multiple materials in the base dataset
        self._mat_ind = mat_ind

        # get selected material's dimension
        keys = list(dataset.inputs.keys())
        self._height, self._width = dataset.inputs[keys[0]][self._mat_ind].shape[1:3]

        # set patch size and compute overlap based on base dataset's patch size (assuming those patches are mapped to
        # 1x1 output patches)
        # FIXME: make this dataset aware of the model's downsampling rate and compute the overlap accordingly
        self._patch_size = patch_size
        self._overlap = dataset.patch_size // 2 + 1

        # select padding mode of large patches
        assert padding in ['none', 'repeat'], 'padding must be one of: "none", "repeat"'
        self.padding = padding

        # minimum size a patch must have (e.g. for being acceptable as network input)
        self.min_patch_size = min_patch_size

        # lists of of x and y index chunks defining the patches
        self._xs, self._ys = None, None
        # number of patches in x and y directions
        self._nx, self._ny = 1, 1
        # actually initialize the above variables
        self.set_patch_size(patch_size)

    def set_patch_size(self, patch_size):
        """compute all x and y chunks for a given patch size over the entire material area"""
        self._patch_size = patch_size
        self._xs, self._ys = split_patches(h=self._height, w=self._width,
                                           patch_size=self._patch_size, min_patch_size=self.min_patch_size,
                                           overlap=self._overlap, padding=self.padding)
        self._nx, self._ny = len(self._xs), len(self._ys)

    def __len__(self):
        return self._nx * self._ny

    def __getitem__(self, index):
        xi = int(min(index % self._nx, self._nx - 1))
        yi = min(index // self._nx, self._ny - 1)
        xs = self._xs[xi]
        ys = self._ys[yi]
        output = self.dataset.__getitem__(index=None, xs=xs, ys=ys, mat_ind=self._mat_ind)
        output['xi'] = xi
        output['yi'] = yi
        output['xs'] = xs
        output['ys'] = ys
        output['overlap'] = self._overlap
        output['patch_size'] = self._patch_size
        return output