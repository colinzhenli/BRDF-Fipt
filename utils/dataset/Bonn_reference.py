import numpy as np
import pyexr
import scipy.io as spio

# for visualizations only
from matplotlib import pyplot as plt

def loadmat(filename):
	"""wrapper around scipy.io.loadmat that avoids conversion of nested matlab structs to np.arrays"""
	mat = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
	for key in mat:
	    if isinstance(mat[key], spio.matlab.mio5_params.mat_struct):
		mat[key] = toDict(mat[key])
	return mat

def toDict(matobj):
	"""construct python dictionary from matobject"""
	newDict = {}
	for fn in matobj._fieldnames:
	    val = matobj.__dict__[fn]
	    if isinstance(val, spio.matlab.mio5_params.mat_struct):
		newDict[fn] = toDict(val)
	    else:
		newDict[fn] = val
	return newDict

def normalize(dirs):
	return dirs / np.linalg.norm(dirs, axis=2)[:, :, None]

# load geometric calibration data
calib = loadmat('mat0377_calibration.mat')
xyz = pyexr.read('mat0003_xyz_rot000.exr')

# camera 1 position for 45 degrees turntable rotation:
V01 = calib['rot045']['cv01']

# LED 9 position for 45 degrees turntable rotation:
L09 = calib['rot045']['il09']

# linear light source corner positions for 22 degrees LLS angle and turntable rotation 45 degrees:
ind = np.nonzero(calib['llsAnglesDegrees'] == 22)[0][0]
llsCorners022Degrees = calib['rot045']['llsCorners'][:, :, ind] # 3 x 4 array

# compute normalized light and view directions for all pixels:
L = normalize(L09[None, None, :] - xyz) # height x width x 3 array
V = normalize(V01[None, None, :] - xyz) # height x width x 3 array

# LLS directions should be sampled in the quad spanned by the 4 corners
# read polychromatic -> panchromatic calibration data
poly2pan = loadmat('mat0003_poly2pan.mat')
poly2panIndices = poly2pan['poly2panIndices']
poly2panWeights = poly2pan['poly2panWeights']

# read all polychromatic channels in IEEE 754 half precision float format
poly = pyexr.open('mat0003_poly.exr')
polyChannels = poly.channel_map['all']
poly = poly.get(group='all', precision=pyexr.HALF)

# read all panchromatic channels in IEEE 754 half precision float format
pan = pyexr.open('mat0003_pan.exr')
panChannels = pan.channel_map['all']
pan = pan.get(group='all', precision=pyexr.HALF)

# extract first polychromatic image
print(polyChannels[:3])
poly_cv01_il026_rot000 = poly[:, :, :3]

# get panchromatic image corresponding to first polychromatic image
panInd = poly2panIndices[0]
pan_cv01_il026_rot000 = pan[:, :, panInd - 1]

# convert polychromatic to panchromatic image using the camera- and light-specific conversion weights
pan_cv01_il026_rot000_converted = np.einsum('c,xyc->xy', poly2panWeights['cv01_il026'].astype(np.float16), poly_cv01_il026_rot000)

# show first polychromatic image
scale = 10
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(scale * poly_cv01_il026_rot000.astype(np.float32))
plt.title(polyChannels[0][:-2])
plt.subplot(1, 3, 2)
plt.imshow(scale * pan_cv01_il026_rot000[:, :, None].repeat(3, axis=2).astype(np.float32), cmap=plt.get_cmap('gray'))
plt.title(panChannels[panInd - 1])
plt.subplot(1, 3, 3)
plt.imshow(scale * pan_cv01_il026_rot000_converted[:, :, None].repeat(3, axis=2).astype(np.float32), cmap=plt.get_cmap('gray'))
plt.title('poly -> pan')