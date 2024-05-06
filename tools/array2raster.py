import rasterio as rio
import numpy as np
from skimage.io import imread
import warnings
warnings.filterwarnings("ignore")

def array2raster(arr, ref_file, new_file):
    """Converts a numpy array to a raster file.
    Parameters
    ----------
    arr : numpy array
        The array to convert.
    ref_file : str
        The path to the reference raster file.
    new_file : str
        The path to the new raster file.
    """
    # ref_img = imread(ref_file)
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=-1)
    band = arr.shape[-1]
    # print(arr.shape)
    with rio.open(ref_file) as src:
        profile = src.profile
        profile.update(count=band)
        profile.update(dtype=arr.dtype)
        with rio.open(new_file, 'w', **profile) as dst:
            for i in range(band):
                dst.write(arr[:,:,i], i+1)


if __name__ == '__main__':
    # Create a numpy array
    input = 'sample_inputs/ds_height.tif'
    arr = imread(input)
    print(arr.shape)
    # arr = np.random.randint(0, 100, (10, 10))
    # Convert the array to a raster file
    path_ref_file = 'sample_inputs/ds_rgb.tif'
    path_new_file = 'sample_inputs/ds_height_rf.tif'

    # array2raster(arr, path_ref_file, path_new_file)
