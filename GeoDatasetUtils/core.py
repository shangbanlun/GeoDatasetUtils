from typing import Optional, Union
from enum import Enum
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import rasterio
import fiona
import rasterio.mask


def normalize(array: np.ndarray, range_: tuple):
    max_value = range_[0]
    min_value = range_[1]
    array_copy = array.copy()
    array[array_copy > max_value] = max_value
    array[array_copy < min_value] = min_value
    return (array - min_value) / (max_value - min_value)

class GeoTiff():
    def __init__(
            self,
            path: str,
        ) -> None:
        
        f = rasterio.open(path)
        self.__reader = f
        self.__meta = f.meta.copy()
        self.__bands: np.ndarray = f.read()

    @property
    def reader(self):
        return self.__reader
    
    @property
    def meta(self):
        return self.__meta
    
    @property
    def band_num(self):
        return self.meta['count']

    def write(self, path: str):
        with rasterio.open(path, 'w', **self.__meta) as f:
            f.write(self.__bands)

    def get_image_for_show(self, render_pattern= None):
        
        image = []
        for idx, band in enumerate(self.__bands):
            if render_pattern != None:
                band = normalize(band, render_pattern[idx])
                image.append(band)
            else:
                max_value = np.max(band[0])
                min_value = np.min(band[0])
                band = normalize(band, (max_value, min_value))
                image.append(band)
                
        image = np.stack(image, axis= 0)
        image = np.transpose(image, (1, 2, 0))    # * (channel, height, width) -> (heigth, width, channel)
        
        return image

render_pattern = ((-4.1, -23.28), (-12.01, -30.98), (0.85, -0.11))

def get_image_for_show(bands: np.ndarray, render_pattern= None):
        
        image = []
        for idx, band in enumerate(bands):
            if render_pattern != None:
                band = normalize(band, render_pattern[idx])
                image.append(band)
            else:
                max_value = np.max(band[0])
                min_value = np.min(band[0])
                band = normalize(band, (max_value, min_value))
                image.append(band)
                
        image = np.stack(image, axis= 0)
        image = np.transpose(image, (1, 2, 0))    # * (channel, height, width) -> (heigth, width, channel)
        
        return image

class OutputType(Enum):
    GeoTiff = 'geotiff'
    NumPy = 'numpy'

def geotiff_cropping(input_tiff: GeoTiff, features, output_folder: Union[str, Path], no_data= -9999., output_type= OutputType.NumPy):
    
    if output_type == OutputType.GeoTiff:
        output_meta = input_tiff.meta.copy()

    if isinstance(output_folder, str):
            output_folder = Path(output_folder)
    if output_folder.is_file() : raise ValueError('the output_folder must be a folder.')
    if not output_folder.exists():
        output_folder.mkdir()

    # * iterate each polygon.
    for feature in features:
        id = feature['properties']['OBJECTID']
        ploygon = feature['geometry']
        shapes = [ploygon]

        img: np.ndarray
        
        if output_type == OutputType.NumPy:
            img, out_transform = rasterio.mask.mask(input_tiff.reader, shapes, crop=True, nodata= no_data)
            output_path = output_folder / f'{id}.npz'
            np.savez_compressed(output_path, img)

        elif output_type == OutputType.GeoTiff:
            img, out_transform = rasterio.mask.mask(input_tiff.reader, shapes, crop=True)
            output_meta.update({
                "driver": "GTiff",
                "height": img.shape[1],
                "width": img.shape[2],
                "transform": out_transform
            })

            output_path = output_folder / f'{id}.tif'
            with rasterio.open(output_path, 'w', **output_meta) as f:
                f.write(img)


# * Paths to the input GeoTIFF and shapefile.
input_tiff_path = 'D:\Work\Vector\output.tif'
shapefile_path = 'D:\\Work\Vector\ChongMing\\for_cropping_test\\for_cropping_test.shp'
features = fiona.open(shapefile_path, 'r')


tiff = GeoTiff(input_tiff_path)
# render_pattern = ((-4.1, -23.28), (-12.01, -30.98), (0.85, -0.11))
# image = tiff.get_image_for_show(render_pattern= render_pattern)


# plt.imshow(image)
# plt.axis('off')
# plt.show()


geotiff_cropping(tiff, features, 'D:\\Work\\Vector\\tiff', output_type= OutputType.GeoTiff)

# img = np.load('')
# image = get_image_for_show(img, render_pattern)
# plt.imshow(image)
# plt.axis('off')
# plt.show()