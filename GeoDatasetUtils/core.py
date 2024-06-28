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
            input: Union[str, np.ndarray],
            name: str = None,
            meta: dict = None
        ) -> None:
        
        # if not isinstance(path, str):
        #     try:
        #         path = str(path)
        #     except :
        #         raise ValueError('The path parameter you input for GeoTiff should be string or can be converted into string by str() function.')
        if not isinstance(input, np.ndarray):
            f = rasterio.open(input)
            self.__reader = f
            self.__meta = f.meta.copy()
            self.__name = Path(f.name).stem

            self.__bands: np.ndarray = f.read()
            self.__band_num = f.count
            self.__shape = f.shape

        else:
            self.__reader = None
            self.__name = name
            if input.ndim != 3:
                raise ValueError('The dimention of ndarray should be 3!')
            
            self.__bands = input
            self.__band_num = input.shape[0]
            self.__shape = (input.shape[1], input.shape[2])

            meta = meta.copy()
            meta.update(count= self.__band_num)
            meta.update(height= self.__shape[0])
            meta.update(width= self.__shape[1])
            self.__meta = meta


    @property
    def reader(self):
        return self.__reader
    
    @property
    def meta(self):
        return self.__meta
    
    @property
    def name(self):
        return self.__name

    @property
    def bands(self):
        return self.__bands

    @property
    def band_num(self):
        return self.__band_num
    
    @property
    def shape(self):
        return self.__shape


    def write(self, path: str):
        with rasterio.open(path, 'w', **self.meta) as f:
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