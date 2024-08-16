from typing import Union
from enum import Enum
from pathlib import Path
import numpy as np
import rasterio
import rasterio.mask
from rasterio.enums import Resampling


def _normalize(array: np.ndarray, range_: tuple):
    max_value = range_[0]
    min_value = range_[1]
    
    array = (array - min_value) / (max_value - min_value)
    array_copy = array.copy()
    
    array[array_copy > 1.] = 1.
    array[array_copy < 0.] = 0.
    
    return array

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
            self.__height = f.height
            self.__width = f.width

        else:
            self.__reader = None
            self.__name = name
            if input.ndim != 3:
                raise ValueError('The dimention of ndarray should be 3!')
            
            self.__bands = input
            self.__band_num = input.shape[0]
            self.__shape = (input.shape[1], input.shape[2])
            self.__height = input.shape[1]
            self.__width = input.shape[2]

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
    
    @property
    def height(self):
        return self.__height
    
    @property
    def width(self):
        return self.__width


    def write(self, path: str):
        with rasterio.open(path, 'w', **self.meta) as f:
            f.write(self.__bands)

    def to_numpy(self, norm_pattern= None):
        
        image = []
        for idx, band in enumerate(self.__bands):
            if norm_pattern is not None:
                band = _normalize(band, norm_pattern[idx])
                image.append(band)
            else:
                max_value = np.max(band[0])
                min_value = np.min(band[0])
                band = _normalize(band, (max_value, min_value))
                image.append(band)
                
        image = np.stack(image, axis= 0)
        # image = np.transpose(image, (1, 2, 0))    # * (channel, height, width) -> (heigth, width, channel)
        
        return image
    
    def resolution_increase(self, scale_factor: float):
        data = self.reader.read(
            out_shape=(
                    self.band_num,
                    int(self.height * scale_factor),
                    int(self.width * scale_factor)
                ),
            resampling=Resampling.nearest
        )
    
        # Update the metadata to reflect the new resolution
        transform = self.reader.transform * self.reader.transform.scale(1./scale_factor, 1./scale_factor)

        # Define the metadata for the new GeoTIFF
        meta = self.meta
        meta.update({
            'height': data.shape[1],
            'width': data.shape[2],
            'transform': transform
        })

        return GeoTiff(data, meta= meta)

def get_image_for_show(bands: np.ndarray, render_pattern= None):
        
        image = []
        for idx, band in enumerate(bands):
            if render_pattern is not None:
                band = _normalize(band, render_pattern[idx])
                image.append(band)
            else:
                max_value = np.max(band[0])
                min_value = np.min(band[0])
                band = _normalize(band, (max_value, min_value))
                image.append(band)
                
        image = np.stack(image, axis= 0)
        image = np.transpose(image, (1, 2, 0))    # * (channel, height, width) -> (heigth, width, channel)
        
        return image

class OutputType(Enum):
    GeoTiff = 'geotiff'
    NumPy = 'numpy'

def geotiff_cropping(input_tiff: GeoTiff, features, field_name: str, output_folder: Union[str, Path], no_data= -9999., output_type= OutputType.NumPy):
    
    if output_type == OutputType.GeoTiff:
        output_meta = input_tiff.meta.copy()

    if isinstance(output_folder, str):
            output_folder = Path(output_folder)

    if not output_folder.exists():
        output_folder.mkdir()
    assert output_folder.is_dir(), 'The output_folder must be a directory.'

    # * iterate each polygon. 
    for feature in features:
        id = feature['properties'][field_name]
        id = f'{id:05d}'
        ploygon = feature['geometry']
        shapes = [ploygon]

        img: np.ndarray
        
        if output_type == OutputType.NumPy:
            img, out_transform = rasterio.mask.mask(input_tiff.reader, shapes, crop=True, nodata= no_data)
            output_path = output_folder / f'{id}.npz'
            np.savez_compressed(output_path, img)

        elif output_type == OutputType.GeoTiff:
            img, out_transform = rasterio.mask.mask(input_tiff.reader, shapes, crop=True, nodata= no_data)
            output_meta.update({
                "driver": "GTiff",
                "height": img.shape[1],
                "width": img.shape[2],
                "transform": out_transform
            })

            output_path = output_folder / f'{id}.tif'
            with rasterio.open(output_path, 'w', **output_meta) as f:
                f.write(img)