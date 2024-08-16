from typing import Union, Optional
from enum import Enum
from pathlib import Path
import numpy as np
import rasterio
import rasterio.mask
from rasterio.enums import Resampling
import fiona


def _normalize(array: np.ndarray, range_: tuple):
    max_value = range_[0]
    min_value = range_[1]
    
    array = (array - min_value) / (max_value - min_value)
    array_copy = array.copy()
    
    array[array_copy > 1.] = 1.
    array[array_copy < 0.] = 0.
    
    return array


class OutputType(Enum):
    GeoTiff = 'geotiff'
    NumPy = 'numpy'


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
        """write the GeoTiff into the specified path.

        Args:
            path (str): the path for storing the GeoTiff, must be ended with .tif suffix.
        """
        
        with rasterio.open(path, 'w', **self.meta) as f:
            f.write(self.__bands)


    def to_normlized_numpy(self, norm_pattern= None):
        """return the bands normlized into 0 ~ 1.

        Args:
            norm_pattern (list, optional): the norm_pattern for how to normalize the bands, consist of three tuple, each spcefiys the max and min value of the band.
                Defaults to None, means the norm_pattern will depend on the max, min value of the band itself.

        Returns:
            np.ndarray: the result bands with the shape of (height x width x channels)
        """
        
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


    def resampling(self, scale_factor: float):
        """return a GeoTiff with a scaled resolution.

        Args:
            scale_factor (float): scale factor for resample operation, for example, if the scale factor is 2 and the shape of result GeoTiff will be double.

        Returns:
            GeoTiff: the result GeoTiff with the shape of (scale_factor*height x scale_factor*width)
        """
        
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
    
    
    def cropping_by_shapefile(self, shapefile_path: Union[str, Path], field_name: str, output_folder: Union[str, Path], no_data: Optional[float]= -9999., output_type: Optional[OutputType]= OutputType.NumPy):
        """cropping the GeoTiff by polygons in shapefile and save the results into the output_folder.

        Args:
            shapefile_path (Union[str, Path]): path to the shapefile.
            field_name (str): the name of target field of shapefile table used for naming different result files.
            output_folder (Union[str, Path]): path to the folder for storing the result files.
            no_data (Optional[float], optional): the value for the pixel stay outside the polygon. Defaults to -9999..
            output_type (Optional[OutputType], optional): the type of result file, can by npz or tif file. Defaults to OutputType.NumPy.
        """
        
        features = fiona.open(shapefile_path)
        
        if output_type == OutputType.GeoTiff:
            output_meta = self.meta.copy()

        if isinstance(output_folder, str):
                output_folder = Path(output_folder)

        if not output_folder.exists():
            output_folder.mkdir()
        assert output_folder.is_dir(), 'The output_folder must be a directory.'

        # * iterate each polygon. 
        for feature in features:
            id = feature['properties'][field_name]
            ploygon = feature['geometry']
            shapes = [ploygon]

            img: np.ndarray
            
            if output_type == OutputType.NumPy:
                img, out_transform = rasterio.mask.mask(self.reader, shapes, crop=True, nodata= no_data)
                output_path = output_folder / f'{id}.npz'
                np.savez_compressed(output_path, img)

            elif output_type == OutputType.GeoTiff:
                img, out_transform = rasterio.mask.mask(self.reader, shapes, crop=True, nodata= no_data)
                output_meta.update({
                    "driver": "GTiff",
                    "height": img.shape[1],
                    "width": img.shape[2],
                    "transform": out_transform
                })

                output_path = output_folder / f'{id}.tif'
                with rasterio.open(output_path, 'w', **output_meta) as f:
                    f.write(img)


def geotiff_cropping(input_tiff: GeoTiff, shapefile_path: str, field_name: str, output_folder: Union[str, Path], no_data= -9999., output_type= OutputType.NumPy):
    """cropping the GeoTiff by polygons in shapefile and save the results into the output_folder.

        Args:
            shapefile_path (Union[str, Path]): path to the shapefile.
            field_name (str): the name of target field of shapefile table used for naming different result files.
            output_folder (Union[str, Path]): path to the folder for storing the result files.
            no_data (Optional[float], optional): the value for the pixel stay outside the polygon. Defaults to -9999..
            output_type (Optional[OutputType], optional): the type of result file, can by npz or tif file. Defaults to OutputType.NumPy.
    """
    
    features = fiona.open(shapefile_path)
    
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
                