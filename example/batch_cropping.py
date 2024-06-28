from typing import List, Optional, Callable
from pathlib import Path
import numpy as np
import fiona
from matplotlib import pyplot as plt
import sys
sys.path.append('./')

from GeoDatasetUtils.core import GeoTiff, geotiff_cropping, OutputType


def batch_process(days: List[Path], input_folder_name: str, output_folder_name: str, func: Callable):
    for idx, day in enumerate(days):
        input_folder_path = day / input_folder_name
        files = [file for file in input_folder_path.iterdir() if file.is_file()]

        if len(files) != 1: raise FileExistsError('the number of file in the input folder shoud be one.')

        input = GeoTiff(files[0])
        output_folder_path = day / output_folder_name
        if not output_folder_path.exists() : output_folder_path.mkdir()
        output_path = day / output_folder_name / f'{input.name}_rgb.tif'
        func(input, output_path)

        print(f'INFO:: ({idx+1}/{len(days)}) day: {day.name} has completed.')


def convert_2_rgb(input: GeoTiff, output_path: str):

    bands = input.bands
    (sigma0_VH, sigma0_VV) = bands
    bands = np.stack([sigma0_VV, sigma0_VH, sigma0_VV / sigma0_VH], axis= 0)
    tiff_rgb = GeoTiff(bands, input.name, input.meta)

    tiff_rgb.write(output_path)


def save_image(input: GeoTiff, output_path: str):
    render_pattern = ((-4.1, -23.28), (-12.01, -30.98), (0.85, -0.11))
    image = input.get_image_for_show(render_pattern= render_pattern)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(output_path, dpi = 1000)


def batch_save_image(days: List[Path], input_folder_name: str, output_folder_name: str, func: Callable):
    for idx, day in enumerate(days):
        input_folder_path = day / input_folder_name
        files = [file for file in input_folder_path.iterdir() if file.is_file()]

        if len(files) != 1: raise FileExistsError('the number of file in the input folder shoud be one.')

        input = GeoTiff(files[0])

        output_folder_path = Path(output_folder_name)
        if not output_folder_path.exists() : output_folder_path.mkdir()
        
        output_path = output_folder_path / f'{input.name}.jpg'
        func(input, output_path)

        print(f'INFO:: ({idx+1}/{len(days)}) day: {day.name} has completed.')


def main():
    home_folder = Path('../Data/Image')
    days = [day for day in home_folder.iterdir() if not day.is_file()]
    days.sort()
    # days = [days[0]]
    batch_process(days, 'GeoTiff', 'GeoTiff_rgb', convert_2_rgb)
    batch_save_image(days, 'GeoTiff_rgb', 'image', save_image)
    

main()