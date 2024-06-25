from osgeo import gdal


# Open the original GeoTIFF
input_tif = 'input.tif'
output_tif = 'output.tif'
new_pixel_size = (0.5, 0.5)  # Example new pixel size

# Open the input dataset
dataset = gdal.Open(input_tif)

# Get the current geo-transform
geo_transform = list(dataset.GetGeoTransform())

# Update the pixel size in the geo-transform
geo_transform[1] = new_pixel_size[0]  # Pixel width
geo_transform[5] = -new_pixel_size[1]  # Pixel height (negative for north-up images)

# Create the output dataset with the updated geo-transform
driver = gdal.GetDriverByName('GTiff')
output_dataset = driver.Create(
    output_tif,
    int((dataset.RasterXSize * geo_transform[1]) / new_pixel_size[0]),
    int((dataset.RasterYSize * geo_transform[5]) / new_pixel_size[1]),
    dataset.RasterCount,
    gdal.GDT_Byte
)
output_dataset.SetGeoTransform(geo_transform)
output_dataset.SetProjection(dataset.GetProjection())

# Perform the resampling
gdal.ReprojectImage(dataset, output_dataset, None, None, gdal.GRA_Bilinear)

# Close the datasets
dataset = None
output_dataset = None