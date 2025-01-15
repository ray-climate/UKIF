#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    step-1-data-preparation.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        14/01/2025 17:53

from osgeo import gdal
import os

sentinel_data = '/gws/pw/j07/nceo_aerosolfire/rsong/project/UKIF/PM25-ML/data_preprocess/S2L1C_London/Sentinel2_L1C_20180807_CloudMasked.tif'
pm25_data =     '/gws/pw/j07/nceo_aerosolfire/rsong/project/UKIF/PM25-ML/data_preprocess/modis_pm25_data_reproject_crop/GHAP_PM2.5_D1K_20180807_V1_cropped_projected.nc'
output_cropped_sentinel_file = './data/Sentinel2_L1C_20180807_CloudMasked_cropped.tif'

crop_width = 500 # pixels
crop_height = 500 # pixels

sentinel_dataset = gdal.Open(sentinel_data)
target_lat = 51.518 # top left corner of the selected box
target_lon = -0.136 # top left corner of the selected box

# Get the geotransform and raster size
geotransform = sentinel_dataset.GetGeoTransform()
origin_x, pixel_width, _, origin_y, _, pixel_height = geotransform
raster_x_size = sentinel_dataset.RasterXSize
raster_y_size = sentinel_dataset.RasterYSize

# Calculate pixel coordinates for the given latitude and longitude
pixel_x = int((target_lon - origin_x) / pixel_width)
pixel_y = int((target_lat - origin_y) / pixel_height)

# Ensure the coordinates are within the raster bounds
pixel_x = max(0, min(raster_x_size - 1, pixel_x))
pixel_y = max(0, min(raster_y_size - 1, pixel_y))

print(f"Closest pixel index to lat/lon ({target_lat}, {target_lon}):")
print(f"X: {pixel_x}, Y: {pixel_y}")

# Define the cropping box (top-left corner and dimensions)
crop_x = pixel_x
crop_y = pixel_y
crop_x_end = min(crop_x + crop_width, raster_x_size)
crop_y_end = min(crop_y + crop_height, raster_y_size)

# Read the data from the cropping region
cropped_data = sentinel_dataset.ReadAsArray(crop_x, crop_y, crop_x_end - crop_x, crop_y_end - crop_y)
if cropped_data is None:
    raise ValueError("Failed to read the cropped data from the Sentinel dataset.")

# Create a new GeoTIFF file for the cropped data
driver = gdal.GetDriverByName('GTiff')
out_raster = driver.Create(output_cropped_sentinel_file, crop_x_end - crop_x, crop_y_end - crop_y,
                           sentinel_dataset.RasterCount, sentinel_dataset.GetRasterBand(1).DataType)

# Set the geotransform and projection for the cropped raster
new_geotransform = (
    origin_x + crop_x * pixel_width,
    pixel_width,
    0,
    origin_y + crop_y * pixel_height,
    0,
    pixel_height
)
out_raster.SetGeoTransform(new_geotransform)
out_raster.SetProjection(sentinel_dataset.GetProjection())

# Write the cropped data to the new file
for band in range(1, sentinel_dataset.RasterCount + 1):
    out_band = out_raster.GetRasterBand(band)
    out_band.WriteArray(cropped_data[band - 1])

# Close datasets
out_raster.FlushCache()
del out_raster
sentinel_dataset = None

print(f"Cropped raster saved to: {output_cropped_sentinel_file}")

#-----------------------------------------------------------------
# reproject MODIS PM2.5
# Define the output file for the reprojected PM2.5 data
reprojected_pm25_file = './data/PM25_20180807_reprojected.nc'

# Use gdalwarp to reproject the PM2.5 data
gdalwarp_command = [
    'gdalwarp',
    '-t_srs', f'"{gdal.Info(output_cropped_sentinel_file, options=["-proj4"])}"',  # Match the target projection
    '-te',
    f'{new_geotransform[0]}',  # xmin
    f'{new_geotransform[3] + (crop_y_end - crop_y) * pixel_height}',  # ymin
    f'{new_geotransform[0] + (crop_x_end - crop_x) * pixel_width}',  # xmax
    f'{new_geotransform[3]}',  # ymax
    '-tr', f'{pixel_width}', f'{abs(pixel_height)}',  # Target resolution
    pm25_data,
    reprojected_pm25_file
]

# Execute the gdalwarp command
print("Running gdalwarp for reprojecting PM2.5 data...")
os.system(' '.join(gdalwarp_command))
