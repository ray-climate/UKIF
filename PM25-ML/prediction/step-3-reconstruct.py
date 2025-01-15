#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    step-3-reconstruct.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        15/01/2025 22:21

from rasterio.windows import Window
from osgeo import gdal
import numpy as np
import subprocess
import rasterio
import os

sentinel_data = '/gws/pw/j07/nceo_aerosolfire/rsong/project/UKIF/PM25-ML/data_preprocess/S2L1C_London/Sentinel2_L1C_20180807_CloudMasked.tif'
output_cropped_sentinel_file = './data/Sentinel2_L1C_20180807_CloudMasked_cropped.tif'
predicted_pm25_dir = './data_patch_predicted'

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
print(f"Raster size: {raster_x_size} x {raster_y_size}")

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

predicted_pm25_array = np.zeros((crop_height, crop_width))

for row in range(crop_y, crop_y_end):
    # for col in range(crop_x, crop_x_end):
    for col in range(5011, 5015):
        # read the value from the predicted_pm25_dir
        patch_file = os.path.join(predicted_pm25_dir, f'patch_{col:05d}_{row:05d}.npy')
        # Load the data
        data = np.load(patch_file, allow_pickle=True)
        data_dict = data.item()
        predicted_pm25 = data_dict['pm25_predicted']
        print(f"Predicted PM2.5 value at ({col}, {row}): {predicted_pm25}")
        predicted_pm25_array[row - crop_y, col - crop_x] = predicted_pm25

# Create the output GeoTIFF file
with rasterio.open(sentinel_data) as src:
    # Get the metadata from the source file
    meta = src.meta.copy()

    # Update the metadata for the new file
    meta.update({
        'driver': 'GTiff',
        'height': crop_height,
        'width': crop_width,
        'count': 1,  # Single band for PM2.5
        'dtype': 'float32',
        'transform': rasterio.transform.from_origin(
            origin_x + (crop_x * pixel_width),
            origin_y + (crop_y * pixel_height),
            pixel_width,
            pixel_height
        )
    })

    # Create the new file with updated metadata
    output_pm25_file = output_cropped_sentinel_file.replace('.tif', '_pm25.tif')
    with rasterio.open(output_pm25_file, 'w', **meta) as dst:
        dst.write(predicted_pm25_array.astype('float32'), 1)

print(f"Saved predicted PM2.5 values to: {output_pm25_file}")