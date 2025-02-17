#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    step-3-reconstruct.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        15/01/2025 22:21

from rasterio.windows import Window
from osgeo import gdal
import pandas as pd
import numpy as np
import subprocess
import rasterio
import os

date_i = '20180507'

sentinel_data = '/gws/pw/j07/nceo_aerosolfire/rsong/project/UKIF/PM25-ML/data_preprocess/S2L1C_London/Sentinel2_L1C_%s_CloudMasked.tif'%date_i
output_cropped_sentinel_file = './data/Sentinel2_L1C_%s_CloudMasked_cropped.tif'%date_i
predicted_pm25_dir = './data_chunk_new_predicted_csv'

dataframes = []

# Loop through all files in the folder
for filename in os.listdir(predicted_pm25_dir):
    if filename.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(predicted_pm25_dir, filename)  # Get the full file path
        df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
        dataframes.append(df)  # Append the DataFrame to the list
        print(f"Loaded data from: {file_path}")

# Combine all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)
# Display the combined DataFrame
pm25_dict = dict(zip(combined_df['patch_name'], combined_df['predicted_pm25']))

crop_width = 2000 # pixels
crop_height = 2000 # pixels

sentinel_dataset = gdal.Open(sentinel_data)
target_lat = 51.612559 # top left corner of the selected box
target_lon = -0.264561 # top left corner of the selected box

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

# for row in range(crop_y, crop_y_end):
#     for col in range(crop_x, crop_x_end):
#     # for col in range(5011, 5015):
#         # read the value from the predicted_pm25_dir
#         predicted_pm25 = combined_df.loc[(combined_df['patch_name'] == f'patch_{col:05d}_{row:05d}')]['predicted_pm25'].values[0]
#         print(f"Predicted PM2.5 value at ({col}, {row}): {predicted_pm25}")
#         predicted_pm25_array[row - crop_y, col - crop_x] = predicted_pm25

for row in range(crop_y, crop_y_end):
    for col in range(crop_x, crop_x_end):
        patch_name = f'patch_{col:05d}_{row:05d}'
        predicted_pm25_array[row - crop_y, col - crop_x] = pm25_dict.get(patch_name, 0)
        print(f"Predicted PM2.5 value at ({col}, {row}): {predicted_pm25_array[row - crop_y, col - crop_x]}")

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
            abs(pixel_height)
        )
    })

    # Create the new file with updated metadata
    output_pm25_file = output_cropped_sentinel_file.replace('.tif', '_pm25.tif')
    with rasterio.open(output_pm25_file, 'w', **meta) as dst:
        dst.write(predicted_pm25_array.astype('float32'), 1)

print(f"Saved predicted PM2.5 values to: {output_pm25_file}")