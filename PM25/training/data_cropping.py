#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    data_cropping.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        28/10/2024 17:03

import rasterio
from rasterio.windows import Window

# Paths to your datasets
path_10m = 'Sentinel2_L1C_20180117_CloudMasked.tif'
path_1km = 'GHAP_PM2.5_D1K_20180117_V1_cropped.nc'
output_path = 'GHAP_PM2.5_D1K_20180117_V1_cropped_to_10m_extent.nc'

# Open the 10m data to get its bounds and CRS
with rasterio.open(path_10m) as src10m:
    bounds10m = src10m.bounds
    crs10m = src10m.crs

# Open the 1km data and set its CRS if it's not defined
with rasterio.open(path_1km) as src1km:
    if src1km.crs is None:
        src1km.crs = crs10m  # Assuming both datasets use WGS84
    transform1km = src1km.transform
    width1km = src1km.width
    height1km = src1km.height

    # Get row and column indices for the bounds
    row_min, col_min = src1km.index(bounds10m.left, bounds10m.top)
    row_max, col_max = src1km.index(bounds10m.right, bounds10m.bottom)

    # Ensure the indices are within the dataset dimensions
    row_min = max(0, min(row_min, height1km))
    row_max = max(0, min(row_max, height1km))
    col_min = max(0, min(col_min, width1km))
    col_max = max(0, min(col_max, width1km))

    # Swap indices if necessary
    if row_max < row_min:
        row_min, row_max = row_max, row_min
    if col_max < col_min:
        col_min, col_max = col_max, col_min

    # Define the window
    window = Window(col_off=col_min, row_off=row_min,
                    width=col_max - col_min, height=row_max - row_min)

    # Read the data within the window
    data1km_cropped = src1km.read(1, window=window)

    # Update the transform for the cropped data
    transform_cropped = src1km.window_transform(window)

    # Define metadata for the output file
    out_meta = src1km.meta.copy()
    out_meta.update({
        "height": window.height,
        "width": window.width,
        "transform": transform_cropped
    })

    # Write the cropped data to a new file
    with rasterio.open(output_path, 'w', **out_meta) as dest:
        dest.write(data1km_cropped, 1)

print(f"Cropped data saved to {output_path}")
