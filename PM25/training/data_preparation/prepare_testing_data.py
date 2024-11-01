#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    prepare_testing_data.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        01/11/2024 17:35

import os
import rasterio
import numpy as np

# Path to your GeoTIFF file
tif_file = './S2_London_2018/Sentinel2_L1C_20180807_CloudMasked.tif'
save_dirt = './testing_data'
os.makedirs(save_dirt, exist_ok=True)

# Open the GeoTIFF file
with rasterio.open(tif_file) as src:
    num_bands = src.count
    band_data = {}
    for i in range(1, num_bands + 1):
        # Get the band description (e.g., 'B1', 'B2', ...)
        band_name = src.descriptions[i - 1]
        # Read the band data
        data = src.read(i)
        # Store the data in the dictionary
        band_data[band_name] = data

# Save all bands into an NPZ file
np.savez(os.path.join(save_dirt, 'Sentinel2_L1C_20180807_CloudMasked.npz'), **band_data)

