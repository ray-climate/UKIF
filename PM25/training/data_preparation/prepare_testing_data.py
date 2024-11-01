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

# Open the GeoTIFF file and read all bands into one variable 'data'
with rasterio.open(tif_file) as src:
    data = src.read()  # This reads all bands

# Save all bands into an NPZ file under the variable name 'data'
np.savez(os.path.join(save_dirt, 'Sentinel2_L1C_20180807_CloudMasked.npz'), data=data)


