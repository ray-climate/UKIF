#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    step-1-data-prep.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        04/08/2025 11:24

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box, mapping
from pyproj import Transformer

# ------------------ 1. Paths ------------------
base_dir = './'
data_dirs = {
    'NDVI': os.path.join(base_dir, 'data_NDVI/data'),
    'LST': os.path.join(base_dir, 'data_LST/data'),
    'Precip': os.path.join(base_dir, 'data_precip/data'),
}
ukceh_ref_path = os.path.join(base_dir, 'data_UKCEH/data/SPEI_2020_08.tif')
output_dir = os.path.join(base_dir, 'prepared_npy')
os.makedirs(output_dir, exist_ok=True)

# ------------------ 2. Define ROI ------------------
# Lat/lon bounds
lon_min, lat_min = -0.5571, 52.5952
lon_max, lat_max =  0.1202, 52.9032

# Transform to EPSG:27700 (BNG)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
x_min, y_min = transformer.transform(lon_min, lat_min)
x_max, y_max = transformer.transform(lon_max, lat_max)
roi_geom = [mapping(box(x_min, y_min, x_max, y_max))]

# ------------------ 3. Load UKCEH Grid Reference ------------------
with rasterio.open(ukceh_ref_path) as ref:
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_shape = ref.shape
    ref_dtype = ref.dtypes[0]

# ------------------ 4. Function to Align One Raster ------------------
def align_raster_to_ref(path):
    with rasterio.open(path) as src:
        src_data, _ = mask(src, roi_geom, crop=True)
        src_meta = src.meta.copy()
        dest = np.full(ref_shape, np.nan, dtype=np.float32)

        reproject(
            source=src_data[0],
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )

        dest[dest <= -1e20] = np.nan
        return dest

# ------------------ 5. Process All Data Types ------------------
for var_name, folder in data_dirs.items():
    print(f'Processing: {var_name}')
    files = sorted([f for f in os.listdir(folder) if f.endswith('.tif')])
    stack = []

    for fname in files:
        fpath = os.path.join(folder, fname)
        aligned = align_raster_to_ref(fpath)
        stack.append(aligned)

    array = np.stack(stack, axis=0)  # Shape: (time, rows, cols)
    out_npy = os.path.join(output_dir, f'{var_name}_JanJul.npy')
    np.save(out_npy, array)
    print(f'Saved: {out_npy} | shape={array.shape}')
