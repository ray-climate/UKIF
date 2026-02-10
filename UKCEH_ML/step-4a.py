#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    pred-step-1.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        05/08/2025 14:52

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box, mapping
from pyproj import Transformer
import matplotlib.pyplot as plt

# ------------------- 1. Configuration -------------------
base_dir = '/Users/rs/Projects/UKIF/UKCEH_ML'
folders = {
    'NDVI': os.path.join(base_dir, 'data_NDVI/data'),
    'LST': os.path.join(base_dir, 'data_LST/data'),
    'Precip': os.path.join(base_dir, 'data_precip/data'),
}
output_dir = os.path.join(base_dir, 'prepared_npy')
os.makedirs(output_dir, exist_ok=True)

# ------------------- 2. ROI in lat/lon -------------------
lon_min, lat_min = -0.5571, 52.5952
lon_max, lat_max =  0.1202, 52.9032

# ------------------- 3. Reference NDVI for 10m grid -------------------
ndvi_ref_path = os.path.join(folders['NDVI'], 'NDVI_2022_05.tif')
with rasterio.open(ndvi_ref_path) as ref:
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_shape = ref.shape
    ref_dtype = ref.dtypes[0]
    print(f"[INFO] Using NDVI as 10-m reference: shape={ref_shape}, crs={ref_crs}")

# Convert ROI to reference pixel window
transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
x_min, y_min = transformer.transform(lon_min, lat_min)
x_max, y_max = transformer.transform(lon_max, lat_max)

from rasterio.transform import rowcol
row_start, col_start = rowcol(ref_transform, x_min, y_max)
row_stop, col_stop = rowcol(ref_transform, x_max, y_min)
row_start = max(0, row_start)
row_stop = min(ref_shape[0], row_stop)
col_start = max(0, col_start)
col_stop = min(ref_shape[1], col_stop)
print(f"[INFO] Crop window: rows {row_start}:{row_stop}, cols {col_start}:{col_stop}")

# ------------------- 4. Reprojection -------------------
def reproject_to_ref(src_path):
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(f"Missing CRS: {src_path}")

        data = src.read(1).astype(np.float32)
        data[data <= -1e20] = np.nan

        dest = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(
            source=data,
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )
        return dest

# ------------------- 5. Find file -------------------
def find_file(folder, year, month):
    tag = f"{year}_{month:02d}"
    for f in os.listdir(folder):
        if tag in f and f.endswith('.tif'):
            return os.path.join(folder, f)
    return None

# ------------------- 6. Build 10-m stack -------------------
def build_highres_stack(year):
    months = [5, 6, 7]
    stack = []

    for month in months:
        layers = []

        for var in ['NDVI', 'LST', 'Precip']:
            path = find_file(folders[var], year, month)
            if not path:
                raise FileNotFoundError(f"[ERROR] Missing {var} {year}-{month:02d}")

            print(f"[INFO] Loading {var}: {os.path.basename(path)}")

            if var == 'NDVI':
                with rasterio.open(path) as src:
                    img = src.read(1).astype(np.float32)
                    img[img == 0] = np.nan
                    data = img
            else:
                data = reproject_to_ref(path)

                if var == 'LST':
                    data[data < 250] = np.nan

            cropped = data[row_start:row_stop, col_start:col_stop]
            print(f"[DEBUG] {var} shape after crop: {cropped.shape}")
            layers.append(cropped)

        month_stack = np.stack(layers, axis=-1)  # (H, W, 3)
        stack.append(month_stack)

    return np.stack(stack, axis=0)  # (3, H, W, 3)

# ------------------- 7. Run -------------------
X_2022_10m = build_highres_stack(2022)
out_path = os.path.join(output_dir, 'X_2022_10m.npy')
np.save(out_path, X_2022_10m)

print(f"[SUCCESS] Saved: {out_path} | shape={X_2022_10m.shape}")
