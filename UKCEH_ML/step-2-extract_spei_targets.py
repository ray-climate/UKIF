#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    step-1-extract_spei_targets.py
# @Author:      Dr. Rui Song
# @Time:        07/08/2025

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from shapely.geometry import box, mapping
from pyproj import Transformer
import matplotlib.pyplot as plt

# ------------------- 1. Config -------------------
base_dir = '/Users/rs/Projects/UKIF/UKCEH_ML'
data_dir = os.path.join(base_dir, 'data_UKCEH/data')
output_dir = os.path.join(base_dir, 'prepared_npy')
os.makedirs(output_dir, exist_ok=True)

spei_files = {
    2020: os.path.join(data_dir, 'spei01_hadukgrid_uk_1km_mon_202001-202012.nc'),
    2022: os.path.join(data_dir, 'spei01_hadukgrid_uk_1km_mon_202201-202212.nc'),
}

# ------------------- 2. Reference grid from UKCEH -------------------
ref_path = spei_files[2020]
with rasterio.open(ref_path) as ref:
    ref_crs = 'EPSG:27700'
    ref_transform = ref.transform
    ref_shape = ref.shape
    print(f"[INFO] Reference grid: shape={ref_shape}, transform={ref_transform}")

# ------------------- 3. Define ROI (lat/lon → EPSG:27700) -------------------
# lon_min, lat_min = -0.564, 52.473
# lon_max, lat_max =  0.374, 53.118
lon_min, lat_min = -0.5571, 52.5952
lon_max, lat_max =  0.1202, 52.9032

transformer = Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
x1, y1 = transformer.transform(lon_min, lat_min)
x2, y2 = transformer.transform(lon_max, lat_max)
x_min, x_max = min(x1, x2), max(x1, x2)
y_min, y_max = min(y1, y2), max(y1, y2)

# ------------------- 4. Final SPEI extractor -------------------
def extract_spei(path, band_index, year):
    with rasterio.open(path) as src:
        print(f"\n[INFO] Reading {os.path.basename(path)}")

        # Force CRS if missing
        src_crs = src.crs
        if src_crs is None:
            print(f"[WARNING] src.crs is missing. Assigning EPSG:27700 manually.")
            src_crs = 'EPSG:27700'

        # Read full band
        band = src.read(band_index).astype(np.float32)
        band[band <= -1e20] = np.nan

        # Reproject to target grid
        dest = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(
            source=band,
            destination=dest,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )

        # Mask to ROI
        transform = ref_transform
        x0, y0 = ~transform * (x_min, y_max)
        x1, y1 = ~transform * (x_max, y_min)
        row_start, row_stop = int(min(y0, y1)), int(max(y0, y1))
        col_start, col_stop = int(min(x0, x1)), int(max(x0, x1))
        row_start = max(0, row_start)
        row_stop = min(ref_shape[0], row_stop)
        col_start = max(0, col_start)
        col_stop = min(ref_shape[1], col_stop)

        cropped = dest[row_start:row_stop, col_start:col_stop]

        print(f"[DEBUG] y_{year} crop window: rows {row_start}:{row_stop}, cols {col_start}:{col_stop}")
        print(f"[DEBUG] y_{year} shape after crop: {cropped.shape}")

        if cropped.size == 0 or np.all(np.isnan(cropped)):
            print(f"[WARNING] Cropped region is empty or all NaN for year {year}. Skipping.")
            return

    # Save .npy
    out_reg = os.path.join(output_dir, f'y_{year}.npy')
    np.save(out_reg, cropped)

    # Binary label
    cls = np.where(cropped < -1, 1, 0).astype(np.uint8)
    cls[np.isnan(cropped)] = 255
    out_cls = os.path.join(output_dir, f'y_{year}_cls.npy')
    np.save(out_cls, cls)

    # Plot
    plt.figure(figsize=(6, 6))
    im = plt.imshow(cropped, cmap='RdYlBu_r', vmin=-2.5, vmax=2.5)
    plt.colorbar(im, label='SPEI')
    plt.title(f'SPEI – August {year}')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'y_{year}_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] y_{year}.npy | y_{year}_cls.npy")
    print(f"[STATS] shape={cropped.shape}, min={np.nanmin(cropped):.2f}, max={np.nanmax(cropped):.2f}, mean={np.nanmean(cropped):.2f}, NaNs={np.count_nonzero(np.isnan(cropped))}")

# ------------------- 5. Run -------------------
extract_spei(spei_files[2020], 8, 2020)
extract_spei(spei_files[2022], 8, 2022)
