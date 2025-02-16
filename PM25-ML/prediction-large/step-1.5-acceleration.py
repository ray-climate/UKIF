#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    step-1.5-acceleration.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        15/01/2025 21:27

from rasterio.windows import Window
from osgeo import gdal
import numpy as np
import rasterio
import h5py
import sys
import os

def extract_patch(file_path, col, row, patch_size=128):
    """
    Extract a patch of size patch_size x patch_size x 12 bands centered at the given pixel coordinates.
    """
    with rasterio.open(file_path) as src:
        half_size = patch_size // 2
        window = Window(col - half_size, row - half_size, patch_size, patch_size)
        patch = src.read(window=window)
        return patch


def process_chunk(start_idx, chunk_size, total_pixels, crop_x, crop_y, crop_x_end, crop_y_end, date_i):

    sentinel_data = '/gws/pw/j07/nceo_aerosolfire/rsong/project/UKIF/PM25-ML/data_preprocess/S2L1C_London/Sentinel2_L1C_%s_CloudMasked.tif' %date_i
    save_patch_data_dir = './data_chunk/'
    os.makedirs(save_patch_data_dir, exist_ok=True)

    # Calculate the actual pixels this process should handle
    pixels = []
    for row in range(crop_y, crop_y_end):
        for col in range(crop_x, crop_x_end):
            pixels.append((col, row))

    # Process only this chunk's portion
    chunk_start = start_idx
    chunk_end = min(start_idx + chunk_size, len(pixels))

    # Create an HDF5 file to store all patches
    hdf5_file = os.path.join(save_patch_data_dir, f'chunk_{job_id}.h5')
    with h5py.File(hdf5_file, 'w') as f:
        for idx in range(chunk_start, chunk_end):
            if idx >= len(pixels):
                break
            col, row = pixels[idx]
            patch = extract_patch(sentinel_data, col, row)
            f.create_dataset(f'patch_{col:05d}_{row:05d}', data=patch)
            print(f"Saved patch {col:05d}_{row:05d} to {hdf5_file}")

    for idx in range(chunk_start, chunk_end):
        if idx >= len(pixels):
            break
        col, row = pixels[idx]
        patch = extract_patch(sentinel_data, col, row)
        patch_file = os.path.join(save_patch_data_dir, f'patch_{col:05d}_{row:05d}.npy')
        np.save(patch_file, patch)
        print(f"Saved patch to {patch_file}")


if __name__ == "__main__":

    date_i = '20180507'

    if len(sys.argv) != 3:
        print("Usage: python process_chunk.py <job_id> <total_jobs>")
        sys.exit(1)

    # Get job_id and total_jobs from command line arguments
    job_id = int(sys.argv[1])
    total_jobs = int(sys.argv[2])

    # These values should match your original script
    target_lat = 51.612559  # top left corner of the selected box
    target_lon = -0.264561  # top left corner of the selected box
    crop_width = 2000  # pixels
    crop_height = 2000  # pixels

    sentinel_data = '/gws/pw/j07/nceo_aerosolfire/rsong/project/UKIF/PM25-ML/data_preprocess/S2L1C_London/Sentinel2_L1C_%s_CloudMasked.tif'%date_i

    # Calculate initial coordinates
    sentinel_dataset = gdal.Open(sentinel_data)
    geotransform = sentinel_dataset.GetGeoTransform()
    origin_x, pixel_width, _, origin_y, _, pixel_height = geotransform
    raster_x_size = sentinel_dataset.RasterXSize
    raster_y_size = sentinel_dataset.RasterYSize

    pixel_x = int((target_lon - origin_x) / pixel_width)
    pixel_y = int((target_lat - origin_y) / pixel_height)

    pixel_x = max(0, min(raster_x_size - 1, pixel_x))
    pixel_y = max(0, min(raster_y_size - 1, pixel_y))

    crop_x = pixel_x
    crop_y = pixel_y
    crop_x_end = min(crop_x + crop_width, raster_x_size)
    crop_y_end = min(crop_y + crop_height, raster_y_size)

    # Calculate total number of pixels
    total_pixels = (crop_x_end - crop_x) * (crop_y_end - crop_y)

    # Calculate chunk size for each job
    chunk_size = -(-total_pixels // total_jobs)  # Ceiling division
    start_idx = job_id * chunk_size
    print(f"Job {job_id}: Processing pixels {start_idx} to {start_idx + chunk_size}")
    process_chunk(start_idx, chunk_size, total_pixels, crop_x, crop_y, crop_x_end, crop_y_end, date_i)