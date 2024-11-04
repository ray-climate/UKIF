#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    prepare_testing_patch.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        03/11/2024 21:41

from rasterio.windows import Window
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import rasterio
import os

MODIS_dir = './modis_pm25_data_crop_testing'
Sentinel_data = './S2_London_2018'

savefig_dir = './testing_figs_all_2018'
savedata_dir = './testing_data_all_2018'

os.makedirs(savefig_dir, exist_ok=True)
os.makedirs(savedata_dir, exist_ok=True)

date_list = []
for file in os.listdir(MODIS_dir):
    if file.endswith('.nc') & ('20180807' in file):
        MODIS_file = os.path.join(MODIS_dir, file)
        YYYYMMDD = file.split('_')[3]
        date_list.append(YYYYMMDD)

for date_i in date_list:

    MODIS_file = os.path.join(MODIS_dir, f'GHAP_PM2.5_D1K_{date_i}_V1_cropped_projected.nc')
    Sentinel_file = os.path.join(Sentinel_data, f'Sentinel2_L1C_{date_i}_CloudMasked.tif')

    # Open the NetCDF file
    dataset = nc.Dataset(MODIS_file, mode='r')

    # Extract the latitude and longitude variables
    lat_modis_array = dataset.variables['lat'][:]
    lon_modis_array = dataset.variables['lon'][:]
    pm25_array = dataset.variables['Band1'][:]
    # Get scale factor and fill value
    scale_factor = dataset.variables['Band1'].scale_factor
    fill_value = dataset.variables['Band1']._FillValue

    # Close the dataset
    dataset.close()

    # create a list of the latitudes and longitudes
    lat_modis_list = []
    lon_modis_list = []
    PM25_list = []

    for i in range(len(lat_modis_array)):
        for j in range(len(lon_modis_array)):
            lat_modis_list.append(lat_modis_array[i])
            lon_modis_list.append(lon_modis_array[j])
            PM25_list.append(pm25_array[i, j])


    def latlon_to_pixel(lat, lon, transform):
        """
        Convert latitude and longitude to pixel coordinates in the image.
        """
        x, y = rasterio.transform.rowcol(transform, lon, lat)
        return x, y

    def extract_patch(file_path, lat, lon, patch_size=128):
        """
        Extract a patch of size patch_size x patch_size x 12 bands centered at the pixel location corresponding
        to the given latitude and longitude.
        """
        # Open the raster file
        with rasterio.open(file_path) as src:
            # Transform for converting lat/lon to pixel coordinates
            transform = src.transform

            # Get pixel coordinates
            px, py = latlon_to_pixel(lat, lon, transform)
            # get the shape of the image
            shape = src.shape
            print(shape)
            print(px, py)

            # Calculate window boundaries
            half_size = patch_size // 2
            window = Window(px - half_size, py - half_size, patch_size, patch_size)

            # Read the window for all 12 bands
            patch = src.read(window=window)

        return patch

    def save_rgb_quickview(patch, filename):
        """
        Save an RGB quickview using bands 4, 3, and 2 of the patch as a PNG file.
        """
        # Select bands 4, 3, and 2 (assuming 1-indexed for bands in Sentinel-2)
        rgb_patch = np.stack([patch[3], patch[2], patch[1]], axis=-1)  # Bands 4, 3, and 2
        # Normalize bands for better visualization (0-1 range)
        rgb_patch = (rgb_patch - rgb_patch.min()) / (rgb_patch.max() - rgb_patch.min())
        rgb_patch = np.clip(rgb_patch, 0, 1)
        plt.imshow(rgb_patch)
        # Save the RGB image
        plt.axis('off')
        plt.savefig(f'{savefig_dir}/{filename}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    def bad_value_percentage(patch):
        """
        Calculate the percentage of pixels where all 13 bands are zero.
        """
        total_pixels = patch.shape[1] * patch.shape[2]
        bad_pixels = np.all(patch == 0, axis=0).sum()
        return (bad_pixels / total_pixels) * 100

    total_number = 0

    for i in range(len(lat_modis_list)):
        patch = extract_patch(Sentinel_file, lat_modis_list[i], lon_modis_list[i])
        if patch.shape[1] == 128 and patch.shape[2] == 128 and PM25_list[i] > 0:
            bad_percentage = bad_value_percentage(patch)
            if bad_percentage < 10:  # Save only if bad values are below 10%
                print(f"Date {date_i} Patch {total_number} has bad value percentage: {bad_percentage:.2f}% ------------ Found")
                save_rgb_quickview(patch, f'date_{date_i}_patch_{total_number}')
                # Save patch and PM2.5 value into npz file
                np.savez(f'{savedata_dir}/date_{date_i}_patch_{total_number}.npz', patch=patch, pm25=PM25_list[i])
            else:
                print(f"Date {date_i} Patch {total_number} has bad value percentage: {bad_percentage:.2f}% -------------- Cloudy")
        else:
            print(patch.shape)
            print(f"Patch {total_number} is not 128x128 or PM2.5 value is 0 {lat_modis_list[i], lon_modis_list[i]}")

        total_number += 1

# test
