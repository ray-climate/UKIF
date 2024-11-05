#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    rebuild_prediction.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        03/11/2024 22:40

from rasterio.windows import Window
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import rasterio
import os

MODIS_dir = './modis_pm25_data_crop_testing'
Sentinel_data = './S2_London_2018'

savefig_dir = './testing_data_all_2018_rotate'
savedata_dir = './testing_data_all_2018_rotate'

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
    PM25_predicted_list = []

    for i in range(len(lat_modis_array)):
        for j in range(len(lon_modis_array)):
            lat_modis_list.append(lat_modis_array[i])
            lon_modis_list.append(lon_modis_array[j])
            PM25_list.append(pm25_array[i, j])
            PM25_predicted_list.append(np.nan)

    total_number = 0

    for i in range(len(lat_modis_list)):
        # check if date_{date_i}_patch_{total_number}.npz exists
        if os.path.exists(f'{savedata_dir}/date_{date_i}_patch_{total_number}.npz'):
            print('loading predication data')
            data = np.load(f'{savedata_dir}/date_{date_i}_patch_{total_number}.npz')
            PM25_predicted_list[i] = data['pm25_predicted']
        total_number += 1

    # convert list to array
    PM25_predicted_array = np.array(PM25_predicted_list).reshape(pm25_array.shape)
    PM25_array = np.array(PM25_list).reshape(pm25_array.shape)

    # Save the predicted PM2.5 array with lat and lon as NetCDF file
    output_file = (f'PM25_predicted_{date_i}.nc')

    with nc.Dataset(output_file, 'w', format='NETCDF4') as ncfile:
        # Create dimensions
        lat_dim = ncfile.createDimension('lat', len(lat_modis_array))
        lon_dim = ncfile.createDimension('lon', len(lon_modis_array))

        # Create variables
        latitudes = ncfile.createVariable('lat', np.float32, ('lat',))
        longitudes = ncfile.createVariable('lon', np.float32, ('lon',))
        pm25_predicted = ncfile.createVariable('PM25_predicted', np.float32, ('lat', 'lon'), fill_value=np.nan)
        pm25_true = ncfile.createVariable('PM25_true', np.float32, ('lat', 'lon'), fill_value=np.nan)

        # Assign data to variables
        latitudes[:] = lat_modis_array
        longitudes[:] = lon_modis_array
        pm25_predicted[:, :] = PM25_predicted_array
        pm25_true[:, :] = PM25_array

        # Add attributes
        latitudes.units = 'degrees_north'
        longitudes.units = 'degrees_east'
        pm25_predicted.units = 'µg/m³'
        pm25_predicted.description = 'Predicted PM2.5 concentrations'
        pm25_true.units = 'µg/m³'
        pm25_true.description = 'True PM2.5 concentrations'

        # Global attributes
        ncfile.title = f'Predicted PM2.5 Data for {date_i}'
        ncfile.source = 'Generated by rebuild_prediction.py'

# test
