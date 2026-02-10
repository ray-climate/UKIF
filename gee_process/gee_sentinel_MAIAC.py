#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    test.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        18/02/2025 12:28

import pandas as pd
import ee

# Initialize Earth Engine
ee.Initialize()

# 1) Define the bounding box as a Rectangle
# Note: ee.Geometry.Rectangle() expects [xmin, ymin, xmax, ymax] in [longitude, latitude].
# top-left    = lat=51.75721 , lon=-0.59863
# bottom-right= lat=51.18992 , lon= 0.32341
region = ee.Geometry.Rectangle([
    -0.59863,    # xmin (west)
    51.18992,    # ymin (south)
    0.32341,     # xmax (east)
    51.75721     # ymax (north)
])

# read csv file
df = pd.read_csv('./date/S2-useable-scenes-V2-2015-2024.csv')
datetime = df['Date']

# convert datetime to string
start_datetime = pd.to_datetime(datetime, format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
# end_datetime is 1 day after start_datetime
end_datetime = (pd.to_datetime(datetime, format='%d/%m/%Y') + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')

print(start_datetime)
print(end_datetime)
quit()

print(start_datetime)
quit()
# 2) Set date range
start_date = '2015-09-10'
end_date   = '2020-09-11'

# 3) Retrieve SENTINEL-2 SR (HARMONIZED), median composite
s2_col = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterDate(start_date, end_date)
      .filterBounds(region)
)
s2_image = s2_col.median().clip(region)

# 4) Retrieve MODIS AOD (MCD19A2.006), 'Optical_Depth_055'
modis_col = (
    ee.ImageCollection('MODIS/061/MCD19A2_GRANULES')
      .select('Optical_Depth_055')   # Only the AOD band at 550 nm
      .filterDate(start_date, end_date)
      .filterBounds(region)
)
# Take the mean over the date range, rename for clarity
modis_aod = modis_col.mean().multiply(0.001).rename('AOD_550').clip(region)

# 5) Merge Sentinel-2 and MODIS AOD into a single image
final_image = s2_image.addBands(modis_aod)

# Convert server-side region to a client-side object for export
region_coords = region.getInfo()['coordinates']

# 6) Export to Google Drive at 1000 m resolution
task = ee.batch.Export.image.toDrive(
    image           = final_image,
    description     = 'Sentinel2_MODIS_1km',
    folder          = 'EarthEngineExports',       # Adjust to your preferred Drive folder
    fileNamePrefix  = f'S2_MODIS_export_1km_{start_date}',
    region          = region_coords,
    scale           = 1000,                       # 1 km resolution
    crs             = 'EPSG:4326',
    maxPixels       = 1e13
)
task.start()

print("Export started. Check your GEE Tasks or Google Drive for the results.")
