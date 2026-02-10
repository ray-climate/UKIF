#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    test.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        19/04/2025 16:38

import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

# Directory containing NDVI GeoTIFF files
base_dir = "./data/Lincs_NDVI"

# Month abbreviation to number mapping
month_map = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
    "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
}


def extract_mean_ndvi(file_path):
    """Extract mean NDVI from a GeoTIFF file, excluding NoData values."""
    dataset = gdal.Open(file_path)
    if dataset is None:
        print(f"Error: Could not open {file_path}")
        return None, None

    band = dataset.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = -3.4028235e+38  # Default NoData value

    data = band.ReadAsArray()
    valid_mask = data != nodata_value
    valid_data = data[valid_mask]

    mean_ndvi = np.mean(valid_data) if valid_data.size > 0 else np.nan

    dataset = None
    return mean_ndvi


def get_date_from_filename(filename):
    """Extract year and month from filename (e.g., Lincs_2016_Aug_NDVI.tif)."""
    parts = os.path.basename(filename).split('_')
    year = parts[1]  # e.g., 2016
    month_abbr = parts[2]  # e.g., Aug
    month = month_map.get(month_abbr, "01")  # Convert to number
    return datetime.strptime(f"{year}-{month}-01", "%Y-%m-%d")


# Collect NDVI data
ndvi_data = []
dates = []

# Iterate over years (2016 to 2024)
for year in range(2016, 2025):
    year_dir = os.path.join(base_dir, str(year))
    if not os.path.exists(year_dir):
        print(f"Directory {year_dir} does not exist, skipping.")
        continue

    # Find all NDVI files in the year directory
    files = glob.glob(os.path.join(year_dir, "Lincs_*NDVI.tif"))
    for file_path in files:
        mean_ndvi = extract_mean_ndvi(file_path)
        if mean_ndvi is not None:
            date = get_date_from_filename(file_path)
            ndvi_data.append(mean_ndvi)
            dates.append(date)
            print(f"Processed {file_path}: Mean NDVI = {mean_ndvi:.3f}")

# Sort data by date
sorted_pairs = sorted(zip(dates, ndvi_data))
dates, ndvi_data = zip(*sorted_pairs)

# Plot time series
plt.figure(figsize=(12, 6))
plt.plot(dates, ndvi_data, marker='o', linestyle='-', color='purple')
plt.title("Mean NDVI Lincolnshire (2016–2024)", fontsize=18)
plt.xlabel("Date", fontsize=18)
plt.ylabel("Mean NDVI", fontsize=18)
plt.grid(False)
plt.xticks(rotation=45)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plot a horizontal line at y=0.78
# plt.axhline(y=0.78, color='red', linestyle='--')
# set xlimt between 2013-01-01 and 2025-01-01
plt.xlim(datetime(2016, 1, 1), datetime(2025, 1, 1))
plt.tight_layout()

# Save the plot
plt.savefig("ndvi_time_series.png")