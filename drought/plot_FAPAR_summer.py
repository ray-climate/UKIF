#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    test.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        19/04/2025 16:38

import matplotlib.pyplot as plt
from datetime import datetime
from osgeo import gdal
import numpy as np
import glob
import os

# Directory containing NDVI GeoTIFF files
base_dir = "./data/Lincs_NDVI"

# Month abbreviation to number mapping
month_map = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
    "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
}

def extract_mean_ndvi(file_path):
    dataset = gdal.Open(file_path)
    if dataset is None:
        print(f"Error: Could not open {file_path}")
        return None

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
    parts = os.path.basename(filename).split('_')
    year = parts[1]
    month_abbr = parts[2]
    month = month_map.get(month_abbr, "01")
    return datetime.strptime(f"{year}-{month}-01", "%Y-%m-%d"), month_abbr

# Initialize container for all 12 months
monthly_data = {abbr: {"years": [], "ndvi": []} for abbr in month_map.keys()}

# Iterate through files and group by month
for year in range(2016, 2025):
    year_dir = os.path.join(base_dir, str(year))
    if not os.path.exists(year_dir):
        print(f"Directory {year_dir} does not exist, skipping.")
        continue

    files = glob.glob(os.path.join(year_dir, "Lincs_*NDVI.tif"))
    for file_path in files:
        mean_ndvi = extract_mean_ndvi(file_path)
        if mean_ndvi is not None:
            date, month_abbr = get_date_from_filename(file_path)
            if month_abbr in monthly_data:
                monthly_data[month_abbr]["years"].append(date.year)
                monthly_data[month_abbr]["ndvi"].append(mean_ndvi)
                print(f"Processed {file_path}: Mean NDVI = {mean_ndvi:.3f}")

# Plot each month separately
for month_abbr in month_map.keys():
    years = monthly_data[month_abbr]["years"]
    ndvi = monthly_data[month_abbr]["ndvi"]

    if not years:
        print(f"No data for {month_abbr}, skipping plot.")
        continue

    # Sort by year
    sorted_pairs = sorted(zip(years, ndvi))
    sorted_years, sorted_ndvi = zip(*sorted_pairs)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_years, sorted_ndvi, marker='o', linestyle='-', color='purple')
    plt.title(f"{month_abbr} Mean NDVI Lincolnshire (2016–2024)", fontsize=18)
    plt.xlabel("Year", fontsize=18)
    plt.ylabel("Mean NDVI", fontsize=18)
    plt.ylim(0.2, 0.9)
    # plt.axhline(y=0.78, color='red', linestyle='--')
    plt.grid(False)

    # Year-based x-axis ticks
    plt.xticks(range(2016, 2025), fontsize=18)
    plt.yticks(fontsize=18)

    plt.xlim(2015.5, 2024.5)
    plt.tight_layout()

    # Save plot
    out_filename = f"ndvi_{month_abbr}_time_series.png"
    plt.savefig(out_filename)
    plt.close()
    print(f"Saved plot: {out_filename}")
