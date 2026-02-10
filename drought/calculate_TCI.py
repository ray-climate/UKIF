#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    generate_vci_map_with_plot.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        19/04/2025 16:38

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime
from osgeo import gdal
import numpy as np
import glob
import os

# Directory containing FAPAR GeoTIFF files
base_dir = "./data/Lincs_FAPAR"

# Month abbreviation to number mapping
month_map = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
    "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
}

def get_date_from_filename(filename):
    parts = os.path.basename(filename).split('_')
    year = parts[1]
    month_abbr = parts[2]
    month = month_map.get(month_abbr, "01")
    return datetime.strptime(f"{year}-{month}-01", "%Y-%m-%d"), month_abbr

def read_fapar_array(file_path):
    dataset = gdal.Open(file_path)
    if dataset is None:
        print(f"Error: Could not open {file_path}")
        return None, None
    band = dataset.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = -3.4028235e+38  # Default NoData value
    data = band.ReadAsArray()
    data = np.where(data == nodata_value, np.nan, data)
    return data, dataset

# Collect all August FAPAR files from 2016 to 2024
august_files = []
for year in range(2016, 2025):
    year_dir = os.path.join(base_dir, str(year))
    if os.path.exists(year_dir):
        files = glob.glob(os.path.join(year_dir, "Lincs_*_Aug_FAPAR_COG.tif"))
        if files:
            august_files.append(files[0])
    else:
        print(f"Directory {year_dir} does not exist, skipping.")

if len(august_files) < 2:
    print("Error: Insufficient August FAPAR files found to compute VCI.")
    exit(1)

print(f"Found {len(august_files)} August FAPAR files.")

# Read FAPAR arrays for all August files
fapar_arrays = []
for file in august_files:
    data, _ = read_fapar_array(file)
    if data is not None:
        fapar_arrays.append(data)

# Stack the arrays and compute FAPAR_min and FAPAR_max
fapar_stack = np.stack(fapar_arrays, axis=0)
fapar_min = np.nanmin(fapar_stack, axis=0)
fapar_max = np.nanmax(fapar_stack, axis=0)

# Find and read the August 2022 file
august_2022_file = None
for file in august_files:
    date, _ = get_date_from_filename(file)
    if date.year == 2022:
        august_2022_file = file
        break

if august_2022_file is None:
    print("Error: August 2022 FAPAR file not found.")
    exit(1)

print(f"August 2022 file: {august_2022_file}")
fapar_2022, dataset_2022 = read_fapar_array(august_2022_file)
if fapar_2022 is None:
    print("Error: Could not read August 2022 FAPAR data.")
    exit(1)

# Compute VCI
denominator = fapar_max - fapar_min
vci = np.where((denominator > 0) & (~np.isnan(fapar_2022)),
               (fapar_2022 - fapar_min) / denominator * 100, np.nan)

# Save VCI as GeoTIFF
driver = gdal.GetDriverByName('GTiff')
out_file = 'Lincs_VCI_2022_Aug.tif'
out_dataset = driver.Create(out_file, dataset_2022.RasterXSize, dataset_2022.RasterYSize, 1, gdal.GDT_Float32)
out_dataset.SetGeoTransform(dataset_2022.GetGeoTransform())
out_dataset.SetProjection(dataset_2022.GetProjection())
out_band = out_dataset.GetRasterBand(1)
out_band.SetNoDataValue(-9999)
vci_to_write = np.where(np.isnan(vci), -9999, vci)
out_band.WriteArray(vci_to_write)
out_dataset = None

print(f"Saved VCI map to {out_file}")

# Plot VCI with categorical colors
plt.figure(figsize=(10, 8))

# Define VCI ranges and colors
bounds = [0, 20, 35, 50, 100]
colors = ['red', 'orange', 'yellow', 'green']
labels = ['Severe Drought (<20)', 'Moderate Drought (20-35)', 'Mild Drought (35-50)', 'No Drought (≥50)']
cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, cmap.N)

# Plot VCI with colormap
im = plt.imshow(vci, cmap=cmap, norm=norm)

# Add colorbar with labels
cbar = plt.colorbar(im, ticks=[10, 27.5, 42.5, 75])
cbar.ax.set_yticklabels(labels)
cbar.set_label('VCI and Drought Category', fontsize=12)

plt.title('VCI Map for Lincolnshire - August 2022', fontsize=14)
plt.xlabel('Pixel X', fontsize=12)
plt.ylabel('Pixel Y', fontsize=12)

# Save plot
plot_file = './figures/Lincs_VCI_2022_Aug_plot_fapar.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved VCI plot to {plot_file}")