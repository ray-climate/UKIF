#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    step-2-model.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        02/08/2025 23:09

import pathlib
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import warnings

# --- 1. DEFINE FILE PATHS AND SETUP ---
LST_FOLDER = pathlib.Path("/Users/rs/Projects/UKIF/GEE-drought/LST_data")
NDVI_BASE_FOLDER = pathlib.Path("/Users/rs/Projects/UKIF/drought/data/Lincs_NDVI")

# Create output directories
OUTPUT_LST_FOLDER = pathlib.Path("./aligned_LST_10m")
OUTPUT_NDVI_ANOMALY_FOLDER = pathlib.Path("./NDVI_anomaly_10m")
OUTPUT_VHI_FOLDER = pathlib.Path("./VHI_10m")
OUTPUT_PLOTS_FOLDER = pathlib.Path("./plots_detailed")
PHENOLOGY_PLOTS_FOLDER = pathlib.Path("./phenology_plots")  # New folder for phenology plots
OUTPUT_LST_FOLDER.mkdir(exist_ok=True)
OUTPUT_NDVI_ANOMALY_FOLDER.mkdir(exist_ok=True)
OUTPUT_VHI_FOLDER.mkdir(exist_ok=True)
OUTPUT_PLOTS_FOLDER.mkdir(exist_ok=True)
PHENOLOGY_PLOTS_FOLDER.mkdir(exist_ok=True)  # Create the new directory

print("--- Configuration ---")
print(f"LST Data Path: {LST_FOLDER}")
print(f"NDVI Data Path: {NDVI_BASE_FOLDER}")

# --- 2. FIND AND PARSE ALL DATA FILES ---

MONTH_MAP = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
    'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}


def find_files(folder_path, pattern, name_parser):
    files = []
    for f in folder_path.glob(pattern):
        try:
            year, month = name_parser(f.name)
            files.append({'year': int(year), 'month': int(month), 'path': f})
        except (ValueError, KeyError):
            print(f"Could not parse date from: {f.name}")
    return files


def parse_lst_name(filename):
    parts = filename.replace('.tif', '').split('_')
    return parts[2], parts[3]


def parse_ndvi_name(filename):
    parts = filename.replace('_COG.tif', '').split('_')
    return parts[1], MONTH_MAP[parts[2]]


lst_files = find_files(LST_FOLDER, "VIIRSv2_LST_*.tif", parse_lst_name)
ndvi_files = find_files(NDVI_BASE_FOLDER, "**/Lincs_*_NDVI_COG.tif", parse_ndvi_name)

lst_df = pd.DataFrame(lst_files).rename(columns={'path': 'lst_path'})
ndvi_df = pd.DataFrame(ndvi_files).rename(columns={'path': 'ndvi_path'})

matched_df = pd.merge(lst_df, ndvi_df, on=['year', 'month']).sort_values(by=['year', 'month']).reset_index(drop=True)
print(f"\nFound {len(matched_df)} matching monthly pairs to process.")

# --- 3. LOAD ALL DATA AND MODEL PHENOLOGY ---
print("\n--- Loading all data and modeling pixel phenology ---")

# Load all NDVI data into a 3D stack (time, height, width)
all_ndvi_data = []
template_profile = None
for index, row in matched_df.iterrows():
    with rasterio.open(row['ndvi_path']) as src:
        if template_profile is None:
            template_profile = src.profile
        all_ndvi_data.append(src.read(1))
ndvi_stack = np.stack(all_ndvi_data, axis=0)

# Apply Savitzky-Golay filter to smooth the time-series for each pixel
# This creates the "expected" phenology curve
# A window length of 13 and polyorder of 3 is a good starting point for monthly data
# The filter needs to see at least a full cycle, so we pad the data
padded_ndvi = np.pad(ndvi_stack, ((6, 6), (0, 0), (0, 0)), mode='wrap')
smoothed_ndvi_padded = savgol_filter(padded_ndvi, window_length=13, polyorder=3, axis=0)
smoothed_ndvi = smoothed_ndvi_padded[6:-6, :, :]

# Calculate anomalies for the entire dataset
anomalies = ndvi_stack - smoothed_ndvi

# --- 3.5. VISUALIZE PHENOLOGY FOR A SAMPLE PIXEL ---
print("\n--- Generating a sample phenology plot ---")

# Select a pixel from the center of the image for visualization
center_y = ndvi_stack.shape[1] // 2
center_x = ndvi_stack.shape[2] // 2

# Extract the time series for this single pixel
sample_pixel_raw = ndvi_stack[:, center_y, center_x]
sample_pixel_smoothed = smoothed_ndvi[:, center_y, center_x]

# Create a proper time axis for plotting
time_axis = pd.to_datetime(matched_df['year'].astype(str) + '-' + matched_df['month'].astype(str))

# Create the plot
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(time_axis, sample_pixel_raw, 'o', label='Raw Monthly NDVI', markersize=4, alpha=0.6)
ax.plot(time_axis, sample_pixel_smoothed, '-', label='Modeled Phenology (Smoothed)', color='red', linewidth=2)

ax.set_title(f'Phenology Model for Sample Pixel (Row: {center_y}, Col: {center_x})', fontsize=16)
ax.set_xlabel('Date')
ax.set_ylabel('NDVI')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# Save the figure
plot_filename = PHENOLOGY_PLOTS_FOLDER / "sample_pixel_phenology.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Sample phenology plot saved to: {plot_filename}")

# --- 4. CALCULATE LONG-TERM EXTREMES ---
print("\n--- Calculating long-term min/max values for each month ---")

monthly_extremes = {}
for month_num in range(1, 13):
    month_indices = matched_df[matched_df['month'] == month_num].index
    if len(month_indices) == 0:
        continue

    # Align all LST data for this month
    lst_stack_month = []
    for i in month_indices:
        with rasterio.open(matched_df.loc[i, 'lst_path']) as lst_src:
            resampled_lst = np.empty((template_profile['height'], template_profile['width']), dtype='float32')
            reproject(
                source=rasterio.band(lst_src, 1), destination=resampled_lst,
                src_transform=lst_src.transform, src_crs=lst_src.crs,
                dst_transform=template_profile['transform'], dst_crs=template_profile['crs'],
                resampling=Resampling.bilinear
            )
            resampled_lst[resampled_lst <= 0] = np.nan
            lst_stack_month.append(resampled_lst)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        monthly_extremes[month_num] = {
            'anomaly_min': np.nanmin(anomalies[month_indices], axis=0),
            'anomaly_max': np.nanmax(anomalies[month_indices], axis=0),
            'lst_min': np.nanmin(np.stack(lst_stack_month), axis=0),
            'lst_max': np.nanmax(np.stack(lst_stack_month), axis=0),
        }

print("\n--- Phenology modeling complete. Starting main processing loop. ---")

# --- 5. PROCESS EACH MONTH: CALCULATE ANOMALY-BASED VHI AND PLOT ---

for index, row in matched_df.iterrows():
    date_key = f"{row['year']}-{row['month']:02d}"
    print(f"\nProcessing: {date_key}")

    # --- a. Get current data ---
    current_ndvi = ndvi_stack[index]
    current_anomaly = anomalies[index]

    with rasterio.open(row['lst_path']) as lst_src:
        current_lst_aligned = np.empty((template_profile['height'], template_profile['width']), dtype='float32')
        reproject(
            source=rasterio.band(lst_src, 1), destination=current_lst_aligned,
            src_transform=lst_src.transform, src_crs=lst_src.crs,
            dst_transform=template_profile['transform'], dst_crs=template_profile['crs'],
            resampling=Resampling.bilinear
        )

    # --- b. Calculate Anomaly-based VCI, TCI, and VHI ---
    extremes = monthly_extremes.get(row['month'])
    if not extremes:
        continue

    # VCI based on anomaly
    anomaly_range = extremes['anomaly_max'] - extremes['anomaly_min']
    vci = np.divide(100 * (current_anomaly - extremes['anomaly_min']), anomaly_range, where=anomaly_range != 0)

    # TCI (standard)
    lst_range = extremes['lst_max'] - extremes['lst_min']
    tci = np.divide(100 * (extremes['lst_max'] - current_lst_aligned), lst_range, where=lst_range != 0)

    vhi = 0.5 * vci + 0.5 * tci
    vhi = np.clip(vhi, 0, 100)

    # --- c. Mask and save outputs ---
    valid_mask = ~np.isnan(current_ndvi) & ~np.isnan(current_lst_aligned)
    output_nodata = -9999.0

    current_anomaly[~valid_mask] = output_nodata
    vhi[~valid_mask] = output_nodata

    output_profile = template_profile.copy()
    output_profile.update(count=1, dtype='float32', nodata=output_nodata)

    with rasterio.open(OUTPUT_NDVI_ANOMALY_FOLDER / f"NDVI_anomaly_{date_key}.tif", 'w', **output_profile) as dst:
        dst.write(current_anomaly.astype('float32'), 1)

    with rasterio.open(OUTPUT_VHI_FOLDER / f"VHI_{date_key}.tif", 'w', **output_profile) as dst:
        dst.write(vhi.astype('float32'), 1)

    # --- d. Generate detailed 2x2 plot ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f"Drought Analysis for {date_key}", fontsize=20)

    # Plot NDVI Anomaly
    im1 = axes[0, 0].imshow(current_anomaly, cmap='RdBu', vmin=-0.3, vmax=0.3)
    axes[0, 0].set_title("NDVI Anomaly", fontsize=14)
    fig.colorbar(im1, ax=axes[0, 0], label='Deviation from Normal')

    # Plot TCI
    tci_plot = tci.copy()
    tci_plot[~valid_mask] = np.nan
    im2 = axes[0, 1].imshow(tci_plot, cmap='plasma', vmin=0, vmax=100)
    axes[0, 1].set_title("Temperature Condition Index (TCI)", fontsize=14)
    fig.colorbar(im2, ax=axes[0, 1], label='Thermal Condition')

    # Plot VCI
    vci_plot = vci.copy()
    vci_plot[~valid_mask] = np.nan
    im3 = axes[1, 0].imshow(vci_plot, cmap='viridis', vmin=0, vmax=100)
    axes[1, 0].set_title("Vegetation Condition Index (VCI)", fontsize=14)
    fig.colorbar(im3, ax=axes[1, 0], label='Vegetation Condition')

    # Plot VHI
    vhi_plot = vhi.copy()
    vhi_plot[vhi == output_nodata] = np.nan
    im4 = axes[1, 1].imshow(vhi_plot, cmap='RdYlGn', vmin=0, vmax=100)
    axes[1, 1].set_title("Vegetation Health Index (VHI)", fontsize=14)
    fig.colorbar(im4, ax=axes[1, 1], label='Overall Health (0=Stress)')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_PLOTS_FOLDER / f"plot_detailed_{date_key}.png", dpi=150)
    plt.close(fig)

print("\n--- Processing complete! ---")
