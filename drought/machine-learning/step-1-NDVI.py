#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    step-1-NDVI-only.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        09/06/2025 00:00

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    step-1-NDVI-only.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        09/06/2025 00:00

import os
import glob
import rasterio
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- STEP 1: Load all NDVI files ----------
root_dir = "/Users/rs/Projects/UKIF/drought/data/Lincs_NDVI/"
ndvi_dict = {}

for year in range(2016, 2025):
    year_dir = os.path.join(root_dir, str(year))
    tif_files = glob.glob(os.path.join(year_dir, "*NDVI.tif"))

    for tif_path in tif_files:
        try:
            fname = os.path.basename(tif_path)
            parts = fname.split("_")
            y = int(parts[1])
            m_str = parts[2]
            month = datetime.strptime(m_str, "%b").month
            date = datetime(y, month, 1)

            with rasterio.open(tif_path) as src:
                ndvi = src.read(1)
                nodata = src.nodata
                transform = src.transform
                meta = src.meta

            valid_mask = ndvi != nodata
            ndvi_clean = np.where(valid_mask, ndvi, np.nan)

            ndvi_dict[date] = {
                "ndvi": ndvi_clean,
                "transform": transform,
                "meta": meta
            }

            print(f"Loaded NDVI for {date.strftime('%Y-%m')}")

        except Exception as e:
            print(f"Error processing {tif_path}: {e}")

# ---------- STEP 2 (UPDATED): Compute VCI using month-specific climatology ----------
ndvi_by_month = {i: [] for i in range(1, 13)}
date_by_month = {i: [] for i in range(1, 13)}

for date in sorted(ndvi_dict.keys()):
    month = date.month
    ndvi_by_month[month].append(ndvi_dict[date]["ndvi"])
    date_by_month[month].append(date)

vci_dict = {}

print("Computing month-specific VCI...")
for month in tqdm(range(1, 13), desc="Month"):
    ndvi_list = ndvi_by_month[month]
    if len(ndvi_list) == 0:
        continue

    ndvi_stack_month = np.stack(ndvi_list, axis=0)
    ndvi_min = np.nanmin(ndvi_stack_month, axis=0)
    ndvi_max = np.nanmax(ndvi_stack_month, axis=0)
    denominator = ndvi_max - ndvi_min
    denominator[denominator == 0] = np.nan

    for i, date in enumerate(date_by_month[month]):
        ndvi_t = ndvi_stack_month[i]
        vci = (ndvi_t - ndvi_min) / denominator * 100
        vci_dict[date] = vci

# ---------- STEP 3: Binary Drought Mask ----------
vci_thresh = 35
binary_mask_dict = {}
for date, vci in vci_dict.items():
    binary_mask_dict[date] = np.where(vci < vci_thresh, 1, 0)

# ---------- STEP 4: Save figures with VCI + Binary Mask ----------
fig_dir = "./VCI_threshold_figures"
os.makedirs(fig_dir, exist_ok=True)

cmap_vci = "YlGn"
cmap_mask = "gray_r"
vmin_vci, vmax_vci = 0, 100

print("Saving VCI + binary mask figures...")
for date in sorted(vci_dict.keys()):
    vci = vci_dict[date]
    mask = binary_mask_dict[date]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(vci, cmap=cmap_vci, vmin=vmin_vci, vmax=vmax_vci)
    axes[0].set_title(f"VCI - {date.strftime('%Y-%m')}")
    axes[0].axis('off')
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("VCI")

    im2 = axes[1].imshow(mask, cmap=cmap_mask, vmin=0, vmax=1)
    axes[1].set_title("Drought Mask (VCI < 35)")
    axes[1].axis('off')
    cbar2 = fig.colorbar(im2, ax=axes[1], ticks=[0, 1], fraction=0.046, pad=0.04)
    cbar2.ax.set_yticklabels(['No Drought', 'Drought'])

    fname = f"VCI_mask_{date.strftime('%Y_%m')}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()

print(f"Saved all figures to: {fig_dir}")
