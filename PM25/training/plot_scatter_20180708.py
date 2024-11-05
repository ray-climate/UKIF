#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    plot_density_20180708.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        03/11/2024 22:27

import numpy as np
import os
import matplotlib.pyplot as plt

# Define the data folder for testing data
test_data_folder = './data_preparation/testing_data_all_2018_rotate'
output_fig = './training_history_figs'
os.makedirs(output_fig, exist_ok=True)

# Get list of .npz files
npz_files = [os.path.join(test_data_folder, f) for f in os.listdir(test_data_folder) if f.endswith('.npz')]

# Initialize lists to store true and predicted PM2.5 values
pm25_true = []
pm25_predicted = []

# Iterate over each file and extract pm25 and pm25_predicted
for file in npz_files:
    # Load the data
    data = np.load(file)

    # Check if 'pm25_predicted' exists in the file
    if 'pm25_predicted' in data:
        pm25_true.append(float(data['pm25']))  # Convert to float
        pm25_predicted.append(float(data['pm25_predicted']))
    else:
        print(f"Predicted PM2.5 not found in file: {file}")

# Convert lists to numpy arrays
pm25_true = np.array(pm25_true)
pm25_predicted = np.array(pm25_predicted)

# Create density plot
plt.figure(figsize=(15, 12))
plt.hexbin(pm25_true, pm25_predicted, gridsize=50, cmap='RdBu_r')
plt.colorbar(extend='max', aspect=20, fraction=0.03, pad=0.03)
plt.xlabel('True PM2.5 [µg/m³]', fontsize=18)
plt.ylabel('Predicted PM2.5 [µg/m³]', fontsize=18)
plt.title('Density Plot: Predicted vs True PM2.5 (Testing Data)', fontsize=18)
plt.plot([0, 50], [0, 50], 'r--', linewidth=2)
plt.xlim([0, 50])
plt.ylim([0, 50])
# Increase x and y ticks font size
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(output_fig, 'density_predicted_vs_true_testing_v4_20180807.png'))

# Optionally, display the plot
# plt.show()
