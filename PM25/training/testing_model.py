#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ml_predict_v2.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        01/11/2024 02:30

import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm  # For progress bar

# Load the data
data = np.load('./data_preparation/testing_data/Sentinel2_L1C_20180807_CloudMasked.npz')['data']  # Replace 'your_array' with the actual key in your npz file

# Ensure data has shape (5443, 10122, 13)

# Define the patch size
patch_height = 128
patch_width = 128
channels = 13

# Compute the number of patches
n_patches_h = data.shape[1] // patch_height
n_patches_w = data.shape[2] // patch_width

total_patches = n_patches_h * n_patches_w

print('Number of patches along height:', n_patches_h)
print('Number of patches along width:', n_patches_w)
print('Total number of patches:', total_patches)

# Load the trained model
model = load_model('./pm25_model_v2.h5')

# Prepare to store results
pm25_predictions = np.zeros((n_patches_h, n_patches_w))

batch_size = 32
batch_patches = []
batch_indices = []
processed_patches = 0

# Initialize progress bar
pbar = tqdm(total=total_patches, desc='Processing Patches')

for i in range(n_patches_h):
    for j in range(n_patches_w):
        # Extract the patch
        patch = data[:, i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
        patch = np.transpose(patch, (1, 2, 0))
        # Append to batch
        batch_patches.append(patch)
        batch_indices.append((i, j))
        # When batch is full, predict
        if len(batch_patches) == batch_size:
            batch_patches_array = np.array(batch_patches)
            pm25_preds = model.predict(batch_patches_array)
            for idx, pm25_pred in enumerate(pm25_preds):
                i_idx, j_idx = batch_indices[idx]
                pm25_predictions[i_idx, j_idx] = pm25_pred[0]
            # Update progress bar
            processed_patches += len(batch_patches)
            pbar.update(len(batch_patches))
            # Reset batch
            batch_patches = []
            batch_indices = []

# Process remaining patches
if len(batch_patches) > 0:
    batch_patches_array = np.array(batch_patches)
    pm25_preds = model.predict(batch_patches_array)
    for idx, pm25_pred in enumerate(pm25_preds):
        i_idx, j_idx = batch_indices[idx]
        pm25_predictions[i_idx, j_idx] = pm25_pred[0]
    # Update progress bar
    processed_patches += len(batch_patches)
    pbar.update(len(batch_patches))

pbar.close()
print('Prediction completed.')
