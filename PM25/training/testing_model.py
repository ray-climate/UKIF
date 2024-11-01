#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ml_predict_v2.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        01/11/2024 02:30

import numpy as np
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
from tqdm import tqdm  # For progress bar

def masked_mse(y_true, y_pred):
    mask = tf.math.is_nan(y_true)
    y_true = tf.where(mask, 0.0, y_true)
    y_pred = tf.where(mask, 0.0, y_pred)
    count = tf.reduce_sum(tf.cast(~mask, tf.float32))
    mse = tf.reduce_sum(tf.square(y_true - y_pred)) / count
    return mse

# Load the trained model
model = load_model('pm25_model.h5', custom_objects={'masked_mse': masked_mse})

# Load the test data
test_data_path = './training/testing_data/Sentinel2_L1C_20180807_CloudMasked.npz'  # Replace with your test data path
data = np.load(test_data_path)
test_data = data['data']  # Replace 'test_data' with the actual key in your npz file

print(f'Test data shape: {test_data.shape}')  # Should be (13, 5443, 10122)

# Transpose test data to (5443, 10122, 13) for easier indexing
test_data = np.transpose(test_data, (1, 2, 0))

# Define patch size and stride
patch_size = 128
stride = 16  # You can adjust this value
height, width, channels = test_data.shape

# Prepare empty array to hold predictions
predictions = np.full((height, width), np.nan)

# Calculate the number of steps
steps_y = (height - patch_size) // stride + 1
steps_x = (width - patch_size) // stride + 1

# Batch size for prediction
batch_size = 32
patches = []
positions = []

# Collect patches and their positions
for y in tqdm(range(0, height - patch_size + 1, stride), desc="Collecting patches"):
    for x in range(0, width - patch_size + 1, stride):
        patch = test_data[y:y+patch_size, x:x+patch_size, :]
        # Check for NaNs or invalid data if necessary
        patches.append(patch)
        positions.append((y + patch_size // 2, x + patch_size // 2))  # Center position

# Convert patches to numpy array
patches = np.array(patches)
print(f'Number of patches: {len(patches)}')
print(f'Patches shape: {patches.shape}')  # Should be (num_patches, 128, 128, 13)

# Predict in batches
num_patches = len(patches)
num_batches = int(np.ceil(num_patches / batch_size))
predictions_list = []

for i in tqdm(range(num_batches), desc="Predicting"):
    batch_patches = patches[i*batch_size:(i+1)*batch_size]
    batch_predictions = model.predict(batch_patches)
    predictions_list.extend(batch_predictions.flatten())

# Assign predictions to their positions
for idx, (y, x) in enumerate(positions):
    predictions[y, x] = predictions_list[idx]

# Optionally, interpolate predictions to fill in the gaps if stride > 1
# This step can be added based on your requirements

# Save the predictions
np.save('predictions.npy', predictions)
print('Predictions saved to predictions.npy')
