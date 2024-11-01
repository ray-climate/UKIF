#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    testing_model.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        01/11/2024 17:39

import numpy as np
import os
from tensorflow.keras.models import load_model
import tensorflow as tf

def masked_mse(y_true, y_pred):
    mask = tf.math.is_nan(y_true)  # Create a mask for NaN values in y_true
    y_true = tf.where(mask, 0.0, y_true)  # Replace NaNs in y_true with zero for computation
    y_pred = tf.where(mask, 0.0, y_pred)  # Replace corresponding predictions with zero

    # Count valid entries (non-NaNs)
    count = tf.reduce_sum(tf.cast(~mask, tf.float32))
    mse = tf.reduce_sum(tf.square(y_true - y_pred)) / count  # MSE on valid entries only
    return mse

# Load the trained model
model = load_model('pm25_model.h5', custom_objects={'masked_mse': masked_mse})

# Load the test data
test_data_path = './data_preparation/testing_data/Sentinel2_L1C_20180807_CloudMasked.npz'
data = np.load(test_data_path)

test_data = data['data']  # Replace 'test_data' with the actual key in your npz file

print(f'Test data shape: {test_data.shape}')

# Expected input shape for the model
input_shape = model.input_shape  # (None, 128, 128, 13)
expected_shape = input_shape[1:]  # (128, 128, 13)

# Check if test data needs to be reshaped or processed
if test_data.shape[-1] != expected_shape[-1]:
    print("The number of channels in test data does not match the model's expected input channels.")
    # Handle this case accordingly
    exit()

# Process the test data to match the input shape
# Assuming we can reshape the test data into patches of size (128, 128, 13)
# This may involve sliding windows or other methods depending on your data context

# Example of reshaping (this is a placeholder and may not work without proper context)
# Let's assume we can reshape (5000, 1000, 13) into (-1, 128, 128, 13)
# You will need to adjust this according to how your test data can be split into patches
try:
    num_samples = test_data.shape[0] * test_data.shape[1] // (128 * 128)
    test_data_reshaped = test_data.reshape((num_samples, 128, 128, 13))
except ValueError:
    print("Cannot reshape test data to match the model's expected input shape.")
    exit()

print(f'Reshaped test data shape: {test_data_reshaped.shape}')

# Make predictions
predictions = model.predict(test_data_reshaped)

# Reshape predictions back to original dimensions if necessary
# For example, if you want to get predictions of shape (5000, 1000)
predictions = predictions.reshape((5000, 1000))

print(f'Predictions shape: {predictions.shape}')

# Save the predictions
np.save('predictions.npy', predictions)
