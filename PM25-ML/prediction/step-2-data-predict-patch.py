#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    step-2-data-predict-patch.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        15/01/2025 17:03

import tensorflow as tf
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model('../pm25_model_pre_v0.h5')

# Define the data folder for testing data
test_data_folder = './data_patch'
save_dir = './data_patch_predicted'
os.makedirs(save_dir, exist_ok=True)

# Get list of .npz files
npz_files = [os.path.join(test_data_folder, f) for f in os.listdir(test_data_folder) if f.endswith('.npy')]
print(f'Found {len(npz_files)} .npz files in the folder')

# Iterate over each file and make predictions
i = 0
for file in npz_files:
    # Load the data
    data = np.load(file)

    # # Get the patch
    # patch = data['patch'].astype(np.float32)  # Convert to float32
    patch = np.transpose(data, (1, 2, 0))  # Shape becomes (128, 128, 13)
    patch = np.expand_dims(patch, axis=0)  # Shape becomes (1, 128, 128, 13)

    # Make prediction
    predicted_pm25 = model.predict(patch)
    predicted_pm25 = predicted_pm25[0][0]  # Extract scalar value
    print(predicted_pm25)

    data_dict = dict(data)
    data_dict['pm25_predicted'] = predicted_pm25
    data_dict['patch'] = patch

    # Save the data to the new directory
    save_file = os.path.join(save_dir, os.path.basename(file))
    np.save(save_file, **data_dict)
    print(f'Predicted the {i}th patch with PM2.5 value of {predicted_pm25}')
    i += 1
