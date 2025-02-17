#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    step-2-data-predict-patch.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        15/01/2025 17:03

import tensorflow as tf
import numpy as np
import h5py
import sys
import csv
import os

job_id = sys.argv[1]

# Load the trained model
model = tf.keras.models.load_model('../pm25_model_pre_v0.h5')

# Define the data folder for testing data
test_data_folder = './data_chunk_new'
save_dir = './data_chunk_new_predicted_csv'
os.makedirs(save_dir, exist_ok=True)

# Open the HDF5 file for reading
with h5py.File(f'{test_data_folder}/chunk_{job_id}.h5', 'r') as f:

    with open(f'{save_dir}/chunk_{job_id}_predictions.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(['patch_name', 'predicted_pm25'])

        # Loop through all the datasets (patches) in the file
        for patch_name in f.keys():
            # Read the patch data
            patch = f[patch_name][()]
            print('read patch:', patch_name, patch.shape)

            # patch = data['patch'].astype(np.float32)  # Convert to float32
            patch = np.transpose(patch, (1, 2, 0))  # Shape becomes (128, 128, 13)
            patch = np.expand_dims(patch, axis=0)  # Shape becomes (1, 128, 128, 13)

            # Make prediction
            predicted_pm25 = model.predict(patch)
            predicted_pm25 = predicted_pm25[0][0]  # Extract scalar value
            print(predicted_pm25)

            # Save the patch_name and predicted_pm25 to the CSV file
            csvwriter.writerow([patch_name, predicted_pm25])