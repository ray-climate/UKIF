#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    step-2-data-predict-patch.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        15/01/2025 17:03

import tensorflow as tf
import numpy as np
import h5py
import os

job_id = 0

# Load the trained model
model = tf.keras.models.load_model('../pm25_model_pre_v0.h5')

# Define the data folder for testing data
test_data_folder = './data_chunk_new'
save_dir = './data_chunk_new_predicted'
os.makedirs(save_dir, exist_ok=True)

# Open the HDF5 file for reading
with h5py.File(f'{test_data_folder}/chunk_{job_id}.h5', 'r') as f:
    # Create a new HDF5 file for saving predicted data
    with h5py.File(f'{save_dir}/chunk_{job_id}_predicted.h5', 'w') as f_out:
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

            # Save the patch and predicted PM2.5 to the new HDF5 file
            f_out.create_dataset(f'{patch_name}/patch', data=patch[0])  # Save the original patch
            f_out.create_dataset(f'{patch_name}/predicted_pm25', data=predicted_pm25)  # Save the predicted PM2.5

print(f"Predictions saved to {save_dir}/chunk_{job_id}_predicted.h5")

quit()
# Get list of .npz files
npz_files = [os.path.join(test_data_folder, f) for f in os.listdir(test_data_folder) if f.endswith('.npy')]
print(f'Found {len(npz_files)} .npz files in the folder')

# Iterate over each file and make predictions
i = 0
for file in npz_files:
    # Load the data
    data = np.load(file)

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
