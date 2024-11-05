#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ml_prediction.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        03/11/2024 21:58

# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ml_predict_v3.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        03/11/2024 18:00

import numpy as np
import os
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('./pm25_model_v3.h5')

# Define the data folder for testing data
test_data_folder = './data_preparation/testing_data_all_2018_rotate'

# Get list of .npz files
npz_files = [os.path.join(test_data_folder, f) for f in os.listdir(test_data_folder) if f.endswith('.npz')]

print(f'Found {len(npz_files)} .npz files in the folder')

# Iterate over each file and make predictions
i = 0
for file in npz_files:
    # Load the data
    data = np.load(file)

    # Get the patch
    patch = data['patch'].astype(np.float32)  # Convert to float32

    # Prepare the data
    # Original shape: (13, 128, 128)
    # Required shape: (1, 128, 128, 13)
    patch = np.transpose(patch, (1, 2, 0))  # Shape becomes (128, 128, 13)
    patch = np.expand_dims(patch, axis=0)  # Shape becomes (1, 128, 128, 13)

    # Make prediction
    predicted_pm25 = model.predict(patch)
    predicted_pm25 = predicted_pm25[0][0]  # Extract scalar value

    # Add predicted value to the data
    # Convert data to a dictionary to modify
    data_dict = dict(data)
    data_dict['pm25_predicted'] = predicted_pm25

    # Save the data back to the same file (overwrite)
    np.savez(file, **data_dict)
    print(f'Predicted the {i}th patch with PM2.5 value of {predicted_pm25}')
    i += 1
