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

# Get list of .npz files
npz_files = [os.path.join(test_data_folder, f) for f in os.listdir(test_data_folder) if f.endswith('.npy')]
print(f'Found {len(npz_files)} .npz files in the folder')