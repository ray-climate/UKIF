#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    step-2.5-h5-to-csv.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        14/02/2025 15:55

# !/usr/bin/env python
# -*- coding:utf-8 -*-
import h5py
import csv
import sys
import os

# Get the job_id from the command line argument
job_id = sys.argv[1]
# job_id = '0'

# Define the directory where the predicted data is saved
save_dir = './data_chunk_new_predicted'
save_csv_dir = './data_chunk_new_predicted_csv'

# Define the output CSV file path
output_csv_path = f'{save_csv_dir}/chunk_{job_id}_predicted_pm25.csv'

# Open the HDF5 file for reading
with h5py.File(f'{save_dir}/chunk_{job_id}_predicted.h5', 'r') as f:
    # Open the CSV file for writing
    with open(output_csv_path, mode='w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the header row
        csv_writer.writerow(['patch_name', 'predicted_pm25'])

        # Loop through all the datasets (patches) in the file
        for patch_name in f.keys():
            print(f"Processing {patch_name}")
            # Read the predicted PM2.5 value
            predicted_pm25 = f[patch_name]['predicted_pm25'][()]

            # Write the patch_name and predicted_pm25 to the CSV file
            csv_writer.writerow([patch_name, predicted_pm25])

print(f"Data has been successfully saved to {output_csv_path}")
