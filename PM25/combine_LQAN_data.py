#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    combine_LQAN_data.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        25/09/2024 23:39

import os
import pandas as pd

# Directory where the CSV files are stored
directory = './data/LAQN PM2.5/'  # Replace with the path to your directory
save_directory = './combined_LQAN'
os.makedirs(save_directory, exist_ok=True)

# Create an empty list to hold DataFrames
dfs = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        # Read each CSV file and append the DataFrame to the list
        df = pd.read_csv(file_path)
        print(f'Reading {filename}...')
        dfs.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined DataFrame into a new CSV file
combined_df.to_csv(os.path.join(save_directory, 'combined_LQAN_PM25_data.csv'), index=False)
