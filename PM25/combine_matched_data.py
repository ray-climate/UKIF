#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    combine_matched_data.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        26/09/2024 22:05

import pandas as pd
import glob
import os

# Set the folder path where your CSV files are stored
folder_path = './matched_data'  # Change this to your folder path

# Get a list of all CSV files in the folder
all_csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Create an empty list to store DataFrames
df_list = []

# Loop through the list of CSV files and read each one
for file in all_csv_files:
    df = pd.read_csv(file)  # Read CSV file into a DataFrame
    df_list.append(df)      # Append the DataFrame to the list

# Concatenate all DataFrames in the list into one DataFrame
combined_df = pd.concat(df_list, ignore_index=True)

# drop rows with missing values in 'PM 2.5 value (ug m-3)' column
combined_df = combined_df.dropna(subset=['PM 2.5 value (ug m-3)'])

# Save the combined DataFrame to a new CSV file (optional)
combined_df.to_csv(os.path.join(folder_path, 'combined_matched_data.csv'), index=False)

# Display the combined DataFrame
print(combined_df)

