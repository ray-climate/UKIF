#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    plot_single_site.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        22/08/2024 15:43

import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the CSV file
file_path = './data/LAQN PM2.5/bexley_slade_green_2016_2017.csv'
data = pd.read_csv(file_path)

# Convert 'ReadingDateTime' to datetime format
data['ReadingDateTime'] = pd.to_datetime(data['ReadingDateTime'], format='%d/%m/%Y %H:%M')

# Drop rows where 'Value' is NaN
clean_data = data.dropna(subset=['Value'])

# Plotting the data
plt.figure(figsize=(10, 5))
plt.plot(clean_data['ReadingDateTime'], clean_data['Value'], label='PM2.5', marker='.', linestyle='-', markersize=2)
plt.title('PM2.5 Values Over Time')
plt.xlabel('Reading Date Time')
plt.ylabel('Value (ug m-3)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
