#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    match_sentinel2_LQAN.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        25/09/2024 23:49

import pandas as pd
import os

output_dir = './matched_data'
os.makedirs(output_dir, exist_ok=True)

LQAN_file = './combined_LQAN/combined_LQAN_PM25_data.csv'
LQAN_df = pd.read_csv(LQAN_file)

# Convert ReadingDateTime to datetime
LQAN_df['ReadingDateTime'] = pd.to_datetime(LQAN_df['ReadingDateTime'], format='%d/%m/%Y %H:%M')


for file in os.listdir('./S2L1C_data'):

    if file.endswith('.csv'):

        S2L1C_file = os.path.join('./S2L1C_data', file)
        S2L1C_df = pd.read_csv(S2L1C_file)

        # Filter rows in S2L1C_df where all B1 to B12 columns are present (no missing values)
        filtered_S2L1C_df = S2L1C_df.dropna(subset=[f'B{i}' for i in range(1, 13)])

        # Extract the site name from the file name, assuming it is the first 3 characters
        filtered_S2L1C_df['site_name'] = S2L1C_file.split('/')[-1][:3]

        # Convert 'date' in S2L1C_df to datetime
        filtered_S2L1C_df['date'] = pd.to_datetime(filtered_S2L1C_df['date'], format='%Y-%m-%d %H:%M:%S')

        # Function to match data within ±1 hour of the ReadingDateTime
        def match_pm25(row, pm25_data):
            site_data = pm25_data[pm25_data['Site'] == row['site_name']]
            matched_data = site_data[(site_data['ReadingDateTime'] >= row['date'] - pd.Timedelta(hours=1)) &
                                     (site_data['ReadingDateTime'] <= row['date'] + pd.Timedelta(hours=1))]
            print(f'Matched data for {row["site_name"]} at {row["date"]}... ...')
            if not matched_data.empty:
                return matched_data['Value'].mean()
            else:
                return None

        # Apply the matching function to each row in the filtered BL0_S2L1C_TimeSeries
        filtered_S2L1C_df['PM 2.5 value (ug m-3)'] = filtered_S2L1C_df.apply(match_pm25, axis=1, pm25_data=LQAN_df)
        # Sort the DataFrame by 'system:index' and 'date' for grouping and identifying duplicates
        # Extract only the date part (without time)
        filtered_S2L1C_df['date_only'] = filtered_S2L1C_df['date'].dt.date
        print('filtered_S2L1C_df:', filtered_S2L1C_df)
        # Remove duplicate rows based on the 'date_only' columns, keeping the first occurrence
        cleaned_df = filtered_S2L1C_df.drop_duplicates(subset='date_only', keep='first')
        print('cleaned_df:', cleaned_df)
        # Drop the 'date_only' column as it was used for identifying duplicates
        cleaned_df = cleaned_df.drop(columns='date_only')

        # Save the matched data to a new CSV file
        cleaned_df.to_csv(os.path.join(output_dir, f'{file[:-4]}_matched.csv'), index=False)
