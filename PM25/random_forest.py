#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    random_forest.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        26/09/2024 22:08

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
# Load the new data
for file in os.listdir('./matched_data'):
    if file.endswith('.csv'):

        print(f"Processing file: {file}")
        file_path_new = './matched_data/' + file
        data_new = pd.read_csv(file_path_new)

        # Drop rows with NaN values in the target variable
        data_clean_new = data_new.dropna(subset=['PM 2.5 value (ug m-3)'])

        # Preparing the data
        X_new = data_clean_new[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']]
        y_new = data_clean_new['PM 2.5 value (ug m-3)']

        # Splitting the data into training and validation sets
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

        # Creating and training the Random Forest model
        rf_new = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_new.fit(X_train_new, y_train_new)

        # Predicting on the validation set
        y_pred_new = rf_new.predict(X_test_new)

        # Calculating R² and RMSE
        r2_new = r2_score(y_test_new, y_pred_new)
        rmse_new = np.sqrt(mean_squared_error(y_test_new, y_pred_new))

        # Displaying the statistics
        print("R-squared (R²) with new data:", r2_new)
        print("Root Mean Square Error (RMSE) with new data:", rmse_new)
