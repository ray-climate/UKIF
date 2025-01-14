#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    plot_model_pre_v0.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        04/12/2024 17:50

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os

version = 'pre_v0'
# Load the saved model
model = load_model('./pm25_model_%s.h5' % version)

# Generate predictions for validation data
val_true = []
val_pred = []

for X_batch, y_batch in val_generator:
    y_pred_batch = model.predict(X_batch)
    val_true.extend(y_batch)
    val_pred.extend(y_pred_batch.flatten())

val_true = np.array(val_true)
val_pred = np.array(val_pred)

# Plot predicted vs true PM2.5 values for validation data
plt.figure(figsize=(12, 12))
plt.hexbin(val_true, val_pred, gridsize=50, bins='log', cmap='RdBu_r')
plt.colorbar(label='log10(N)')
plt.xlabel('MODIS PM2.5 [µg/m³]', fontsize=18)
plt.ylabel('Predicted PM2.5 [µg/m³]', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Predicted vs True PM2.5 (Validation Data)', fontsize=18)
plt.plot([val_true.min(), val_true.max()], [val_true.min(), val_true.max()], 'r--')
plt.savefig(os.path.join(output_fig, 'predicted_vs_true_density_val_%s.png' % version))

# Repeat for test data
test_true = []
test_pred = []

for X_batch, y_batch in test_generator:
    y_pred_batch = model.predict(X_batch)
    test_true.extend(y_batch)
    test_pred.extend(y_pred_batch.flatten())

test_true = np.array(test_true)
test_pred = np.array(test_pred)

plt.figure(figsize=(12, 12))
plt.hexbin(test_true, test_pred, gridsize=50, bins='log', cmap='RdBu_r')
plt.colorbar(label='log10(N)')
plt.xlabel('MODIS PM2.5 [µg/m³]', fontsize=18)
plt.ylabel('Predicted PM2.5 [µg/m³]', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Predicted vs True PM2.5 (Test Data)', fontsize=18)
plt.plot([test_true.min(), test_true.max()], [test_true.min(), test_true.max()], 'r--')
plt.savefig(os.path.join(output_fig, 'predicted_vs_true_density_test_%s.png' % version))

