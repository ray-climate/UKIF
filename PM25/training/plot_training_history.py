#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    plot_training_history.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        01/11/2024 16:35

import os
import matplotlib.pyplot as plt
import pickle
# import json  # Uncomment if you used JSON to save history

# Path where the training history is saved
history_path = './pm25_model.h5'

# Output directory for saving the plot
output_fig = './training_figs'
os.makedirs(output_fig, exist_ok=True)

# Load the training history
# If you saved it using pickle
with open(history_path, 'rb') as f:
    history = pickle.load(f)

# Plot training and validation loss
plt.figure()
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_fig, 'training_validation_loss.png'))
plt.show()

