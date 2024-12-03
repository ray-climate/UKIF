#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ml_train_v1.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        03/12/2024 10:55

import numpy as np
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import json  # Add this line to import json module

# Define the data folder
data_folder = './data_preprocess/training_data_2018-2020'
output_fig = './training_history_figs'
os.makedirs(output_fig, exist_ok=True)
version = 'v3'

# Get list of .npz files
npz_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npz')]
print(len(npz_files))
# randon keep 50% of the data
npz_files = np.random.choice(npz_files, int(len(npz_files) * 0.5), replace=False)
print(len(npz_files))

# Split the data into training and validation sets
train_files, val_files = train_test_split(npz_files, test_size=0.2, random_state=42)

# Define the data generator
class DataGenerator(Sequence):
    def __init__(self, list_files, batch_size=32, shuffle=True):
        self.list_files = list_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.list_files) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.list_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_patches = []
        batch_pm25 = []
        for file in batch_files:
            data = np.load(file)
            patch = data['patch'].astype(np.float32)  # Convert patch to float to support NaN values
            pm25 = data['pm25']

            batch_patches.append(patch)
            batch_pm25.append(pm25)

        X = np.array(batch_patches)
        X = np.transpose(X, (0, 2, 3, 1))  # Rearrange dimensions to (batch_size, 128, 128, 13)
        y = np.array(batch_pm25)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.list_files)

# Create data generators
batch_size = 32
train_generator = DataGenerator(train_files, batch_size=batch_size, shuffle=True)
val_generator = DataGenerator(val_files, batch_size=batch_size, shuffle=False)

# Build the CNN model
input_shape = (128, 128, 13)
# inputs = Input(shape=input_shape)
#
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
# x = MaxPooling2D((2, 2))(x)
# x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2))(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2))(x)
# x = Flatten()(x)
# x = Dense(128, activation='relu')(x)
# outputs = Dense(1)(x)  # Output is a single value
#
# model = Model(inputs=inputs, outputs=outputs)
# model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
#
# # Train the model
# history = model.fit(train_generator, validation_data=val_generator, epochs=100)
# Load pretrained VGG16 model without top layers
base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)

# Build the modified model
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)  # Dense layer after VGG16 for feature learning
outputs = Dense(1)(x)  # Final output layer for regression

model = Model(inputs=base_model.input, outputs=outputs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])

# Print model summary for verification
model.summary()

# Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=100)

model.save('./pm25_model_%s.h5' % version)

# Save training history to a JSON file
with open('training_history_%s.json' % version, 'w') as f:
    json.dump(history.history, f)

# Plot training and validation loss
plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.savefig(os.path.join(output_fig, 'training_validation_loss_%s.png' % version))

# Evaluate the model on validation data
val_true = []
val_pred = []

for X_batch, y_batch in val_generator:
    y_pred_batch = model.predict(X_batch)
    val_true.extend(y_batch)
    val_pred.extend(y_pred_batch.flatten())

val_true = np.array(val_true)
val_pred = np.array(val_pred)

# Compute evaluation metrics
rmse = np.sqrt(mean_squared_error(val_true, val_pred))
mae = mean_absolute_error(val_true, val_pred)

print(f'Validation RMSE: {rmse}')
print(f'Validation MAE: {mae}')

# Plot predicted vs true PM2.5 values
plt.figure(figsize=(12, 12))
plt.scatter(val_true, val_pred, alpha=0.5)
plt.xlabel('True PM2.5 [µg/m³]', fontsize=18)
plt.ylabel('Predicted PM2.5 [µg/m³]', fontsize=18)
plt.title('Predicted vs True PM2.5', fontsize=18)
plt.plot([val_true.min(), val_true.max()], [val_true.min(), val_true.max()], 'r--')
plt.savefig(os.path.join(output_fig, 'predicted_vs_true_%s.png' % version))
