#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    ml_train.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        30/10/2024 00:47

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input, Lambda, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Custom masking layer to remove zero-band pixels
def mask_zero_pixels(x):
    # Create a mask where any pixel with all zero bands is masked out
    mask = tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True) > 0  # shape (batch, 128, 128, 1)
    mask = tf.cast(mask, x.dtype)
    return x * mask  # Apply mask by element-wise multiplication

# Define the data loading function
def load_data(data_dir):
    patches = []
    pm25_values = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.npz'):
            data = np.load(os.path.join(data_dir, file_name))
            patches.append(data['patch'])
            pm25_values.append(data['pm25'])
    X = np.array(patches)
    y = np.array(pm25_values)
    return X, y

# Load and preprocess the data
data_dir = './data_preparation/training_data'
output_fig = './training_figs'
os.makedirs(output_fig, exist_ok=True)

X, y = load_data(data_dir)

# Reshape and normalize the input data
X = np.moveaxis(X, 1, -1)  # Change shape to (samples, 128, 128, 13)
X = X / 1000.0  # Scale pixel values to [0, 1] range

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
input_shape = (128, 128, 13)
inputs = Input(shape=input_shape)
masked_inputs = Lambda(mask_zero_pixels)(inputs)  # Apply the custom masking layer

# Use VGG16 as the backbone
base_model = VGG16(weights=None, include_top=False, input_tensor=masked_inputs)

# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='linear')(x)  # Linear activation for regression output

model = Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Model summary
model.summary()

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MSE: {test_loss}, Test MAE: {test_mae}")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.title('Training and Validation MAE')

plt.savefig(os.path.join(output_fig, 'training_history.png'))
