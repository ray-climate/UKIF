#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    step-2-flatten_inputs_for_ml.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        04/08/2025 23:43

import os
import numpy as np

# ------------------- 1. Paths -------------------
base_dir = './prepared_npy'
X_2020 = np.load(os.path.join(base_dir, 'X_2020.npy'))  # (7, H, W, 3)
X_2022 = np.load(os.path.join(base_dir, 'X_2022.npy'))  # (7, H, W, 3)
y_2020 = np.load(os.path.join(base_dir, 'y_2020.npy'))  # (H, W)
y_2022 = np.load(os.path.join(base_dir, 'y_2022.npy'))  # (H, W)

# ------------------- 2. Flatten Inputs -------------------
def flatten_inputs(X, y):
    T, H, W, C = X.shape  # e.g. (7, H, W, 3)
    X_flat = X.transpose(1, 2, 0, 3).reshape(H * W, T * C)  # shape: (H*W, 21)
    y_flat = y.flatten()  # shape: (H*W,)
    return X_flat, y_flat

X1, y1 = flatten_inputs(X_2020, y_2020)
X2, y2 = flatten_inputs(X_2022, y_2022)

# ------------------- 3. Combine -------------------
X_all = np.concatenate([X1, X2], axis=0)
y_all = np.concatenate([y1, y2], axis=0)
print(f"[INFO] Combined shape: X={X_all.shape}, y={y_all.shape}")

# ------------------- 4. Remove rows with any NaNs -------------------
valid_mask = np.isfinite(y_all) & np.all(np.isfinite(X_all), axis=1)
X_clean = X_all[valid_mask]
y_reg = y_all[valid_mask]  # regression target

# ------------------- 5. Classification Label: SPEI < -1 → 1 (drought), else 0
y_cls = (y_reg < -1).astype(int)

print(f"[CLEANED] X={X_clean.shape}, y_reg={y_reg.shape}, y_cls={y_cls.shape}")
print(f"[DROUGHT COUNT] Class 1 = {np.sum(y_cls)}, Class 0 = {len(y_cls) - np.sum(y_cls)}")

# ------------------- 6. Save Output -------------------
np.save(os.path.join(base_dir, 'X.npy'), X_clean)
np.save(os.path.join(base_dir, 'y_reg.npy'), y_reg)
np.save(os.path.join(base_dir, 'y_cls.npy'), y_cls)
print("[SAVED] All ML-ready files written to disk.")
