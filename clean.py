import os
import numpy as np

DATA_DIR = "data/mouth_rois"
IMG_SIZE = 96

for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".npy"):
        continue
    path = os.path.join(DATA_DIR, fname)
    arr = np.load(path)
    if arr.shape[2:] != (IMG_SIZE, IMG_SIZE):
        print(f"[REMOVED] {fname} with shape {arr.shape}")
        os.remove(path)
