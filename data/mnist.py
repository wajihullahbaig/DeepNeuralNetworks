# -*- coding: utf-8 -*-
"""
Created on 18th November 2026
Updated on 18th November 2026
@author: Lenovo Laptop

Unified MNIST dataset loader backed by data/mnist.npz (pure Python/NumPy).
"""
import os
import numpy as np
from data.download_mnist import DEFAULT_NPZ_PATH, download_mnist


def load_mnist(npz_path=DEFAULT_NPZ_PATH, auto_download=True):
    """
    Load the unified MNIST dataset from data/mnist.npz.
    Returns a dict with keys: train_x, train_y, test_x, test_y (all uint8 arrays).
    """
    if not os.path.exists(npz_path):
        if not auto_download:
            raise FileNotFoundError(
                f"MNIST archive not found at {npz_path}. Run 'python data/download_mnist.py' first."
            )
        download_mnist(output_path=npz_path)

    print(f"Reading file: {npz_path}")
    with np.load(npz_path) as data:
        dataset = {key: data[key] for key in ("train_x", "train_y", "test_x", "test_y")}
    print("Successfully loaded unified MNIST dataset")
    for k, arr in dataset.items():
        print(f"{k} {arr.dtype.name} {arr.shape}")
    return dataset
