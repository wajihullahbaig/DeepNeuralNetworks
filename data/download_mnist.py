# -*- coding: utf-8 -*-
"""
Created on 18th November 2026
Updated on 18th November 2026
@author: Lenovo Laptop

Download the official MNIST dataset (IDX format) and cache it as a unified
compressed NumPy archive (data/mnist.npz) without MATLAB dependencies.
"""
import argparse
import gzip
import os
import struct
import urllib.request
import numpy as np


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NPZ_PATH = os.path.join(DATA_DIR, "mnist.npz")

MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _fetch_bytes(filename, timeout=30):
    last_err = None
    for base_url in MIRRORS:
        url = base_url + filename
        try:
            print(f"Downloading {url} ...")
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return gzip.decompress(resp.read())
        except Exception as exc:
            last_err = exc
            print(f"Mirror failed ({exc}); trying next mirror...")
    raise RuntimeError(f"Failed to download {filename} from all mirrors: {last_err}")


def _parse_idx_images(raw_bytes):
    magic, num_images, rows, cols = struct.unpack(">IIII", raw_bytes[:16])
    if magic != 2051:
        raise ValueError(f"Invalid MNIST image magic number: {magic}")
    arr = np.frombuffer(raw_bytes[16:], dtype=np.uint8)
    return arr.reshape(num_images, rows * cols).copy()


def _parse_idx_labels_onehot(raw_bytes, num_classes=10):
    magic, num_labels = struct.unpack(">II", raw_bytes[:8])
    if magic != 2049:
        raise ValueError(f"Invalid MNIST label magic number: {magic}")
    labels = np.frombuffer(raw_bytes[8:], dtype=np.uint8)
    onehot = np.zeros((num_labels, num_classes), dtype=np.uint8)
    onehot[np.arange(num_labels), labels] = 1
    return onehot


def download_mnist(output_path=DEFAULT_NPZ_PATH, force=False):
    """Download MNIST IDX files and save unified data/mnist.npz."""
    if os.path.exists(output_path) and not force:
        print(f"MNIST archive already exists at: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_x = _parse_idx_images(_fetch_bytes(FILES["train_images"]))
    train_y = _parse_idx_labels_onehot(_fetch_bytes(FILES["train_labels"]))
    test_x = _parse_idx_images(_fetch_bytes(FILES["test_images"]))
    test_y = _parse_idx_labels_onehot(_fetch_bytes(FILES["test_labels"]))

    np.savez_compressed(
        output_path,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
    )
    print(f"Saved unified MNIST dataset to {output_path}")
    for key, arr in [("train_x", train_x), ("train_y", train_y), ("test_x", test_x), ("test_y", test_y)]:
        print(f"  {key}: {arr.dtype} {arr.shape}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download MNIST dataset into unified data/mnist.npz.")
    parser.add_argument("--force", action="store_true", help="Re-download even if mnist.npz already exists.")
    parser.add_argument("--output", default=DEFAULT_NPZ_PATH, help="Output .npz path.")
    args = parser.parse_args()
    download_mnist(output_path=args.output, force=args.force)


if __name__ == "__main__":
    main()
