# -*- coding: utf-8 -*-
"""
Created on 18th November 2026
Updated on 18th November 2026
@author: Lenovo Laptop

Run the complete end-to-end pipeline:
1. Ensure the unified MNIST dataset (data/mnist.npz) is downloaded.
2. Execute all 5 model verification suites (NN, CNN, DBN, SAE, CAE).
3. Generate all 5 animated training GIFs and summary plots in output/.
"""
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.environ["DNN_NONINTERACTIVE"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")

from data import download_mnist
from tests.test_example_NN import test_example_NN
from tests.test_example_CNN import test_example_CNN
from tests.test_example_DBN import test_example_DBN
from tests.test_example_SAE import test_example_SAE
from tests.test_example_CAE import test_example_CAE
from scripts.generate_training_gifs import main as generate_gifs


def main():
    t0 = time.time()
    print("=== Step 1: Ensuring unified MNIST dataset ===")
    download_mnist.download_mnist()

    print("\n=== Step 2: Running all 5 model test suites ===")
    for name, fn in [
        ("NN", lambda: test_example_NN(interactive=False)),
        ("CNN", lambda: test_example_CNN(interactive=False)),
        ("DBN", lambda: test_example_DBN(interactive=False)),
        ("SAE", lambda: test_example_SAE(interactive=False)),
        ("CAE", lambda: test_example_CAE(interactive=False)),
    ]:
        st = time.time()
        print(f"\n--- Running {name} ---")
        fn()
        print(f"--- {name} finished in {time.time() - st:.1f}s ---")

    print("\n=== Step 3: Generating training animations & output plots ===")
    generate_gifs()

    out_dir = os.path.join(ROOT_DIR, "output")
    files = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []
    print(f"\n=== End-to-end run complete in {time.time() - t0:.1f}s ===")
    print(f"Generated {len(files)} artifact(s) in {out_dir}:")
    for fname in files:
        kb = os.path.getsize(os.path.join(out_dir, fname)) / 1024.0
        print(f"  - {fname} ({kb:.1f} KB)")


if __name__ == "__main__":
    main()
