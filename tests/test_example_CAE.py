# -*- coding: utf-8 -*-
"""
Created on 18th November 2026
Updated on 18th November 2026
@author: Lenovo Laptop

Example and verification suite for Convolutional Autoencoder (CAE) and Stacked CAE (SCAE).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from data import mnist
from CAE.ConvolutionalAutoEncoder import CAE, SCAE
from CAE.caeff import caeff
from CAE.caebp import caebp
from CAE.caetrain import caetrain, scaetrain
from CAE.caetest import caetest, caeunfoldtocnn
from CNN import cnntrain, cnntest
from util.visualize import visualize


def check_cae_gradients():
    """Finite-difference verification of all CAE analytical gradients (BCE and MSE)."""
    np.random.seed(42)
    eps = 1e-6

    for cfg in [
        {"inputMaps": 1, "outputMaps": 3, "kernelSize": 3, "scale": 2, "H": 10, "W": 10, "loss": "bce"},
        {"inputMaps": 2, "outputMaps": 2, "kernelSize": 3, "scale": 1, "H": 8, "W": 8, "loss": "mse"},
    ]:
        cae = CAE(**{k: cfg[k] for k in ("inputMaps", "outputMaps", "kernelSize", "scale", "loss")})
        if cfg["inputMaps"] == 1:
            x = np.random.uniform(0.0, 1.0, (cfg["H"], cfg["W"], 4))
        else:
            x = {i: np.random.uniform(0.0, 1.0, (cfg["H"], cfg["W"], 4)) for i in range(cfg["inputMaps"])}

        caeff(cae, x, training=False)
        caebp(cae)

        def eval_loss():
            caeff(cae, x, training=False)
            return cae.L

        # Check IK and OK kernels
        for param_dict, grad_dict, name in [
            (cae.IK, cae.dIK, "IK"),
            (cae.OK, cae.dOK, "OK"),
        ]:
            for key, arr in param_dict.items():
                for r in range(arr.shape[0]):
                    for c in range(arr.shape[1]):
                        orig = arr[r, c]
                        arr[r, c] = orig + eps
                        l_plus = eval_loss()
                        arr[r, c] = orig - eps
                        l_minus = eval_loss()
                        arr[r, c] = orig
                        num_g = (l_plus - l_minus) / (2.0 * eps)
                        ana_g = grad_dict[key][r, c]
                        rel_err = abs(num_g - ana_g) / max(1e-5, abs(num_g) + abs(ana_g))
                        assert rel_err < 1e-5, f"Gradient mismatch in {name}{key}[{r},{c}]: {ana_g} vs {num_g}"

        # Check B and C biases
        for j in range(cae.OutputMaps):
            orig = cae.B[j]
            cae.B[j] = orig + eps
            l_plus = eval_loss()
            cae.B[j] = orig - eps
            l_minus = eval_loss()
            cae.B[j] = orig
            num_g = (l_plus - l_minus) / (2.0 * eps)
            ana_g = cae.dB[j]
            rel_err = abs(num_g - ana_g) / max(1e-5, abs(num_g) + abs(ana_g))
            assert rel_err < 1e-5, f"Gradient mismatch in B[{j}]: {ana_g} vs {num_g}"

        for i in range(cae.InputMaps):
            orig = cae.C[i]
            cae.C[i] = orig + eps
            l_plus = eval_loss()
            cae.C[i] = orig - eps
            l_minus = eval_loss()
            cae.C[i] = orig
            num_g = (l_plus - l_minus) / (2.0 * eps)
            ana_g = cae.dC[i]
            rel_err = abs(num_g - ana_g) / max(1e-5, abs(num_g) + abs(ana_g))
            assert rel_err < 1e-5, f"Gradient mismatch in C[{i}]: {ana_g} vs {num_g}"

    print("CAE numerical gradient checks (BCE & MSE) passed (< 1e-6 relative error).")


def test_example_CAE(interactive=None):
    if interactive is None:
        interactive = os.environ.get("DNN_NONINTERACTIVE") != "1" and matplotlib.get_backend().lower() != "agg"

    check_cae_gradients()

    dataSet = mnist.load_mnist()
    train_x = np.transpose(dataSet["train_x"].reshape(-1, 28, 28), (1, 2, 0)) / 255.0
    train_y = dataSet["train_y"].T
    test_x = np.transpose(dataSet["test_x"].reshape(-1, 28, 28), (1, 2, 0)) / 255.0
    test_y = dataSet["test_y"].T

    # ex1: Train a denoising Convolutional Autoencoder (12 filters, 5x5, 2x2 pooling)
    np.random.seed(1)
    cae = CAE(inputMaps=1, outputMaps=12, kernelSize=5, scale=2, noise=0.3, momentum=0.5, loss="bce")
    init_mse, _ = caetest(cae, test_x[:, :, :1000])

    options = {"alpha": 4.0, "numepochs": 3, "batchsize": 50}
    cae = caetrain(cae, train_x[:, :, :15000], options)
    final_mse, _ = caetest(cae, test_x[:, :, :1000])
    print(f"CAE ex1 reconstruction MSE: {init_mse:.6f} -> {final_mse:.6f}")
    assert final_mse < init_mse * 0.2, f"CAE reconstruction MSE did not decrease enough: {final_mse}"
    assert final_mse < 0.03, f"Too big CAE reconstruction MSE: {final_mse}"

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    filters = np.column_stack([cae.IK[(0, j)].ravel() for j in range(cae.OutputMaps)])
    ax = visualize(filters, None, 5, 5, save_path=os.path.join(out_dir, "cae_encoder_filters.png"))
    if not interactive:
        plt.close(ax.figure)

    # ex2: Train a 2-layer Stacked CAE (6c5-2s -> 12c5-2s) and unfold into a CNN
    np.random.seed(1)
    scae = SCAE(
        [
            {"outputMaps": 6, "kernelSize": 5, "scale": 2, "noise": 0.2, "loss": "bce"},
            {"outputMaps": 12, "kernelSize": 5, "scale": 2, "noise": 0.0, "loss": "bce"},
        ],
        momentum=0.8,
    )
    scae = scaetrain(scae, train_x[:, :, :15000], {"alpha": 4.0, "numepochs": 2, "batchsize": 50})
    cnn = caeunfoldtocnn(scae, train_x[:, :, :50], train_y[:, :50])
    cnn = cnntrain.cnntrain(cnn, train_x[:, :, :30000], train_y[:, :30000], {"alpha": 1.0, "numepochs": 1, "batchsize": 50})
    err, _ = cnntest.cnntest(cnn, test_x[:, :, :2000], test_y[:, :2000])
    print(f"SCAE-initialized CNN error (30k train, 1 epoch): {err:.6f}")
    assert err < 0.22, f"SCAE unfolded CNN error too high: {err}"

    print("test_example_CAE completed")
    if interactive:
        input("Press [enter] to continue.")


if __name__ == "__main__":
    test_example_CAE()
