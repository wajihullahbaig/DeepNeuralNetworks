# -*- coding: utf-8 -*-
"""
Created on 18th November 2026
Updated on 18th November 2026
@author: Lenovo Laptop

Encoder (upward) pass for Convolutional Autoencoder.
"""
import numpy as np
from util import sigm, conv2d
from CAE.ConvolutionalAutoEncoder import as_map_dict


def caeup(cae, x, training=False):
    x_maps = as_map_dict(x, cae.InputMaps)
    H, W, M = x_maps[0].shape
    K = cae.KernelSize
    S = cae.Scale

    cae.X = {}
    cae.I = {}
    for i in range(cae.InputMaps):
        clean_map = x_maps[i].astype(np.float64, copy=True)
        cae.X[i] = clean_map
        if training and cae.InputZeroMaskedFraction > 0.0:
            mask = np.where(
                np.random.uniform(0.0, 1.0, clean_map.shape) <= cae.InputZeroMaskedFraction,
                0.0,
                1.0,
            )
            cae.I[i] = np.multiply(clean_map, mask)
        else:
            cae.I[i] = clean_map.copy()

    conv_h = H - K + 1
    conv_w = W - K + 1
    assert conv_h > 0 and conv_w > 0, "Input spatial dimensions smaller than kernel size"

    conv_maps = conv2d.conv2d_valid_batch(cae.I, cae.IK, cae.OutputMaps)
    cae.A = {}
    cae.P = {}
    for j in range(cae.OutputMaps):
        cae.A[j] = sigm.sigm(np.add(conv_maps[j], cae.B[j]))

        if S > 1:
            assert conv_h % S == 0 and conv_w % S == 0, (
                f"Convolved map size ({conv_h}, {conv_w}) must be divisible by scale {S}"
            )
            Hp = conv_h // S
            Wp = conv_w // S
            cae.P[j] = cae.A[j].reshape(Hp, S, Wp, S, M).mean(axis=(1, 3))
        else:
            cae.P[j] = cae.A[j].copy()

    return cae
