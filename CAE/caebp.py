# -*- coding: utf-8 -*-
"""
Created on 18th November 2026
Updated on 18th November 2026
@author: Lenovo Laptop

Backpropagation for Convolutional Autoencoder.
"""
import numpy as np
from util import conv2d
from CAE.ConvolutionalAutoEncoder import as_map_dict


def caebp(cae, target=None):
    target_maps = cae.X if target is None else as_map_dict(target, cae.InputMaps)
    M = target_maps[0].shape[2]
    S = cae.Scale
    K = cae.KernelSize

    # Output error & decoder deltas (normalized by total pixel elements across batch)
    cae.E = {}
    cae.D_dec = {}
    total_loss = 0.0
    total_elems = 0
    loss_type = getattr(cae, "LossType", "mse")
    eps = 1e-12
    for i in range(cae.InputMaps):
        cae.E[i] = cae.O[i] - target_maps[i]
        total_elems += cae.E[i].size
        if loss_type == "bce" and cae.Output == "sigm":
            o_clip = np.clip(cae.O[i], eps, 1.0 - eps)
            total_loss -= np.sum(
                target_maps[i] * np.log(o_clip) + (1.0 - target_maps[i]) * np.log(1.0 - o_clip)
            )
            cae.D_dec[i] = cae.E[i]
        elif cae.Output == "sigm":
            total_loss += 0.5 * np.sum(np.power(cae.E[i], 2))
            cae.D_dec[i] = cae.E[i] * cae.O[i] * (1.0 - cae.O[i])
        elif cae.Output == "linear":
            total_loss += 0.5 * np.sum(np.power(cae.E[i], 2))
            cae.D_dec[i] = cae.E[i]
        else:
            raise ValueError(f"Unsupported CAE output activation: {cae.Output}")

    denom = float(total_elems)
    cae.L = np.true_divide(total_loss, denom)

    # Decoder kernel & bias gradients
    dec_grads = conv2d.conv2d_grad_dec_valid(cae.D_dec, cae.U, K, K)
    for i in range(cae.InputMaps):
        cae.dC[i] = float(np.sum(cae.D_dec[i]) / denom)
        for j in range(cae.OutputMaps):
            cae.dOK[(i, j)] = dec_grads[(i, j)] / denom

    # Backpropagate delta from decoder output to unpooled hidden maps U
    cae.D_u = conv2d.conv2d_valid_batch(cae.D_dec, cae.OK, cae.OutputMaps, rot180=True)

    # Backpropagate through unpooling and mean-pooling to encoder sigmoid activations
    cae.D_enc = {}
    for j in range(cae.OutputMaps):
        if S > 1:
            Hp, Wp = cae.P[j].shape[0], cae.P[j].shape[1]
            d_pool = cae.D_u[j].reshape(Hp, S, Wp, S, M).sum(axis=(1, 3))
            d_act_in = np.repeat(np.repeat(d_pool, S, axis=0), S, axis=1) / float(S * S)
        else:
            d_act_in = cae.D_u[j]
        cae.D_enc[j] = d_act_in * cae.A[j] * (1.0 - cae.A[j])

    # Encoder kernel & bias gradients
    enc_grads = conv2d.conv2d_grad_kernels_valid(cae.I, cae.D_enc, K, K)
    for j in range(cae.OutputMaps):
        cae.dB[j] = float(np.sum(cae.D_enc[j]) / denom)
        for i in range(cae.InputMaps):
            cae.dIK[(i, j)] = enc_grads[(i, j)] / denom

    return cae
