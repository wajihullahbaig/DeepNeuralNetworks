# -*- coding: utf-8 -*-
"""
Created on 18th November 2026
Updated on 18th November 2026
@author: Lenovo Laptop

Training routines for CAE and Stacked CAE (SCAE).
"""
import sys
from time import time
import numpy as np
from CAE.caeff import caeff
from CAE.caebp import caebp
from CAE.caeapplygrads import caeapplygrads
from CAE.caeup import caeup
from CAE.ConvolutionalAutoEncoder import as_map_dict


def _slice_batch(x_maps, indices):
    if len(x_maps) == 1:
        return x_maps[0][:, :, indices]
    return {k: v[:, :, indices] for k, v in x_maps.items()}


def caetrain(cae, x, opts, callback=None):
    x_maps = as_map_dict(x, cae.InputMaps)
    m = x_maps[0].shape[2]
    batchsize = int(opts["batchsize"])
    numepochs = int(opts["numepochs"])

    if m % batchsize != 0:
        sys.exit("numbatches not integer")
    numbatches = m // batchsize

    for i in range(numepochs):
        print("epoch ", i + 1, "/", numepochs)
        tic = time()
        kk = np.random.permutation(range(m))
        for l in range(numbatches):
            idx = kk[l * batchsize:(l + 1) * batchsize]
            batch_x = _slice_batch(x_maps, idx)
            cae = caeff(cae, batch_x, training=True)
            cae = caebp(cae)
            cae = caeapplygrads(cae, opts)
            if not cae.rL:
                cae.rL.insert(0, cae.L)
            cae.rL.append(0.99 * cae.rL[-1] + 0.01 * cae.L)
            if callback is not None:
                callback(cae, i, l, cae.L)
        toc = time()
        print("epoch ", i + 1, "/", numepochs, ". Took ", toc - tic, " seconds. Smoothed loss:", cae.rL[-1])
    return cae


def scaetrain(scae, x, opts, callback=None):
    """Greedy layer-wise training of a Stacked Convolutional Autoencoder."""
    curr_x = x
    for idx, cae in enumerate(scae.Layers):
        print(f"Training CAE layer {idx + 1}/{len(scae.Layers)}")
        layer_cb = (
            (lambda net, ep, b, loss_val, layer_idx=idx: callback(scae, layer_idx, ep, b, loss_val))
            if callback is not None else None
        )
        scae.Layers[idx] = caetrain(cae, curr_x, opts, callback=layer_cb)
        caeup(scae.Layers[idx], curr_x, training=False)
        curr_x = {j: m.copy() for j, m in scae.Layers[idx].P.items()}
    return scae
