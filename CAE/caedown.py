# -*- coding: utf-8 -*-
"""
Created on 18th November 2026
Updated on 18th November 2026
@author: Lenovo Laptop

Decoder (downward reconstruction) pass for Convolutional Autoencoder.
"""
import numpy as np
from util import sigm, conv2d
from CAE.ConvolutionalAutoEncoder import as_map_dict


def caedown(cae, code=None):
    if code is None:
        p_maps = cae.P
    else:
        p_maps = as_map_dict(code, cae.OutputMaps)

    S = cae.Scale

    cae.U = {}
    for j in range(cae.OutputMaps):
        if S > 1:
            u = np.repeat(p_maps[j], S, axis=0)
            u = np.repeat(u, S, axis=1)
            cae.U[j] = u
        else:
            cae.U[j] = p_maps[j].copy()

    # Re-index OK[(i, j)] -> (j, i) so source=j maps to destination=i
    ok_src_dst = {(j, i): cae.OK[(i, j)] for i in range(cae.InputMaps) for j in range(cae.OutputMaps)}
    dec_maps = conv2d.conv2d_full_batch(cae.U, ok_src_dst, cae.InputMaps, rot180=False)

    cae.O = {}
    for i in range(cae.InputMaps):
        z = np.add(dec_maps[i], cae.C[i])
        if cae.Output == "sigm":
            cae.O[i] = sigm.sigm(z)
        elif cae.Output == "linear":
            cae.O[i] = z
        else:
            raise ValueError(f"Unsupported CAE output activation: {cae.Output}")

    return cae
