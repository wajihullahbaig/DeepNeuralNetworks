# -*- coding: utf-8 -*-
"""
Created on 18th November 2026
Updated on 18th November 2026
@author: Lenovo Laptop

Full forward pass (encode + decode) and reconstruction loss for Convolutional Autoencoder.
"""
import numpy as np
from CAE.caeup import caeup
from CAE.caedown import caedown
from CAE.ConvolutionalAutoEncoder import as_map_dict


def caeff(cae, x, target=None, training=False):
    cae = caeup(cae, x, training=training)
    cae = caedown(cae)

    target_maps = cae.X if target is None else as_map_dict(target, cae.InputMaps)
    total_loss = 0.0
    total_elems = 0
    cae.E = {}
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
        else:
            total_loss += 0.5 * np.sum(np.power(cae.E[i], 2))

    cae.L = np.true_divide(total_loss, float(total_elems))
    return cae
