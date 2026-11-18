# -*- coding: utf-8 -*-
"""
Created on 18th November 2026
Updated on 18th November 2026
@author: Lenovo Laptop

Evaluation and CNN unfolding utilities for Convolutional Autoencoder.
"""
import numpy as np
from CAE.caeff import caeff
from CNN import ConvolutionalNeuralNetwork, Layers


def caetest(cae, x):
    """Evaluate reconstruction MSE (per pixel) and average sample loss on x."""
    cae = caeff(cae, x, training=False)
    total_sq = 0.0
    total_elems = 0
    for i in range(cae.InputMaps):
        total_sq += np.sum(np.power(cae.E[i], 2))
        total_elems += cae.E[i].size
    mse = total_sq / float(total_elems)
    return mse, cae.L


def caeunfoldtocnn(scae, x, y):
    """Initialize a CNN's convolutional layers from a trained SCAE's encoder weights."""
    cnn_layers = [Layers.InputLayer()]
    for cae in scae.Layers:
        cnn_layers.append(Layers.ConvolutionalLayer(cae.OutputMaps, cae.KernelSize))
        if cae.Scale > 1:
            cnn_layers.append(Layers.ScaleLayer(cae.Scale))

    net = ConvolutionalNeuralNetwork.CNN(x, y, layers=cnn_layers)
    conv_idx = 0
    for l in range(len(net.layers)):
        if isinstance(net.layers[l], Layers.ConvolutionalLayer):
            cae = scae.Layers[conv_idx]
            for j in range(cae.OutputMaps):
                net.layers[l].B[j] = cae.B[j]
                for i in range(cae.InputMaps):
                    net.layers[l].K[i, j] = cae.IK[(i, j)].copy()
            conv_idx += 1
    return net
