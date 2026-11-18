# -*- coding: utf-8 -*-
"""
Created on 18th November 2026
Updated on 18th November 2026
@author: Lenovo Laptop

Apply gradients (SGD with optional momentum) for Convolutional Autoencoder.
"""


def caeapplygrads(cae, opts):
    alpha = float(opts["alpha"])
    momentum = float(opts.get("momentum", cae.Momentum))

    for j in range(cae.OutputMaps):
        cae.vB[j] = momentum * cae.vB[j] + alpha * cae.dB[j]
        cae.B[j] = cae.B[j] - cae.vB[j]
        for i in range(cae.InputMaps):
            cae.vIK[(i, j)] = momentum * cae.vIK[(i, j)] + alpha * cae.dIK[(i, j)]
            cae.IK[(i, j)] = cae.IK[(i, j)] - cae.vIK[(i, j)]

            cae.vOK[(i, j)] = momentum * cae.vOK[(i, j)] + alpha * cae.dOK[(i, j)]
            cae.OK[(i, j)] = cae.OK[(i, j)] - cae.vOK[(i, j)]

    for i in range(cae.InputMaps):
        cae.vC[i] = momentum * cae.vC[i] + alpha * cae.dC[i]
        cae.C[i] = cae.C[i] - cae.vC[i]

    return cae
