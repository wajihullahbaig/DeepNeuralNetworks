# -*- coding: utf-8 -*-
"""
Created on 18th November 2026
Updated on 18th November 2026
@author: Lenovo Laptop

Convolutional Autoencoder (CAE) and Stacked CAE (SCAE) classes.
"""
import numpy as np


class CAE:
    def __init__(self, inputMaps=1, outputMaps=6, kernelSize=5, scale=2,
                 noise=0.0, momentum=0.5, output="sigm", loss="bce"):
        self.InputMaps = int(inputMaps)
        self.OutputMaps = int(outputMaps)
        self.KernelSize = int(kernelSize)
        self.Scale = int(scale)
        self.InputZeroMaskedFraction = float(noise)
        self.Momentum = float(momentum)
        self.Output = output
        self.LossType = loss

        fan_in = self.InputMaps * self.KernelSize * self.KernelSize
        fan_out = self.OutputMaps * self.KernelSize * self.KernelSize
        bound_enc = np.sqrt(6.0 / (fan_in + fan_out))
        bound_dec = np.sqrt(6.0 / (fan_in + fan_out))

        self.IK = {}
        self.vIK = {}
        self.dIK = {}
        self.OK = {}
        self.vOK = {}
        self.dOK = {}
        self.B = {}
        self.vB = {}
        self.dB = {}
        self.C = {}
        self.vC = {}
        self.dC = {}

        for j in range(self.OutputMaps):
            self.B[j] = 0.0
            self.vB[j] = 0.0
            self.dB[j] = 0.0
            for i in range(self.InputMaps):
                self.IK[(i, j)] = (
                    (np.random.uniform(0, 1, [self.KernelSize, self.KernelSize]) - 0.5)
                    * 2.0 * bound_enc
                ).T
                self.vIK[(i, j)] = np.zeros((self.KernelSize, self.KernelSize), dtype=np.float64)
                self.dIK[(i, j)] = np.zeros((self.KernelSize, self.KernelSize), dtype=np.float64)

                self.OK[(i, j)] = (
                    (np.random.uniform(0, 1, [self.KernelSize, self.KernelSize]) - 0.5)
                    * 2.0 * bound_dec
                ).T
                self.vOK[(i, j)] = np.zeros((self.KernelSize, self.KernelSize), dtype=np.float64)
                self.dOK[(i, j)] = np.zeros((self.KernelSize, self.KernelSize), dtype=np.float64)

        for i in range(self.InputMaps):
            self.C[i] = 0.0
            self.vC[i] = 0.0
            self.dC[i] = 0.0

        self.X = {}
        self.I = {}
        self.A = {}
        self.P = {}
        self.U = {}
        self.O = {}
        self.E = {}
        self.D_dec = {}
        self.D_u = {}
        self.D_enc = {}
        self.L = 0.0
        self.rL = []


class SCAE:
    """Stacked Convolutional Autoencoder composed of greedy layer-wise CAE blocks."""
    def __init__(self, layerSpecs, inputMaps=1, noise=0.0, momentum=0.5):
        self.Layers = []
        in_maps = int(inputMaps)
        for spec in layerSpecs:
            out_maps = int(spec["outputMaps"])
            kernel_size = int(spec.get("kernelSize", 5))
            scale = int(spec.get("scale", 2))
            layer_noise = float(spec.get("noise", noise))
            layer_momentum = float(spec.get("momentum", momentum))
            cae = CAE(
                inputMaps=in_maps,
                outputMaps=out_maps,
                kernelSize=kernel_size,
                scale=scale,
                noise=layer_noise,
                momentum=layer_momentum,
                output=spec.get("output", "sigm"),
                loss=spec.get("loss", "bce"),
            )
            self.Layers.append(cae)
            in_maps = out_maps


def as_map_dict(x, expected_maps):
    """Normalize a 3D array (H, W, M) or dict of 3D arrays into {idx: array}."""
    if isinstance(x, dict):
        if len(x) != expected_maps:
            raise ValueError(f"Expected {expected_maps} feature maps, got {len(x)}")
        return x
    if expected_maps != 1:
        raise ValueError(f"Array input only supported when expected_maps==1, got {expected_maps}")
    return {0: x}
