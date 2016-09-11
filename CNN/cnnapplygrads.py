'''
Created on Sep 11, 2016

@author: Wajih-PC
'''
import numpy as np
from CNN import Layers
def cnnapplygrads(net, opts):
    for l in range(1, len(net.layers)):
        if isinstance(net.layers[l], Layers.ConvolutionalLayer):
            for j in range(0,len(net.layers[l].A)):
                for ii in range(0 , len(net.layers[l - 1].A)):
                    net.layers[l].K[ii,j] = net.layers[l].K[ii,j] - opts["alpha"] * net.layers[l].dK[ii,j]                
                net.layers[l].B[j] = net.layers[l].B[j] - opts["alpha"] * net.layers[l].dB[j];

    net.ffW = net.ffW - opts["alpha"] * net.dffW;
    net.ffb = net.ffB - opts["alpha"] * net.dffB;

    return net