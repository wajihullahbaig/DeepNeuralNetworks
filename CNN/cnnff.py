'''
Created on Aug 28, 2016

@author: Wajih-PC

Updated on 18th November 2026
@author: Lenovo Laptop
'''
from CNN import Layers
import numpy as np
from util import sigm, conv2d

def cnnff(net,x):
    n = len(net.layers)
    net.layers[0].A[0] = x.copy();
    inputMaps = 1
    for l in range(1,n):
        if isinstance(net.layers[l],Layers.ConvolutionalLayer):
            conv_maps = conv2d.conv2d_valid_batch(
                net.layers[l - 1].A,
                net.layers[l].K,
                net.layers[l].OutputMaps,
            )
            for j in range(0,net.layers[l].OutputMaps):
                # add bias, pass through nonlinearity
                net.layers[l].A[j] = sigm.sigm(np.add(conv_maps[j],net.layers[l].B[j]))                              
                
            # set number of input maps to this layer's number of output maps
            inputMaps = net.layers[l].OutputMaps        
        elif isinstance(net.layers[l],Layers.ScaleLayer):
            # Downsample via non-overlapping SxS block mean
            s = net.layers[l].Scale
            for j in range(0,inputMaps):
                a = net.layers[l-1].A[j]
                H, W, M = a.shape
                net.layers[l].A[j] = a.reshape(H // s, s, W // s, s, M).mean(axis=(1, 3))

    # Concatenate all end layer feature maps into vector
    net.fV = None
    for j in range(0,len(net.layers[n-1].A)):
        sa = net.layers[n-1].A[j].shape    
        if net.fV is None:
            net.fV = np.reshape(net.layers[n-1].A[j], newshape =(sa[0]*sa[1], sa[2]),order="F")[:]
        else:          
            net.fV = np.append(net.fV,np.reshape(net.layers[n-1].A[j], newshape =(sa[0]*sa[1], sa[2]),order="F"),axis = 0)[:]   
    # Feedforward into output perceptrons
    net.O = sigm.sigm(np.dot(net.ffW,net.fV)+np.tile(net.ffB,(1,net.fV.shape[1])))
       
    return net 