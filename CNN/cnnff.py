'''
Created on Aug 28, 2016

@author: Wajih-PC
'''
from CNN.CNN import Layers
import numpy as np
def cnnff(net,x):
    n = len(net.layers)
    net.layers[1].A[1] = x;
    inputMaps = 1
    
    for l in range(1,n):
        if isinstance(net.layers[l],Layers.ConvolutionalLayer):
            for j in range(0,net.layers[l].OutputMaps):
                # Create a temp output map
                # map shape 
                t = [net.layers[l].KernelSize - 1 ,net.layers[l].KernelSize - 1, 0]
                s = net.layers[l - 1].A[1].shape()
                z = np.zeros(shape = s-t) 
                
            
    
    
    return net 