'''
Created on Sep 11, 2016

@author: Wajih-PC

Updated on 18th November 2026
@author: Lenovo Laptop
'''
import numpy as np
from CNN import Layers
from util import conv2d

def cnnbp(net,y):
    
    n = len(net.layers)
    # Error
    net.E = net.O - y
    
    # Loss Function
    
    net.L = 0.5* np.true_divide(np.sum(np.power(net.E,2)),net.E.shape[1])
    
    # backpropagation deltas
    net.oD = net.E*net.O*(1-net.O) # Output delta 
    net.fVD = np.dot(net.ffW.T,net.oD) # feature vector delta
    # Only convolution layers have sigm function
    if isinstance(net.layers[n-1], Layers.ConvolutionalLayer):
        net.fVD = net.fVD*net.fV*(1-net.fV)
    
    #Reshape feature vector deltas into output map style
    
    sa = net.layers[n-1].A[0].shape    
    fvnum = sa[0]*sa[1]
    
    for j in range(0,len(net.layers[n-1].A)):
        startIndex = j * fvnum
        endIndex = (j+1)*fvnum
        block = net.fVD[startIndex:endIndex,:]
        net.layers[n-1].D[j] = np.reshape(block, newshape =(sa[0],sa[1], sa[2]),order = "F")
    
    for l in range(n-2,-1,-1):
        if isinstance(net.layers[l], Layers.ConvolutionalLayer):
            for j in range(0,len(net.layers[l].A)):
                t1 = net.layers[l].A[j] * (1.0-net.layers[l].A[j])
                # The next two lines with np.repeat produce the Matlab's "expand" function
                t2 = np.repeat(net.layers[l + 1].D[j], net.layers[l + 1].Scale,0)
                t2 = np.repeat(t2, net.layers[l + 1].Scale,1)
                net.layers[l].D[j] = t1*t2/(net.layers[l + 1].Scale*net.layers[l + 1].Scale)
        elif isinstance(net.layers[l], Layers.ScaleLayer):
            k_swap = {
                (j, i): net.layers[l + 1].K[i, j]
                for i in range(len(net.layers[l].A))
                for j in range(len(net.layers[l + 1].A))
            }
            d_maps = conv2d.conv2d_full_batch(
                net.layers[l + 1].D,
                k_swap,
                len(net.layers[l].A),
                rot180=True,
            )
            for i in range(0, len(net.layers[l].A)):
                net.layers[l].D[i] = d_maps[i]
                
    # Calculate gradient
    for l in range(1,n) :
        if isinstance(net.layers[l], Layers.ConvolutionalLayer):
            kh = net.layers[l].KernelSize
            grads = conv2d.conv2d_grad_kernels_valid(net.layers[l - 1].A, net.layers[l].D, kh, kh)
            batch_size = float(net.layers[l].D[0].shape[2])
            for j in range(0,len(net.layers[l].A)):
                for i in range(0,len(net.layers[l-1].A)):
                    net.layers[l].dK[i,j] = grads[(i, j)] / batch_size
                net.layers[l].dB[j] = np.sum(net.layers[l].D[j]) / batch_size
    
    net.dffW = np.dot(net.oD,net.fV.T) / net.oD.shape[1]
    net.dffB = np.mean(net.oD,1)    
    net.dffB = net.dffB[:,None] # Adding singleton dimension for dimensionality consistency  
    return net