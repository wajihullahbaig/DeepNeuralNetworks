'''
Created on Aug 28, 2016

@author: Wajih-PC
'''
from CNN import Layers
import numpy as np
from scipy.ndimage import convolve
from util import sigm
def cnnff(net,x):
    n = len(net.layers)
    net.layers[0].A[0] = x[:];
    inputMaps = 1
    
    for l in range(1,n):
        if isinstance(net.layers[l],Layers.ConvolutionalLayer):
            for j in range(0,net.layers[l].OutputMaps):
                # Create a temp output map
                # map shape 
                t = np.asarray([net.layers[l].KernelSize - 1 ,net.layers[l].KernelSize - 1, 0])
                s = np.asarray(net.layers[l - 1].A[0].shape)
                z = np.zeros(shape = s-t,dtype = np.float64)
                for i in range(0,inputMaps):                    
                    # add a dimension to kernel as the data is in MxMxN
                    kernel = np.expand_dims(net.layers[l].K[i,j],axis=2)
                    valid = [slice(kernel.shape[0]//2, -kernel.shape[0]//2+1), slice(kernel.shape[1]//2, -kernel.shape[1]//2+1)]
                    # Reproducing what convn(...) with 'valid' would give us in matlab
                    convolutionResult =convolve(net.layers[l-1].A[i],kernel)[valid]                    
                    z = np.add(z,convolutionResult)
                # add bias, pass through nonlinearity
                net.layers[l].A[j] = sigm.sigm(np.add(z,net.layers[l].B[j]))                              
                
            # set number of input maps tp this layers number of output maps
            inputMaps = net.layers[l].OutputMaps        
        elif isinstance(net.layers[l],Layers.ScaleLayer):
            # Downsample
            for j in range(0,inputMaps):
                kernel = np.true_divide(np.ones(shape = (net.layers[l].Scale,net.layers[l].Scale)),net.layers[l].Scale*net.layers[l].Scale)
                kernel = np.expand_dims(kernel,axis=2)
                z= None
                z = convolve(net.layers[l-1].A[j],kernel)[1:,1:]
                net.layers[l].A[j] = z[:: net.layers[l].Scale, :: net.layers[l].Scale,:];                
    # Concatenate all end layer feature maps into vector
    for j in range(0,len(net.layers[n-1].A)):
        sa = net.layers[n-1].A[j].shape    
        if net.FV is None:
            net.FV = np.reshape(net.layers[n-1].A[j], newshape =(sa[0]*sa[1], sa[2]))[:]
        else:          
            net.FV = np.append(net.FV,np.reshape(net.layers[n-1].A[j], newshape =(sa[0]*sa[1], sa[2])),axis = 0)   
    # Feedforward into output perceptrons
    net.O = sigm.sigm(np.dot(net.ffW,net.FV)+np.tile(net.ffB,(1,net.FV.shape[1])))   
    return net 