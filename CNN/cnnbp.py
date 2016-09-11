'''
Created on Sep 11, 2016

@author: Wajih-PC
'''
import numpy as np
from scipy.signal import fftconvolve
from CNN import Layers

def cnnbp(net,y):
    
    n = len(net.layers)
    # Error
    net.E = net.O - y
    
    # Loss Function
    
    net.L = 0.5* np.true_divide(np.sum(np.power(net.E,2)),net.E.shape[1])
    
    # backpropagation deltas
    net.oD = np.multiply(net.E,np.multiply(net.O,(1-net.O))) # Output delta 
    net.fVD = np.dot(net.ffW.T,net.oD) # feature vector delta
    # Only convolution layers have sigm function
    if isinstance(net.layers[n-1], Layers.ConvolutionalLayer):
        net.fVD = np.dot(net.fVD,np.dot(net.fV,1-net.fV))
    
    #Reshape feature vector deltas into output map style
    
    sa = net.layers[n-1].A[0].shape    
    fvnum = sa[0]*sa[1]
    
    for j in range(0,len(net.layers[n-1].A)):
        startIndex = j * fvnum
        endIndex = (j+1)*fvnum
        block = net.fVD[startIndex:endIndex,:]
        net.layers[n-1].D[j] = np.reshape(block, newshape =(sa[0],sa[1], sa[2]))
    
    for l in range(n-2,-1,-1):
        if isinstance(net.layers[l], Layers.ConvolutionalLayer):
            for j in range(0,len(net.layers[l].A)):
                t1 = net.layers[l].A[j] * (1-net.layers[l].A[j])
                t2 = np.repeat(net.layers[l + 1].D[j], net.layers[l + 1].Scale,0)
                t2 = np.repeat(t2, net.layers[l + 1].Scale,1)
                net.layers[l].D[j] = t2*(net.layers[l + 1].Scale*net.layers[l + 1].Scale)
        elif isinstance(net.layers[l], Layers.ScaleLayer):
            for i in range(0,len(net.layers[l].A)):
                z = np.zeros(net.layers[l].A[0].shape)
                for j in range (0,len(net.layers[l+1].A)):
                    kernel = np.rot90(net.layers[l + 1].K[i,j],2)
                    # Adding singleton dimension so that we can perform convolution
                    kernel = kernel[:,:,None]
                    # Have to use fftconvolve instead of ndimage.convolve
                    # fftConvolve gives mode='full' option
                    convolutionResult = fftconvolve(net.layers[l+1].D[i],kernel,mode='full')
                    z = np.add(z,convolutionResult)
                net.layers[l].D[i] = z
                
    # Calculate gradient
    for l in range(1,n) :
        if isinstance(net.layers[l], Layers.ConvolutionalLayer):
            for j in range(0,len(net.layers[l].A)):
                for i in range(0,len(net.layers[l-1].A)):
                    # See later that if we need to remove the singleton dimension
                    net.layers[l].dK[i,j] =fftconvolve(np.flipud(net.layers[l-1].A[i]),net.layers[l].D[j],mode='valid')
                net.layers[l].dB[j] = np.sum(net.layers[l].D[j])/net.layers[l].D[j].shape[2]    
    
    net.dffW = np.dot(net.oD,net.FV.T) / net.oD.shape[1]
    net.dffB = np.mean(net.oD,1)    
    net.dffB = net.dffB[:,None] # Adding singleton dimension for dimensionality consistency  
    return net