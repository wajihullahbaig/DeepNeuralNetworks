'''
Created on Sep 25, 2016

@author: Wajih-PC
'''
import numpy as np
from NN import NeuralNetwork
def dbnunfoldtonn(dbn, outputsize):
#DBNUNFOLDTONN Unfolds a DBN to a NN
#   dbnunfoldtonn(dbn, outputsize ) returns the unfolded dbn with a final
#   layer of size outputsize added.
    if outputsize is not None: 
        size = np.append(dbn.dbnSizes, outputsize)
    else:
        size = np.array(dbn.sizes)
            
    nn = NeuralNetwork.NN(size)
    for i in range(0 , len(dbn.RBM)):
        nn.W[i][:,0]=dbn.RBM[i].C.squeeze()
        nn.W[i][:,1:]=dbn.RBM[i].W
    
    return nn