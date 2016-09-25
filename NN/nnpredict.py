'''
Created on Jun 13, 2016

@author: Wajih-PC
'''
import numpy as np
import operator as op
from NN import nnff
def nnpredict(nn,x):
    # NEEDS FIXING
    nn.Testing = True
    nn = nnff.nnff(nn,x,np.zeros(shape = (np.shape(x)[0],nn.Size[-1])))
    nn.Testing = False
    # max of rows for the last layer (output layer) This is where we get our predicted outputs
    labels = nn.A[-1].argmax(axis=1)
    dummy = np.amax(nn.A[-1],axis = 1);
    return labels # Predicted labels
    
    