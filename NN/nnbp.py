'''
Created on Jun 8, 2016

@author: Wajih-PC

NNBP performs Backpropagation
returns a NN with updated weights
'''

import numpy as np
from NN import Collections as collections
def nnbp(nn):
    n = nn.N
    sparsityError = 0
    d = [None for i in range(n)]
    if nn.Output == "sigm":
        temp2 = 1- nn.A[n-1]    
        d[n-1] = - np.multiply(nn.E,np.multiply(nn.A[n-1],temp2))
    elif nn.Output == "softmax" or "linear":
        d[n-1] = - nn.E
    for i in range(n-2,0,-1):
        # Derivative of the activation function
        if nn.ActivationFunction == "sigm":
            temp1 = 1
            temp = np.subtract(temp1,nn.A[i]) 
            d_act = np.multiply(nn.A[i],temp)
        elif nn.ActivationFunction == 'tanh_opt':
            t1 = 1.7159*(2/3)
            t2 = 1/(1.7159*1.7159)
            d_act = t1*(1-(t2*np.power(nn.A[i],2)))
    
    if nn.NonSparsityPenalty > 0:
        pass        
    # Backpropagate first derivatives
    if (i+1) == (n-1): #In this case in d[n-1] there is no bias term to be removed
        sE = np.full((d[i+1].shape[0],nn.W[i].shape[1]),sparsityError,dtype = np.float64)
        d[i] = np.multiply(np.add(np.dot(d[i+1],nn.W[i]),sE),d_act)
    else: # In this case in d[i] the bias term has to be removed
        temp = d[i+1][:,1:]
        d[i] = np.multiply(np.dot(temp,nn.W[i]),d_act)+sparsityError
        
    if nn.DropoutFraction > 0:
        resized = np.ones([nn.DropOutMask[i].shape[0],nn.DropOutMask[i].shape[1]+1], dtype=np.float64)
        resized[:,1:] = nn.DropOutMask[i]
        d[i] = np.multiply(d[i],resized)
    #temp[:,1:] = np.copy(x)
    
    # Allocate space for activation, error and loss
    # Since matlab allows structs to have cells/arrays etc on the fly, 
    # This is not possible with python. We declare members and then use them    
    #nn.A[0] = np.copy(temp);       
    
    for i in range (0,n-1):
        if (i+1) == (n-1):
            term = d[i+1]
            nn.DW[i] = np.dot(term.T,nn.A[i])/d[i+1].shape[0]
        else:
            term = d[i+1][:,1:]            
            nn.DW[i] = np.dot(term.T,nn.A[i])/d[i+1].shape[0]
            
    return nn