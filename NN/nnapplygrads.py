'''
Created on Jun 12, 2016
Update weights and biases with calculated gradients.
@author: Wajih-PC
'''
import numpy as np
def nnapplygrads(nn):
    dW = None;
    for i in range(0,nn.N-1):
        if nn.WeightPenaltyL2 > 0:
            temp = np.zeros(shape = (nn.W[i].shape[0],nn.W[i].shape[1]))
            temp[:,1:] =  nn.W[i][:,1:]
            dW = nn.DW[i] + nn.WeightPenaltyL2*temp
        else:
            dW = nn.DW[i] 
    
        dW = np.dot(nn.LearningRate,dW)
        if nn.Momentum > 0 :
            nn.VW[i] = nn.Momentum*nn.VW[i] + dW;
            dW = nn.VW[i]
        nn.W[i] = nn.W[i] - dW

    
    return nn        