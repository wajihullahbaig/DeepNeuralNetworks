'''
Created on Jun 13, 2016
#This function evaluates the performance of neural network
@author: Wajih-PC
'''
from NN import nnff
from NN import Collections
from NN.nntest import nntest
import numpy as np
def nneval(nn,loss,train_x,train_y,val_x,val_y,epochNumber):
    
    nn.Testing = 1
    # Training performance
    nn = nnff.nnff(nn,train_x,train_y)
    loss.Training.E[epochNumber] = nn.L
    # Validation performance
    if val_x is not None and val_y is not None:
         nn = nnff.nnff(nn,val_x,val_y)
         loss.Validity.E[epochNumber] = nn.L
    nn.Testing = 0
    #Calculate mis-classifcation rate if softmax
    if nn.Output == "softmax":
        er_train,dummy = nntest(nn,train_x,train_y)
        loss.Training.E_Frac[epochNumber] = er_train 
 
        if val_x is not None and val_y is not None:
            er_val,dummy = nntest(nn,val_x,val_y)
            loss.Validity.E_Frac[epochNumber] = er_val; 
    return loss