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
    # NEEDS FIXING
    nn.Testing = 1
    # Training performance
    nn = nnff.nnff(nn,train_x,train_y)
    loss.Training.E[epochNumber] = nn.L
    
    if val_x is not None and val_y is not None:
         nn = nnff.nnff(nn,val_x,val_y)
         loss.Validity.E[epochNumber] = nn.L
    else:
        nn.Testing = 0
        #Calculate mis-classifcation rate if softmax
        if nn.Output == "softmax":
            er_train,dummy = nntest(nn,train_x,train_y)
            loss.Training.E[epochNumber] = er_train 
        
            if val_x is not None and val_y is not None:
                er_val,dummy = nntest(nn,train_x,train_y)
                loss.Validity.E[epochNumber] = er_val; 
    return loss