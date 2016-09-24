'''
Created on Sep 24, 2016

@author: Wajih-PC
'''

import numpy as np
def sigmrnd(input):
    # Declaring variables as np float type to avoid Overflow warnings
    minusone = np.float(-1.0) 
    plusone = np.float(1.0)
    sigmVals = np.add(plusone,np.exp(np.multiply(minusone,input)))
    temp = np.random.uniform(0,1,input.shape)
    truthTable = (sigmVals>temp).astype(int).astype(float)    
    sigmVals = np.true_divide(plusone,(truthTable))    
    return sigmVals
       
    