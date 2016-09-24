'''
Created on Sep 24, 2016

@author: Wajih-PC
'''

import numpy as np
from scipy.special import erfinv 
def sigmrnd(input):
    # Declaring variables as np float type to avoid Overflow warnings
    minusone = np.float(-1.0) 
    plusone = np.float(1.0)
    sigmVals = np.true_divide(plusone,np.add(plusone,np.exp(np.multiply(minusone,input))))
    samples = np.random.uniform(0,1,input.shape)
    samples= np.where(sigmVals>samples,1,0)
    return samples
       
    