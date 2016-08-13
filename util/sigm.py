'''
Created on Jun 8, 2016

@author: Wajih-PC
'''

import numpy as np
def sigm(input):
    # Declaring variables as np float type to avoid Overflow warnings
    minusone = np.float(-1.0) 
    plusone = np.float(1.0)
    return np.true_divide(plusone,(np.add(plusone,np.exp(np.multiply(minusone,input)))))
       
    