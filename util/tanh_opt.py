'''
Created on Jun 8, 2016

@author: Wajih-PC
'''

import numpy as np
def tanh_opt(A):
    v1 = 2/3
    v2 = 1.7159    
    result = np.multiply(A,v1)  
    result = np.tanh(result)  
    return np.multiply(result,v2)
