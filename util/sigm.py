'''
Created on Jun 8, 2016

@author: Wajih-PC

Updated on 18th November 2026
@author: Lenovo Laptop
'''

import numpy as np
def sigm(input):
    # Clip to avoid overflow warnings in exp while preserving float64 precision
    x = np.clip(np.asarray(input, dtype=np.float64), -500.0, 500.0)
    minusone = np.float64(-1.0)
    plusone = np.float64(1.0)
    return np.true_divide(plusone,(np.add(plusone,np.exp(np.multiply(minusone,x)))))
       
    