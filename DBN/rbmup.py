'''
Created on Sep 24, 2016

@author: Wajih-PC
'''

from util import sigm
import numpy as np
def rbmup(rbm, x):
    x = sigm.sigm(np.repeat(rbm.C.T, x.shape[0], 1) + np.dot(x , rbm.W.T))
    return x
             
