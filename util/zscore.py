# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:43:39 2016

@author: Wajih-PC
"""
from numpy import float64


def zscore(x):
    import numpy as np
    mu=np.mean(x,axis = 0)
    epsArray = np.full(np.size(x, 1),np.finfo(float64).eps)
    sigma=np.maximum(np.std(x,axis = 0),epsArray)
    x=np.subtract(x,mu)
    x=np.true_divide(x,sigma)
    return x,mu,sigma