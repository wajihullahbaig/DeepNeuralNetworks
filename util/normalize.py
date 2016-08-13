# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:14:03 2016

@author: Wajih-PC
"""
import numpy as np    
def normalize(x,mu,sigma):
    x = np.subtract(x,mu)
    x = np.true_divide(x,sigma)
    return x