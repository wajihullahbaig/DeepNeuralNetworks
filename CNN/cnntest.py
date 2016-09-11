'''
Created on Sep 11, 2016

@author: Wajih-PC
'''
from CNN import cnnff
import numpy as np
def cnntest(net, x, y):
    #  feedforward
    net = cnnff.cnnff(net, x)
    h = net.O.argmax(axis = 1)
    a = y.argmax(axis = 1)  
    bad = np.where(h  != a)
    er = len(bad) / y.shape[1];

