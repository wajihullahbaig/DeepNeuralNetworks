'''
Created on Sep 11, 2016

@author: Wajih-PC
'''
from CNN import cnnff
import numpy as np
def cnntest(net, x, y):
    #  feedforward
    net = cnnff.cnnff(net, x)
    h = net.O.argmax(axis = 0)
    a = y.argmax(axis = 0)  
    bad = np.where(h  != a)[0]
    err = len(bad) / y.shape[1];
    return err,bad
