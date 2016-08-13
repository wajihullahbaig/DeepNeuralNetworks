'''
Created on Jun 13, 2016

@author: Wajih-PC
'''
import numpy as np
from NN import nnpredict
def nntest(nn,x,y):
    labels = nnpredict.nnpredict(nn,x)
    expected = y.argmax(axis=1)    
    bad = np.where(labels != expected)
    er = np.size(bad)/np.size(x,0)
    return er,bad