'''
Created on Sep 24, 2016

@author: Wajih-PC
'''
from DBN import rbmtrain
from DBN import  rbmup
def dbntrain(dbn,x,opts):
    n = len(dbn.RBM)
    
    dbn.RBM[0] = rbmtrain.rbmtrain(dbn.RBM[0],x,opts)
    
    for i in range (1,n):
        x = rbmup.rbmup(dbn.RBM[i-1],x)
        dbn.RBM[i] = rbmtrain.rbmtrain(dbn.rbm[i],x,opts)