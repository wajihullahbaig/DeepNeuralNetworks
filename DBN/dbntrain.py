'''
Created on Sep 24, 2016

@author: Wajih-PC

Updated on 18th November 2026
@author: Lenovo Laptop
'''
from DBN import rbmtrain
from DBN import  rbmup
def dbntrain(dbn,x,opts,callback=None):
    n = len(dbn.RBM)
    
    dbn.RBM[0] = rbmtrain.rbmtrain(dbn.RBM[0],x,opts,callback=callback,rbm_idx=0)
    for i in range (1,n):
        x = rbmup.rbmup(dbn.RBM[i-1],x)
        dbn.RBM[i] = rbmtrain.rbmtrain(dbn.RBM[i],x,opts,callback=callback,rbm_idx=i)
    return dbn    