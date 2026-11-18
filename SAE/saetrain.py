'''
Created on Sep 26, 2016

@author: Wajih-PC

Updated on 18th November 2026
@author: Lenovo Laptop
'''
from NN.nntrain import nntrain
from NN.nnff import nnff

def saetrain(sae, x, opts, callback=None):
    for i in range(0 , len(sae.AutoEncoder)):
        print ("Training AE " + str(i+1)+ "/" +str(len(sae.AutoEncoder)));
        ae_cb = (lambda nn, epoch, batch, loss_val, step, idx=i: callback(sae, idx, epoch, batch, loss_val, step)) if callback is not None else None
        sae.AutoEncoder[i] = nntrain(sae.AutoEncoder[i], x, x, opts,None,None,None,callback=ae_cb)[0]
        t = nnff(sae.AutoEncoder[i], x, x);
        x = t.A[1];
        #remove bias term
        x = x[:,1::];
    return sae