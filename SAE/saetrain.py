'''
Created on Sep 26, 2016

@author: Wajih-PC
'''
from NN.nntrain import nntrain
from NN.nnff import nnff

def saetrain(sae, x, opts):
    for i in range(0 , len(sae.AutoEncoder)):
        print ("Training AE " + str(i+1)+ "/" +str(len(sae.AutoEncoder)));
        sae.AutoEncoder[i] = nntrain(sae.AutoEncoder[i], x, x, opts,None,None,None)[0]
        t = nnff(sae.AutoEncoder[i], x, x);
        x = t.A[1];
        #remove bias term
        x = x[:,1::];
    return sae