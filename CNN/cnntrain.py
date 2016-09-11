'''
Created on Aug 28, 2016

@author: Wajih-PC
'''
import sys
from time import time
import numpy as np
from CNN import cnnff,cnnbp,cnnapplygrads
def cnntrain(net,x,y,opts):
    m = x.shape[2]
    numbatches = int(m/opts["batchsize"])
    batchsize = opts["batchsize"]
    numepochs = opts["numepochs"]
    if numbatches % 1 != 0:
        sys.exit("numbatches not integer")
    n = 0
    L = np.zeros(shape=(numepochs*numbatches,1),dtype=np.float64)
    for i in range(0, opts["numepochs"]):
        print("epoch " , i+1 ,"/" , opts["numepochs"])
        tic = time()
        kk = np.random.permutation(range(m)) # For testing, go random
        #kk = np.arange(0,m,1); # For debugging, go sequentially
        for l in range (0,numbatches):
            batch_x = x[:,:,kk[l* batchsize  : (l+1) * batchsize]]
            batch_y = y[:,kk[l* batchsize  : (l+1) * batchsize]]
            net = cnnff.cnnff(net,batch_x)
            net = cnnbp.cnnbp(net,batch_y)
            net = cnnapplygrads.cnnapplygrads(net,opts)
    
            if net.rL[0] is None:
                net.rL.insert(0,net.L)
            else:  
                net.rL.insert(n+1,0.99 * net.rL(-1) + 0.01 * net.L)    