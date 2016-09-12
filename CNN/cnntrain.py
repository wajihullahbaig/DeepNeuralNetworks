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
 
    for i in range(0, opts["numepochs"]):
        print("epoch " , i+1 ,"/" , opts["numepochs"])
        tic = time()
        kk = np.random.permutation(range(m)) # For testing, go random
        #kk = np.arange(0,m,1); # For debugging, go sequentially
        for l in range (0,numbatches):
            print ("Batch number:"+str(l)+" out of :"+str(numbatches))
            batch_x = x[:,:,kk[l* batchsize  : (l+1) * batchsize]]
            batch_y = y[:,kk[l* batchsize  : (l+1) * batchsize]]
            net = cnnff.cnnff(net,batch_x)
            net = cnnbp.cnnbp(net,batch_y)
            net = cnnapplygrads.cnnapplygrads(net,opts)
            if not net.rL:
                net.rL.insert(0,net.L)
            net.rL.append(0.99 * net.rL[-1] + 0.01 * net.L)    
        toc = time()
        t = toc-tic
        print("epoch " , i+1 ,"/" , opts["numepochs"], ". Took " ,t, " seconds.")
    return net