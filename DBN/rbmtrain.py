'''
Created on Sep 24, 2016

@author: Wajih-PC
'''
from util import sigmrnd,sigm
import numpy as np
def rbmtrain(rbm, x, opts):
    assert(np.all(isinstance(i, float) for i in x)), "x must be a float"
    assert(np.all(x[:]>=0) and np.all(x[:]<=1)), "all data in x must be in [0:1]"
    m = x.shape[0];
    batchsize = opts["batchsize"]
    numbatches = int(m / batchsize)
    
    assert(numbatches%1 == 0), "numbatches not integer"
 
    for i  in range(0 , opts["numepochs"]):
        kk = np.random.permutation(range(m)) # For testing, go random
        #kk = np.arange(0,m,1); # For debugging, go sequentially
        err = 0
        for l in range(0 , numbatches):
            batch = x[kk[l* batchsize  : (l+1) * batchsize], :]            
            v1 = batch.copy();
            h1 = sigmrnd.sigmrnd(np.tile(rbm.C.T, (batchsize, 1))+np.dot(v1,rbm.W.T))
            v2 = sigmrnd.sigmrnd(np.tile(rbm.B.T, (batchsize, 1))+np.dot(h1,rbm.W))
            h2 = sigm.sigm(np.tile(rbm.C.T, (batchsize, 1))+np.dot(v2,rbm.W.T))
            
            c1 = np.dot(h1.T , v1)
            c2 = np.dot(h2.T , v2)

            rbm.vW = np.add(rbm.Momentum * rbm.vW , rbm.Alpha * (c1 - c2)     / float(batchsize))
            # Add a singleton dimension after sum
            rbm.vB = np.add(rbm.Momentum * rbm.vB , rbm.Alpha * sum(v1 - v2,0).T[:,None] / float(batchsize))
            rbm.vC = np.add(rbm.Momentum * rbm.vC , rbm.Alpha * sum(h1 - h2,0).T[:,None] / float(batchsize))

            rbm.W = rbm.W + rbm.vW;
            rbm.B = rbm.B + rbm.vB;
            rbm.C = rbm.C + rbm.vC;
            
            err = err + np.sum(np.power(v1 - v2 , 2)) / batchsize
        
        print("epoch " +str(i+1) +"/" +str(opts["numepochs"]) + ". Average reconstruction error is: " +str(err / numbatches))
    return rbm       