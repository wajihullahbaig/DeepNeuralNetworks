'''
Created on Jun 8, 2016

@author: Wajih-PC
'''

import numpy as np
from numpy import double, float64
from NN import nnff, nnbp, nnapplygrads, nneval, nnupdatefigures
from time import time
from NN import Collections
import matplotlib.pyplot as plt

def nntrain(nn, train_x, train_y, opts):    
    m = train_x.shape[0]
    batchsize = opts["batchsize"]
    numepochs = opts["numepochs"]
    opts["validation"] = 0
    numbatches = m/batchsize
    loss = Collections.Loss(numepochs)
    figureNo = 1
    if "plot" in opts and opts["plot"]== 1:
        fhandle = plt.figure();
    
    assert (numbatches%1 == 0),"numbatches must be an integer"
    
    L = np.zeros(shape=(numepochs*numbatches,1),dtype=np.float64)
    n = 0
    for i in range(0,numepochs):
        tic = time()
        kk = np.random.permutation(range(m)) # For testing, go random
        #kk = np.arange(0,m,1); # For debugging, go sequentially
        for l in range(0,round(numbatches)):
            batch_x = train_x[kk[l* batchsize  : (l+1) * batchsize], :]
        
            #Add noise to input (for use in denoising autoencoder)
            if (nn.InputZeroMaskedFraction != 0):
                temp = np.random.uniform(0,1,batch_x.shape)
                temp= np.where(temp<=nn.InputZeroMaskedFraction,0,temp)
                batch_x = np.multiply(batch_x,temp)
                
            batch_y = train_y[kk[l* batchsize  : (l+1) * batchsize], :]
            nn = nnff.nnff(nn, batch_x, batch_y)
            nn = nnbp.nnbp(nn)
            nn = nnapplygrads.nnapplygrads(nn)
            
            L[n] = nn.L
            
            n= n+1
        toc = time()
        t = toc-tic
        
        str_pref = ""
        if opts["validation"] == 1:
            loss = nneval.nneval(nn,loss,train_x,train_y,None,None,i)
            str_pref = (":Full batch train mse = {0},val mse = {1}".format(loss.Training.E[i],loss.Validity.E[i]))
        else:
            loss = nneval.nneval(nn,loss,train_x,train_y,None,None,i)
            str_pref = " Full-batch train err = {0}".format(loss.Training.E[i])
            nn.LearningRate = nn.LearningRate*nn.ScalingLearningRate
        if fhandle is not None:
           pass
           nnupdatefigures.nnupdatefigures(nn, figureNo, loss, opts, i)
            
        print("epoch " , i+1 ,"/" , opts["numepochs"], ". Took " ,t, " seconds. Mini-batch mean squared error on training set is " , np.mean(L[n-numbatches:n-1]) ,str_pref)
        nn.LearningRate = nn.LearningRate * nn.ScalingLearningRate;
    return nn,L     