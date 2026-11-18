# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:22:06 2016

@author: Wajih-PC

Updated on 18th November 2026
@author: Lenovo Laptop
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
import matplotlib.pyplot as plt
from DBN.dbntrain import dbntrain
from DBN.dbnunfoldtonn import dbnunfoldtonn
from NN.nntrain import nntrain
from NN.nntest import nntest
from util.visualize import visualize
from data import mnist
from DBN import DeepBeliefNetwork
import numpy as np
def test_example_DBN(interactive=None):
    if interactive is None:
        interactive = os.environ.get("DNN_NONINTERACTIVE") != "1" and matplotlib.get_backend().lower() != "agg"
    dataSet = mnist.load_mnist()
    if dataSet is not None:
        keys = dataSet.keys()
        for key in keys:
            if key == "train_x":
                train_x = dataSet[key]
                train_x = train_x/255.0
            elif key == "train_y":
                train_y = dataSet[key]
                train_y = train_y
            elif key == "test_x":
                test_x = dataSet[key]
                test_x = test_x/255.0
            elif key == "test_y":
                test_y = dataSet[key]
                test_y = test_y            
            
    # ex1 train a 100 hidden unit RBM and visualize its weights       
    np.random.seed(10) # Setting the random seed so that we reproduce results same as in matlab code - default is twister algorithm
    options = {"numepochs":3,"batchsize":100,"momentum":0,"alpha":1}
    dbnSizes = np.array([100]);        
    dbn = DeepBeliefNetwork.DBN(train_x,dbnSizes,options)
    dbn = dbntrain(dbn,train_x,options)    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    ax = visualize(dbn.RBM[0].W.T, None, None, None, save_path=os.path.join(out_dir, "dbn_rbm_weights.png"))
    if not interactive:
        plt.close(ax.figure)
    
    # ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
    np.random.seed(10) # Setting the random seed so that we reproduce results same as in matlab code - default is twister algorithm 
    #train dbn
    dbnSizes = np.array([100,100])
    options = {"numepochs":1,"batchsize":100,"momentum":0,"alpha":1}
    del dbn
    dbn = DeepBeliefNetwork.DBN(train_x,dbnSizes,options)
    dbn = dbntrain(dbn, train_x, options);

    #unfold dbn to nn
    nn = dbnunfoldtonn(dbn, 10);
    nn.ActivationFunction = 'sigm'
    #train nn
    options = {"numepochs":1,"batchsize":100}
    nn = nntrain(nn, train_x, train_y, options,None,None,None)[0]
    er = nntest(nn, test_x, test_y)[0]
    print(f"DBN unfolded NN error: {er:.6f}")

    assert er < 0.10, 'Too big error'
    
    print ("test_example_DBN completed")
    if interactive:
        input("Press [enter] to continue.")          

if __name__ == "__main__":
    test_example_DBN()