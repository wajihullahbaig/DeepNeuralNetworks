# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:22:06 2016

@author: Wajih-PC
"""
import sys
from DBN.dbntrain import dbntrain
from DBN.dbnunfoldtonn import dbnunfoldtonn
from NN.nntrain import nntrain
from NN.nntest import nntest
from util.visualize import visualize
sys.path.append('E:\RnD\Machine Learning\DNN\EclipseWorkSpace\DeepNeuralNetworks')
from data import importMat
from DBN import DeepBeliefNetwork
import numpy as np
def test_example_DBN():
    dataSet = importMat.loadMatFile()
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
    visualize(dbn.RBM[0].W.T,None,None,None) 
    
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
    options = []
    options = {"numepochs":1,"batchsize":100}
    nn = nntrain(nn, train_x, train_y, options,None,None,None)[0]
    er = nntest(nn, test_x, test_y)[0]

    assert er < 0.10, 'Too big error'
    
    print ("test_example_DBN completed")
    input("Press [enter] to continue.")          
test_example_DBN()      
    