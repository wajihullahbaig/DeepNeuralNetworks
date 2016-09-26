# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:22:06 2016

@author: Wajih-PC
"""
import sys
from SAE.StackedAutoEncoder import SAE
from NN.NeuralNetwork import NN
from util.visualize import visualize
from NN.nntrain import nntrain
from NN.nntest import nntest
from SAE.saetrain import saetrain
sys.path.append('E:\RnD\Machine Learning\DNN\EclipseWorkSpace\DeepNeuralNetworks')
from data import importMat
import numpy as np

def test_example_SAE():
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
    
    #  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
    #  Setup and train a stacked denoising autoencoder (SDAE)
    np.random.seed(10) # Setting the random seed so that we reproduce results same as in matlab code - default is twister algorithm
    saeSizes = np.array([784, 100])        
    sae = SAE(saeSizes)
    sae.AutoEncoder[0].ActivationFunction = 'sigm'
    sae.AutoEncoder[0].LearningRate = 1
    sae.AutoEncoder[0].InputZeroMaskedFraction = 0.5 # This causes de-noising
    options = {"numepochs":1,"batchsize":100}    
    sae = saetrain(sae,train_x,options)
    visualize(sae.AutoEncoder[0].W[0][:,1::].T,None,None,None) 
    
    # Use the SDAE to initialize a FFNN
    nn = NN(np.array([784 ,100, 10]));
    nn.ActivationFunction = 'sigm';
    nn.LearningRate = 1;
    nn.W[0] = sae.AutoEncoder[0].W[0];
    # Train the FFNN
    options = {"numepochs":1,"batchsize":100}    
    nn = nntrain(nn, train_x, train_y, options,None,None,None)[0]
    [er, bad] = nntest(nn, test_x, test_y);
    assert er < 0.16, 'Too big error';
    
    print ("test_example_DBN completed")
    input("Press [enter] to continue.")          
test_example_SAE()      
    