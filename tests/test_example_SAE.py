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
from SAE.StackedAutoEncoder import SAE
from NN.NeuralNetwork import NN
from util.visualize import visualize
from NN.nntrain import nntrain
from NN.nntest import nntest
from SAE.saetrain import saetrain
from data import mnist
import numpy as np

def test_example_SAE(interactive=None):
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
    
    #  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
    #  Setup and train a stacked denoising autoencoder (SDAE)
    np.random.seed(0) # Match DeepLearnToolbox rand('state', 0) baseline
    saeSizes = np.array([784, 100])        
    sae = SAE(saeSizes)
    sae.AutoEncoder[0].ActivationFunction = 'sigm'
    sae.AutoEncoder[0].LearningRate = 1
    sae.AutoEncoder[0].InputZeroMaskedFraction = 0.5 # This causes de-noising
    options = {"numepochs":1,"batchsize":100}    
    sae = saetrain(sae,train_x,options)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    ax = visualize(sae.AutoEncoder[0].W[0][:,1::].T, None, None, None, save_path=os.path.join(out_dir, "sae_encoder_weights.png"))
    if not interactive:
        plt.close(ax.figure)
    
    # Use the SDAE to initialize a FFNN
    nn = NN(np.array([784 ,100, 10]));
    nn.ActivationFunction = 'sigm';
    nn.LearningRate = 1;
    nn.W[0] = sae.AutoEncoder[0].W[0];
    # Train the FFNN
    options = {"numepochs":2,"batchsize":100}    
    nn = nntrain(nn, train_x, train_y, options,None,None,None)[0]
    [er, bad] = nntest(nn, test_x, test_y);
    print(f"SAE initialized NN error: {er:.6f}")
    assert er < 0.16, 'Too big error';
    
    print ("test_example_SAE completed")
    if interactive:
        input("Press [enter] to continue.")          

if __name__ == "__main__":
    test_example_SAE()