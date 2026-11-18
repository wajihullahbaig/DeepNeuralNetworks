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
from data import mnist
from util import zscore
from util import normalize
from util.visualize import visualize
from NN import NeuralNetwork, nntrain,nntest
import matplotlib.pyplot as plt
import numpy as np
def test_example_NN(interactive=None):
    if interactive is None:
        interactive = os.environ.get("DNN_NONINTERACTIVE") != "1" and matplotlib.get_backend().lower() != "agg"
    plot_opt = 1 if interactive else 0
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
            
    train_x, mu, sigma = zscore.zscore(train_x)
    test_x =  normalize.normalize(test_x,mu,sigma)
        
    # ex1 vanilla nerual net
    np.random.seed(1) # Setting the random seed so that the weights are generated same as in matlab code - default is twister algorithm        
    nn = NeuralNetwork.NN(np.array([784, 100 ,10]))
    # numepochs = Number of full sweep through data
    # batchsize = Mean gradient step over this many samples
    options = {"numepochs":5,"batchsize":100,"plot":plot_opt}
    nn,L = nntrain.nntrain(nn,train_x,train_y,options,None,None,1)
    er,bad = nntest.nntest(nn,test_x,test_y)
    print(f"NN ex1 (tanh) error: {er:.6f}")
    assert (er < 0.08),"Too big error"
    del options
    del nn
      
    # ex2 neural net with L2 weight decay
    np.random.seed(1) # Setting the random seed so that the weights are generated same as in matlab code - default is twister algorithm            
    nn = NeuralNetwork.NN(np.array([784, 100 ,10]))
    nn.WeightPenaltyL2 = 1e-4
    options = {"numepochs":1,"batchsize":100,"plot":plot_opt}    
    nn,L = nntrain.nntrain(nn,train_x,train_y,options,None,None,2)
    er,bad = nntest.nntest(nn,test_x,test_y)
    print(f"NN ex2 (L2 decay) error: {er:.6f}")
    assert (er < 0.08),"Too big error"
    del nn
      
    # ex3 nerual net with dropout    
    np.random.seed(1) # Setting the random seed so that the weights are generated same as in matlab code - default is twister algorithm            
    nn = NeuralNetwork.NN(np.array([784, 100 ,10]))
    nn.DropoutFraction = 0.5
    nn,L = nntrain.nntrain(nn,train_x,train_y,options,None,None,3)
    er,bad = nntest.nntest(nn,test_x,test_y)
    print(f"NN ex3 (dropout) error: {er:.6f}")
    assert (er < 0.1),"Too big error"
    del nn
      
    # ex4 nerual net with sigmoid activation function    
    np.random.seed(1) # Setting the random seed so that the weights are generated same as in matlab code - default is twister algorithm            
    nn = NeuralNetwork.NN(np.array([784, 100 ,10]))
    nn.ActivationFunction = "sigm"
    nn.LearningRate = 1
    nn,L = nntrain.nntrain(nn,train_x,train_y,options,None,None,4)
    er,bad = nntest.nntest(nn,test_x,test_y)
    print(f"NN ex4 (sigmoid) error: {er:.6f}")
    assert (er < 0.1),"Too big error"
    del nn    
     
    # ex5 nerual net with softmax activation function and plotting functionality
    np.random.seed(1) # Setting the random seed so that the weights are generated same as in matlab code - default is twister algorithm            
    nn = NeuralNetwork.NN(np.array([784, 100 ,10]))
    nn.Output = "softmax" #Use the softmax output
    options = {"numepochs":1,"batchsize":1000,"plot":plot_opt}    
    nn,L = nntrain.nntrain(nn,train_x,train_y,options,None,None,5)
    er,bad = nntest.nntest(nn,test_x,test_y)
    print(f"NN ex5 (softmax) error: {er:.6f}")
    assert (er < 0.1),"Too big error"
    del nn
     
    
    # ex6 nerual net with sigmoid activation function and plotting functionality and training error
    # Split training data into training and validation
    vx = train_x[0:10000,:]    
    tx = train_x[9999:-1,:]
    vy = train_y[0:10000,:]
    ty = train_y[9999:-1,:]
    np.random.seed(1) # Setting the random seed so that the weights are generated same as in matlab code - default is twister algorithm            
    nn = NeuralNetwork.NN(np.array([784, 100 ,10]))
    nn.Output = "softmax" #Use the softmax output
    options = {"numepochs":5,"batchsize":1000,"plot":plot_opt}    
    nn,L = nntrain.nntrain(nn,tx,ty,options,vx,vy,6)
    er,bad = nntest.nntest(nn,test_x,test_y)
    print(f"NN ex6 (softmax + val) error: {er:.6f}")
    assert (er < 0.1),"Too big error"
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    ax = visualize(nn.W[0][:, 1:].T, None, None, None, save_path=os.path.join(out_dir, "nn_receptive_fields.png"))
    plt.close(ax.figure)
    del nn
    print ("test_example_NN completed")
    if interactive:
        input("Press [enter] to continue.")

if __name__ == "__main__":
    test_example_NN()