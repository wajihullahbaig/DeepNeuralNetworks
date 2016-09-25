# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:15:29 2016
The FeedFoward Backpropaagte Neural Network
A neural network structure with n=numel(architecture)
layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]
@author: Wajih-PC
"""
from NN import Collections as collections
import numpy as np
class NN:
    def __init__(self,architecture):
        self.ActivationFunction               = 'tanh_opt';   #  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
        self.LearningRate                     = 2;            #  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
        self.Momentum                         = 0.5;          #  Momentum
        self.ScalingLearningRate              = 1;            #  Scaling factor for the learning rate (each epoch)
        self.WeightPenaltyL2                  = 0;            #  L2 regularization
        self.NonSparsityPenalty               = 0;            #  Non sparsity penalty
        self.SparsityTarget                   = 0.05;         #  Sparsity target
        self.InputZeroMaskedFraction          = 0;            #  Used for Denoising AutoEncoders
        self.DropoutFraction                  = 0;            #  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
        self.Testing                          = False;            #  Internal variable. nntest sets this to one.
        self.Output                           = 'sigm'
        self.DropOutMask                      = None;
        self.Size = architecture
        self.N =(self.Size).size
        self.TotalLayers = len(range(1,self.N))
        #Empty allocations
        self.W = [None for i in range(self.TotalLayers)]
        self.VW = [None for i in range(self.TotalLayers) ]
        self.P = [None for i in range(self.TotalLayers+1)]
        self.DW = [None for i in range(self.TotalLayers+1)]
        self.A = [None for i in range(self.TotalLayers+1)]
        self.E = None
        self.L = None
        self.DropOutMask = [None for i in range(self.TotalLayers+1)]
        for i in range(1,self.TotalLayers+1):
            #Weights and weight momentums
            rows = self.Size[i]
            cols = self.Size[i-1]
            #self.W[i-1].ResizeToFit(rows,cols+1)
            temp  = (np.random.uniform(0,1,[cols+1,rows])-0.5)*2*4*np.sqrt(6/(rows+cols))
            # Transpose the random values to make the same matrix as in Matlab            
            self.W[i-1] = np.transpose(temp) 
            self.VW[i-1] = np.zeros(self.W[i-1].shape)
            
            #average activations (for use with sparsity)
                      