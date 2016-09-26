'''
Created on Sep 26, 2016

@author: Wajih-PC
'''
from NN.NeuralNetwork import NN
import numpy as np
class SAE: # This also works as a denoising SAE
    def __init__(self,size):
        self.AutoEncoder = {} 
        for u in range(1,len(size)):
            self.AutoEncoder[u-1] = NN(np.array([size[u-1],size[u],size[u-1]])) 