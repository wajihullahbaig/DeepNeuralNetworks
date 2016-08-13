# -*- coding: utf-8 -*-
"""
Created on Tue May 24 20:40:54 2016

@author: Wajih-PC
"""
import numpy as np

class Training:
    E = None;
    E_Frac = None
    def __init__(self, epochs):
        self.E = np.zeros([epochs,1],dtype = np.float64 );
        self.E_Frac = np.zeros([epochs,1],dtype = np.float64 );
        
class Validity:
    E = None;
    E_Frac = None
    def __init__(self, epochs):
        self.E = np.zeros([epochs,1],dtype = np.float64 );
        self.E_Frac = np.zeros([epochs,1],dtype = np.float64 );
    
        
class Loss:
    Training = None;
    Validity = None;
    def __init__(self,epochs):
        self.Training = Training(epochs)
        self.Validity = Validity(epochs)
    
