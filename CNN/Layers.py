'''
Created on Aug 25, 2016
This is where we define the convolutional neural network layers
@author: Wajih-PC
'''

# Python newbie alert!
# Note that the variable 'A' was shared through out the different layers
# This is probably pythonic behaviour, unlike C#,C++,Java where inheritance 
# from 'Layer' assured independent memeber 'A' in the following class structure.
# Missing a call to Layer.__init__(self) cause the single copy of 'A' to be shared 
# by the layers
class Layer:
   
    def __init__(self): 
        A = None
        D = None
        self.A = {}
        self.D = {}
    
    
class InputLayer(Layer):
    def __init__(self):        
        Layer.__init__(self)

class ConvolutionalLayer(Layer):
        
    def __init__(self,outputMaps,kernelSize):
        Layer.__init__(self)
        self.OutputMaps = 0
        self.KernelSize = 0
        self.K = {}     
        self.B = {}    
        self.dK = {}
        self.dB = {}
        self.OutputMaps = outputMaps
        self.KernelSize = kernelSize

class ScaleLayer(Layer):
    def __init__(self,scale):
        Layer.__init__(self)        
        self.B = {}    
        self.Scale = scale

