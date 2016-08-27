'''
Created on Aug 25, 2016
This is where we define the convolutional neural network layers
@author: Wajih-PC
'''


class Layer:pass
    
class InputLayer(Layer):
    def __init__(self):        
        pass

class ConvolutionalLayer(Layer):
    OutputMaps = 0
    KernelSize = 0
    K = {}     
    B = {}    
    def __init__(self,outputMaps,kernelSize):
        self.OutputMaps = outputMaps
        self.KernelSize = kernelSize

class ScaleLayer(Layer):
    Scale = 0
    B = {}
    def __init__(self,scale):
        self.Scale = scale

