'''
Created on Aug 26, 2016

@author: Wajih-PC
'''

from CNN import Layers
import numpy as np
class CNN:
    # Define the deep layers        
    layers = [Layers.InputLayer(),Layers.ConvolutionalLayer(6,5),Layers.ScaleLayer(2),Layers.ConvolutionalLayer(12,5),Layers.ScaleLayer(2)]
    inputMaps = 1
    mapsize = 0   
    ffB = None #Bias
    ffW = None #Weights
    FV = None # feature vector
    rL = 1*[None]
    L = 0; # Loss
    def __init__(self,x,y):        
        self.mapsize =  np.asarray(np.shape(x[:,:,0]))
        for l in range(0,len(self.layers)):
            if isinstance(self.layers[l], Layers.ScaleLayer):
                self.mapsize = np.divide(self.mapsize,self.layers[l].Scale) 
                assert (all(np.floor(self.mapsize)== self.mapsize)),"Layer "+ str(l) +" size must be integer. Actual: "+ str(self.mapsize)                
                for j in range(0 , self.inputMaps):
                    self.layers[l].B[0,j] = 0.0                
            elif isinstance(self.layers[l],Layers.ConvolutionalLayer):
                self.mapsize = self.mapsize - self.layers[l].KernelSize + 1;    
                fan_out = self.layers[l].OutputMaps*self.layers[l].KernelSize*self.layers[l].KernelSize                
                for j in range(0,self.layers[l].OutputMaps):
                    fan_in = self.inputMaps * self.layers[l].KernelSize*self.layers[l].KernelSize
                    for i in range(0,self.inputMaps):
                            # Ramdom initialization. Transpose it so that we have same sequence as in Matlab
                            self.layers[l].K[i,j] =  ((np.random.uniform(0,1,[self.layers[l].KernelSize,self.layers[l].KernelSize])-0.5)*2*np.sqrt(6/(fan_in+fan_out))).T                                        
                    self.layers[l].B[j] = 0.0
                     
                self.inputMaps = self.layers[l].OutputMaps
            
        # 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
        # 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
        # 'ffb' is the biases of the output neurons.
        # 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
                
        fvnum = np.prod(self.mapsize,axis = 0) * self.inputMaps
        onum =  np.shape(y)[0]       
        self.ffB = np.zeros(shape = (onum,1),dtype = np.float64)
        self.ffW = ((np.random.uniform(0,1,[fvnum,onum])-0.5)*2*np.sqrt(6/(onum+fvnum))).T           