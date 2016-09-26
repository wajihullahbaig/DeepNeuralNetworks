'''
Created on Aug 25, 2016

@author: Wajih-PC
'''
import sys
from CNN import cnntrain, cnntest
sys.path.append('E:\RnD\Machine Learning\DNN\EclipseWorkSpace\DeepNeuralNetworks')
import matplotlib.pyplot as plt
from data import importMat
from CNN import ConvolutionalNeuralNetwork
import numpy as np

    
def test_example_CNN():
    dataSet = importMat.loadMatFile()
    if dataSet is not None:
        keys = dataSet.keys()
        for key in keys:
            if key == "train_x":
                train_x = dataSet[key]
                train_x = np.reshape(train_x.T,(28,28,-1),order="F")/255.0 # Fortran ordering to make sure we have the same as in Matlab
            elif key == "train_y":
                train_y = dataSet[key]
                train_y = train_y.T
            elif key == "test_x":
                test_x = dataSet[key]
                test_x = np.reshape(test_x.T,(28,28,-1),order="F")/255.0 # F ordering to make sure we have the same as in Matlab                
            elif key == "test_y":
                test_y = dataSet[key]
                test_y = test_y.T            
                    
    np.random.seed(1) # Setting the random seed so that the weights are generated same as in matlab code - default is twister algorithm            
    options = {"alpha":1,"numepochs":1,"batchsize":50}
    #ex1 Train a 6c-2s-12c-2s Convolutional neural network 
    #will run 1 epoch in about 200 second and get around 11% error. 
    #With 100 epochs you'll get around 1.2% error
    cnn = ConvolutionalNeuralNetwork.CNN(train_x,train_y)     
    cnn = cnntrain.cnntrain(cnn, train_x, train_y, options)
    err = cnntest.cnntest(cnn,test_x,test_y)[0]
    print("Error:"+str(err))    
    plt.figure()
    plt.plot(cnn.rL)   
    plt.show()
    assert err < 0.12,"Too big error"
    print ("test_example_CNN completed")
    input("Press [enter] to continue.") 
test_example_CNN()      
    