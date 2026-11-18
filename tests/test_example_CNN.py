'''
Created on Aug 25, 2016

@author: Wajih-PC

Updated on 18th November 2026
@author: Lenovo Laptop
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
import matplotlib.pyplot as plt
from CNN import cnntrain, cnntest
from data import mnist
from CNN import ConvolutionalNeuralNetwork
import numpy as np

    
def test_example_CNN(interactive=None):
    if interactive is None:
        interactive = os.environ.get("DNN_NONINTERACTIVE") != "1" and matplotlib.get_backend().lower() != "agg"
    dataSet = mnist.load_mnist()
    if dataSet is not None:
        keys = dataSet.keys()
        for key in keys:
            if key == "train_x":
                train_x = dataSet[key]
                train_x = np.transpose(train_x.reshape(-1, 28, 28), (1, 2, 0)) / 255.0
            elif key == "train_y":
                train_y = dataSet[key]
                train_y = train_y.T
            elif key == "test_x":
                test_x = dataSet[key]
                test_x = np.transpose(test_x.reshape(-1, 28, 28), (1, 2, 0)) / 255.0
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
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure()
    plt.plot(cnn.rL)   
    plt.title("CNN Training Loss (rL)")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    fig.savefig(os.path.join(out_dir, "cnn_training_loss.png"), bbox_inches="tight", dpi=150)
    if interactive:
        plt.show()
    else:
        plt.close(fig)
    assert err < 0.12,"Too big error"
    print ("test_example_CNN completed")
    if interactive:
        input("Press [enter] to continue.") 

if __name__ == "__main__":
    test_example_CNN()