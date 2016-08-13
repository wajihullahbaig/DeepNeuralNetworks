# -*- coding: utf-8 -*-
"""
Created on Sun May 22 14:14:55 2016

@author: Wajih-PC
"""
import scipy.io
import sys
def loadMatFile():
    path = "E:\RnD\Machine Learning\DNN\EclipseWorkSpace\DeepNeuralNetworks\data"
    filename = "\mnist_uint8"
    fullname = path+filename
    mat = None
    print("Reading file:"+fullname)
    try:
        mat = scipy.io.loadmat(fullname)
    except IOError as err:
         print ("I/O error({0}): {1}".format(err.errno, err.strerror))
         sys.exit();
    else:
        print("Successfully loaded mat file")  
        for k in mat.keys():
            if not k.startswith('__'):
                print(k + " " + mat[k].dtype.name + " " + str(mat[k].shape))
    return mat