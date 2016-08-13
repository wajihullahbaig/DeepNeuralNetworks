'''
Created on Aug 10, 2016

@author: Wajih-PC
'''
import matplotlib.pyplot as plt
import numpy as np
def nnupdatefigures(nn,fhandle,L,options,i):
    if i > 0: #Dont plot the first points, its only a Point
        x_ax = np.zeros([1,i+1],dtype = np.float64)
        for v in range(i+1):
            x_ax[0,v] = v
            
        # Create legend
        if options["validation"] == 1:
            M = ["Training","Validation"]
        else:
            M = ["Training"]

        # Create data fop plots
        if nn.Output == "softmax":
            plot_x = x_ax.T
            plot_ye = L.Training.E         
            plot_yfrac = L.Training.E_Frac
        else:
            plot_x = x_ax.T
            plot_ye = L.Training.E
        
        #add error on validation data if present
        if options["validation"] == 1:
            plot_x       = [plot_x, x_ax.T]
            plot_ye      = [plot_ye,L.val.e.T]
        
        #add classification error on validation data if present
        if options["validation"] == 1 and nn.Output == "softmax":
            plot_yfrac   = [plot_yfrac, L.val.E_Frac.T];        
        
        #plotting
        plt.plot(plot_x,plot_ye)
        plt.show()