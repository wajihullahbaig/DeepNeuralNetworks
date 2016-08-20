'''
Created on Aug 10, 2016

@author: Wajih-PC
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import ioff, ion
def nnupdatefigures(nn,figureNo,L,opts,i):
    if i > 0: #Dont plot the first points, its only a Point
        x_ax = np.zeros((1,i+1),dtype = np.float64)
        for v in range(i+1):
            x_ax[0,v] = v+1 # Add a one, this helps in show epochs starting from 1 in plots
            
        # Create legend
        if opts["validation"] == 1:
            M = ["Training","Validation"]
        else:
            M = ["Training"]

        # Create data fop plots
        if nn.Output == "softmax":
            plot_x = x_ax.T
            plot_ye = L.Training.E[0:i+1]         
            plot_yfrac = L.Training.E_Frac[0:i+1]
        else:
            plot_x = x_ax.T
            plot_ye = L.Training.E[0:i+1]
        
        #add error on validation data if present
        if opts["validation"] == 1:
            plot_x       = np.stack((plot_x, x_ax.T)) # Combine arrays
            plot_x =     plot_x.squeeze() # Remove the unwanted dimension
            plot_x = plot_x.T
            plot_ye      = np.stack((plot_ye,L.Validity.E[0:i+1]))
            plot_ye = plot_ye.squeeze()
            plot_ye = plot_ye.T
        #add classification error on validation data if present
        if opts["validation"] == 1 and nn.Output == "softmax":
            plot_yfrac   = np.stack((plot_yfrac, L.Validity.E_Frac[0:i+1]));
            plot_yfrac = plot_yfrac.squeeze()  
            plot_yfrac = plot_yfrac.T
        
        #plotting
        if plt.fignum_exists(str(figureNo)) == False:
            plt.ion()
        plt.figure(figureNo)     
        if nn.Output == "softmax":
            ax = plt.subplot(121)
            if len(M) < 2:
                if i == 1:
                    plt.plot(plot_x,plot_ye,label = M, color = 'b')
                    plt.legend(loc="upper right") 
            if len(M) > 1:
                    ax.plot(plot_x[:,0],plot_ye[:,0],label= M[0],color = 'blue')
                    ax.plot(plot_x[:,1],plot_ye[:,1],label= M[1],color = 'red') 
                    plt.legend(M,loc="upper right")

                     
            plt.xlabel("Number of epochs")
            plt.ylabel("Error")
            plt.title("Error")      
            plt.xlim(0, opts["numepochs"]+1)                  
            plt.draw()
            plt.pause(0.1)
            
            
            ax = plt.subplot(122)
            if len(M) < 2:
                if i == 1:
                    plt.plot(plot_x,plot_yfrac,label = M[0], color = 'b') 
            if len(M) > 1:
                    ax.plot(plot_x[:,0],plot_ye[:,0],label= M[0],color = 'blue')
                    ax.plot(plot_x[:,1],plot_ye[:,1],label= M[1],color = 'red') 
                    plt.legend(M,loc="upper right")                
                                
            plt.xlabel("Number of epochs")
            plt.ylabel("Misclassification rate")
            plt.title("Misclassification rate")
            plt.xlim(0, opts["numepochs"]+1)
            plt.draw()
            plt.pause(0.1)
            
        else:
            plt.xlabel("Number of epochs")
            plt.ylabel("Error")
            plt.title("Error")      
            plt.plot(plot_x,plot_ye,label = M if i == 1 else "",color = 'b')
            plt.xlim(0, opts["numepochs"]+1)        
            plt.legend(loc="upper right")
            plt.draw()
            plt.pause(0.01)
        