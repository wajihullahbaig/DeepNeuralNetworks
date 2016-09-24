'''
Created on Sep 24, 2016

@author: Wajih-PC
'''
import numpy as np
class RBM:
    def __init__(self):
        self.Alpha = None
        self.Momentum = None                
        self.W  = None
        self.vW = None
        self.B  = None
        self.vB = None
        self.C  = None
        self.vC = None


class DBN:
    def __init__(self,x,dbnSizes,options):
        totalRBMs = len(dbnSizes)
        self.RBM = [ RBM() for i in range(totalRBMs)]
        self.dbnSizes = x.shape[1]
        self.dbnSizes = np.append(self.dbnSizes,dbnSizes)
        for u in range(0,totalRBMs):
            
            self.RBM[u].Alpha = options["alpha"]
            self.RBM[u].Momentum = options["momentum"]
            
            self.RBM[u].W = np.zeros(shape=(self.dbnSizes[u+1],self.dbnSizes[u]),dtype=np.float64)
            self.RBM[u].vW = np.zeros(shape=(self.dbnSizes[u+1],self.dbnSizes[u]),dtype=np.float64)
        
            self.RBM[u].B = np.zeros(shape=(self.dbnSizes[u],1),dtype=np.float64)
            self.RBM[u].vB = np.zeros(shape=(self.dbnSizes[u],1),dtype=np.float64)
        
            self.RBM[u].C = np.zeros(shape=(self.dbnSizes[u+1],1),dtype=np.float64)
            self.RBM[u].vC = np.zeros(shape=(self.dbnSizes[u+1],1),dtype=np.float64)
                