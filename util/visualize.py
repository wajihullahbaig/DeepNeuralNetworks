'''
Created on Sep 24, 2016

@author: Wajih-PC

Updated on 18th November 2026
@author: Lenovo Laptop
'''
import os
import matplotlib.pyplot as plt
import numpy as np
from util.scaledimage import scaledimage
def visualize(X, mm=None, s1=None, s2=None, ax=None, save_path=None):
    
    if mm is None:
        mm = [np.min(X[:]), np.max(X[:])]
            
    if s1 is None:
        s1 = 0;
    if s2 is None:
        s2 = 0;
        
    D,N= X.shape[0],X.shape[1]
    s=np.sqrt(D);
    if s==np.floor(s) or (s1 !=0 and s2 !=0):
        if s1 ==0 or s2 ==0:
            s1 = int(s); s2 = int(s);
        else:
            s1 = int(s1); s2 = int(s2);
        #its a square, so data is probably an image
        num=int(np.ceil(np.sqrt(N)));
        a=mm[1]*np.ones(shape= (num*s2+num-1,num*s1+num-1))
        x=0;
        y=0;
        for i in range (0,N):
            # Add singleton dimension to im
            im = np.reshape(X[:,i][:,None],(s1,s2))
            #scaledimage(im.T)
            a[int(x*s2+x) : int(x*s2+s2+x), int(y*s1+y) : int(y*s1+s1+y)]=im[:,:]
            x=x+1;
            if x>=num:
                x=0;
                y=y+1;
                    
        d= True;
    else:    
        #there is not much we can do
        a=X;
    
    
    out_ax = scaledimage(a, ax=ax)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out_ax.figure.savefig(save_path, bbox_inches="tight", dpi=150)
    return out_ax
    