'''
Created on Sep 24, 2016

@author: Wajih-PC
'''
import numpy as np
from util.scaledimage import scaledimage
def visualize(X, mm, s1, s2):
    
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
            s1 = s; s2 = s;        
        #its a square, so data is probably an image
        num=np.ceil(np.sqrt(N));
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
    
    
    scaledimage(a)    
    