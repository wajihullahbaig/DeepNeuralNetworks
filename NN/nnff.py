import numpy as np
from NN import NeuralNetwork
from util import tanh_opt, sigm
def nnff(nn,x,y):
    #NNFF performs a feed forward pass
    # nn = nnff(nn,x,y) returns a neural network structure with updated layer activations,
    # error and loss (nn.a,nn.e,nn.L)
    
    n = nn.N
    m = x.shape[0]
    temp = np.ones([m,x.shape[1]+1], dtype=np.float64)
    temp[:,1:] = x;
    #temp[:,1:] = np.copy(x)
    
    # Allocate space for activation, error and loss
    # Since matlab allows structs to have cells/arrays etc on the fly, 
    # This is not possible with python. We declare members and then use them    
    nn.A[0] = np.copy(temp);
    
    # Feedforward pass
    for i in range(1,n-1):
        if nn.ActivationFunction == 'sigm':
            #Caluclate the unit's outputs (including the units bias term)
            r = np.dot(nn.A[i-1],nn.W[i-1].T)
            nn.A[i] = sigm.sigm(r)                       
        elif nn.ActivationFunction == 'tanh_opt':
            r = np.dot(nn.A[i-1],nn.W[i-1].T)
            nn.A[i] = tanh_opt.tanh_opt(r)
            
        # Droput
        if nn.DropoutFraction > 0:
            if nn.Testing == True:
                nn.A[i] = np.multiply(nn.A[i],(1-nn.DropoutFraction))
            else:
                nn.DropOutMask[i] = np.random.uniform(1,0,nn.A[i].shape)
                # Get the indices and convert them to floats
                indices = [nn.DropOutMask[i] > nn.DropoutFraction]
                tempMask = np.where(indices,1,0).astype(np.float64) # convert binary to float
                nn.DropOutMask[i] = tempMask.squeeze() # Remove the extra unwanted dimension
                nn.A[i] = np.multiply(nn.A[i],nn.DropOutMask[i])
                nn.A[i][nn.A[i]==0.] = 0 # Remove the negative zeros that propagate to cause large errors in testing phase
                 

        # Calculate running exponential activations for use with sparsity
        
        if nn.NonSparsityPenalty > 0:
            pass
        
        #Add Bias term
        biasTerm = np.ones(shape = (nn.A[i].shape[0],nn.A[i].shape[1]+1.0), dtype=np.float64)
        biasTerm[:,1:] = nn.A[i]
        nn.A[i] = biasTerm
    
    if nn.Output == 'sigm':       
        nn.A[n-1] = sigm.sigm(np.dot(nn.A[n-2],nn.W[n-2].T))
    elif nn.Output == 'linear':
        nn.A[n-1] = np.dot(nn.A[n-2],nn.W[n-2].T)
    elif nn.Output == 'softmax':
        nn.A[n-1] = np.dot(nn.A[n-2],nn.W[n-2].T)
        maxVector = nn.A[n-1].argmax(axis=1) # Returns a flattened array
        maxVector = maxVector.reshape(maxVector.shape[0],1) # Convert to coloumn vector Mx1
        nn.A[n-1] = np.exp(np.subtract(nn.A[n-1],maxVector))
        sumVector = np.sum(nn.A[n-1],1)
        sumVector = sumVector.reshape(sumVector.shape[0],1) # Convert to coloumn vector Mx1
        nn.A[n-1] = np.true_divide(nn.A[n-1],sumVector)
    
    # Error and Loss
    nn.E = y - nn.A[n-1]
    
    if nn.Output == 'sigm' or nn.Output == 'linear':
        nn.L = 0.5* np.sum(np.power(nn.E,2))/m
    elif nn.Output == "softmax":
        # Try removing values that are too small. This helps countering the invalid value encountered in log warning
        eps = 1e-50
        nn.A[n-1][nn.A[n-1]<eps]=eps
        nn.L = -1.0 * np.sum(np.multiply(y,np.log(nn.A[n-1])))/m
    return nn
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     