# The optimising algorithm for sparse coding that will be used is ISTA
#Iterative shrinkage-Thresholding algorithm is a iretative method for soloving sparse problem
#The goal of ISTA is to find a coeffecient vector a that both reconstructs X well and stays sparse
#ISTA does two things:

#1. First it improves reconstruction:  Starting form the current estimate a, it looks at the residual X-Da and uses
#it to move a in a direction that reduces reconstruction error 
# in other words it moves towards the negetive gradient direction 

#2. It performs soft thresholding to enforce sparsity

# and repeats it again and agian, updates all coeffecietly simultaneously
# but problem is it requires many iterations 
import numpy as np
from synthetic_data import synthetic_data
from threshold import soft_threshold

def ista(X,D,lam,n_iterations):
    eta=1/(np.linalg.norm(D,2)**2)
    a=np.zeros(D.shape[1])#this is our arbitary a to start with having all zero coeffecient maching the given D sahpe
    for i in range(n_iterations):
        reconstruction=D@a
        residual=X-reconstruction
        direction=D.T@residual
        a_temp=a+eta*direction
        a=soft_threshold(a_temp,eta*lam)
    
    return a




