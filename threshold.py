import numpy as np 


#after we have a candidate vector a to recontruct the signal, we apply soft thresholding 
# decreases the magnitude of every coefficient by the threshold, and sets coefficients whose magnitude is below the threshold to zero.
#this applies the sparsity penalty to all the coeffecient to induce sparcity 

def soft_threshold(z,threshold):
    T=np.sign(z)*np.maximum(np.abs(z)-threshold,0)
    return T

'''
#For example:
z=np.array([1,-2,3,0.1,1])
threshold=0.2

T=soft_threshold(z,threshold)
print(T) # gives output [ 0.8 -1.8  2.8  0.   0.8]
'''