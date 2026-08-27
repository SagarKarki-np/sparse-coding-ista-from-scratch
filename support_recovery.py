import numpy as np
from ista import ista
from synthetic_data import synthetic_data

def support_recovery(n_iterations,lam=None):
    D,a_true,X=synthetic_data()
    if lam is None:
        lam=np.linspace(0,2,101)
    estimated_coeffecient=[]
    for x in lam:
        a=ista(X,D,x,n_iterations)
        estimated_coeffecient.append(a)
    E=np.array(estimated_coeffecient)
    number_of_active_c=np.sum(np.abs(E)>1e-6,axis=1)
    active_in_a_true=np.sum(np.abs(a_true)>1e-6)
    a_with_same_active_count=E[number_of_active_c==active_in_a_true]
    estimated_indices=[]
    for x in range(a_with_same_active_count.shape[0]):
        indices=np.where(np.abs(a_with_same_active_count[x])>1e-6)[0]
        estimated_indices.append(indices)
    true_indices=np.where(np.abs(a_true)>1e-6)[0]
    lam_values=lam[number_of_active_c == active_in_a_true]
    for i, j in zip(lam_values,estimated_indices):
        if np.array_equal(true_indices,j):
                print(
                   "lamda:",i,'--',j,'=','exact recovery'
                )
        else:
                print(
                    "lamda:",i,'--',j,'=','not exact recovery'
                )
        
        

    

support_recovery(50)
