import numpy as np
from ista import ista

from synthetic_data import synthetic_data
import matplotlib.pyplot as plt


def lambda_sweep(
    n_iterations,lam=None):
    D,a_true,X=synthetic_data()
    if lam is None:
        lam=np.linspace(0,2,101)
    estimated_coeffecient=[]
    errors=[]
    for x in lam:
        a=ista(X,D,x,n_iterations)
        estimated_coeffecient.append(a)
        error=np.sum((X-D@a)**2)
        errors.append(error)
    E=np.array(estimated_coeffecient)
    number_active=np.sum(np.abs(E)>1e-6,axis=1)
    y1=number_active
    y2=np.array(errors)
    fig,(ax1,ax2)=plt.subplots(ncols=1,nrows=2,sharex=True)
    ax1.plot(lam,y1,label='active coeffecient')
    ax1.set_title('active coeffecient vs lambda')
    ax1.set_ylabel('active coeffecients')
    ax2.plot(lam,y2,label='recontruction error')
    ax2.set_title('reconstruction error vs lamda')
    ax2.set_xlabel('lamda')
    ax2.set_ylabel('reconstruction error')
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(
    "figures/lambda_sweep.png",
    dpi=300,
    bbox_inches="tight"
    )
    plt.show()





    
    
