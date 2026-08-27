import numpy as np
import matplotlib.pyplot as plt
from threshold import soft_threshold
from synthetic_data import synthetic_data
from pathlib import Path

def evaluate_convergence(lam,n_iterations):
    D,a_true,X=synthetic_data()
    a=np.zeros(D.shape[1])
    eta=1/(np.linalg.norm(D,2)**2)
    errors=[]
    iterations=[]
    O=[]
    for i in range(n_iterations):
        reconstruction=D@a
        residual=X-reconstruction
        direction=D.T@residual
        a_temp=a+eta*direction
        a=soft_threshold(a_temp,eta*lam)
        error=np.sum((X-D@a)**2)
        objective=0.5*(np.sum((X-D@a)**2))+lam*(np.sum(np.abs(a)))
        errors.append(error)
        iterations.append(i)
        O.append(objective)
    x_points=np.array(iterations)
    y1_points=np.array(errors)
    y2_points=np.array(O)
    fig,(ax1,ax2)=plt.subplots(ncols=1,nrows=2,sharex=True)
    ax1.plot(x_points,y1_points,label='reconstruction error')
    ax1.set_title('Reconstruction error vs iterations')
    ax1.set_ylabel('reconstuction error')
    ax2.plot(x_points,y2_points,label='objective value')
    ax2.set_title('Objective value vs iteration')
    ax2.set_ylabel('objective_value')
    ax2.set_xlabel('iterations')
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(
    "figures/convergence.png",
    dpi=300,
    bbox_inches="tight"
 )

    plt.show()

