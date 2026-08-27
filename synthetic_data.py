
import numpy as np
'''
np.random.seed(40) #for reproducibility.

#Creating a random dictionary D
D=np.random.randn(10,20)

#randn because to decrease dictionary coherance, randin is a good choice here because positive and negative entries usually give less mutually similar atoms than rand()
#which would make the sparse recovery easier
#this creates D with 20 atoms each with 10 cordinates (different lengths and directions)

D=D/np.linalg.norm(D,axis=0,keepdims=True) 
#D can have different lengths and hence longer lengths would get less penalty and get more advantage
#hence we normalise D to make them unit vectors and hence remove bias along axis=0 as atoms are the coloumns

#next we create vector a to create the signal using just 3 atoms of the dictionary 
a_true=np.array([
    0,0,0,0,0,0,0,0,0,0,-1,0,0,1,0,0,2,0,0,0
],dtype=float)

#now we generate the signal 
X=D@a_true
'''

def synthetic_data(
    D=None,
    a_true=None,
    signal_dimensions=10,
    n_atoms=20,
    seed=40
):

    # If no dictionary is provided, generate one
    if D is None:

        np.random.seed(seed)

        D = np.random.randn(signal_dimensions, n_atoms)

    # Normalize every dictionary atom
    D = D / np.linalg.norm(D, axis=0, keepdims=True)

    # If no true coefficient vector is provided, create one
    if a_true is None:

        a_true = np.array([
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            -1, 0, 0, 1, 0,
            0, 2, 0, 0, 0
        ], dtype=float)

    # Check that number of coefficients matches number of atoms
    if len(a_true) != D.shape[1]:
        raise ValueError(
            "Length of a_true must match the number of dictionary atoms."
        )

    # Generate synthetic signal
    X = D @ a_true

    return D, a_true, X

D,a_true,X = synthetic_data()
active_in_a_true=np.sum(abs(a_true)>1e-6)
print(active_in_a_true)