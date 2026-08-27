import numpy as np

from ista import ista
from synthetic_data import synthetic_data


def objective_analysis(lam, n_iterations):

    D, a_true, X = synthetic_data()

    a_estimated = ista(X, D, lam, n_iterations)

    # ----- TRUE COEFFICIENT VECTOR -----

    reconstruction_error_true = 0.5 * np.sum((X - D @ a_true) ** 2)

    sparsity_penalty_true = lam * np.sum(np.abs(a_true))

    objective_true = reconstruction_error_true + sparsity_penalty_true


    # ----- ESTIMATED COEFFICIENT VECTOR -----

    reconstruction_error_estimated = 0.5 * np.sum(
        (X - D @ a_estimated) ** 2
    )

    sparsity_penalty_estimated = lam * np.sum(
        np.abs(a_estimated)
    )

    objective_estimated = (
        reconstruction_error_estimated
        + sparsity_penalty_estimated
    )


    print("Lambda:", lam)

    print("\nTrue coefficients:")
    print(a_true)

    print("\nEstimated coefficients:")
    print(a_estimated)

    print("\nTRUE SOLUTION")
    print("Reconstruction term:", reconstruction_error_true)
    print("Sparsity term:", sparsity_penalty_true)
    print("Total objective:", objective_true)

    print("\nESTIMATED SOLUTION")
    print("Reconstruction term:", reconstruction_error_estimated)
    print("Sparsity term:", sparsity_penalty_estimated)
    print("Total objective:", objective_estimated)
    
objective_analysis(
    lam=0.8,
    n_iterations=200
)