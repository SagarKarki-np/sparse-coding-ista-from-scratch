import numpy as np

from ista import ista
from synthetic_data import synthetic_data


def coefficient_recovery(lam, n_iterations):

    D, a_true, X = synthetic_data()

    a_estimated = ista(X, D, lam, n_iterations)

    true_indices = np.where(np.abs(a_true) > 1e-6)[0]
    estimated_indices = np.where(np.abs(a_estimated) > 1e-6)[0]

    coefficient_error = np.linalg.norm(
        a_true - a_estimated
    )

    print("Lambda:", lam)

    print("\nTrue support:")
    print(true_indices)

    print("\nEstimated support:")
    print(estimated_indices)

    print("\nTrue coefficients:")
    print(a_true)

    print("\nEstimated coefficients:")
    print(a_estimated)

    print("\nCoefficient recovery error:")
    print(coefficient_error)

    if np.array_equal(true_indices, estimated_indices):

        print("\nExact support recovery achieved.")

        print("\nComparison of active coefficients:")

        for index in true_indices:

            print(
                "Atom:", index,
                "| true:", a_true[index],
                "| estimated:", a_estimated[index]
            )

    else:
        print("\nSupport was not recovered exactly.")

