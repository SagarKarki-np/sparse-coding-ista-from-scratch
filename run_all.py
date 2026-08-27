import os
from pathlib import Path
from synthetic_data import synthetic_data
from convergence import evaluate_convergence
from lambda_sweep import lambda_sweep
from support_recovery import support_recovery
from objective_analysis import objective_analysis
from coefficient_recovery import coefficient_recovery


def run_all():

    # Create figures directory if it does not already exist
    os.makedirs("figures", exist_ok=True)

    print("\n==============================")
    print("SPARSE CODING WITH ISTA")
    print("==============================")

    D, a_true, X = synthetic_data()

    print("\n1. SYNTHETIC DATA")
    print("-----------------")
    print("Dictionary shape:", D.shape)
    print("True coefficient shape:", a_true.shape)
    print("Signal shape:", X.shape)

    true_indices = (abs(a_true) > 1e-6).nonzero()[0]

    print("True active indices:", true_indices)

    n_iterations = 200

    print("\n2. CONVERGENCE ANALYSIS")
    print("-----------------------")

    evaluate_convergence(
        lam=1.0,
        n_iterations=n_iterations
    )

    print("\nFigure saved:")
    print("figures/convergence.png")

    print("\n3. LAMBDA SWEEP")
    print("----------------")

    lambda_sweep(
        n_iterations=n_iterations
    )

    print("\nFigure saved:")
    print("figures/lambda_sweep.png")

    print("\n4. SUPPORT RECOVERY")
    print("-------------------")

    support_recovery(
        n_iterations=n_iterations
    )

    print("\n5. OBJECTIVE ANALYSIS")
    print("---------------------")

    analysis_lambda = 0.5

    objective_analysis(
        lam=analysis_lambda,
        n_iterations=n_iterations
    )

    print("\n6. COEFFICIENT RECOVERY")
    print("-----------------------")

    coefficient_recovery(
        lam=analysis_lambda,
        n_iterations=n_iterations
    )

    print("\n==============================")
    print("ALL EXPERIMENTS COMPLETE")
    print("==============================")
if __name__ == "__main__":
    run_all()