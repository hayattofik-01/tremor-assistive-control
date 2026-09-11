import numpy as np
from scipy.linalg import eig


def eigenvalue_analysis(A):
    return eig(A)[0]


def controllability_matrix(A, B):
    return np.hstack([B, A @ B])


def observability_matrix(A, C):
    return np.vstack([C, C @ A])


def structural_summary(A, B, C):
    eigvals = eigenvalue_analysis(A)
    Cmat = controllability_matrix(A, B)
    Omat = observability_matrix(A, C)
    return {
        "eigenvalues": eigvals,
        "controllability_rank": np.linalg.matrix_rank(Cmat),
        "observability_rank": np.linalg.matrix_rank(Omat),
        "controllable": np.linalg.matrix_rank(Cmat) == A.shape[0],
        "observable": np.linalg.matrix_rank(Omat) == A.shape[0],
    }
