import numpy as np
from scipy.linalg import expm


def discretize_zoh(A, B, dt):
    """Zero-order-hold discretization of a continuous-time LTI pair (A, B)."""
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd
