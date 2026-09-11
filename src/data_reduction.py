import numpy as np

FRACTIONS = (1.0, 0.5, 0.2, 0.1)


def subsample_dataset(X, Y, fraction, seed=0):
    """Uniformly subsample a representative fraction of the training set."""
    n = len(X)
    n_keep = max(2, int(round(fraction * n)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_keep, replace=False)
    return X[idx], Y[idx]
