import time

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_kernel_model(alpha=1e-2, gamma=1.0):
    krr = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
    return make_pipeline(StandardScaler(), krr)


def train_kernel_model(X_train, Y_train, alpha=1e-2, gamma=1.0):
    model = build_kernel_model(alpha, gamma)
    start = time.perf_counter()
    model.fit(X_train, Y_train)
    train_time = time.perf_counter() - start
    return model, train_time


def predict_one_step(model, X):
    return model.predict(X)


def predict_next_state(model, z):
    return model.predict(np.asarray(z, dtype=float).reshape(1, -1))[0]
