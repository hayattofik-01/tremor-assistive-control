import time

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_nn_model(hidden_layer_sizes=(32, 32), seed=0, max_iter=2000):
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="tanh",
        alpha=1e-4,
        max_iter=max_iter,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=20,
    )
    return make_pipeline(StandardScaler(), mlp)


def train_nn_model(X_train, Y_train, hidden_layer_sizes=(32, 32), seed=0, max_iter=2000):
    model = build_nn_model(hidden_layer_sizes, seed, max_iter)
    start = time.perf_counter()
    model.fit(X_train, Y_train)
    train_time = time.perf_counter() - start
    return model, train_time


def predict_one_step(model, X):
    return model.predict(X)


def predict_next_state(model, z):
    """z = [theta, thetadot, u] -> predicted [theta_next, thetadot_next]."""
    return model.predict(np.asarray(z, dtype=float).reshape(1, -1))[0]
