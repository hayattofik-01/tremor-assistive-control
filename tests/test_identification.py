import numpy as np

from src.config import get_config
from src.data_generation import generate_identification_data, split_by_trajectory
from src.data_reduction import subsample_dataset
from src.kernel_identifier import train_kernel_model
from src.metrics import rmse
from src.nn_identifier import train_nn_model


def _dataset():
    cfg = get_config()
    X, Y, traj_ids, cond_ids = generate_identification_data(
        cfg, n_trajectories=4, horizon=60, noise=0.0005, seed=7
    )
    train_idx, val_idx, test_idx = split_by_trajectory(traj_ids, seed=3)
    return X, Y, train_idx, test_idx


def test_trajectory_split_has_no_overlap_and_covers_all_samples():
    cfg = get_config()
    X, Y, traj_ids, cond_ids = generate_identification_data(
        cfg, n_trajectories=3, horizon=30, seed=1
    )
    train_idx, val_idx, test_idx = split_by_trajectory(traj_ids, seed=5)
    assert not np.any(train_idx & val_idx)
    assert not np.any(train_idx & test_idx)
    assert not np.any(val_idx & test_idx)
    assert np.all(train_idx | val_idx | test_idx)


def test_nn_identifier_predicts_reasonably_well():
    X, Y, train_idx, test_idx = _dataset()
    model, train_time = train_nn_model(X[train_idx], Y[train_idx], hidden_layer_sizes=(16, 16), max_iter=300)
    pred = model.predict(X[test_idx])
    error = rmse(pred, Y[test_idx])
    assert error < 0.2
    assert train_time >= 0.0


def test_kernel_identifier_predicts_reasonably_well():
    X, Y, train_idx, test_idx = _dataset()
    model, train_time = train_kernel_model(X[train_idx], Y[train_idx])
    pred = model.predict(X[test_idx])
    error = rmse(pred, Y[test_idx])
    assert error < 0.2
    assert train_time >= 0.0


def test_subsample_dataset_produces_requested_fraction():
    X, Y, train_idx, test_idx = _dataset()
    X_train, Y_train = X[train_idx], Y[train_idx]
    for frac in (1.0, 0.5, 0.2, 0.1):
        Xs, Ys = subsample_dataset(X_train, Y_train, frac, seed=0)
        assert len(Xs) == max(2, int(round(frac * len(X_train))))
        assert len(Xs) == len(Ys)
