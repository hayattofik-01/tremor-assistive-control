"""Experiment F: effect of reducing the identification dataset to 100/50/20/10% on
one-step RMSE, multi-step rollout RMSE, and training time (Figure 10)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import get_config
from src.data_generation import DEFAULT_CONDITIONS, generate_identification_data, split_by_trajectory
from src.data_reduction import FRACTIONS, subsample_dataset
from src.kernel_identifier import train_kernel_model
from src.metrics import rmse, rollout_rmse
from src.nn_identifier import train_nn_model

RESULTS_DIR = "results"


def run():
    cfg = get_config()
    X, Y, traj_ids, cond_ids = generate_identification_data(
        cfg, n_trajectories=25, horizon=150, noise=0.001, conditions=DEFAULT_CONDITIONS, seed=cfg["seed"]
    )
    train_idx, val_idx, test_idx = split_by_trajectory(traj_ids, seed=cfg["seed"])
    X_train_full, Y_train_full = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    test_traj_id = traj_ids[test_idx][0]
    mask = traj_ids == test_traj_id
    X_traj, Y_traj = X[mask], Y[mask]
    x0 = X_traj[0, :2]
    U_traj = X_traj[:, 2]

    rows = []
    for frac in FRACTIONS:
        X_sub, Y_sub = subsample_dataset(X_train_full, Y_train_full, frac, seed=cfg["seed"])

        nn_model, nn_time = train_nn_model(X_sub, Y_sub, hidden_layer_sizes=(32, 32), seed=cfg["seed"])
        kr_model, kr_time = train_kernel_model(X_sub, Y_sub, alpha=1e-2, gamma=1.0)

        nn_one_step = rmse(nn_model.predict(X_test), Y_test)
        kr_one_step = rmse(kr_model.predict(X_test), Y_test)
        nn_roll, _ = rollout_rmse(lambda z: nn_model.predict(z.reshape(1, -1))[0], x0, U_traj, Y_traj)
        kr_roll, _ = rollout_rmse(lambda z: kr_model.predict(z.reshape(1, -1))[0], x0, U_traj, Y_traj)

        rows.append({
            "fraction": frac, "n_train": len(X_sub),
            "nn_one_step_rmse": nn_one_step, "nn_rollout_rmse": nn_roll, "nn_train_time": nn_time,
            "kernel_one_step_rmse": kr_one_step, "kernel_rollout_rmse": kr_roll, "kernel_train_time": kr_time,
        })
        print(rows[-1])

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/tables/data_reduction.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(df["fraction"] * 100, df["nn_one_step_rmse"], "o-", label="NN one-step")
    axes[0].plot(df["fraction"] * 100, df["kernel_one_step_rmse"], "s-", label="Kernel one-step")
    axes[0].plot(df["fraction"] * 100, df["nn_rollout_rmse"], "o--", label="NN rollout")
    axes[0].plot(df["fraction"] * 100, df["kernel_rollout_rmse"], "s--", label="Kernel rollout")
    axes[0].set_xlabel("% of training data")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[0].set_title("Prediction error vs. training-data fraction")

    axes[1].plot(df["fraction"] * 100, df["nn_train_time"], "o-", label="NN")
    axes[1].plot(df["fraction"] * 100, df["kernel_train_time"], "s-", label="Kernel ridge")
    axes[1].set_xlabel("% of training data")
    axes[1].set_ylabel("training time [s]")
    axes[1].legend()
    axes[1].set_title("Training time vs. training-data fraction")

    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/figures/fig10_data_reduction.png", dpi=150)
    plt.close(fig)

    return df


if __name__ == "__main__":
    run()
