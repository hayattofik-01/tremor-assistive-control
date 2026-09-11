"""Experiment D: neural-network vs kernel-ridge identification of the nonlinear
dynamics x_{k+1} = f(x_k, u_k), evaluated on one-step and multi-step rollout error."""
import matplotlib.pyplot as plt
import numpy as np

from src.config import get_config
from src.data_generation import DEFAULT_CONDITIONS, generate_identification_data, split_by_trajectory
from src.dynamics import rk4_step
from src.kernel_identifier import train_kernel_model
from src.metrics import rmse, rollout_rmse
from src.nn_identifier import train_nn_model

RESULTS_DIR = "results"


def _true_next_state(z, cfg, d):
    x = np.array([z[0], z[1]])
    return rk4_step(x, z[2], d, cfg["dt"], cfg)


def run():
    cfg = get_config()
    X, Y, traj_ids, cond_ids = generate_identification_data(
        cfg, n_trajectories=25, horizon=150, noise=0.001, conditions=DEFAULT_CONDITIONS, seed=cfg["seed"]
    )
    train_idx, val_idx, test_idx = split_by_trajectory(traj_ids, seed=cfg["seed"])
    print(f"train={train_idx.sum()} val={val_idx.sum()} test={test_idx.sum()}")

    nn_model, nn_time = train_nn_model(X[train_idx], Y[train_idx], hidden_layer_sizes=(32, 32), seed=cfg["seed"])
    kr_model, kr_time = train_kernel_model(X[train_idx], Y[train_idx], alpha=1e-2, gamma=1.0)

    nn_pred = nn_model.predict(X[test_idx])
    kr_pred = kr_model.predict(X[test_idx])
    nn_one_step_rmse = rmse(nn_pred, Y[test_idx])
    kr_one_step_rmse = rmse(kr_pred, Y[test_idx])
    print(f"NN one-step RMSE = {nn_one_step_rmse:.5f} (train time {nn_time:.3f}s)")
    print(f"Kernel one-step RMSE = {kr_one_step_rmse:.5f} (train time {kr_time:.3f}s)")

    # Multi-step rollout on one held-out test trajectory.
    test_traj_id = traj_ids[test_idx][0]
    mask = traj_ids == test_traj_id
    X_traj, Y_traj = X[mask], Y[mask]
    x0 = X_traj[0, :2]
    U_traj = X_traj[:, 2]

    nn_roll_rmse, nn_roll_preds = rollout_rmse(lambda z: nn_model.predict(z.reshape(1, -1))[0], x0, U_traj, Y_traj)
    kr_roll_rmse, kr_roll_preds = rollout_rmse(lambda z: kr_model.predict(z.reshape(1, -1))[0], x0, U_traj, Y_traj)
    print(f"NN rollout RMSE = {nn_roll_rmse:.5f}")
    print(f"Kernel rollout RMSE = {kr_roll_rmse:.5f}")

    # Figure 7: one-step prediction (theta channel) on the held-out test set.
    fig7, ax = plt.subplots(figsize=(7, 5))
    order = np.argsort(Y[test_idx][:, 0])
    ax.plot(Y[test_idx][order, 0], Y[test_idx][order, 0], "k--", label="perfect prediction")
    ax.scatter(Y[test_idx][:, 0], nn_pred[:, 0], s=6, alpha=0.4, label="NN")
    ax.scatter(Y[test_idx][:, 0], kr_pred[:, 0], s=6, alpha=0.4, label="Kernel ridge")
    ax.set_xlabel(r"true $\theta_{k+1}$")
    ax.set_ylabel(r"predicted $\theta_{k+1}$")
    ax.legend()
    ax.set_title("Experiment D: one-step prediction (test set)")
    fig7.tight_layout()
    fig7.savefig(f"{RESULTS_DIR}/figures/fig07_one_step_prediction.png", dpi=150)
    plt.close(fig7)

    # Figure 8: multi-step rollout comparison on the held-out trajectory.
    t_traj = np.arange(len(Y_traj)) * cfg["dt"]
    fig8, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_traj, Y_traj[:, 0], "k-", label="true")
    ax.plot(t_traj, nn_roll_preds[:, 0], label="NN rollout")
    ax.plot(t_traj, kr_roll_preds[:, 0], label="Kernel rollout")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$\theta$")
    ax.legend()
    ax.set_title("Experiment D: multi-step rollout on a held-out trajectory")
    fig8.tight_layout()
    fig8.savefig(f"{RESULTS_DIR}/figures/fig08_rollout.png", dpi=150)
    plt.close(fig8)

    return {
        "nn": {"one_step_rmse": nn_one_step_rmse, "rollout_rmse": nn_roll_rmse, "train_time": nn_time},
        "kernel": {"one_step_rmse": kr_one_step_rmse, "rollout_rmse": kr_roll_rmse, "train_time": kr_time},
    }


if __name__ == "__main__":
    run()
