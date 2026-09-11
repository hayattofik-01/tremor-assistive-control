"""Experiment E: train NN/kernel identifiers on a set of disturbance conditions and
test on an unseen, more demanding disturbance condition (Figure 9)."""
import matplotlib.pyplot as plt
import numpy as np

from src.config import get_config
from src.data_generation import (
    DEFAULT_CONDITIONS,
    GENERALIZATION_CONDITIONS,
    generate_identification_data,
)
from src.kernel_identifier import train_kernel_model
from src.metrics import rmse
from src.nn_identifier import train_nn_model

RESULTS_DIR = "results"


def run():
    cfg = get_config()

    X_train, Y_train, _, _ = generate_identification_data(
        cfg, n_trajectories=25, horizon=150, noise=0.001, conditions=DEFAULT_CONDITIONS, seed=cfg["seed"]
    )
    X_unseen, Y_unseen, _, _ = generate_identification_data(
        cfg, n_trajectories=10, horizon=150, noise=0.001, conditions=GENERALIZATION_CONDITIONS, seed=cfg["seed"] + 1
    )

    nn_model, _ = train_nn_model(X_train, Y_train, hidden_layer_sizes=(32, 32), seed=cfg["seed"])
    kr_model, _ = train_kernel_model(X_train, Y_train, alpha=1e-2, gamma=1.0)

    # In-distribution test set: held out from the *training* conditions.
    X_indist, Y_indist, _, _ = generate_identification_data(
        cfg, n_trajectories=5, horizon=150, noise=0.001, conditions=DEFAULT_CONDITIONS, seed=cfg["seed"] + 2
    )

    results = {
        "nn": {
            "in_distribution_rmse": rmse(nn_model.predict(X_indist), Y_indist),
            "unseen_condition_rmse": rmse(nn_model.predict(X_unseen), Y_unseen),
        },
        "kernel": {
            "in_distribution_rmse": rmse(kr_model.predict(X_indist), Y_indist),
            "unseen_condition_rmse": rmse(kr_model.predict(X_unseen), Y_unseen),
        },
    }
    for name, m in results.items():
        print(name, m)

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["in-distribution", "unseen disturbance"]
    x_pos = np.arange(len(labels))
    width = 0.35
    ax.bar(x_pos - width / 2, [results["nn"]["in_distribution_rmse"], results["nn"]["unseen_condition_rmse"]],
           width, label="NN")
    ax.bar(x_pos + width / 2, [results["kernel"]["in_distribution_rmse"], results["kernel"]["unseen_condition_rmse"]],
           width, label="Kernel ridge")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("one-step RMSE")
    ax.set_title("Experiment E: generalization to an unseen disturbance condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/figures/fig09_generalization.png", dpi=150)
    plt.close(fig)

    return results


if __name__ == "__main__":
    run()
