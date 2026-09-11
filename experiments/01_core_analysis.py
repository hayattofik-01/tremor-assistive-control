"""Experiment A: open-loop tremor disturbance, nonlinear-vs-linearized comparison,
and the structural analysis (stability / controllability / observability)."""
import matplotlib.pyplot as plt
import numpy as np

from src.analysis import structural_summary
from src.config import get_config
from src.data_generation import make_reference_trajectory
from src.dynamics import linearize_system, nonlinear_dynamics, simulate_system, tremor_disturbance

RESULTS_DIR = "results"


def run():
    cfg = get_config()
    A, B, C = linearize_system(cfg)
    summary = structural_summary(A, B, C)

    print("Linearized A =\n", A)
    print("Linearized B =\n", B)
    print("Linearized C =\n", C)
    print("Eigenvalues =", summary["eigenvalues"])
    print("Controllability rank =", summary["controllability_rank"])
    print("Observability rank =", summary["observability_rank"])

    n_steps = int(3.0 / cfg["dt"])
    t = np.arange(n_steps) * cfg["dt"]
    disturbance = tremor_disturbance(t, cfg)
    ref = make_reference_trajectory(t, cfg["reference_amplitude"], cfg["reference_frequency"])
    x0 = np.array([0.2, 0.0])

    # Figure 1: open-loop response to tremor disturbance (uncontrolled, u = 0)
    states_ol, _ = simulate_system(x0, np.zeros(n_steps), cfg, disturbance=disturbance)
    error_ol = states_ol[:-1, 0] - ref[0]

    fig1, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(t, ref[0], "k--", label="intended motion r(t)")
    axes[0].plot(t, states_ol[:-1, 0], label="actual motion (open loop)")
    axes[0].set_ylabel(r"$\theta$")
    axes[0].legend()
    axes[1].plot(t, disturbance, color="tab:red")
    axes[1].set_ylabel("disturbance d(t)")
    axes[2].plot(t, error_ol, color="tab:orange")
    axes[2].set_ylabel("tracking error")
    axes[2].set_xlabel("time [s]")
    fig1.suptitle("Experiment A: open-loop tremor disturbance")
    fig1.tight_layout()
    fig1.savefig(f"{RESULTS_DIR}/figures/fig01_open_loop.png", dpi=150)
    plt.close(fig1)

    # Figure 2: nonlinear vs linearized free response (u=0, no disturbance)
    n_steps2 = int(4.0 / cfg["dt"])
    t2 = np.arange(n_steps2) * cfg["dt"]
    x0_large = np.array([1.0, 0.0])
    states_nl, _ = simulate_system(x0_large, np.zeros(n_steps2), cfg)

    x_lin = x0_large.copy()
    lin_states = [x_lin.copy()]
    for _ in range(n_steps2):
        x_lin = x_lin + cfg["dt"] * (A @ x_lin)
        lin_states.append(x_lin.copy())
    lin_states = np.array(lin_states)

    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t2, states_nl[:-1, 0], label="nonlinear")
    ax.plot(t2, lin_states[:-1, 0], "--", label="linearized")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$\theta$")
    ax.set_title("Experiment: nonlinear vs. linearized free response " r"($\theta_0=1.0$ rad)")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(f"{RESULTS_DIR}/figures/fig02_nonlinear_vs_linear.png", dpi=150)
    plt.close(fig2)

    # Figure 3: eigenvalues in the complex plane
    fig3, ax = plt.subplots(figsize=(5, 5))
    eigvals = summary["eigenvalues"]
    ax.axvline(0, color="gray", lw=0.8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.scatter(eigvals.real, eigvals.imag, marker="x", s=80, color="tab:blue")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Eigenvalues of the linearized system")
    fig3.tight_layout()
    fig3.savefig(f"{RESULTS_DIR}/figures/fig03_eigenvalues.png", dpi=150)
    plt.close(fig3)

    return summary


if __name__ == "__main__":
    run()
