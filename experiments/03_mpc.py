"""Experiment C: LQR vs constrained linear MPC on the nonlinear plant, plus the
Luenberger observer estimate-vs-true-state comparison (Figures 5 and 6)."""
import matplotlib.pyplot as plt
import numpy as np
from control import place

from src.config import get_config
from src.controllers import discrete_lqr_controller, lqr_controller, simulate_closed_loop
from src.data_generation import make_reference_trajectory
from src.discretization import discretize_zoh
from src.dynamics import linearize_system, rk4_step, tremor_disturbance
from src.metrics import constraint_violations, control_effort, residual_tremor_rms, tracking_rmse
from src.mpc import simulate_linear_mpc
from src.observer import simulate_with_observer

RESULTS_DIR = "results"


def run():
    cfg = get_config()
    A, B, C = linearize_system(cfg)
    Ad, Bd = discretize_zoh(A, B, cfg["dt"])

    # Same Q/R as the "medium" tuning in Experiment B, whose unconstrained response
    # needs |u| up to ~3.8 -- comfortably inside the u=[-4,4] actuator range used
    # there, but outside a tighter range, which is what we impose here.
    Q = np.diag([100.0, 5.0])
    R = np.array([[0.5]])
    K_lqr, S, _ = lqr_controller(A, B, Q, R)

    # Tight actuator limit to make the constraint bind for LQR but not for MPC.
    u_min, u_max = -1.0, 1.0
    horizon = 15
    # Terminal cost must be the *discrete* Riccati solution (not the continuous S):
    # it is the fixed point of the discrete backward Riccati recursion, so MPC with
    # this terminal weight reduces to the infinite-horizon discrete LQR law even for
    # a short horizon. Using the continuous S here made the MPC drastically
    # under-actuate (verified empirically before this fix).
    _, P_discrete, _ = discrete_lqr_controller(Ad, Bd, Q, R)

    n_steps = int(3.0 / cfg["dt"])
    t = np.arange(n_steps) * cfg["dt"]
    disturbance = tremor_disturbance(t, cfg)
    ref = make_reference_trajectory(t, cfg["reference_amplitude"], cfg["reference_frequency"])
    ref_padded = make_reference_trajectory(
        np.arange(n_steps + horizon) * cfg["dt"], cfg["reference_amplitude"], cfg["reference_frequency"]
    )
    x0 = np.array([0.0, 0.0])

    def plant_step(x, u, d):
        return rk4_step(x, u, d, cfg["dt"], cfg)

    # LQR with the same hard clipping applied (naive saturation, no anti-windup).
    states_lqr = np.zeros((n_steps + 1, 2))
    controls_lqr = np.zeros(n_steps)
    x = x0.copy()
    states_lqr[0] = x
    for k in range(n_steps):
        u = float(-K_lqr @ (x - ref[:, k]))
        u = float(np.clip(u, u_min, u_max))
        controls_lqr[k] = u
        x = plant_step(x, u, disturbance[k])
        states_lqr[k + 1] = x

    states_mpc, controls_mpc = simulate_linear_mpc(
        Ad, Bd, cfg, x0, ref_padded, disturbance, Q, R, horizon,
        u_min=u_min, u_max=u_max, theta_max=cfg["theta_max"], P=P_discrete, plant_step=plant_step,
    )

    metrics = {}
    for name, (states, controls) in {
        "lqr_saturated": (states_lqr, controls_lqr),
        "mpc": (states_mpc, controls_mpc),
    }.items():
        cviol = constraint_violations(controls, u_min, u_max)
        metrics[name] = {
            "tracking_rmse": tracking_rmse(states[:-1, 0], ref[0]),
            "residual_tremor_rms": residual_tremor_rms(states[:-1, 0], cfg["dt"]),
            "control_effort": control_effort(controls),
            **cviol,
        }
        print(name, metrics[name])

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, ref[0], "k--", label="reference")
    axes[0].plot(t, states_lqr[:-1, 0], label="LQR (saturated)")
    axes[0].plot(t, states_mpc[:-1, 0], label="MPC (constrained)")
    axes[0].set_ylabel(r"$\theta$")
    axes[0].legend()
    axes[0].set_title("Experiment C: LQR vs. constrained MPC")

    axes[1].plot(t, controls_lqr, label="LQR (saturated)")
    axes[1].plot(t, controls_mpc, label="MPC (constrained)")
    axes[1].axhline(u_min, color="gray", ls=":")
    axes[1].axhline(u_max, color="gray", ls=":")
    axes[1].set_ylabel("u")
    axes[1].set_xlabel("time [s]")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/figures/fig05_lqr_vs_mpc.png", dpi=150)
    plt.close(fig)

    # --- Observer: state estimate vs true state (Figure 6) ---
    observer_poles = [-15.0, -18.0]
    L = place(A.T, C.T, observer_poles).T
    ref_full = ref.T
    states_true, estimates, _ = simulate_with_observer(
        K_lqr.ravel(), L, A, B, C, x0, np.array([0.0, 0.0]), cfg, disturbance,
        ref_states=ref_full, measurement_noise_std=0.01,
    )
    observer_rmse = tracking_rmse(states_true[:, 1], estimates[:, 1])
    print("observer velocity-estimate RMSE =", observer_rmse)

    fig6, axes6 = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes6[0].plot(t, states_true[:-1, 0], label="true $\\theta$")
    axes6[0].plot(t, estimates[:-1, 0], "--", label="estimated $\\theta$")
    axes6[0].set_ylabel(r"$\theta$")
    axes6[0].legend()
    axes6[0].set_title("Luenberger observer: estimate vs. true state (position-only measurement)")
    axes6[1].plot(t, states_true[:-1, 1], label="true $\\dot\\theta$")
    axes6[1].plot(t, estimates[:-1, 1], "--", label="estimated $\\dot\\theta$")
    axes6[1].set_ylabel(r"$\dot\theta$")
    axes6[1].set_xlabel("time [s]")
    axes6[1].legend()
    fig6.tight_layout()
    fig6.savefig(f"{RESULTS_DIR}/figures/fig06_observer.png", dpi=150)
    plt.close(fig6)

    metrics["observer_velocity_rmse"] = observer_rmse
    return metrics


if __name__ == "__main__":
    run()
