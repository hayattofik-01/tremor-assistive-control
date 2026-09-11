"""Experiment B: uncontrolled vs pole-placement vs LQR, tracking a reference motion
under tremor-like disturbance."""
import matplotlib.pyplot as plt
import numpy as np

from src.config import get_config
from src.controllers import lqr_controller, pole_placement_controller, simulate_closed_loop
from src.data_generation import make_reference_trajectory
from src.dynamics import linearize_system, tremor_disturbance
from src.metrics import control_effort, residual_tremor_rms, tracking_rmse

RESULTS_DIR = "results"


def run():
    cfg = get_config()
    A, B, C = linearize_system(cfg)

    desired_poles = [-3.0, -4.0]
    K_pole = pole_placement_controller(A, B, desired_poles)

    Q = np.diag([100.0, 5.0])
    R = np.array([[0.5]])
    K_lqr, S, E = lqr_controller(A, B, Q, R)

    n_steps = int(3.0 / cfg["dt"])
    t = np.arange(n_steps) * cfg["dt"]
    disturbance = tremor_disturbance(t, cfg)
    ref = make_reference_trajectory(t, cfg["reference_amplitude"], cfg["reference_frequency"]).T
    x0 = np.array([0.0, 0.0])

    K_zero = np.zeros_like(K_lqr)
    states_open, controls_open = simulate_closed_loop(K_zero, x0, cfg, disturbance, ref_states=ref)
    states_pole, controls_pole = simulate_closed_loop(K_pole, x0, cfg, disturbance, ref_states=ref)
    states_lqr, controls_lqr = simulate_closed_loop(K_lqr, x0, cfg, disturbance, ref_states=ref)

    results = {}
    for name, (states, controls) in {
        "uncontrolled": (states_open, controls_open),
        "pole_placement": (states_pole, controls_pole),
        "lqr": (states_lqr, controls_lqr),
    }.items():
        results[name] = {
            "tracking_rmse": tracking_rmse(states[:-1, 0], ref[:, 0]),
            "residual_tremor_rms": residual_tremor_rms(states[:-1, 0], cfg["dt"]),
            "control_effort": control_effort(controls),
        }

    print("Pole-placement gain =", K_pole)
    print("LQR gain =", K_lqr)
    for name, m in results.items():
        print(name, m)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, ref[:, 0], "k--", label="reference")
    axes[0].plot(t, states_open[:-1, 0], label="uncontrolled")
    axes[0].plot(t, states_pole[:-1, 0], label="pole placement")
    axes[0].plot(t, states_lqr[:-1, 0], label="LQR")
    axes[0].set_ylabel(r"$\theta$")
    axes[0].legend()
    axes[0].set_title("Tracking under tremor-like disturbance")

    axes[1].plot(t, controls_open, label="uncontrolled")
    axes[1].plot(t, controls_pole, label="pole placement")
    axes[1].plot(t, controls_lqr, label="LQR")
    axes[1].set_ylabel("u")
    axes[1].set_xlabel("time [s]")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/figures/fig04_pole_vs_lqr.png", dpi=150)
    plt.close(fig)

    # LQR Q/R sensitivity study (Section 9): light / medium / aggressive weighting.
    sensitivity = {}
    for name, (qtheta, qdot, r) in {
        "light": (10.0, 1.0, 1.0),
        "medium": (100.0, 5.0, 0.5),
        "aggressive": (200.0, 10.0, 0.2),
    }.items():
        Qs = np.diag([qtheta, qdot])
        Rs = np.array([[r]])
        Ks, _, _ = lqr_controller(A, B, Qs, Rs)
        states_s, controls_s = simulate_closed_loop(Ks, x0, cfg, disturbance, ref_states=ref)
        sensitivity[name] = {
            "Q": (qtheta, qdot), "R": r,
            "tracking_rmse": tracking_rmse(states_s[:-1, 0], ref[:, 0]),
            "control_effort": control_effort(controls_s),
            "max_abs_u": float(np.max(np.abs(controls_s))),
            "exceeds_actuator_limit": bool(np.max(np.abs(controls_s)) > cfg["u_max"]),
        }
        print("Q/R sensitivity:", name, sensitivity[name])

    return results, sensitivity


if __name__ == "__main__":
    run()
