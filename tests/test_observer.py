import numpy as np
from control import place

from src.config import get_config
from src.controllers import lqr_controller
from src.dynamics import linearize_system, tremor_disturbance
from src.observer import simulate_with_observer


def test_observer_estimate_converges_to_true_state():
    cfg = get_config()
    A, B, C = linearize_system(cfg)
    Q = np.diag([10.0, 1.0])
    R = np.array([[1.0]])
    K, _, _ = lqr_controller(A, B, Q, R)

    observer_poles = [-15.0, -18.0]
    L = place(A.T, C.T, observer_poles).T

    n_steps = 300
    t = np.arange(n_steps) * cfg["dt"]
    disturbance = tremor_disturbance(t, cfg)
    x0 = np.array([0.2, -0.1])
    x_hat0 = np.array([0.0, 0.0])

    states, estimates, _ = simulate_with_observer(
        K.ravel(), L, A, B, C, x0, x_hat0, cfg, disturbance, measurement_noise_std=0.0
    )

    early_error = np.mean(np.abs(states[:20] - estimates[:20]))
    late_error = np.mean(np.abs(states[-20:] - estimates[-20:]))
    assert late_error < early_error
