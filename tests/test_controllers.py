import numpy as np

from src.config import get_config
from src.controllers import lqr_controller, pole_placement_controller, simulate_closed_loop
from src.dynamics import linearize_system


def test_pole_placement_matches_desired_poles():
    cfg = get_config()
    A, B, C = linearize_system(cfg)
    desired = [-3.0, -4.0]
    K = pole_placement_controller(A, B, desired)
    closed_loop_eigs = np.linalg.eigvals(A - B @ K)
    assert np.allclose(sorted(closed_loop_eigs.real), sorted(desired), atol=1e-6)


def test_lqr_gain_stabilizes_closed_loop():
    cfg = get_config()
    A, B, C = linearize_system(cfg)
    Q = np.diag([10.0, 1.0])
    R = np.array([[1.0]])
    K, S, E = lqr_controller(A, B, Q, R)
    closed_loop_eigs = np.linalg.eigvals(A - B @ K)
    assert np.all(closed_loop_eigs.real < 0)


def test_lqr_rejects_disturbance_better_than_open_loop():
    cfg = get_config()
    A, B, C = linearize_system(cfg)
    Q = np.diag([10.0, 1.0])
    R = np.array([[1.0]])
    K, _, _ = lqr_controller(A, B, Q, R)

    n_steps = 200
    disturbance = cfg["disturbance_amplitude"] * np.sin(
        2 * np.pi * cfg["disturbance_frequency"] * np.arange(n_steps) * cfg["dt"]
    )
    x0 = np.array([0.2, 0.0])

    states_controlled, _ = simulate_closed_loop(K, x0, cfg, disturbance)
    states_uncontrolled, _ = simulate_closed_loop(np.zeros_like(K), x0, cfg, disturbance)

    rmse_controlled = np.sqrt(np.mean(states_controlled[:, 0] ** 2))
    rmse_uncontrolled = np.sqrt(np.mean(states_uncontrolled[:, 0] ** 2))
    assert rmse_controlled < rmse_uncontrolled
