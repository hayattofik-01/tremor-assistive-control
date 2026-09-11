import numpy as np

from src.config import get_config
from src.dynamics import linearize_system, nonlinear_dynamics, rk4_step, simulate_system, tremor_disturbance


def test_equilibrium_is_fixed_point():
    cfg = get_config()
    xdot = nonlinear_dynamics(np.zeros(2), 0.0, 0.0, cfg)
    assert np.allclose(xdot, 0.0)


def test_linearization_matches_nonlinear_near_equilibrium():
    cfg = get_config()
    A, B, C = linearize_system(cfg)
    x = np.array([1e-4, -1e-4])
    u = 1e-4
    nonlinear = nonlinear_dynamics(x, u, 0.0, cfg)
    linear = A @ x + B[:, 0] * u
    assert np.allclose(nonlinear, linear, atol=1e-6)


def test_rk4_step_shape_and_finite():
    cfg = get_config()
    x_next = rk4_step(np.array([0.1, 0.0]), 0.5, 0.0, cfg["dt"], cfg)
    assert x_next.shape == (2,)
    assert np.all(np.isfinite(x_next))


def test_simulate_system_matches_manual_rk4():
    cfg = get_config()
    x0 = np.array([0.2, -0.1])
    controls = np.array([0.1, -0.2, 0.3])
    disturbance = np.array([0.01, 0.02, -0.01])
    states, _ = simulate_system(x0, controls, cfg, disturbance=disturbance)

    x = x0.copy()
    for u, d in zip(controls, disturbance):
        x = rk4_step(x, u, d, cfg["dt"], cfg)
    assert np.allclose(states[-1], x)


def test_tremor_disturbance_is_bounded_sinusoid():
    cfg = get_config()
    t = np.linspace(0, 5, 500)
    d = tremor_disturbance(t, cfg)
    assert np.max(np.abs(d)) <= cfg["disturbance_amplitude"] + 1e-9
