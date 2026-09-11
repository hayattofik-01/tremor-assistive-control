import numpy as np

from src.config import get_config
from src.discretization import discretize_zoh
from src.dynamics import linearize_system
from src.metrics import constraint_violations, control_effort, residual_tremor_rms, rmse, rollout_rmse


def test_rmse_zero_for_identical_arrays():
    a = np.array([1.0, 2.0, 3.0])
    assert rmse(a, a) == 0.0


def test_control_effort_matches_sum_of_squares():
    u = np.array([1.0, -2.0, 3.0])
    assert control_effort(u) == 1.0 + 4.0 + 9.0


def test_constraint_violations_detects_out_of_bounds():
    u = np.array([0.5, -2.0, 3.0, 0.9])
    result = constraint_violations(u, -1.0, 1.0)
    assert result["n_violations"] == 2
    assert result["max_abs_control"] == 3.0


def test_rollout_rmse_perfect_model_is_zero():
    def perfect_model(z):
        return np.array([z[0] + z[2] * 0.01, z[1]])

    x0 = np.array([0.0, 0.0])
    U = np.array([1.0, 1.0, 1.0])
    Y_true = np.array([[0.01, 0.0], [0.02, 0.0], [0.03, 0.0]])
    error, preds = rollout_rmse(perfect_model, x0, U, Y_true)
    assert error < 1e-9


def test_residual_tremor_rms_isolates_the_tremor_band():
    dt = 0.01
    t = np.arange(0, 5.0, dt)
    # A signal that is *only* a slow (0.3 Hz) reference-like motion should have
    # almost no energy left after band-passing around the 3-7 Hz tremor band.
    slow_only = 0.4 * np.sin(2 * np.pi * 0.3 * t)
    assert residual_tremor_rms(slow_only, dt) < 0.01

    # A signal that is a pure 5 Hz tremor-band sinusoid of amplitude A should have
    # RMS close to A/sqrt(2) after band-passing, independent of any reference.
    amplitude = 0.25
    tremor_only = amplitude * np.sin(2 * np.pi * 5.0 * t)
    result = residual_tremor_rms(tremor_only, dt)
    assert abs(result - amplitude / np.sqrt(2)) < 0.02

    # It must not depend on a reference/tracking signal at all (unlike tracking RMSE).
    combined = slow_only + tremor_only
    assert abs(residual_tremor_rms(combined, dt) - result) < 0.02


def test_discretize_zoh_matches_scipy_for_small_dt():
    cfg = get_config()
    A, B, C = linearize_system(cfg)
    Ad, Bd = discretize_zoh(A, B, cfg["dt"])
    # For small dt, ZOH discretization should approximate forward Euler.
    assert np.allclose(Ad, np.eye(2) + cfg["dt"] * A, atol=1e-3)
    assert np.allclose(Bd, cfg["dt"] * B, atol=1e-3)
