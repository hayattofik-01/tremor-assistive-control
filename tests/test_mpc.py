import numpy as np
from control import dlqr

from src.config import get_config
from src.discretization import discretize_zoh
from src.dynamics import linearize_system, rk4_step
from src.mpc import simulate_linear_mpc


def test_mpc_respects_actuator_constraints():
    cfg = get_config()
    A, B, C = linearize_system(cfg)
    Ad, Bd = discretize_zoh(A, B, cfg["dt"])
    Q = np.diag([10.0, 1.0])
    R = np.array([[0.5]])

    n_steps = 40
    disturbance = np.zeros(n_steps)
    ref = np.zeros((2, n_steps + 20))
    x0 = np.array([1.0, 0.0])

    u_min, u_max = -1.0, 1.0
    _, controls = simulate_linear_mpc(
        Ad, Bd, cfg, x0, ref, disturbance, Q, R, horizon=10, u_min=u_min, u_max=u_max
    )
    assert np.all(controls >= u_min - 1e-6)
    assert np.all(controls <= u_max + 1e-6)


def test_mpc_tracks_nonzero_reference():
    """Linear MPC drives the nonlinear plant most of the way to a step reference and
    settles there; a small steady-state offset is expected because the controller's
    internal model is linear while the simulated plant has an extra cubic restoring
    term (the deliberate model-mismatch scenario described in the project scope)."""
    cfg = get_config()
    A, B, C = linearize_system(cfg)
    Ad, Bd = discretize_zoh(A, B, cfg["dt"])
    Q = np.diag([50.0, 1.0])
    R = np.array([[0.1]])

    n_steps = 300
    disturbance = np.zeros(n_steps)
    x_target = np.array([0.5, 0.0])
    ref = np.tile(x_target.reshape(-1, 1), (1, n_steps + 20))
    x0 = np.array([0.0, 0.0])

    def plant_step(x, u, d):
        return rk4_step(x, u, d, cfg["dt"], cfg)

    states, controls = simulate_linear_mpc(
        Ad, Bd, cfg, x0, ref, disturbance, Q, R, horizon=15,
        u_min=cfg["u_min"], u_max=cfg["u_max"], plant_step=plant_step,
    )
    settled = states[-20:]
    assert np.ptp(settled[:, 0]) < 0.02
    assert 0.2 < states[-1, 0] < x_target[0] + 0.05
    assert np.all(np.isfinite(states))


def test_mpc_matches_lqr_in_unconstrained_regulation():
    """With loose bounds and the discrete-LQR terminal cost, finite-horizon MPC
    regulation should reduce to the discrete-time infinite-horizon LQR law,
    since the DARE solution is a fixed point of the backward Riccati recursion."""
    cfg = get_config()
    A, B, C = linearize_system(cfg)
    Ad, Bd = discretize_zoh(A, B, cfg["dt"])
    Q = np.diag([10.0, 1.0])
    R = np.array([[1.0]])
    K, S, _ = dlqr(Ad, Bd, Q, R)
    K = np.asarray(K)

    n_steps = 60
    disturbance = np.zeros(n_steps)
    ref = np.zeros((2, n_steps + 30))
    x0 = np.array([0.3, 0.0])

    states_mpc, _ = simulate_linear_mpc(
        Ad, Bd, cfg, x0, ref, disturbance, Q, R, horizon=30,
        u_min=-1e3, u_max=1e3, P=S,
    )

    x = x0.copy()
    states_lqr = [x.copy()]
    for _ in range(n_steps):
        u = float((-K @ x).item())
        x = Ad @ x + Bd[:, 0] * u
        states_lqr.append(x.copy())
    states_lqr = np.array(states_lqr)

    assert np.allclose(states_mpc, states_lqr, atol=1e-2)
