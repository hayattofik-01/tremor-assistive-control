import numpy as np
from control import dlqr, lqr, place


def pole_placement_controller(A, B, desired_poles):
    K = place(A, B, desired_poles)
    return K


def lqr_controller(A, B, Q, R):
    """Continuous-time LQR gain and Riccati solution, for the continuous plant."""
    K, S, E = lqr(A, B, Q, R)
    return K, S, E


def discrete_lqr_controller(Ad, Bd, Q, R):
    """Discrete-time LQR gain and Riccati solution for a ZOH-discretized model.

    This P is the correct terminal-cost weight for linear MPC on (Ad, Bd): since it
    is the fixed point of the discrete backward Riccati recursion, using it as the
    terminal cost makes finite-horizon MPC reduce to the infinite-horizon discrete
    LQR law even for short horizons.
    """
    K, S, E = dlqr(Ad, Bd, Q, R)
    return np.asarray(K), np.asarray(S), E


def state_feedback(x, K, x_ref=None):
    """Tracking state-feedback law: u = -K (x - x_ref)."""
    x_ref = np.zeros_like(x) if x_ref is None else np.asarray(x_ref, dtype=float)
    return float(-np.dot(K.ravel(), np.asarray(x, dtype=float) - x_ref))


def simulate_closed_loop(K, x0, cfg, disturbance, ref_states=None, plant_step=None):
    """Closed-loop simulation of u = -K(x - x_ref) against a plant step function.

    plant_step(x, u, d) -> x_next; defaults to RK4 integration of the nonlinear plant.
    ref_states: array (n_steps+1, 2) of reference [theta_ref, thetadot_ref], or None for zero.
    """
    from src.dynamics import rk4_step

    n_steps = len(disturbance)
    if plant_step is None:
        def plant_step(x, u, d):
            return rk4_step(x, u, d, cfg["dt"], cfg)

    x = np.asarray(x0, dtype=float).copy()
    states = np.zeros((n_steps + 1, 2))
    controls = np.zeros(n_steps)
    states[0] = x

    for i in range(n_steps):
        x_ref = ref_states[i] if ref_states is not None else None
        u = state_feedback(x, K, x_ref)
        controls[i] = u
        x = plant_step(x, u, disturbance[i])
        states[i + 1] = x

    return states, controls
