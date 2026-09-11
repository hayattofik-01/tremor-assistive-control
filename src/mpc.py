import cvxpy as cp
import numpy as np


def linear_mpc_step(Ad, Bd, x_ref, x_current, Q, R, horizon, u_min, u_max,
                     theta_max=None, P=None, previous_u=0.0):
    """Solve one finite-horizon linear-MPC problem and return the first control action.

    x_ref: array of shape (n, horizon+1) or (n,) for a constant reference.
    P: optional terminal-cost matrix (e.g. the LQR Riccati solution).
    """
    n = Ad.shape[0]
    m = Bd.shape[1]

    x_ref = np.asarray(x_ref, dtype=float)
    if x_ref.ndim == 1:
        x_ref = np.tile(x_ref.reshape(-1, 1), (1, horizon + 1))

    U = cp.Variable((m, horizon))
    X = cp.Variable((n, horizon + 1))

    constraints = [X[:, 0] == x_current]
    cost = 0.0
    for i in range(horizon):
        err = X[:, i] - x_ref[:, i]
        cost += cp.quad_form(err, Q)
        cost += cp.quad_form(U[:, i], R)
        constraints.append(X[:, i + 1] == Ad @ X[:, i] + Bd @ U[:, i])
        constraints.append(U[:, i] >= u_min)
        constraints.append(U[:, i] <= u_max)
        if theta_max is not None:
            constraints.append(cp.abs(X[0, i]) <= theta_max)

    err_N = X[:, horizon] - x_ref[:, horizon]
    if P is not None:
        cost += cp.quad_form(err_N, P)
    else:
        cost += cp.quad_form(err_N, Q)
    if theta_max is not None:
        constraints.append(cp.abs(X[0, horizon]) <= theta_max)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate") or U.value is None:
        return previous_u

    return float(U.value[0, 0])


def simulate_linear_mpc(Ad, Bd, cfg, x0, ref_trajectory, disturbance, Q, R, horizon,
                         u_min, u_max, theta_max=None, P=None, plant_step=None):
    """Closed-loop simulation applying linear MPC to (optionally nonlinear) plant_step.

    ref_trajectory: array (n, n_steps+1) of reference states.
    plant_step(x, u, d) -> x_next; defaults to the discrete linear model.
    """
    n_steps = len(disturbance)
    n = Ad.shape[0]

    if plant_step is None:
        def plant_step(x, u, d):
            return Ad @ x + Bd[:, 0] * u

    states = np.zeros((n_steps + 1, n))
    controls = np.zeros(n_steps)
    states[0] = x0
    x = np.asarray(x0, dtype=float).copy()
    prev_u = 0.0

    for k in range(n_steps):
        remaining = ref_trajectory[:, k:k + horizon + 1]
        if remaining.shape[1] < horizon + 1:
            pad = np.tile(remaining[:, -1:], (1, horizon + 1 - remaining.shape[1]))
            remaining = np.hstack([remaining, pad])
        u = linear_mpc_step(Ad, Bd, remaining, x, Q, R, horizon, u_min, u_max,
                             theta_max=theta_max, P=P, previous_u=prev_u)
        u = float(np.clip(u, u_min, u_max))
        controls[k] = u
        x = plant_step(x, u, disturbance[k])
        states[k + 1] = x
        prev_u = u

    return states, controls
