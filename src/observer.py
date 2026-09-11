import numpy as np


def luenberger_observer_step(x_hat, y_meas, u, A, B, C, L, dt):
    """y_meas is the scalar (or vector) measured output C x, not the full state."""
    y_meas = np.atleast_1d(np.asarray(y_meas, dtype=float))
    x_hat = np.asarray(x_hat, dtype=float)
    x_hat_dot = A @ x_hat + B[:, 0] * u + L @ (y_meas - C @ x_hat)
    x_hat_next = x_hat + dt * x_hat_dot
    return x_hat_next


def simulate_with_observer(K, L, A, B, C, x0, x_hat0, cfg, disturbance, ref_states=None,
                            plant_step=None, measurement_noise_std=0.0, seed=0):
    """Closed loop with output-feedback: u = -K(x_hat - x_ref), state estimated by a
    Luenberger observer from noisy position-only measurements y = C x + noise."""
    from src.dynamics import rk4_step

    rng = np.random.default_rng(seed)
    n_steps = len(disturbance)
    if plant_step is None:
        def plant_step(x, u, d):
            return rk4_step(x, u, d, cfg["dt"], cfg)

    x = np.asarray(x0, dtype=float).copy()
    x_hat = np.asarray(x_hat0, dtype=float).copy()
    states = np.zeros((n_steps + 1, 2))
    estimates = np.zeros((n_steps + 1, 2))
    controls = np.zeros(n_steps)
    states[0] = x
    estimates[0] = x_hat

    for i in range(n_steps):
        x_ref = ref_states[i] if ref_states is not None else np.zeros(2)
        u = float(-K @ (x_hat - x_ref))
        controls[i] = u

        x = plant_step(x, u, disturbance[i])
        y_meas = C @ x + measurement_noise_std * rng.normal()
        x_hat = luenberger_observer_step(x_hat, y_meas, u, A, B, C, L, cfg["dt"])

        states[i + 1] = x
        estimates[i + 1] = x_hat

    return states, estimates, controls
