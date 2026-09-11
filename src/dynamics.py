import numpy as np


def nonlinear_dynamics(x, u, d, cfg):
    theta = x[0]
    theta_dot = x[1]
    J = cfg["J"]
    k = cfg["k"]
    k3 = cfg["k3"]
    c = cfg["c"]
    return np.array([
        theta_dot,
        -(k / J) * theta - (k3 / J) * theta ** 3 - (c / J) * theta_dot + (1.0 / J) * u + (1.0 / J) * d,
    ], dtype=float)


def linearize_system(cfg):
    J = cfg["J"]
    k = cfg["k"]
    c = cfg["c"]
    A = np.array([[0.0, 1.0],
                  [-(k / J), -(c / J)]], dtype=float)
    B = np.array([[0.0],
                  [1.0 / J]], dtype=float)
    C = np.array([[1.0, 0.0]], dtype=float)
    return A, B, C


def rk4_step(x, u, d, dt, cfg):
    k1 = nonlinear_dynamics(x, u, d, cfg)
    k2 = nonlinear_dynamics(x + 0.5 * dt * k1, u, d, cfg)
    k3 = nonlinear_dynamics(x + 0.5 * dt * k2, u, d, cfg)
    k4 = nonlinear_dynamics(x + dt * k3, u, d, cfg)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x_next


def simulate_system(initial_state, controls, cfg, dt=None, disturbance=None, return_time=False):
    dt = cfg["dt"] if dt is None else dt
    controls = np.asarray(controls, dtype=float)
    n_steps = len(controls)
    states = np.zeros((n_steps + 1, 2), dtype=float)
    states[0] = np.asarray(initial_state, dtype=float)

    if disturbance is None:
        disturbance = np.zeros(n_steps, dtype=float)
    else:
        disturbance = np.asarray(disturbance, dtype=float)
        if disturbance.ndim == 0:
            disturbance = np.full(n_steps, float(disturbance), dtype=float)

    for i in range(n_steps):
        states[i + 1] = rk4_step(states[i], controls[i], disturbance[i], dt, cfg)

    if return_time:
        t = np.arange(n_steps + 1) * dt
        return states, t
    return states, None


def tremor_disturbance(t, cfg, amplitude=None, frequency=None):
    amplitude = cfg["disturbance_amplitude"] if amplitude is None else amplitude
    frequency = cfg["disturbance_frequency"] if frequency is None else frequency
    return amplitude * np.sin(2.0 * np.pi * frequency * t)
