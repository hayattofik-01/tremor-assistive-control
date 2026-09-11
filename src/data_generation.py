import numpy as np

from src.dynamics import rk4_step


def make_reference_signal(t, amplitude=0.8, frequency=0.7):
    return amplitude * np.sin(2.0 * np.pi * frequency * t)


def make_reference_trajectory(t, amplitude=0.8, frequency=0.7):
    """Reference state trajectory [theta_ref, thetadot_ref] over time vector t."""
    omega = 2.0 * np.pi * frequency
    theta_ref = amplitude * np.sin(omega * t)
    thetadot_ref = amplitude * omega * np.cos(omega * t)
    return np.vstack([theta_ref, thetadot_ref])


DEFAULT_CONDITIONS = [
    {"amplitude": 0.15, "frequency": 4.0},
    {"amplitude": 0.25, "frequency": 5.0},
    {"amplitude": 0.35, "frequency": 6.0},
]

GENERALIZATION_CONDITIONS = [
    {"amplitude": 0.45, "frequency": 8.0},
]


def _simulate_trajectory(cfg, rng, horizon, disturbance_cond, noise_std, control_mode):
    dt = cfg["dt"]
    x = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.5, 0.5)], dtype=float)

    if control_mode == "chirp":
        f0, f1 = 0.2, 3.0
        u_scale = rng.uniform(0.3, 1.0) * cfg["u_max"]
        phase = rng.uniform(0, 2 * np.pi)

    X_in, Y_out, X_true = [], [], []
    for k in range(horizon):
        t = k * dt
        d = disturbance_cond["amplitude"] * np.sin(2.0 * np.pi * disturbance_cond["frequency"] * t)

        if control_mode == "random":
            u = rng.uniform(cfg["u_min"], cfg["u_max"])
        elif control_mode == "chirp":
            f_t = f0 + (f1 - f0) * (t / (horizon * dt))
            u = u_scale * np.sin(2.0 * np.pi * f_t * t + phase)
        else:
            raise ValueError(f"unknown control_mode {control_mode}")
        u = float(np.clip(u, cfg["u_min"], cfg["u_max"]))

        x_next = rk4_step(x, u, d, dt, cfg)
        y_meas = x_next + noise_std * rng.normal(size=2)

        X_in.append(np.array([x[0], x[1], u], dtype=float))
        Y_out.append(y_meas)
        X_true.append(x_next)
        x = x_next

    return np.array(X_in), np.array(Y_out), np.array(X_true)


def generate_identification_data(cfg, n_trajectories=20, horizon=200, noise=0.0,
                                  conditions=None, control_modes=("random", "chirp"),
                                  seed=None):
    """Generate identification data z_k=[theta,thetadot,u] -> y_k=[theta_{k+1},thetadot_{k+1}].

    Varies initial conditions, control inputs (random + chirp exploration) and
    tremor-like disturbance conditions. Returns flattened (X, Y) plus per-sample
    trajectory ids and disturbance-condition ids so a caller can split by trajectory
    (no leakage) or hold out a disturbance condition for a generalization test.
    """
    conditions = DEFAULT_CONDITIONS if conditions is None else conditions
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)

    X_all, Y_all, traj_ids, cond_ids = [], [], [], []
    traj_id = 0
    for cond_idx, cond in enumerate(conditions):
        for _ in range(n_trajectories):
            mode = control_modes[traj_id % len(control_modes)]
            X_in, Y_out, _ = _simulate_trajectory(cfg, rng, horizon, cond, noise, mode)
            X_all.append(X_in)
            Y_all.append(Y_out)
            traj_ids.append(np.full(len(X_in), traj_id))
            cond_ids.append(np.full(len(X_in), cond_idx))
            traj_id += 1

    X = np.vstack(X_all)
    Y = np.vstack(Y_all)
    traj_ids = np.concatenate(traj_ids)
    cond_ids = np.concatenate(cond_ids)
    return X, Y, traj_ids, cond_ids


def split_by_trajectory(traj_ids, train_frac=0.6, val_frac=0.2, seed=0):
    """Split sample indices into train/val/test sets at the trajectory level."""
    rng = np.random.default_rng(seed)
    unique_traj = np.unique(traj_ids)
    rng.shuffle(unique_traj)

    n = len(unique_traj)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train_traj = set(unique_traj[:n_train])
    val_traj = set(unique_traj[n_train:n_train + n_val])
    test_traj = set(unique_traj[n_train + n_val:])

    train_idx = np.isin(traj_ids, list(train_traj))
    val_idx = np.isin(traj_ids, list(val_traj))
    test_idx = np.isin(traj_ids, list(test_traj))
    return train_idx, val_idx, test_idx
