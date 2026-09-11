import numpy as np
from scipy.signal import butter, filtfilt


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def tracking_rmse(theta, theta_ref):
    return rmse(theta, theta_ref)


def residual_tremor_rms(theta, dt, band=(3.0, 7.0)):
    """RMS of the motion energy inside the tremor frequency band.

    This is deliberately independent of tracking error: it band-pass filters the
    *actual* motion around the tremor band (default 3-7 Hz, bracketing the 4-6 Hz
    disturbance conditions used elsewhere) and reports the RMS of what remains, so
    it measures how much tremor-frequency content survives in the output
    regardless of how well the slow reference is tracked.
    """
    theta = np.asarray(theta, dtype=float)
    fs = 1.0 / dt
    nyquist = fs / 2.0
    low, high = band
    if high >= nyquist:
        raise ValueError(f"tremor band upper edge {high} Hz must be below Nyquist {nyquist} Hz")
    b, a = butter(N=4, Wn=[low / nyquist, high / nyquist], btype="bandpass")
    filtered = filtfilt(b, a, theta)
    return float(np.sqrt(np.mean(filtered ** 2)))


def control_effort(u):
    u = np.asarray(u, dtype=float)
    return float(np.sum(u ** 2))


def constraint_violations(u, u_min, u_max, tol=1e-6):
    u = np.asarray(u, dtype=float)
    n_violations = int(np.sum((u < u_min - tol) | (u > u_max + tol)))
    max_abs = float(np.max(np.abs(u))) if len(u) else 0.0
    return {"max_abs_control": max_abs, "n_violations": n_violations}


def one_step_rmse(y_true, y_pred):
    return rmse(y_true, y_pred)


def rollout_rmse(model_predict_fn, X0, U, Y_true):
    """Multi-step rollout: feed predictions back in as the next state.

    model_predict_fn(z) -> next state, where z = [theta, thetadot, u].
    U: array of controls applied along the rollout (horizon,).
    Y_true: true states (horizon, 2) to compare against.
    """
    x = np.asarray(X0, dtype=float).copy()
    preds = np.zeros((len(U), 2))
    for k in range(len(U)):
        z = np.array([x[0], x[1], U[k]])
        x = model_predict_fn(z)
        preds[k] = x
    return rmse(preds, Y_true), preds
