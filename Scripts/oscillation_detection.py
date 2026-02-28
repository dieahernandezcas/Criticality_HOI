# functions/oscillation_detection.py

import numpy as np

def detect_limit_cycle_poincare(E_signal, I_signal, dt, threshold_ratio=0.05, period_cv_threshold=0.10):
    """Detect limit cycle using Poincaré section method."""
    E_mean = np.mean(E_signal)
    E_centered = E_signal - E_mean

    crossings = []
    for i in range(len(E_centered) - 1):
        if E_centered[i] <= 0 < E_centered[i+1]:
            alpha = -E_centered[i] / (E_centered[i+1] - E_centered[i])
            I_cross = I_signal[i] + alpha * (I_signal[i+1] - I_signal[i])
            t_cross = (i + alpha) * dt
            crossings.append((t_cross, I_cross))

    if len(crossings) < 3:
        return False, None, np.inf, len(crossings)

    times = np.array([c[0] for c in crossings])
    I_values = np.array([c[1] for c in crossings])

    I_range = np.max(I_signal) - np.min(I_signal)
    if I_range == 0:
        return False, None, np.inf, len(crossings)

    dispersion = np.std(I_values) / I_range
    periods = np.diff(times)
    period_mean = np.mean(periods)
    period_cv = np.std(periods) / period_mean if period_mean > 0 else np.inf

    is_limit_cycle = (dispersion < threshold_ratio and period_cv < period_cv_threshold)

    return is_limit_cycle, period_mean, dispersion, len(crossings)
