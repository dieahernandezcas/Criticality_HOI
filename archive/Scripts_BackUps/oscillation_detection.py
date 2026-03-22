# functions/oscillation_detection.py

import numpy as np


def detect_limit_cycle_poincare(
    E_signal,
    I_signal,
    dt,
    threshold_ratio=0.05,
    period_cv_threshold=0.10
):
    """Detect limit cycle using Poincaré section method (upward or downward crossings)."""

    E_centered = E_signal - np.mean(E_signal)

    def analyze_crossings(sign=1):
        crossings = []

        for i in range(len(E_centered) - 1):
            if sign == 1:
                condition = E_centered[i] <= 0 < E_centered[i + 1]
            else:
                condition = E_centered[i] >= 0 > E_centered[i + 1]

            if condition:
                denom = E_centered[i + 1] - E_centered[i]
                if denom == 0:
                    continue

                alpha = -E_centered[i] / denom
                t_cross = (i + alpha) * dt
                I_cross = I_signal[i] + alpha * (I_signal[i + 1] - I_signal[i])

                crossings.append((t_cross, I_cross))

        if len(crossings) < 3:
            return False, None, np.inf, len(crossings)

        times = np.array([c[0] for c in crossings])
        I_vals = np.array([c[1] for c in crossings])

        I_range = np.max(I_signal) - np.min(I_signal)
        if I_range == 0:
            return False, None, np.inf, len(crossings)

        dispersion = np.std(I_vals) / I_range

        periods = np.diff(times)
        period_mean = np.mean(periods)
        period_cv = np.std(periods) / period_mean if period_mean > 0 else np.inf

        is_cycle = (dispersion < threshold_ratio and period_cv < period_cv_threshold)

        return is_cycle, period_mean, dispersion, len(crossings)

    # Ascending crossings
    up_cycle, up_period, up_disp, up_n = analyze_crossings(sign=1)

    # Descending crossings
    down_cycle, down_period, down_disp, down_n = analyze_crossings(sign=-1)

    # Final decision
    is_limit_cycle = up_cycle or down_cycle

    # Choose the most stable estimate if available
    if up_cycle and down_cycle:
        period_mean = min(up_period, down_period)
        dispersion = min(up_disp, down_disp)
        n_crossings = min(up_n, down_n)
    elif up_cycle:
        period_mean = up_period
        dispersion = up_disp
        n_crossings = up_n
    elif down_cycle:
        period_mean = down_period
        dispersion = down_disp
        n_crossings = down_n
    else:
        period_mean = None
        dispersion = np.inf
        n_crossings = max(up_n, down_n)

    return is_limit_cycle, period_mean, dispersion, n_crossings
