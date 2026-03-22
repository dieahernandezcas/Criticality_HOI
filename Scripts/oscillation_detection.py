"""
oscillation_detection.py
========================
Algorithms for detecting sustained oscillations (limit cycles) in simulated
Wilson–Cowan time series, as used in:

    Hernández et al. (2025) "Higher-order statistical structure emerges from
    nonlinear dynamics without explicit higher-order coupling."

The classification of each (P, K) grid point into "oscillatory" or "fixed
point" regime is essential for interpreting the TC−DTC and cumulant maps.

Method
------
The Poincaré section method is used:
    1. Identify all upward zero-crossings of the mean-subtracted E signal.
    2. Record the value of the inhibitory signal I at each crossing time
       (interpolated linearly between time steps).
    3. Classify the trajectory as a limit cycle if:
       - The dispersion of I at the crossings is small relative to the
         overall I range (below *threshold_ratio*).
       - The inter-crossing periods are nearly constant (coefficient of
         variation below *period_cv_threshold*).

This simple test is robust for the Wilson–Cowan regime explored in the paper
and does not require computing power spectra or autocorrelation functions.
"""

import numpy as np


def detect_limit_cycle_poincare(
    E_signal: np.ndarray,
    I_signal: np.ndarray,
    dt: float,
    threshold_ratio: float = 0.05,
    period_cv_threshold: float = 0.10,
):
    """Detect a limit cycle using the Poincaré section method.

    A single node's (E, I) trajectory is classified as a limit cycle if the
    Poincaré section (defined by upward crossings of E through its mean) shows
    small dispersion and regular inter-crossing intervals.

    Parameters
    ----------
    E_signal : np.ndarray, shape (T,)
        Excitatory time series for one node (transient already removed).
    I_signal : np.ndarray, shape (T,)
        Inhibitory time series for the same node.
    dt : float
        Integration time step (seconds).
    threshold_ratio : float, optional
        Maximum allowed dispersion of I at crossings, normalized by the
        total I range.  Smaller values are stricter (default 0.05).
    period_cv_threshold : float, optional
        Maximum allowed coefficient of variation of the crossing periods.
        Smaller values require more regular oscillations (default 0.10).

    Returns
    -------
    is_limit_cycle : bool
        True if the trajectory satisfies both criteria.
    period_mean : float or None
        Mean period (seconds) of the detected oscillation, or None if fewer
        than 3 crossings were found.
    dispersion : float
        Normalized dispersion of I at the Poincaré section crossings
        (std(I_cross) / range(I)).  Returns np.inf if detection fails.
    n_crossings : int
        Total number of upward zero-crossings detected.
    """
    # Center E around its mean to define the Poincaré section at zero
    E_mean = np.mean(E_signal)
    E_centered = E_signal - E_mean

    # --- Find upward zero-crossings (Poincaré section) ---
    crossings = []
    for i in range(len(E_centered) - 1):
        # Upward crossing: E goes from non-positive to positive
        if E_centered[i] <= 0 < E_centered[i + 1]:
            # Linear interpolation to find the precise crossing time
            alpha = -E_centered[i] / (E_centered[i + 1] - E_centered[i])
            I_cross = I_signal[i] + alpha * (I_signal[i + 1] - I_signal[i])
            t_cross = (i + alpha) * dt
            crossings.append((t_cross, I_cross))

    # Need at least 3 crossings to estimate period and dispersion
    if len(crossings) < 3:
        return False, None, np.inf, len(crossings)

    times = np.array([c[0] for c in crossings])
    I_values = np.array([c[1] for c in crossings])

    # --- Dispersion criterion ---
    I_range = np.max(I_signal) - np.min(I_signal)
    if I_range == 0:
        return False, None, np.inf, len(crossings)

    # Normalized dispersion of I at section crossings
    dispersion = float(np.std(I_values) / I_range)

    # --- Period regularity criterion ---
    periods = np.diff(times)
    period_mean = float(np.mean(periods))
    # Coefficient of variation of the inter-crossing intervals
    period_cv = float(np.std(periods) / period_mean) if period_mean > 0 else np.inf

    # Both criteria must be satisfied simultaneously
    is_limit_cycle = (dispersion < threshold_ratio) and (period_cv < period_cv_threshold)

    return is_limit_cycle, period_mean, dispersion, len(crossings)
