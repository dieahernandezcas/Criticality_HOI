"""
simulation_10nodes.py
=====================
Euler–Maruyama stochastic integration for the 10-node Wilson–Cowan
experiments.  Provides one stochastic simulation function per coupling
architecture (4 total), each accepting an arbitrary connectivity matrix /
tensor so the same function works for both topologies (ring-SW and random).

Functions
---------
simulate_wc_stochastic_10n               — 2nd-order additive
simulate_wc_additive_stochastic_10n      — 2nd-order diffusive
simulate_wc_ho_stochastic_10n            — 3rd-order additive
simulate_wc_ho_additive_stochastic_10n   — 3rd-order diffusive
"""

import numpy as np
from .dynamics_10nodes import (
    wc_rhs,
    wc_rhs_additive,
    wc_rhs_higher_order,
    wc_rhs_higher_order_additive,
)


# =====================================================================
# Generic RK4 step (copied for self-containment)
# =====================================================================

def rk4_step(f, state, dt, *args):
    k1 = f(state, *args)
    k2 = f(state + 0.5 * dt * k1, *args)
    k3 = f(state + 0.5 * dt * k2, *args)
    k4 = f(state + dt * k3, *args)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# =====================================================================
# 1. Second-order additive — stochastic
# =====================================================================

def simulate_wc_stochastic_10n(state0, P, K, M, T, dt,
                                sigma_E=0.03):
    """Euler–Maruyama simulation with 2nd-order additive coupling.

    Parameters
    ----------
    state0 : (2N,) initial condition [E | I]
    P      : scalar or (N,) external drive
    K      : 2nd-order coupling strength
    M      : (N, N) connectivity matrix
    T      : total time (s)
    dt     : time step (s)
    sigma_E: noise amplitude on E

    Returns
    -------
    t      : (n_steps,)
    states : (n_steps, 2N)
    """
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        deriv = wc_rhs(states[i - 1], P, K, M)
        dW = np.random.randn(N) * np.sqrt(dt)
        states[i, :N]  = states[i - 1, :N] + deriv[:N] * dt + sigma_E * dW
        states[i, N:]  = states[i - 1, N:] + deriv[N:] * dt

    return t, states


# =====================================================================
# 2. Second-order diffusive — stochastic
# =====================================================================

def simulate_wc_additive_stochastic_10n(state0, P, K, M, T, dt,
                                         sigma_E=0.03):
    """Euler–Maruyama simulation with 2nd-order diffusive coupling."""
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        deriv = wc_rhs_additive(states[i - 1], P, K, M)
        dW = np.random.randn(N) * np.sqrt(dt)
        states[i, :N]  = states[i - 1, :N] + deriv[:N] * dt + sigma_E * dW
        states[i, N:]  = states[i - 1, N:] + deriv[N:] * dt

    return t, states


# =====================================================================
# 3. Third-order additive — stochastic
# =====================================================================

def simulate_wc_ho_stochastic_10n(state0, P, K3, T_ho, T, dt,
                                   sigma_E=0.03):
    """Euler–Maruyama simulation with 3rd-order additive coupling."""
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        deriv = wc_rhs_higher_order(states[i - 1], P, K3, T_ho)
        dW = np.random.randn(N) * np.sqrt(dt)
        states[i, :N]  = states[i - 1, :N] + deriv[:N] * dt + sigma_E * dW
        states[i, N:]  = states[i - 1, N:] + deriv[N:] * dt

    return t, states


# =====================================================================
# 4. Third-order diffusive — stochastic
# =====================================================================

def simulate_wc_ho_additive_stochastic_10n(state0, P, K3, T_ho, T, dt,
                                            sigma_E=0.03):
    """Euler–Maruyama simulation with 3rd-order diffusive coupling."""
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        deriv = wc_rhs_higher_order_additive(states[i - 1], P, K3, T_ho)
        dW = np.random.randn(N) * np.sqrt(dt)
        states[i, :N]  = states[i - 1, :N] + deriv[:N] * dt + sigma_E * dW
        states[i, N:]  = states[i - 1, N:] + deriv[N:] * dt

    return t, states
