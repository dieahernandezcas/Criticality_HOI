"""
simulation.py
=============
Numerical integration routines for the Wilson–Cowan network described in:

    Hernández et al. (2025) "Higher-order statistical structure emerges from
    nonlinear dynamics without explicit higher-order coupling."

This module provides:
    - A generic 4th-order Runge–Kutta (RK4) step.
    - Deterministic (noise-free) simulation functions for each of the four
      coupling architectures.
    - Stochastic simulation functions using the Euler–Maruyama method
      (additive Gaussian noise on the excitatory population only).
    - Variants for the sigmoid-slope sweeps (A–K phase diagrams).

Naming convention
-----------------
    simulate_wc_*                  → deterministic integration
    simulate_wc_*_stochastic       → stochastic integration (Euler–Maruyama)
    simulate_wc_*_a / *_stochastic_a → sigmoid-slope-sweep variants (P fixed)

All functions return (t, states) where:
    t      : np.ndarray, shape (n_steps,)   time vector
    states : np.ndarray, shape (n_steps, 2N)  state matrix [E | I]
"""

import numpy as np
from .dynamics import (
    wc_rhs,
    wc_rhs_additive,
    wc_rhs_higher_order,
    wc_rhs_higher_order_additive,
    wc_rhs_a,
    wc_rhs_higher_order_a,
    wc_rhs_additive_a,
    wc_rhs_higher_order_additive_a,
)
from .parameters import *


# ---------------------------------------------------------------------------
# Generic RK4 integrator
# ---------------------------------------------------------------------------

def rk4_step(f, state: np.ndarray, dt: float, *args) -> np.ndarray:
    """Perform one step of the classical 4th-order Runge–Kutta method.

    Integrates  dy/dt = f(y, *args)  from time t to t + dt.

    Parameters
    ----------
    f : callable
        Right-hand-side function f(state, *args) → dstate/dt.
    state : np.ndarray, shape (2N,)
        Current state vector.
    dt : float
        Integration time step.
    *args
        Additional arguments forwarded to *f*.

    Returns
    -------
    np.ndarray, shape (2N,)
        Updated state at t + dt.
    """
    k1 = f(state, *args)
    k2 = f(state + 0.5 * dt * k1, *args)
    k3 = f(state + 0.5 * dt * k2, *args)
    k4 = f(state + dt * k3, *args)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ===========================================================================
# Second-order additive coupling  — deterministic and stochastic
# ===========================================================================

def simulate_wc(state0: np.ndarray, P, K: float, M: np.ndarray,
                T: float, dt: float):
    """Deterministic simulation with second-order additive coupling.

    Parameters
    ----------
    state0 : np.ndarray, shape (2N,)
        Initial condition [E_0,...,E_{N-1}, I_0,...,I_{N-1}].
    P : float or np.ndarray, shape (N,)
        External drive.
    K : float
        Second-order coupling strength.
    M : np.ndarray, shape (N, N)
        Structural connectivity matrix.
    T : float
        Total simulation time (seconds).
    dt : float
        Integration time step (seconds).

    Returns
    -------
    t : np.ndarray, shape (n_steps,)
    states : np.ndarray, shape (n_steps, 2N)
    """
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs, states[i - 1], dt, P, K, M)

    return t, states


def simulate_wc_stochastic(state0: np.ndarray, P, K: float, M: np.ndarray,
                            T: float, dt: float, sigma_E: float = 0.01):
    """Stochastic simulation with second-order additive coupling (Euler–Maruyama).

    Adds additive Gaussian noise to the excitatory population:
        E_i(t + dt) = E_i(t) + dE_i/dt * dt + σ_E * √dt * ξ_i
    where ξ_i ~ N(0, 1) are independent white-noise increments.

    Parameters
    ----------
    state0 : np.ndarray, shape (2N,)
    P : float or np.ndarray, shape (N,)
    K : float
    M : np.ndarray, shape (N, N)
    T : float
    dt : float
    sigma_E : float, optional
        Noise amplitude (default 0.01).

    Returns
    -------
    t : np.ndarray, shape (n_steps,)
    states : np.ndarray, shape (n_steps, 2N)
    """
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        # RK4 deterministic update
        states[i] = rk4_step(wc_rhs, states[i - 1], dt, P, K, M)

        # Euler–Maruyama noise correction (applied on top of the RK4 step)
        deriv = wc_rhs(states[i - 1], P, K, M)
        dE_dt = deriv[:N]
        dI_dt = deriv[N:]

        # Independent Brownian increments for each excitatory node
        dW_E = np.random.randn(N) * np.sqrt(dt)

        E_new = states[i - 1, :N] + dE_dt * dt + sigma_E * dW_E
        I_new = states[i - 1, N:] + dI_dt * dt
        states[i] = np.concatenate([E_new, I_new])

    return t, states


# ===========================================================================
# Second-order diffusive coupling  — deterministic and stochastic
# ===========================================================================

def simulate_wc_additive(state0: np.ndarray, P, K: float, M: np.ndarray,
                          T: float, dt: float):
    """Deterministic simulation with second-order diffusive coupling.

    Parameters
    ----------
    state0 : np.ndarray, shape (2N,)
    P : float or np.ndarray, shape (N,)
    K : float
    M : np.ndarray, shape (N, N)
    T : float
    dt : float

    Returns
    -------
    t : np.ndarray, shape (n_steps,)
    states : np.ndarray, shape (n_steps, 2N)
    """
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_additive, states[i - 1], dt, P, K, M)

    return t, states


def simulate_wc_additive_stochastic(state0: np.ndarray, P, K: float,
                                     M: np.ndarray, T: float, dt: float,
                                     sigma_E: float = 0.01):
    """Stochastic simulation with second-order diffusive coupling (Euler–Maruyama).

    Parameters
    ----------
    state0 : np.ndarray, shape (2N,)
    P : float or np.ndarray, shape (N,)
    K : float
    M : np.ndarray, shape (N, N)
    T : float
    dt : float
    sigma_E : float, optional

    Returns
    -------
    t : np.ndarray, shape (n_steps,)
    states : np.ndarray, shape (n_steps, 2N)
    """
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_additive, states[i - 1], dt, P, K, M)
        deriv = wc_rhs_additive(states[i - 1], P, K, M)

        dE_dt = deriv[:N]
        dI_dt = deriv[N:]

        dW_E = np.random.randn(N) * np.sqrt(dt)

        E_new = states[i - 1, :N] + dE_dt * dt + sigma_E * dW_E
        I_new = states[i - 1, N:] + dI_dt * dt
        states[i] = np.concatenate([E_new, I_new])

    return t, states


# ===========================================================================
# Third-order additive coupling  — deterministic and stochastic
# ===========================================================================

def simulate_wc_higher_order(state0: np.ndarray, P, K3: float,
                              T_ho: np.ndarray, T: float, dt: float):
    """Deterministic simulation with third-order additive coupling.

    Parameters
    ----------
    state0 : np.ndarray, shape (2N,)
    P : float or np.ndarray, shape (N,)
    K3 : float
        Higher-order coupling strength.
    T_ho : np.ndarray, shape (N, N, N)
        Third-order interaction tensor.
    T : float
    dt : float

    Returns
    -------
    t : np.ndarray, shape (n_steps,)
    states : np.ndarray, shape (n_steps, 2N)
    """
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_higher_order, states[i - 1], dt, P, K3, T_ho)

    return t, states


def simulate_wc_higher_order_stochastic(state0: np.ndarray, P, K3: float,
                                         T_ho: np.ndarray, T: float, dt: float,
                                         sigma_E: float = 0.01):
    """Stochastic simulation with third-order additive coupling.

    Parameters
    ----------
    state0 : np.ndarray, shape (2N,)
    P : float or np.ndarray, shape (N,)
    K3 : float
    T_ho : np.ndarray, shape (N, N, N)
    T : float
    dt : float
    sigma_E : float, optional

    Returns
    -------
    t : np.ndarray, shape (n_steps,)
    states : np.ndarray, shape (n_steps, 2N)
    """
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_higher_order, states[i - 1], dt, P, K3, T_ho)
        deriv = wc_rhs_higher_order(states[i - 1], P, K3, T_ho)

        dE_dt = deriv[:N]
        dI_dt = deriv[N:]

        dW_E = np.random.randn(N) * np.sqrt(dt)

        E_new = states[i - 1, :N] + dE_dt * dt + sigma_E * dW_E
        I_new = states[i - 1, N:] + dI_dt * dt
        states[i] = np.concatenate([E_new, I_new])

    return t, states


# ===========================================================================
# Third-order diffusive coupling  — deterministic and stochastic
# ===========================================================================

def simulate_wc_higher_order_additive(state0: np.ndarray, P, K3: float,
                                       T_ho: np.ndarray, T: float, dt: float):
    """Deterministic simulation with third-order diffusive coupling.

    Parameters
    ----------
    state0 : np.ndarray, shape (2N,)
    P : float or np.ndarray, shape (N,)
    K3 : float
    T_ho : np.ndarray, shape (N, N, N)
    T : float
    dt : float

    Returns
    -------
    t : np.ndarray, shape (n_steps,)
    states : np.ndarray, shape (n_steps, 2N)
    """
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_higher_order_additive,
                              states[i - 1], dt, P, K3, T_ho)

    return t, states


def simulate_wc_higher_order_additive_stochastic(state0: np.ndarray, P,
                                                  K3: float, T_ho: np.ndarray,
                                                  T: float, dt: float,
                                                  sigma_E: float = 0.01):
    """Stochastic simulation with third-order diffusive coupling.

    Parameters
    ----------
    state0 : np.ndarray, shape (2N,)
    P : float or np.ndarray, shape (N,)
    K3 : float
    T_ho : np.ndarray, shape (N, N, N)
    T : float
    dt : float
    sigma_E : float, optional

    Returns
    -------
    t : np.ndarray, shape (n_steps,)
    states : np.ndarray, shape (n_steps, 2N)
    """
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_higher_order_additive,
                              states[i - 1], dt, P, K3, T_ho)
        deriv = wc_rhs_higher_order_additive(states[i - 1], P, K3, T_ho)

        dE_dt = deriv[:N]
        dI_dt = deriv[N:]

        dW_E = np.random.randn(N) * np.sqrt(dt)

        E_new = states[i - 1, :N] + dE_dt * dt + sigma_E * dW_E
        I_new = states[i - 1, N:] + dI_dt * dt
        states[i] = np.concatenate([E_new, I_new])

    return t, states


# ===========================================================================
# Sigmoid-slope sweep variants  (A and K as free parameters, P fixed at 7)
# ===========================================================================

def simulate_wc_stochastic_a(state0: np.ndarray, A, K: float, M: np.ndarray,
                              T: float, dt: float,
                              sigma_E: float = 0.01, P: float = 7):
    """Stochastic second-order additive simulation with a free sigmoid slope A.

    Used for the a_E–K phase diagrams (Fig. 3 of the manuscript).

    Parameters
    ----------
    state0 : np.ndarray, shape (2N,)
    A : float or np.ndarray, shape (N,)
        Sigmoid slope for the excitatory population.
    K : float
        Second-order coupling strength.
    M : np.ndarray, shape (N, N)
    T : float
    dt : float
    sigma_E : float, optional
    P : float, optional
        Fixed external drive (default 7).

    Returns
    -------
    t : np.ndarray, shape (n_steps,)
    states : np.ndarray, shape (n_steps, 2N)
    """
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_a, states[i - 1], dt, A, K, M, P)
        deriv = wc_rhs_a(states[i - 1], A, K, M, P)

        dE_dt = deriv[:N]
        dI_dt = deriv[N:]

        dW_E = np.random.randn(N) * np.sqrt(dt)

        E_new = states[i - 1, :N] + dE_dt * dt + sigma_E * dW_E
        I_new = states[i - 1, N:] + dI_dt * dt
        states[i] = np.concatenate([E_new, I_new])

    return t, states


def simulate_wc_higher_order_stochastic_a(state0: np.ndarray, A, K3: float,
                                           T_ho: np.ndarray, T: float, dt: float,
                                           sigma_E: float = 0.01, P: float = 7):
    """Stochastic third-order additive simulation with a free sigmoid slope A.

    Parameters
    ----------
    state0 : np.ndarray, shape (2N,)
    A : float or np.ndarray, shape (N,)
        Sigmoid slope for the excitatory population.
    K3 : float
        Higher-order coupling strength.
    T_ho : np.ndarray, shape (N, N, N)
    T : float
    dt : float
    sigma_E : float, optional
    P : float, optional
        Fixed external drive (default 7).

    Returns
    -------
    t : np.ndarray, shape (n_steps,)
    states : np.ndarray, shape (n_steps, 2N)
    """
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_higher_order_a,
                              states[i - 1], dt, A, K3, T_ho, P)
        deriv = wc_rhs_higher_order_a(states[i - 1], A, K3, T_ho, P)

        dE_dt = deriv[:N]
        dI_dt = deriv[N:]

        dW_E = np.random.randn(N) * np.sqrt(dt)

        E_new = states[i - 1, :N] + dE_dt * dt + sigma_E * dW_E
        I_new = states[i - 1, N:] + dI_dt * dt
        states[i] = np.concatenate([E_new, I_new])

    return t, states


# ===========================================================================
# Diffusive sigmoid-slope sweep variants  (A and K as free parameters, P fixed)
# ===========================================================================

def simulate_wc_additive_stochastic_a(state0: np.ndarray, A, K: float,
                                       M: np.ndarray, T: float, dt: float,
                                       sigma_E: float = 0.01,
                                       P: float = 7):
    """Stochastic second-order diffusive simulation with a free sigmoid slope A.

    Used for the a_E–K phase diagrams of the diffusive model.

    Parameters
    ----------
    state0 : np.ndarray, shape (2N,)
    A : float or np.ndarray, shape (N,)
        Sigmoid slope for the excitatory population.
    K : float
        Second-order coupling strength.
    M : np.ndarray, shape (N, N)
    T : float
    dt : float
    sigma_E : float, optional
    P : float, optional
        Fixed external drive (default 7).

    Returns
    -------
    t : np.ndarray, shape (n_steps,)
    states : np.ndarray, shape (n_steps, 2N)
    """
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_additive_a, states[i - 1], dt, A, K, M, P)
        deriv = wc_rhs_additive_a(states[i - 1], A, K, M, P)

        dE_dt = deriv[:N]
        dI_dt = deriv[N:]

        dW_E = np.random.randn(N) * np.sqrt(dt)

        E_new = states[i - 1, :N] + dE_dt * dt + sigma_E * dW_E
        I_new = states[i - 1, N:] + dI_dt * dt
        states[i] = np.concatenate([E_new, I_new])

    return t, states


def simulate_wc_higher_order_additive_stochastic_a(
        state0: np.ndarray, A, K3: float,
        T_ho: np.ndarray, T: float, dt: float,
        sigma_E: float = 0.01, P: float = 7):
    """Stochastic third-order diffusive simulation with a free sigmoid slope A.

    Parameters
    ----------
    state0 : np.ndarray, shape (2N,)
    A : float or np.ndarray, shape (N,)
        Sigmoid slope for the excitatory population.
    K3 : float
        Third-order coupling strength.
    T_ho : np.ndarray, shape (N, N, N)
    T : float
    dt : float
    sigma_E : float, optional
    P : float, optional
        Fixed external drive (default 7).

    Returns
    -------
    t : np.ndarray, shape (n_steps,)
    states : np.ndarray, shape (n_steps, 2N)
    """
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_higher_order_additive_a,
                              states[i - 1], dt, A, K3, T_ho, P)
        deriv = wc_rhs_higher_order_additive_a(states[i - 1], A, K3, T_ho, P)

        dE_dt = deriv[:N]
        dI_dt = deriv[N:]

        dW_E = np.random.randn(N) * np.sqrt(dt)

        E_new = states[i - 1, :N] + dE_dt * dt + sigma_E * dW_E
        I_new = states[i - 1, N:] + dI_dt * dt
        states[i] = np.concatenate([E_new, I_new])

    return t, states
