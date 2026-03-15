# functions/simulation.py
import numpy as np
from .dynamics import *
from .parameters import *


def rk4_step(f, state, dt, *args):
    """Fourth-order Runge-Kutta step."""
    k1 = f(state, *args)
    k2 = f(state + 0.5 * dt * k1, *args)
    k3 = f(state + 0.5 * dt * k2, *args)
    k4 = f(state + dt * k3, *args)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_wc(state0, P, K, M, T, dt):
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs, states[i-1], dt, P, K, M)

    return t, states

def simulate_wc_stochastic(state0, P, K, M, T, dt, sigma_E=0.01):
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs, states[i-1], dt, P, K, M)
        # derivada en el estado previo
        deriv = wc_rhs(states[i-1], P, K, M)
        dE_dt = deriv[:N]
        dI_dt = deriv[N:]

        # ruido browniano
        dW_E = np.random.randn(N) * np.sqrt(dt)

        E_new = states[i-1, :N] + dE_dt * dt + sigma_E * dW_E
        I_new = states[i-1, N:] + dI_dt * dt

        states[i] = np.concatenate([E_new, I_new])

    return t, states


def simulate_wc_higher_order(state0, P, K3, T_ho, T, dt):
    """Simulate Wilson-Cowan network with higher-order interactions."""
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_higher_order, states[i-1], dt, P, K3, T_ho)

    return t, states


def simulate_wc_higher_order_stochastic(state0, P, K3, T_ho, T, dt, sigma_E=0.01):
    """Simulate Wilson-Cowan network with higher-order interactions."""
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_higher_order, states[i-1], dt, P, K3, T_ho)
        deriv = wc_rhs_higher_order(states[i-1], P, K3, T_ho)

        dE_dt = deriv[:N]
        dI_dt = deriv[N:]

        # ruido browniano
        dW_E = np.random.randn(N) * np.sqrt(dt)

        E_new = states[i-1, :N] + dE_dt * dt + sigma_E * dW_E
        I_new = states[i-1, N:] + dI_dt * dt

        states[i] = np.concatenate([E_new, I_new])

    return t, states


def simulate_wc_additive(state0, P, K, M, T, dt):
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_additive, states[i-1], dt, P, K, M)

    return t, states


def simulate_wc_additive_stochastic(state0, P, K, M, T, dt, sigma_E=0.01):
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_additive, states[i-1], dt, P, K, M)
        deriv = wc_rhs_additive(states[i-1], P, K, M)

        dE_dt = deriv[:N]
        dI_dt = deriv[N:]

        # ruido browniano
        dW_E = np.random.randn(N) * np.sqrt(dt)

        E_new = states[i-1, :N] + dE_dt * dt + sigma_E * dW_E
        I_new = states[i-1, N:] + dI_dt * dt

        states[i] = np.concatenate([E_new, I_new])

    return t, states


def simulate_wc_higher_order_additive(state0, P, K3, T_ho, T, dt):
    """Simulate Wilson-Cowan network with higher-order interactions."""
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_higher_order_additive, states[i-1], dt, P, K3, T_ho)

    return t, states

def simulate_wc_higher_order_additive_stochastic(state0, P, K3, T_ho, T, dt, sigma_E=0.01):
    """Simulate Wilson-Cowan network with higher-order interactions."""
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_higher_order_additive, states[i-1], dt, P, K3, T_ho)
        deriv = wc_rhs_higher_order_additive(states[i-1], P, K3, T_ho)

        dE_dt = deriv[:N]
        dI_dt = deriv[N:]

        # ruido browniano
        dW_E = np.random.randn(N) * np.sqrt(dt)

        E_new = states[i-1, :N] + dE_dt * dt + sigma_E * dW_E
        I_new = states[i-1, N:] + dI_dt * dt

        states[i] = np.concatenate([E_new, I_new])

    return t, states



##moving parameters A and K:

def simulate_wc_higher_order_stochastic_a(state0, A, K3, T_ho, T, dt, sigma_E=0.01, P=7):
    """Simulate Wilson-Cowan network with higher-order interactions."""
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_higher_order_a, states[i-1], dt, A, K3, T_ho, P)
        deriv = wc_rhs_higher_order_a(states[i-1], A, K3, T_ho, P)

        dE_dt = deriv[:N]
        dI_dt = deriv[N:]

        # ruido browniano
        dW_E = np.random.randn(N) * np.sqrt(dt)

        E_new = states[i-1, :N] + dE_dt * dt + sigma_E * dW_E
        I_new = states[i-1, N:] + dI_dt * dt

        states[i] = np.concatenate([E_new, I_new])

    return t, states


def simulate_wc_stochastic_a(state0, A, K, M, T, dt, sigma_E=0.01, P=7):
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_a, states[i-1], dt, A, K, M, P)
        # derivada en el estado previo
        deriv = wc_rhs_a(states[i-1], A, K, M, P)
        dE_dt = deriv[:N]
        dI_dt = deriv[N:]

        # ruido browniano
        dW_E = np.random.randn(N) * np.sqrt(dt)

        E_new = states[i-1, :N] + dE_dt * dt + sigma_E * dW_E
        I_new = states[i-1, N:] + dI_dt * dt

        states[i] = np.concatenate([E_new, I_new])

    return t, states








#########Backup functions

def simulate_wc_stochastic_backup(state0, P, K, M, T, dt, higher_order=False, T_ho=None, sigma_E=0.01):
    """
    Simulate Wilson-Cowan network with stochastic excitatory dynamics (noise in E).

    Parameters
    ----------
    state0 : ndarray, shape (2*N,)
        Initial state (E and I for each node)
    P : float or ndarray, shape (N,)
        External input, can be scalar or vector per node
    K : float
        Coupling strength for pairwise interactions
    M : ndarray, shape (N, N)
        Pairwise structural connectivity
    T : float
        Total simulation time
    dt : float
        Time step
    higher_order : bool, optional
        If True, use higher-order interactions (requires T_ho)
    T_ho : ndarray, optional, shape (N, N, N)
        Higher-order interaction tensor
    sigma_E : float, optional
        Noise amplitude for E (Brownian)
    
    Returns
    -------
    t : ndarray
        Time vector
    states : ndarray, shape (n_steps, 2*N)
        Simulated states (E and I)
    """
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        # calcular RHS según tipo de acoplamiento
        if higher_order:
            if T_ho is None:
                raise ValueError("T_ho must be provided for higher-order interactions.")
            dstate_dt = wc_rhs_higher_order(states[i-1], P, K, T_ho)
        else:
            dstate_dt = wc_rhs(states[i-1], P, K, M)

        dE_dt = dstate_dt[:N]
        dI_dt = dstate_dt[N:]

        # ruido Browniano solo en E
        dW_E = np.random.randn(N) * np.sqrt(dt)

        # si P es escalar, mantener igual, si es vector, multiplicar por nodo
        E_new = states[i-1, :N] + dE_dt * dt + (sigma_E * dW_E)
        I_new = states[i-1, N:] + dI_dt * dt

        states[i] = np.concatenate([E_new, I_new])

    return t, states




def simulate_wc_stochastic_additive_backup(state0, P, K, M, T, dt, higher_order=False, T_ho=None, sigma_E=0.01):
    """
    Simulate Wilson-Cowan network with stochastic excitatory dynamics (noise in E).

    Parameters
    ----------
    state0 : ndarray, shape (2*N,)
        Initial state (E and I for each node)
    P : float or ndarray, shape (N,)
        External input, can be scalar or vector per node
    K : float
        Coupling strength for pairwise interactions
    M : ndarray, shape (N, N)
        Pairwise structural connectivity
    T : float
        Total simulation time
    dt : float
        Time step
    higher_order : bool, optional
        If True, use higher-order interactions (requires T_ho)
    T_ho : ndarray, optional, shape (N, N, N)
        Higher-order interaction tensor
    sigma_E : float, optional
        Noise amplitude for E (Brownian)
    
    Returns
    -------
    t : ndarray
        Time vector
    states : ndarray, shape (n_steps, 2*N)
        Simulated states (E and I)
    """
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    if higher_order:
        if T_ho is None:
            raise ValueError("T_ho must be provided for higher-order interactions.")
        dstate_dt = simulate_wc_higher_order_additive_stochastic(state0, P, K3, T_ho, T, dt)
    else:
        dstate_dt = simulate_wc_additive_stochastic(states[i-1], P, K, M)

    """
    for i in range(1, n_steps):
        # calcular RHS según tipo de acoplamiento
        if higher_order:
            if T_ho is None:
                raise ValueError("T_ho must be provided for higher-order interactions.")
            dstate_dt = wc_rhs_higher_order_additive(states[i-1], P, K, T_ho)
        else:
            dstate_dt = wc_rhs_additive(states[i-1], P, K, M)

        dE_dt = dstate_dt[:N]
        dI_dt = dstate_dt[N:]

        # ruido Browniano solo en E
        dW_E = np.random.randn(N) * np.sqrt(dt)

        # si P es escalar, mantener igual, si es vector, multiplicar por nodo
        E_new = states[i-1, :N] + dE_dt * dt + (sigma_E * dW_E)
        I_new = states[i-1, N:] + dI_dt * dt

        states[i] = np.concatenate([E_new, I_new])
        """

    return t, dstate_dt #states
