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


def simulate_wc_higher_order(state0, P, K3, T_ho, T=10.0, dt=1e-3):
    """
    Simulate Wilson-Cowan network with higher-order interactions (node-specific params).

    Parameters
    ----------
    state0 : ndarray, shape (2*N,)
        Initial state (E and I for each node)
    P : float or ndarray, shape (N,)
        External input, scalar or vector per node
    K3 : float
        Coupling strength for higher-order interactions
    T_ho : ndarray, shape (N, N, N)
        Higher-order interaction tensor
    T : float
        Total simulation time
    dt : float
        Time step
    """
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2*N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(wc_rhs_higher_order, states[i-1], dt, P, K3, T_ho)

    return t, states



def euler_maruyama_step(state, dt, rhs_function, *args, sigma_E=0.01):
    """Euler-Maruyama step for Wilson-Cowan with noise in E."""
    N = len(state) // 2

    dstate_dt = rhs_function(state, *args)
    dE_dt = dstate_dt[:N]
    dI_dt = dstate_dt[N:]

    dW_E = np.random.randn(N) * np.sqrt(dt)

    E_new = state[:N] + dE_dt * dt + sigma_E * dW_E
    I_new = state[N:] + dI_dt * dt

    return np.concatenate([E_new, I_new])



def simulate_wc_stochastic(state0, P, K, M, T, dt, higher_order=False, T_ho=None, sigma_E=0.01):
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



def simulate_wc_stochastic_scaled(state0, P, K, M, T, dt,
                           higher_order=False,
                           T_ho=None,
                           sigma_E=1.0,
                           N_pop=5000):
    """
    Stochastic Wilson-Cowan with finite-size multiplicative noise
    consistent with mesoscopic derivation.

    Noise scales as 1/sqrt(N_pop) and vanishes at E=0.
    """

    n_steps = int(T / dt)
    N_nodes = M.shape[0]

    states = np.zeros((n_steps, 2 * N_nodes))
    states[0] = state0

    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):

        if higher_order:
            if T_ho is None:
                raise ValueError("T_ho must be provided for higher-order interactions.")
            dstate_dt = wc_rhs_higher_order(states[i-1], P, K, T_ho)
        else:
            dstate_dt = wc_rhs(states[i-1], P, K, M)

        dE_dt = dstate_dt[:N_nodes]
        dI_dt = dstate_dt[N_nodes:]

        E_prev = states[i-1, :N_nodes]
        I_prev = states[i-1, N_nodes:]

        # --------- Multiplicative finite-size noise ----------
        dW = np.random.randn(N_nodes) * np.sqrt(dt)

        noise_term = (sigma_E / np.sqrt(N_pop)) \
                     * np.sqrt(np.maximum(E_prev * (1 - E_prev), 0)) \
                     * dW

        # Euler-Maruyama update
        E_new = E_prev + dE_dt * dt + noise_term
        I_new = I_prev + dI_dt * dt

        # Ensure E stays in [0,1]
        E_new = np.clip(E_new, 0, 1)

        states[i] = np.concatenate([E_new, I_new])

    return t, states





def simulate_wc_with_avalanches(state0, P, K, M, T, dt, 
                                avalanche_amp=0.5, avalanche_period=None):
    """
    Simulate Wilson-Cowan network with periodic avalanche inputs in E.
    
    Parameters
    ----------
    state0 : ndarray, shape (2*N,)
        Initial state (E and I for each node)
    P : float or ndarray, shape (N,)
        External input (scalar or vector per node)
    K : float
        Pairwise coupling strength
    M : ndarray, shape (N, N)
        Pairwise structural connectivity
    T : float
        Total simulation time
    dt : float
        Time step
    avalanche_amp : float, optional
        Amplitude of periodic avalanche input
    avalanche_period : float, optional
        Period of avalanche injection. If None, defaults to T/3.
    
    Returns
    -------
    t : ndarray
        Time vector
    states : ndarray, shape (n_steps, 2*N)
        Simulated states (E and I)
    """
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2*N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)
    
    if avalanche_period is None:
        avalanche_period = T / 3.0
    
    avalanche_steps = int(avalanche_period / dt)
    
    for i in range(1, n_steps):
        # Paso de Runge-Kutta normal
        states[i] = rk4_step(wc_rhs, states[i-1], dt, P, K, M)
        
        # Cada 'avalanche_steps', sumamos un impulso a E
        if i % avalanche_steps == 0:
            states[i, :N] += avalanche_amp
    
    return t, states



def simulate_wc_with_avalanches_stochastic(
    state0, P, K, M, T, dt,
    avalanche_amp=0.5,
    avalanche_period=None,
    sigma_E=0.01
):
    """
    Wilson–Cowan with periodic step avalanches + Brownian noise in E.
    """
    n_steps = int(T / dt)
    N = M.shape[0]

    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    if avalanche_period is None:
        avalanche_period = T / 3.0

    avalanche_steps = int(avalanche_period / dt)

    for i in range(1, n_steps):

        # 1paso determinista (RK4)
        states[i] = rk4_step(
            wc_rhs,
            states[i-1],
            dt,
            P,
            K,
            M
        )

        # ruido browniano SOLO en E
        dW_E = np.random.randn(N) * np.sqrt(dt)
        states[i, :N] += sigma_E * dW_E

        # avalancha tipo step (evento exógeno)
        if i % avalanche_steps == 0:
            states[i, :N] += avalanche_amp

    return t, states



def simulate_wc_with_avalanches_general(
    state0,
    P,
    K=None,
    M=None,
    K3=None,
    T_ho=None,
    T=10.0,
    dt=1e-3,
    avalanche_amp=0.5,
    avalanche_period=None,
    sigma_E=0.0,
    higher_order=False
):
    """
    Wilson–Cowan network with avalanches + optional noise + pairwise or higher-order interactions.

    Parameters
    ----------
    state0 : ndarray, shape (2*N,)
        Initial state (E and I)
    P : float or ndarray
        External input (scalar or per node)
    K : float, optional
        Pairwise coupling strength
    M : ndarray, optional
        Pairwise connectivity matrix
    K3 : float, optional
        Higher-order coupling strength
    T_ho : ndarray, optional
        Higher-order interaction tensor
    T : float
        Total simulation time
    dt : float
        Time step
    avalanche_amp : float
        Amplitude of periodic avalanche
    avalanche_period : float
        Period of avalanches. If None, defaults to T/3
    sigma_E : float
        Noise amplitude in E
    higher_order : bool
        Use higher-order coupling if True
    
    Returns
    -------
    t : ndarray
        Time vector
    states : ndarray
        Simulated states (E and I)
    """
    n_steps = int(T / dt)
    N = len(state0) // 2
    states = np.zeros((n_steps, 2*N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    if avalanche_period is None:
        avalanche_period = T / 3.0
    avalanche_steps = int(avalanche_period / dt)

    for i in range(1, n_steps):
        # 1️⃣ Paso determinista
        if higher_order:
            if T_ho is None or K3 is None:
                raise ValueError("T_ho and K3 must be provided for higher-order interactions")
            states[i] = rk4_step(wc_rhs_higher_order, states[i-1], dt, P, K3, T_ho)
        else:
            if M is None or K is None:
                raise ValueError("M and K must be provided for pairwise interactions")
            states[i] = rk4_step(wc_rhs, states[i-1], dt, P, K, M)

        # 2️⃣ Ruido Browniano solo en E
        if sigma_E > 0.0:
            dW_E = np.random.randn(N) * np.sqrt(dt)
            states[i, :N] += sigma_E * dW_E

        # 3️⃣ Avalancha tipo step
        if i % avalanche_steps == 0:
            states[i, :N] += avalanche_amp

    return t, states
