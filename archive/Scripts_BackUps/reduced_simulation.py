import numpy as np
from .parameters import *
from .reduced_dynamics import *
from .simulation import *

def simulate_reduced_wc(state0, P, K, M, M_reduced, D_reduced, W_reduced, T, dt):
    """
    Simula el modelo reducido con interacciones pairwise.
    """
    n_steps = int(T / dt)
    N = M.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(reduced_wc_rhs, states[i-1], dt, P, K, M, M_reduced, D_reduced, W_reduced)

    return t, states

def simulate_reduced_wc_higher_order(state0, P, K3, T_ho, M_reduced, T, dt):
    """
    Simula el modelo reducido con interacciones de alto orden.
    """
    n_steps = int(T / dt)
    N = T_ho.shape[0]
    states = np.zeros((n_steps, 2 * N))
    states[0] = state0
    t = np.linspace(0, T, n_steps)

    for i in range(1, n_steps):
        states[i] = rk4_step(reduced_wc_rhs_higher_order, states[i-1], dt, P, K3, T_ho, M_reduced)

    return t, states
