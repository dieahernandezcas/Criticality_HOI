# functions/dynamics.py

import numpy as np
from .parameters import *


def S(x, a=aE, theta=thetaE):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))


def wc_rhs(state, P, K, M):
    """Right-hand side of Wilson-Cowan equations with pairwise interactions."""
    N = M.shape[0]
    E = state[:N]
    I = state[N:]

    coupling = K * M @ E
    dE_dt = (-E + S(cEE * E - cEI * I + P + coupling)) / tauE
    dI_dt = (-I + S(cIE * E - cII * I + Q_baseline)) / tauI

    return np.concatenate([dE_dt, dI_dt])


def wc_rhs_higher_order(state, P, K3, T_ho):
    N = T_ho.shape[0]

    # Split state into excitatory and inhibitory components
    E = state[:N]
    I = state[N:]

    # Compute higher-order coupling term for each node
    higher_order_input = np.zeros(N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                higher_order_input[i] += T_ho[i, j, k] * E[j] * E[k]

    # Excitatory population dynamics (sin el término de acoplamiento de pares)
    dE_dt = (-E + S(cEE * E - cIE * I + P + K3 * higher_order_input)) / tauE

    # Inhibitory population dynamics
    dI_dt = (-I + S(cIE * E - cII * I + Q_baseline)) / tauI

    return np.concatenate([dE_dt, dI_dt])

