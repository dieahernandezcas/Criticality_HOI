import numpy as np
from .parameters import *
from .dynamics import *

def reduced_wc_rhs(state, P, K, M, M_reduced, D_reduced, W_reduced):
    """
    Dinámica reducida de Wilson-Cowan con interacciones pairwise.
    Args:
        state: Vector de estado completo [E1, E2, E3, I1, I2, I3].
        P: Entrada externa.
        K: Fuerza del acoplamiento.
        M: Matriz de conectividad original.
        M_reduced: Matriz de reducción (2 x N_nodes).
        D_reduced: Matriz de decaimiento reducida.
        W_reduced: Matriz de acoplamiento lineal reducida.
    Returns:
        dstate_dt: Derivada del estado completo [dE1/dt, dE2/dt, dE3/dt, dI1/dt, dI2/dt, dI3/dt].
    """
    N = M.shape[0]
    E = state[:N]
    I = state[N:]

    # Reducción de E a X1, X2
    X = M_reduced.T @ E  # Vector de tamaño 2
    X1, X2 = X[0], X[1]  # Desempaquetar correctamente

    # Reconstrucción de E a partir de X1, X2
    E_reconstructed = M_reduced @ np.array([X1, X2])

    # Dinámica de I (local para cada área)
    dI_dt = (-I + S(cIE * E_reconstructed - cII * I + Q_baseline, a=aI, theta=thetaI)) / tauI

    # Dinámica de E (con acoplamiento)
    coupling = K * M @ E_reconstructed
    dE_dt = (-E_reconstructed + S(cEE * E_reconstructed - cIE * I + P + coupling, a=aE, theta=thetaE)) / tauE

    # Combinar las derivadas
    dstate_dt = np.concatenate([dE_dt, dI_dt])
    return dstate_dt


def reduced_wc_rhs_higher_order(state, P, K3, T_ho, M_reduced):
    """
    Dinámica reducida con interacciones de alto orden.
    """
    N = T_ho.shape[0]
    E = state[:N]
    I = state[N:]

    # Reducción de E a X1, X2
    X = M_reduced.T @ E  # Vector de tamaño 2
    X1, X2 = X[0], X[1]  # Desempaquetar correctamente

    # Reconstrucción de E a partir de X1, X2
    E_reconstructed = M_reduced @ np.array([X1, X2])

    # Término de alto orden
    higher_order_input = np.zeros(N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                higher_order_input[i] += T_ho[i, j, k] * E_reconstructed[j] * E_reconstructed[k]

    # Dinámica de I
    dI_dt = (-I + S(cIE * E_reconstructed - cII * I + Q_baseline, a=aI, theta=thetaI)) / tauI

    # Dinámica de E
    dE_dt = (-E_reconstructed + S(cEE * E_reconstructed - cIE * I + P + K3 * higher_order_input, a=aE, theta=thetaE)) / tauE

    # Combinar las derivadas
    dstate_dt = np.concatenate([dE_dt, dI_dt])
    return dstate_dt

