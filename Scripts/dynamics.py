# functions/dynamics.py
import numpy as np
from .parameters import param_nodes


def S(x, a, theta):
    #return 1.0 / (1.0 + np.exp(-a * (x - theta)))
    return 1.0 / (1.0 + np.exp(-a * (x)))


def wc_rhs(state, P, K, M):
    """
    Wilson–Cowan with pairwise coupling.
    P can be scalar or array of length N.
    """
    N = M.shape[0]
    E = state[:N]
    I = state[N:]

    coupling = K * (M @ E)

    dE_dt = np.zeros(N)
    dI_dt = np.zeros(N)

    for i in range(N):
        p = param_nodes[i]

        P_i = P if np.isscalar(P) else P[i]

        dE_dt[i] = (
            -E[i]
            + S(
                p["cEE"] * E[i]
                - p["cEI"] * I[i]
                + P_i
                + coupling[i],
                p["aE"],
                p["thetaE"],
            )
        ) / p["tauE"]

        dI_dt[i] = (
            -I[i]
            + S(
                p["cIE"] * E[i]
                - p["cII"] * I[i]
                + p["Q"],
                p["aI"],
                p["thetaI"],
            )
        ) / p["tauI"]

    return np.concatenate([dE_dt, dI_dt])


def wc_rhs_higher_order(state, P, K3, T_ho):
    """
    Wilson-Cowan RHS with higher-order interactions and node-specific parameters.
    state: [E0, E1, ..., I0, I1, ...]
    P: external input (scalar or vector)
    K3: higher-order coupling scalar
    T_ho: higher-order tensor
    """
    N = T_ho.shape[0]
    E = state[:N]
    I = state[N:]
    
    dE_dt = np.zeros(N)
    dI_dt = np.zeros(N)
    
    for i in range(N):
        p = param_nodes[i]  # parámetros nodo i
        
        # external input dinámico por nodo
        P_i = P if np.isscalar(P) else P[i]
        
        # término de acoplamiento de orden superior
        higher_order_input = 0.0
        for j in range(N):
            for k in range(N):
                higher_order_input += T_ho[i, j, k] * E[j] * E[k]
        
        dE_dt[i] = (-E[i] + S(p['cEE']*E[i] - p['cEI']*I[i] + P_i + K3*higher_order_input,
                              a=p['aE'], theta=p['thetaE'])) / p['tauE']
        
        dI_dt[i] = (-I[i] + S(p['cIE']*E[i] - p['cII']*I[i] + p['Q'],
                              a=p['aI'], theta=p['thetaI'])) / p['tauI']
        
    return np.concatenate([dE_dt, dI_dt])





def numerical_jacobian(f, x0, eps=1e-6):
    n = len(x0)
    J = np.zeros((n, n))
    f0 = f(x0)

    for i in range(n):
        x_eps = x0.copy()
        x_eps[i] += eps
        fi = f(x_eps)
        J[:, i] = (fi - f0) / eps

    return J




def wc_rhs_additive(state, P, K, M):
    """
    Wilson–Cowan with pairwise coupling.
    P can be scalar or array of length N.
    """
    N = M.shape[0]
    E = state[:N]
    I = state[N:]

    degree = M.sum(axis=1)
    coupling = K * (M @ E - degree * E)

    dE_dt = np.zeros(N)
    dI_dt = np.zeros(N)

    for i in range(N):
        p = param_nodes[i]

        P_i = P if np.isscalar(P) else P[i]

        dE_dt[i] = (
            -E[i]
            + S(
                p["cEE"] * E[i]
                - p["cEI"] * I[i]
                + P_i
                + coupling[i],
                p["aE"],
                p["thetaE"],
            )
        ) / p["tauE"]

        dI_dt[i] = (
            -I[i]
            + S(
                p["cIE"] * E[i]
                - p["cII"] * I[i]
                + p["Q"],
                p["aI"],
                p["thetaI"],
            )
        ) / p["tauI"]

    return np.concatenate([dE_dt, dI_dt])


def wc_rhs_higher_order_additive(state, P, K3, T_ho):
    """
    Wilson-Cowan RHS with higher-order interactions and node-specific parameters.
    state: [E0, E1, ..., I0, I1, ...]
    P: external input (scalar or vector)
    K3: higher-order coupling scalar
    T_ho: higher-order tensor
    """
    N = T_ho.shape[0]
    E = state[:N]
    I = state[N:]
    
    dE_dt = np.zeros(N)
    dI_dt = np.zeros(N)
    
    for i in range(N):
        p = param_nodes[i]
        P_i = P if np.isscalar(P) else P[i]
        
        # ---- Higher-order Laplacian term ----
        Ti = T_ho[i]           # NxN slice
        diff = E - E[i]        # (Ej - Ei)
        higher_order_input = diff @ Ti @ diff
        
        dE_dt[i] = (-E[i] + S(p['cEE']*E[i] - p['cEI']*I[i] + P_i + K3*higher_order_input,
                              a=p['aE'], theta=p['thetaE'])) / p['tauE']
        
        dI_dt[i] = (-I[i] + S(p['cIE']*E[i] - p['cII']*I[i] + p['Q'],
                              a=p['aI'], theta=p['thetaI'])) / p['tauI']
        
    return np.concatenate([dE_dt, dI_dt])