"""
dynamics_10nodes.py
===================
Right-hand-side (RHS) functions for the 10-node Wilson–Cowan experiments.

These are identical to the 4 coupling architectures in dynamics.py but read
the node parameters from ``parameters_10nodes.param_nodes`` (10 heterogeneous
nodes) instead of ``parameters.param_nodes`` (3 nodes).

Coupling architectures
----------------------
1. ``wc_rhs``                        — Second-order additive
2. ``wc_rhs_additive``               — Second-order diffusive
3. ``wc_rhs_higher_order``           — Third-order additive
4. ``wc_rhs_higher_order_additive``  — Third-order diffusive
"""

import numpy as np
from .parameters_10nodes import param_nodes


# ---------------------------------------------------------------------------
# Sigmoid transfer function
# ---------------------------------------------------------------------------

def S(x, a, theta):
    """Logistic sigmoid: S(x; a) = 1 / (1 + exp(-a * x))."""
    return 1.0 / (1.0 + np.exp(-a * x))


# ---------------------------------------------------------------------------
# 1. Second-order additive coupling
# ---------------------------------------------------------------------------

def wc_rhs(state, P, K, M):
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
            + S(p["cEE"] * E[i] - p["cEI"] * I[i] + P_i + coupling[i],
                p["aE"], p["thetaE"])
        ) / p["tauE"]

        dI_dt[i] = (
            -I[i]
            + S(p["cIE"] * E[i] - p["cII"] * I[i] + p["Q"],
                p["aI"], p["thetaI"])
        ) / p["tauI"]

    return np.concatenate([dE_dt, dI_dt])


# ---------------------------------------------------------------------------
# 2. Second-order diffusive coupling
# ---------------------------------------------------------------------------

def wc_rhs_additive(state, P, K, M):
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
            + S(p["cEE"] * E[i] - p["cEI"] * I[i] + P_i + coupling[i],
                p["aE"], p["thetaE"])
        ) / p["tauE"]

        dI_dt[i] = (
            -I[i]
            + S(p["cIE"] * E[i] - p["cII"] * I[i] + p["Q"],
                p["aI"], p["thetaI"])
        ) / p["tauI"]

    return np.concatenate([dE_dt, dI_dt])


# ---------------------------------------------------------------------------
# 3. Third-order additive coupling
# ---------------------------------------------------------------------------

def wc_rhs_higher_order(state, P, K3, T_ho):
    N = T_ho.shape[0]
    E = state[:N]
    I = state[N:]

    dE_dt = np.zeros(N)
    dI_dt = np.zeros(N)

    for i in range(N):
        p = param_nodes[i]
        P_i = P if np.isscalar(P) else P[i]

        # Bilinear HOI input: sum_{j,k} T[i,j,k] E_j E_k
        # Vectorised as E^T @ T_i @ E
        ho_input = E @ T_ho[i] @ E

        dE_dt[i] = (
            -E[i]
            + S(p["cEE"] * E[i] - p["cEI"] * I[i] + P_i + K3 * ho_input,
                p["aE"], p["thetaE"])
        ) / p["tauE"]

        dI_dt[i] = (
            -I[i]
            + S(p["cIE"] * E[i] - p["cII"] * I[i] + p["Q"],
                p["aI"], p["thetaI"])
        ) / p["tauI"]

    return np.concatenate([dE_dt, dI_dt])


# ---------------------------------------------------------------------------
# 4. Third-order diffusive coupling
# ---------------------------------------------------------------------------

def wc_rhs_higher_order_additive(state, P, K3, T_ho):
    N = T_ho.shape[0]
    E = state[:N]
    I = state[N:]

    dE_dt = np.zeros(N)
    dI_dt = np.zeros(N)

    for i in range(N):
        p = param_nodes[i]
        P_i = P if np.isscalar(P) else P[i]

        diff = E - E[i]
        ho_input = diff @ T_ho[i] @ diff

        dE_dt[i] = (
            -E[i]
            + S(p["cEE"] * E[i] - p["cEI"] * I[i] + P_i + K3 * ho_input,
                p["aE"], p["thetaE"])
        ) / p["tauE"]

        dI_dt[i] = (
            -I[i]
            + S(p["cIE"] * E[i] - p["cII"] * I[i] + p["Q"],
                p["aI"], p["thetaI"])
        ) / p["tauI"]

    return np.concatenate([dE_dt, dI_dt])
