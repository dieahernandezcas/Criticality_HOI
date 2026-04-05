"""
parameters_10nodes.py
=====================
Network topology and node-level parameters for the 10-node Wilson–Cowan
experiments.  Two connectivity architectures are defined:

    1. **Ring–small-world** (``M_ring``, ``T_ho_ring``):
       Lattice ring where each node connects to its nearest *and*
       next-nearest neighbours (1-hop shortcut), giving a uniform degree
       of 4.  Two additional long-range ("diagonal") edges are added
       between nodes 0↔5 and 2↔7 to introduce small-world structure.
       Those two nodes therefore have degree 5; the rest keep degree 4.

    2. **Random** (``M_rand``, ``T_ho_rand``):
       A fixed random topology (seed 2025) with heterogeneous degree
       distribution, providing a contrast to the structured ring.

Node-level biophysical parameters are based on the 3-node paper values
but with small per-node perturbations of θ_E and θ_I so that every node
is slightly different while remaining near the Hopf boundary.

Usage
-----
    from Scripts.parameters_10nodes import (
        N_nodes, param_nodes,
        M_ring, T_ho_ring,
        M_rand, T_ho_rand,
        dt, sigma_E, P_values, K_values, K3_values,
    )
"""

import numpy as np


# =====================================================================
# Reproducibility
# =====================================================================
_rng = np.random.default_rng(2025)

# =====================================================================
# Network size
# =====================================================================
N_nodes = 10

# =====================================================================
# Integration defaults
# =====================================================================
dt = 1e-3
sigma_E = 0.03

# =====================================================================
# Parameter sweep grids  (same as 3-node experiments)
# =====================================================================
P_values  = np.linspace(0.0, 10.0, 20)
K_values  = np.linspace(0.0, 1.0, 20)
K3_values = np.linspace(0.0, 1.0, 20)


# =====================================================================
# 1.  Ring–Small-World topology
# =====================================================================
# Base lattice ring: each node i connects to i±1 and i±2  (mod 10)
M_ring = np.zeros((N_nodes, N_nodes))
for i in range(N_nodes):
    for offset in [1, 2]:                       # nearest + next-nearest
        j_fwd = (i + offset) % N_nodes
        j_bwd = (i - offset) % N_nodes
        M_ring[i, j_fwd] = 1.0
        M_ring[j_fwd, i] = 1.0
        M_ring[i, j_bwd] = 1.0
        M_ring[j_bwd, i] = 1.0

# Add two long-range ("diagonal") edges for small-world property
#   0 ↔ 5   (distance 5 on the ring → antipodal)
#   2 ↔ 7   (distance 5 on the ring → antipodal)
M_ring[0, 5] = 1.0;  M_ring[5, 0] = 1.0
M_ring[2, 7] = 1.0;  M_ring[7, 2] = 1.0

# Ensure no self-connections
np.fill_diagonal(M_ring, 0.0)

# =====================================================================
# 2.  Random topology (fixed seed for reproducibility)
# =====================================================================
# Erdős–Rényi-like: each off-diagonal pair has independent probability
# p = 0.45 of being connected.  This gives E[degree] ≈ 4.05 (similar
# average degree to the ring) but with a non-uniform distribution.
M_rand = np.zeros((N_nodes, N_nodes))
for i in range(N_nodes):
    for j in range(i + 1, N_nodes):
        if _rng.random() < 0.45:
            M_rand[i, j] = 1.0
            M_rand[j, i] = 1.0
np.fill_diagonal(M_rand, 0.0)

# If any node ended up isolated (degree 0), connect it to a random
# neighbour so the network is connected.
for i in range(N_nodes):
    if M_rand[i].sum() == 0:
        j = _rng.integers(0, N_nodes)
        while j == i:
            j = _rng.integers(0, N_nodes)
        M_rand[i, j] = 1.0
        M_rand[j, i] = 1.0


# =====================================================================
# Higher-order interaction tensors  (derived from each M)
# =====================================================================
def _build_T_ho(M: np.ndarray) -> np.ndarray:
    """Build the third-order interaction tensor from a connectivity matrix.

    T_ho[i, j, k] = 1  iff  i ≠ j ≠ k ≠ i  AND  M[i,j] ≠ 0  AND  M[i,k] ≠ 0.
    """
    N = M.shape[0]
    T = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if i != j and i != k and j != k:
                    if M[i, j] != 0 and M[i, k] != 0:
                        T[i, j, k] = 1.0
    return T


T_ho_ring = _build_T_ho(M_ring)
T_ho_rand = _build_T_ho(M_rand)


# =====================================================================
# Node-level biophysical parameters  (10 heterogeneous nodes)
# =====================================================================
# Base values (from the 3-node paper):
_base = {
    "tauE":   0.12,
    "tauI":   0.25,
    "aE":     1.5,
    "aI":     1.5,
    "thetaE": 2.0,
    "thetaI": 3.0,
    "cEE":    6.0,
    "cEI":    15.0,
    "cIE":    7.0,
    "cII":    0.4,
    "P":      5.0,
    "Q":     -1.5,
}

# Small deterministic perturbations of θ_E and θ_I per node
# (drawn once from seed 2025 so they are fixed across runs).
_delta_thetaE = _rng.uniform(-0.3, 0.3, N_nodes)
_delta_thetaI = _rng.uniform(-0.5, 0.5, N_nodes)

param_nodes = []
for i in range(N_nodes):
    node = dict(_base)                          # copy the base
    node["thetaE"] = _base["thetaE"] + _delta_thetaE[i]
    node["thetaI"] = _base["thetaI"] + _delta_thetaI[i]
    param_nodes.append(node)


# =====================================================================
# Diagnostics (printed once on import)
# =====================================================================
def _print_topo_info(name: str, M: np.ndarray):
    degrees = M.sum(axis=1).astype(int)
    print(f"  {name}:  degrees = {degrees.tolist()}, "
          f"mean = {degrees.mean():.1f}, "
          f"edges = {int(M.sum()) // 2}")

print(f"[10-node] N = {N_nodes}")
_print_topo_info("Ring-SW", M_ring)
_print_topo_info("Random ", M_rand)
print(f"[10-node] thetaE perturbations: "
      f"{[round(d, 3) for d in _delta_thetaE]}")
print(f"[10-node] thetaI perturbations: "
      f"{[round(d, 3) for d in _delta_thetaI]}")
