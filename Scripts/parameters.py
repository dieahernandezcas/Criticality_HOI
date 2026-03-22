"""
parameters.py
=============
Network and node-level parameter configuration for the three-node Wilson–Cowan
simulations described in:

    Hernández et al. (2025) "Higher-order statistical structure emerges from
    nonlinear dynamics without explicit higher-order coupling."

This module defines:
    - The all-to-all pairwise connectivity matrix M (3×3).
    - The higher-order interaction tensor T_ho (3×3×3), derived from M.
    - Node-specific biophysical parameters (time constants, local connectivity
      weights, sigmoid slope, external input Q).
    - Default numerical integration parameters (dt, T, sigma_E).
    - Utility functions for effective-rank analysis of the connectivity matrix.

Usage
-----
    from Scripts.parameters import param_nodes, M, T_ho, dt, T, sigma_E

Notes
-----
- param_nodes is a list of three dictionaries, one per node.
- The heterogeneous parameter sets across the three nodes were chosen to
  produce a dynamical regime where the system can transition between a stable
  fixed point and a limit cycle (Hopf bifurcation) as P or K are varied.
- All parameters are consistent with the notation in the manuscript.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Effective-rank utilities
# ---------------------------------------------------------------------------

def stable_rank(matrix: np.ndarray) -> float:
    """Return the stable rank of *matrix*.

    The stable rank is defined as ||A||_F^2 / ||A||_2^2 and provides a
    continuous, noise-robust estimate of the number of non-negligible
    singular values.

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n)
        Input matrix.

    Returns
    -------
    float
        Stable rank value.
    """
    _, Sigma, _ = np.linalg.svd(matrix)
    return float((Sigma ** 2).sum() / (Sigma[0] ** 2))


def nuclear_rank(matrix: np.ndarray) -> float:
    """Return the nuclear (trace) rank of *matrix*.

    Defined as ||A||_* / ||A||_2, i.e., the ratio of the nuclear norm to the
    spectral norm.

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n)
        Input matrix.

    Returns
    -------
    float
        Nuclear rank value.
    """
    _, Sigma, _ = np.linalg.svd(matrix)
    return float(Sigma.sum() / Sigma[0])


def energy_ratio(matrix: np.ndarray, threshold: float = 0.95):
    """Return the effective rank based on cumulative singular-value energy.

    The effective rank is the smallest integer k such that the top-k singular
    values capture at least *threshold* fraction of the total Frobenius energy.

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n)
        Input matrix.
    threshold : float, optional
        Energy fraction required (default 0.95).

    Returns
    -------
    effective_rank : int
        Smallest k satisfying the energy criterion.
    cumulative_energy : np.ndarray
        Cumulative fractional energy per singular value.
    """
    _, Sigma, _ = np.linalg.svd(matrix)
    total_energy = (Sigma ** 2).sum()
    cumulative_energy = np.cumsum(Sigma ** 2) / total_energy
    effective_rank = int(np.argmax(cumulative_energy >= threshold) + 1)
    return effective_rank, cumulative_energy


# ---------------------------------------------------------------------------
# Network topology
# ---------------------------------------------------------------------------

# Number of Wilson–Cowan nodes
N_nodes = 3

# All-to-all pairwise structural connectivity matrix (no self-connections).
# M[i, j] = 1 means node j provides pairwise input to node i.
M = np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0]
])

# ---------------------------------------------------------------------------
# Higher-order interaction tensor
# ---------------------------------------------------------------------------
# T_ho[i, j, k] = 1 if nodes i, j, k are all distinct AND both
# edges (i,j) and (i,k) exist in M.  This defines the third-order coupling
# architecture used in the explicit HOI models (see Sec. II-C of the paper).
T_ho = np.zeros((N_nodes, N_nodes, N_nodes))
for i in range(N_nodes):
    for j in range(N_nodes):
        for k in range(N_nodes):
            if i != j and i != k and j != k:
                if M[i, j] != 0 and M[i, k] != 0:
                    T_ho[i, j, k] = 1.0

# ---------------------------------------------------------------------------
# Noise and integration defaults
# ---------------------------------------------------------------------------

# Additive Gaussian noise amplitude for the excitatory population (σ_E).
# The stochastic term is σ_E * √dt * ξ_i, where ξ_i ~ N(0,1).
sigma_E = 0.01

# Default integration time step (seconds)
dt = 1e-3

# Default total simulation time (seconds)
T = 10.0

# ---------------------------------------------------------------------------
# Node-specific biophysical parameters
# ---------------------------------------------------------------------------
# Each entry in param_nodes is a dictionary describing one Wilson–Cowan node.
# Parameters follow the notation in Eqs. (1)–(4) of the manuscript.
#
# Key:      Symbol in paper    Description
# ----      ----------------   -----------
# tauE      τ_E                Excitatory time constant (s)
# tauI      τ_I                Inhibitory time constant (s)
# aE        a_E                Sigmoid slope for excitatory population
# aI        a_I                Sigmoid slope for inhibitory population
# thetaE    θ_E                Sigmoid threshold for excitatory population
# thetaI    θ_I                Sigmoid threshold for inhibitory population
# cEE       c_EE               Local excitatory self-connection weight
# cEI       c_EI               Local E→I connection weight (E drives I)
# cIE       c_IE               Local I→E connection weight (I inhibits E)
# cII       c_II               Local inhibitory self-connection weight
# P         P_i                Node-level external input (baseline drive)
# Q         Q_i                External input to the inhibitory population
#
# The three nodes are intentionally heterogeneous to avoid fully synchronous
# dynamics while remaining close to the Hopf bifurcation boundary.

param_nodes = [
    {   # Node 0
        "tauE":   0.125,
        "tauI":   0.25,
        "aE":     1.5,    # Sigmoid slope; swept from 0.5 to 3 in aE-K experiments
        "aI":     1.0,
        "thetaE": 2.0,
        "thetaI": 3.0,
        "cEE":    6.0,
        "cEI":    15.0,
        "cIE":    8.0,
        "cII":    0.4,
        "P":      5.0,
        "Q":     -2.0,
    },
    {   # Node 1
        "tauE":   0.12,
        "tauI":   0.25,
        "aE":     2.0,
        "aI":     2.0,
        "thetaE": 1.5,
        "thetaI": 3.5,
        "cEE":    6.0,
        "cEI":    15.0,
        "cIE":    6.0,
        "cII":    0.4,
        "P":      3.0,
        "Q":     -1.0,
    },
    {   # Node 2
        "tauE":   0.12,
        "tauI":   0.25,
        "aE":     2.0,
        "aI":     2.0,
        "thetaE": 2.0,
        "thetaI": 1.0,
        "cEE":    6.0,
        "cEI":    15.0,
        "cIE":    6.0,
        "cII":    0.4,
        "P":      3.0,
        "Q":     -1.0,
    },
]

# ---------------------------------------------------------------------------
# Effective-rank diagnostics for the connectivity matrix
# ---------------------------------------------------------------------------
# These values are printed once when the module is imported and serve as a
# quick sanity check on the topology used throughout the simulations.
srank = stable_rank(M)
nrank = nuclear_rank(M)
erank, cumulative_energy = energy_ratio(M)

print(f"Stable Rank  (M): {srank:.2f}")
print(f"Nuclear Rank (M): {nrank:.2f}")
print(f"Effective Rank (95% energy, M): {erank}")
