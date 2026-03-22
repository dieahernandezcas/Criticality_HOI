"""
parameters_random.py
====================
Homogeneous (single-set) parameter configuration for the three-node
Wilson–Cowan network described in:

    Hernández et al. (2025) "Higher-order statistical structure emerges from
    nonlinear dynamics without explicit higher-order coupling."

Unlike ``parameters.py``, which assigns distinct biophysical parameters to
each node, this module uses a single globally shared parameter set.  It is
used in exploratory sweeps and robustness checks where node heterogeneity
is not the focus.

This module also defines the parameter grids for the (P, K) and (a_E, K)
sweeps that produce the phase diagrams shown in Figs. 1–3 of the paper.

Usage
-----
    from Scripts.parameters_random import (
        N_nodes, M, T_ho, dt, T, sigma_E,
        P_values, K_values, K3_values,
        tauE, tauI, aE, aI, cEE, cEI, cIE, cII
    )
"""

import numpy as np


# ---------------------------------------------------------------------------
# Effective-rank utilities (duplicated here so this module is self-contained)
# ---------------------------------------------------------------------------

def stable_rank(matrix: np.ndarray) -> float:
    """Return the stable rank ||A||_F^2 / ||A||_2^2 of *matrix*."""
    _, Sigma, _ = np.linalg.svd(matrix)
    return float((Sigma ** 2).sum() / (Sigma[0] ** 2))


def nuclear_rank(matrix: np.ndarray) -> float:
    """Return the nuclear rank ||A||_* / ||A||_2 of *matrix*."""
    _, Sigma, _ = np.linalg.svd(matrix)
    return float(Sigma.sum() / Sigma[0])


def energy_ratio(matrix: np.ndarray, threshold: float = 0.95):
    """Return the effective rank based on cumulative singular-value energy.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.
    threshold : float, optional
        Fractional energy threshold (default 0.95).

    Returns
    -------
    effective_rank : int
    cumulative_energy : np.ndarray
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

# ---------------------------------------------------------------------------
# Global (homogeneous) biophysical parameters
# ---------------------------------------------------------------------------
# These values are shared across all three nodes.

# Excitatory time constant (s)  — τ_E in the manuscript
tauE = 0.12

# Inhibitory time constant (s)  — τ_I in the manuscript
tauI = 0.25

# Sigmoid slope for excitatory population  — a_E
# Range explored in the a_E–K sweep: [0.5, 3.0]
aE = 0.8

# Sigmoid slope for inhibitory population  — a_I
aI = 0.8

# Sigmoid threshold for excitatory population  — θ_E
thetaE = 2.0

# Sigmoid threshold for inhibitory population  — θ_I
thetaI = 8.0

# Local excitatory self-connection weight  — c_EE
cEE = 8.0

# Local E→I (excitatory drives inhibitory) connection weight  — c_EI
cEI = 16.0

# Local I→E (inhibitory suppresses excitatory) connection weight  — c_IE
cIE = 8.0

# Local inhibitory self-connection weight  — c_II
cII = 0.4

# Baseline external input to the excitatory population  — P
P_baseline = 1.5

# External input to the inhibitory population  — Q
Q_baseline = -2.0

# ---------------------------------------------------------------------------
# Noise and integration defaults
# ---------------------------------------------------------------------------

# Additive Gaussian noise amplitude for the excitatory population (σ_E)
sigma_E = 0.01

# Integration time step (seconds)
dt = 1e-3

# Total simulation time (seconds)
T = 10.0

# ---------------------------------------------------------------------------
# Parameter sweep grids
# ---------------------------------------------------------------------------
# These ranges define the (P × K) and (a_E × K) phase diagrams.

# External drive values (rows of the phase diagram)
P_values = np.linspace(1.0, 10.0, 20)

# Pairwise coupling strength values (columns of the phase diagram)
K_values = np.linspace(0.0, 1.0, 20)

# Third-order coupling strength values (used for HOI sweeps)
K3_values = np.linspace(0.0, 1.0, 20)

# ---------------------------------------------------------------------------
# Network topology
# ---------------------------------------------------------------------------

# All-to-all pairwise structural connectivity matrix (no self-connections)
M = np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0]
])

# ---------------------------------------------------------------------------
# Higher-order interaction tensor
# ---------------------------------------------------------------------------
# T_ho[i, j, k] = 1 if the three nodes are all distinct and both edges
# (i,j) and (i,k) are present in M.  See Eq. (4) in the manuscript.
T_ho = np.zeros((N_nodes, N_nodes, N_nodes))
for i in range(N_nodes):
    for j in range(N_nodes):
        for k in range(N_nodes):
            if i != j and i != k and j != k:
                if M[i, j] != 0 and M[i, k] != 0:
                    T_ho[i, j, k] = 1.0

# ---------------------------------------------------------------------------
# Effective-rank diagnostics for the connectivity matrix
# ---------------------------------------------------------------------------
srank = stable_rank(M)
nrank = nuclear_rank(M)
erank, cumulative_energy = energy_ratio(M)

print(f"Stable Rank  (M): {srank:.2f}")
print(f"Nuclear Rank (M): {nrank:.2f}")
print(f"Effective Rank (95% energy, M): {erank}")
