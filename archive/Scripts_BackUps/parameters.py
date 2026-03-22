# functions/parameters.py
import numpy as np


def stable_rank(matrix):
    """Calculate the stable rank of a matrix."""
    U, Sigma, VT = np.linalg.svd(matrix)
    return (Sigma ** 2).sum() / (Sigma[0] ** 2)


def nuclear_rank(matrix):
    """Calculate the nuclear rank of a matrix."""
    U, Sigma, VT = np.linalg.svd(matrix)
    return Sigma.sum() / Sigma[0]


def energy_ratio(matrix, threshold=0.95):
    """Calculate the energy ratio and the effective rank."""
    U, Sigma, VT = np.linalg.svd(matrix)
    total_energy = (Sigma ** 2).sum()
    cumulative_energy = np.cumsum(Sigma ** 2) / total_energy
    effective_rank = np.argmax(cumulative_energy >= threshold) + 1
    return effective_rank, cumulative_energy


# Number of nodes
N_nodes = 3

# Time constants
tauE = 0.12
tauI = 0.25

# Sigmoid parameters
aE = 0.8
aI = 0.8
thetaE = 2.0
thetaI = 8.0

# Local connectivity coefficients
cEE = 8.0
cEI = 16.0
cIE = 8.0
cII = 0.4

# Baseline external inputs
P_baseline = 1.5
Q_baseline = -2.0

# Noise strength
sigma_E = 0.01

# Numerical integration parameters
dt = 1e-3
T = 10.0

# Parameter ranges for oscillation detection
P_values = np.linspace(1.0, 10.0, 20)
K_values = np.linspace(0.0, 1.0, 20)
K3_values = np.linspace(0.0, 1.0, 20)

# Pairwise structural connectivity
M = np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0]
])

# Higher-order interaction tensor
T_ho = np.zeros((N_nodes, N_nodes, N_nodes))
for i in range(N_nodes):
    for j in range(N_nodes):
        for k in range(N_nodes):
            if i != j and i != k and j != k:
                if M[i, j] != 0 and M[i, k] != 0:
                    T_ho[i, j, k] = 1.0

# Cálculo de rangos efectivos para M
srank = stable_rank(M)
nrank = nuclear_rank(M)
erank, cumulative_energy = energy_ratio(M)

print(f"Stable Rank: {srank:.2f}")
print(f"Nuclear Rank: {nrank:.2f}")
print(f"Effective Rank (Energy Ratio 95%): {erank}")
