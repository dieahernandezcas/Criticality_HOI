import numpy as np
from .parameters import *

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

def calculate_reduction_matrix(M, effective_rank=2):
    """Calculate the reduction matrix using SVD."""
    U, Sigma, VT = np.linalg.svd(M)
    V = VT.T
    M_reduced = V[:effective_rank, :].T  # Matriz 2x3
    return M_reduced

def calculate_D_reduced(M_reduced, tauE):
    """Calculate the reduced decay matrix D."""
    N_original = M.shape[0]
    D = np.diag([1/tauE] * N_original)
    print(D)
    print(M_reduced)
    print(M_reduced.T)
    D_reduced = -M_reduced.T @ D @ M_reduced
    print(D_reduced)
    return D_reduced

def calculate_W_reduced(M_reduced, M, K):
    """Calculate the reduced linear coupling matrix W."""
    W_reduced = M_reduced.T @ (K * M) @ M_reduced
    return W_reduced

def calculate_T_reduced(M_reduced, T_ho):
    """Calculate the reduced higher-order coupling tensor T."""
    N_reduced = M_reduced.shape[0]
    T_reduced = np.zeros((N_reduced, N_reduced, N_reduced))
    for mu in range(N_reduced):
        for nu in range(N_reduced):
            for kappa in range(N_reduced):
                for i in range(M.shape[0]):
                    for j in range(M.shape[0]):
                        for k in range(M.shape[0]):
                            T_reduced[mu, nu, kappa] += M_reduced[mu, i] * T_ho[i, j, k] * M_reduced[nu, j] * M_reduced[kappa, k]
    return T_reduced
