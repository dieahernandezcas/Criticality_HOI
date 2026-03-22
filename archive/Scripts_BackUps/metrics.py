# functions/metrics.py

import numpy as np

def entropy_gauss(Sigma):
    """Gaussian entropy of an n-dimensional normal distribution."""
    n = Sigma.shape[0]
    Sigma += 1e-6 * np.eye(n)
    det = np.linalg.det(Sigma)
    if det <= 0:
        det = 1e-6
    return (n / 2) * np.log(2 * np.pi * np.e) + 0.5 * np.log(det)

def TC_gauss(Sigma):
    """Total Correlation (TC) for a Gaussian distribution."""
    n = Sigma.shape[0]
    H_marginals = 0.0
    for i in range(n):
        H_marginals += entropy_gauss(np.array([[Sigma[i, i]]]))
    H_Sigma = entropy_gauss(Sigma)
    return H_marginals - H_Sigma if not np.isnan(H_Sigma) else 0.0

def DTC_gauss(Sigma):
    """Dual Total Correlation (DTC) for a Gaussian distribution."""
    n = Sigma.shape[0]
    H_sum = 0.0
    indices = np.arange(n)
    for i in range(n):
        J = np.delete(indices, i)
        H_sum += entropy_gauss(Sigma[np.ix_(J, J)])
    H_Sigma = entropy_gauss(Sigma)
    return H_sum - (n - 1) * H_Sigma if not np.isnan(H_Sigma) else 0.0

def cumulants(X):
    """Third-order cumulant (coskewness) for three signals."""
    if X.shape[1] != 3:
        raise ValueError("Cumulants require exactly 3 signals.")
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    return np.mean(Xz[:, 0] * Xz[:, 1] * Xz[:, 2])

def powercorr(X):
    """Triple power correlation for three signals."""
    if X.shape[1] != 3:
        raise ValueError("Power correlation requires exactly 3 signals.")

    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    x1 = (x1 - x1.mean()) / (x1.std() + 1e-9)
    x2 = (x2 - x2.mean()) / (x2.std() + 1e-9)
    x3 = (x3 - x3.mean()) / (x3.std() + 1e-9)

    p1 = np.mean((x1 * x2 * x3) * np.conj(x1 * x2 * x3))

    p2 = np.zeros((3, 3))
    p2[0, 0] = np.mean((x1 * x2) * np.conj(x1 * x2)) * np.mean(x3 * np.conj(x3))
    p2[0, 1] = np.mean((x1 * x2) * np.conj(x2 * x3)) * np.mean(x3 * np.conj(x1))
    p2[0, 2] = np.mean((x1 * x2) * np.conj(x1 * x3)) * np.mean(x3 * np.conj(x2))
    p2[1, 0] = np.mean(x1 * np.conj(x1)) * np.mean((x2 * x3) * np.conj(x2 * x3))
    p2[1, 1] = np.mean(x1 * np.conj(x2)) * np.mean((x2 * x3) * np.conj(x1 * x3))
    p2[1, 2] = np.mean(x1 * np.conj(x3)) * np.mean((x2 * x3) * np.conj(x1 * x2))
    p2[2, 0] = np.mean(x2 * np.conj(x2)) * np.mean((x1 * x3) * np.conj(x1 * x3))
    p2[2, 1] = np.mean(x2 * np.conj(x1)) * np.mean((x1 * x3) * np.conj(x2 * x3))
    p2[2, 2] = np.mean(x2 * np.conj(x3)) * np.mean((x1 * x3) * np.conj(x1 * x2))

    p3 = np.zeros(6)
    p3[0] = np.mean(x1 * np.conj(x1)) * np.mean(x2 * np.conj(x2)) * np.mean(x3 * np.conj(x3))
    p3[1] = np.mean(x1 * np.conj(x2)) * np.mean(x2 * np.conj(x1)) * np.mean(x3 * np.conj(x3))
    p3[2] = np.mean(x1 * np.conj(x1)) * np.mean(x2 * np.conj(x3)) * np.mean(x3 * np.conj(x2))
    p3[3] = np.mean(x1 * np.conj(x2)) * np.mean(x2 * np.conj(x3)) * np.mean(x3 * np.conj(x1))
    p3[4] = np.mean(x1 * np.conj(x3)) * np.mean(x2 * np.conj(x2)) * np.mean(x3 * np.conj(x1))
    p3[5] = np.mean(x1 * np.conj(x3)) * np.mean(x2 * np.conj(x1)) * np.mean(x3 * np.conj(x2))

    return np.real(p1 - np.sum(p2) + 2 * np.sum(p3))
