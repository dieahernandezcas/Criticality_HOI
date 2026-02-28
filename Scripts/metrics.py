# functions/metrics.py

import numpy as np
from scipy.ndimage import generic_filter
from scipy.interpolate import griddata

def entropy_gauss(Sigma):
    """Gaussian entropy of an n-dimensional normal distribution."""
    n = Sigma.shape[0]
    det = np.linalg.det(Sigma)
    #if det <= 0:
    #    det = 1e-6
    return (n / 2) * np.log(2 * np.pi * np.e) + 0.5 * np.log(det)

def TC_gauss(Sigma):
    """Total Correlation (TC) for a Gaussian distribution."""
    n = Sigma.shape[0]
    H_marginals = 0.0
    for i in range(n):
        H_marginals += entropy_gauss(np.array([[Sigma[i, i]]]))
    H_Sigma = entropy_gauss(Sigma)
    return H_marginals - H_Sigma  #if not np.isnan(H_Sigma) else 0.0

def DTC_gauss(Sigma):
    """Dual Total Correlation (DTC) for a Gaussian distribution."""
    n = Sigma.shape[0]
    H_sum = 0.0
    indices = np.arange(n)
    for i in range(n):
        J = np.delete(indices, i)
        H_sum += entropy_gauss(Sigma[np.ix_(J, J)])
    H_Sigma = entropy_gauss(Sigma)
    return H_sum - (n - 1) * H_Sigma #if not np.isnan(H_Sigma) else 0.0

def cumulants(X):
    """Third-order cumulant (coskewness) for three signals.
    if X.shape[1] != 3:
        raise ValueError("Cumulants require exactly 3 signals.")
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    return np.mean(Xz[:, 0] * Xz[:, 1] * Xz[:, 2])
    """

    """
    Computes normalized third-order cumulant (coskewness)
    for a 3-column data matrix.

    Parameters
    ----------
    X : array (T, 3)
        Data matrix (each column is a signal)

    Returns
    -------
    C : float
        Third-order cumulant (coskewness)
    """

    if X.shape[1] != 3:
        raise ValueError("Input must have exactly 3 columns")

    # Z-score normalization (column-wise)
    # Xz = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    Xz = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)

    x1 = Xz[:, 0]
    x2 = Xz[:, 1]
    x3 = Xz[:, 2]

    # Third-order cumulant (coskewness)
    C = np.mean(x1 * x2 * x3)

    return C

def dev_gauss(X):
    
    if X.shape[1] != 3:
        raise ValueError("Input must have exactly 3 columns")

    # Z-score normalization (column-wise)
    Xz = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)

    x1 = np.mean(Xz[:, 0]*Xz[:, 0]*Xz[:, 0])
    x2 = np.mean(Xz[:, 1]*Xz[:, 1]*Xz[:, 1])
    x3 = np.mean(Xz[:, 2]*Xz[:, 2]*Xz[:, 2])

    skew = (x1 + x2 + x3)/3

    y1 = np.mean(Xz[:,0]**4)-3
    y2 = np.mean(Xz[:,1]**4)-3
    y3 = np.mean(Xz[:,2]**4)-3

    kurt = (y1+y2+y3)/3

    return skew, kurt



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


def clean_rowwise_signed(matrix, n_std=2):
    """
    Limpia una matriz fila por fila:
    1. Interpola NaN con vecinos en la misma fila.
    2. Detecta outliers muy positivos o negativos (> mean + n_std*std o < mean - n_std*std)
       y reemplaza siguiendo la tendencia lineal de la fila.
    
    Parameters
    ----------
    matrix : 2D np.array
    n_std : float
        Número de desviaciones estándar para detectar outliers
    
    Returns
    -------
    cleaned : 2D np.array
    """
    M = matrix.copy().astype(float)
    cleaned = M.copy()
    
    for i in range(M.shape[0]):
        row = M[i, :]
        
        # --- Interpolar NaN ---
        nan_mask = np.isnan(row)
        if np.any(nan_mask):
            x = np.arange(len(row))
            valid = ~nan_mask
            if np.sum(valid) > 1:
                row[nan_mask] = np.interp(x[nan_mask], x[valid], row[valid])
            else:
                row[nan_mask] = 0.0  # si toda la fila es NaN
        
        # --- Detectar outliers (positivos y negativos) ---
        mean = np.mean(row)
        std = np.std(row)
        outlier_mask = (row > mean + n_std*std) | (row < mean - n_std*std)
        
        # --- Reemplazar outliers siguiendo tendencia ---
        for idx in np.where(outlier_mask)[0]:
            if idx == 0:
                # primer valor, reemplazamos con la media
                row[idx] = mean
            else:
                # tendencia lineal simple: valor anterior + delta
                delta = row[idx-1] - row[idx-2] if idx > 1 else 0
                row[idx] = row[idx-1] + delta
        
        cleaned[i, :] = row
    
    return cleaned