"""
metrics.py
==========
Information-theoretic and statistical metrics for quantifying higher-order
statistical dependencies in multivariate time series.  Used throughout:

    Hernández et al. (2025) "Higher-order statistical structure emerges from
    nonlinear dynamics without explicit higher-order coupling."

The main measures implemented here are:
    - Gaussian entropy H(Σ)
    - Total Correlation (TC) — also known as multivariate mutual information
    - Dual Total Correlation (DTC) — also known as binding information
    - Third-order cumulant (co-skewness) κ_{123}
    - Auxiliary measures: skewness, excess kurtosis, triple power correlation
    - Row-wise outlier cleaning for smooth phase diagrams

Mathematical background
-----------------------
For a multivariate Gaussian X ~ N(0, Σ):

    H(Σ)  = (n/2) log(2πe) + (1/2) log det(Σ)

    TC(X) = Σ_i H(X_i) − H(X_1, ..., X_n)
           (measures redundancy; positive when variables are dependent)

    DTC(X) = Σ_i H(X_{-i}) − (n-1) H(X_1, ..., X_n)
           (measures synergy; related to the binding information)

    TC − DTC > 0 indicates net redundancy (O-information > 0)
    TC − DTC < 0 indicates net synergy   (O-information < 0)

    κ_{123} = E[X_1 X_2 X_3] (after z-scoring) — the normalized third-order
    cumulant, also called co-skewness or coskewness.

References
----------
Varley et al. (2023), Stramaglia et al. (2021), Hindriks et al. (2024).
"""

import numpy as np
from scipy.ndimage import generic_filter
from scipy.interpolate import griddata
from scipy.special import digamma
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Gaussian entropy and higher-order information measures
# ---------------------------------------------------------------------------

def entropy_gauss(Sigma: np.ndarray) -> float:
    """Differential entropy of a zero-mean multivariate Gaussian.

    H(X) = (n/2) log(2πe) + (1/2) log det(Σ)

    Parameters
    ----------
    Sigma : np.ndarray, shape (n, n)
        Covariance matrix.

    Returns
    -------
    float
        Gaussian differential entropy in nats.
    """
    n = Sigma.shape[0]
    det = np.linalg.det(Sigma)
    return (n / 2) * np.log(2 * np.pi * np.e) + 0.5 * np.log(det)


def TC_gauss(Sigma: np.ndarray) -> float:
    """Total Correlation (TC) for a multivariate Gaussian.

    TC(X) = Σ_i H(X_i) − H(X_1, ..., X_n)

    TC ≥ 0, with equality iff all variables are mutually independent.
    TC quantifies the net shared (redundant) information across all variables.

    Parameters
    ----------
    Sigma : np.ndarray, shape (n, n)
        Full covariance matrix of the n-dimensional distribution.

    Returns
    -------
    float
        Total correlation in nats.
    """
    n = Sigma.shape[0]

    # Sum of marginal entropies (each marginal is a 1-D Gaussian)
    H_marginals = sum(
        entropy_gauss(np.array([[Sigma[i, i]]]))
        for i in range(n)
    )

    # Joint entropy
    H_Sigma = entropy_gauss(Sigma)
    return H_marginals - H_Sigma


def DTC_gauss(Sigma: np.ndarray) -> float:
    """Dual Total Correlation (DTC) for a multivariate Gaussian.

    DTC(X) = Σ_i H(X_{-i}) − (n−1) H(X_1, ..., X_n)

    where X_{-i} denotes all variables except X_i.  DTC ≥ 0 and quantifies
    the synergistic information shared across the full joint distribution.

    Parameters
    ----------
    Sigma : np.ndarray, shape (n, n)
        Full covariance matrix.

    Returns
    -------
    float
        Dual total correlation in nats.
    """
    n = Sigma.shape[0]
    indices = np.arange(n)

    # Sum of leave-one-out marginal entropies H(X_{-i})
    H_sum = sum(
        entropy_gauss(Sigma[np.ix_(np.delete(indices, i), np.delete(indices, i))])
        for i in range(n)
    )

    H_Sigma = entropy_gauss(Sigma)
    return H_sum - (n - 1) * H_Sigma


# ---------------------------------------------------------------------------
# Third-order cumulant (normalized co-skewness)
# ---------------------------------------------------------------------------

def cumulants(X: np.ndarray) -> float:
    """Normalized third-order cumulant (co-skewness) for three signals.

    Computes κ_{123} = E[Z_1 Z_2 Z_3] where Z_i = (X_i − μ_i) / σ_i are
    z-scored versions of the input signals.  For a jointly Gaussian
    distribution κ_{123} = 0, so deviations from zero indicate genuine
    third-order statistical structure.

    Parameters
    ----------
    X : np.ndarray, shape (T, 3)
        Data matrix; each column is an independent signal of length T.

    Returns
    -------
    float
        Normalized third-order cumulant (dimensionless).

    Raises
    ------
    ValueError
        If X does not have exactly 3 columns.
    """
    if X.shape[1] != 3:
        raise ValueError("Input must have exactly 3 columns (one per signal).")

    # Z-score each signal independently (avoid division by zero with small ε)
    Xz = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)

    x1, x2, x3 = Xz[:, 0], Xz[:, 1], Xz[:, 2]

    # Third-order mixed moment E[Z_1 Z_2 Z_3]
    return float(np.mean(x1 * x2 * x3))


# ---------------------------------------------------------------------------
# Marginal skewness and excess kurtosis
# ---------------------------------------------------------------------------

def dev_gauss(X: np.ndarray):
    """Marginal skewness and excess kurtosis averaged across the three signals.

    Used as a diagnostic to assess how far the marginal distributions deviate
    from Gaussianity.

    Parameters
    ----------
    X : np.ndarray, shape (T, 3)
        Data matrix.

    Returns
    -------
    skew : float
        Mean normalized skewness E[Z^3] averaged over the three signals.
    kurt : float
        Mean excess kurtosis (E[Z^4] − 3) averaged over the three signals.

    Raises
    ------
    ValueError
        If X does not have exactly 3 columns.
    """
    if X.shape[1] != 3:
        raise ValueError("Input must have exactly 3 columns.")

    # Z-score each column
    Xz = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)

    # Third standardized moment (skewness) for each signal
    skew_per_node = [np.mean(Xz[:, i] ** 3) for i in range(3)]
    skew = float(np.mean(skew_per_node))

    # Fourth standardized moment minus 3 (excess kurtosis) for each signal
    kurt_per_node = [np.mean(Xz[:, i] ** 4) - 3 for i in range(3)]
    kurt = float(np.mean(kurt_per_node))

    return skew, kurt


# ---------------------------------------------------------------------------
# Triple power correlation
# ---------------------------------------------------------------------------

def powercorr(X: np.ndarray) -> float:
    """Triple power correlation for three signals.

    Computes a third-order analogue of the power spectral coherence using
    the mixed third moment of the z-scored signals.  Unlike the simple
    cumulant, this measure includes cross-products of pairs and exploits
    complex-valued correlation structure.

    Parameters
    ----------
    X : np.ndarray, shape (T, 3)
        Data matrix.

    Returns
    -------
    float
        Real part of the triple power correlation.

    Raises
    ------
    ValueError
        If X does not have exactly 3 columns.
    """
    if X.shape[1] != 3:
        raise ValueError("Input must have exactly 3 columns.")

    # Z-score each signal
    x1, x2, x3 = (
        (X[:, i] - X[:, i].mean()) / (X[:, i].std() + 1e-9) for i in range(3)
    )
    x1 = (X[:, 0] - X[:, 0].mean()) / (X[:, 0].std() + 1e-9)
    x2 = (X[:, 1] - X[:, 1].mean()) / (X[:, 1].std() + 1e-9)
    x3 = (X[:, 2] - X[:, 2].mean()) / (X[:, 2].std() + 1e-9)

    # Third-order product moment ⟨x1 x2 x3, x1* x2* x3*⟩
    p1 = np.mean((x1 * x2 * x3) * np.conj(x1 * x2 * x3))

    # Second-order terms (all 9 combinations of products)
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

    # First-order (product of three individual power spectra / cross-spectra)
    p3 = np.zeros(6)
    p3[0] = np.mean(x1 * np.conj(x1)) * np.mean(x2 * np.conj(x2)) * np.mean(x3 * np.conj(x3))
    p3[1] = np.mean(x1 * np.conj(x2)) * np.mean(x2 * np.conj(x1)) * np.mean(x3 * np.conj(x3))
    p3[2] = np.mean(x1 * np.conj(x1)) * np.mean(x2 * np.conj(x3)) * np.mean(x3 * np.conj(x2))
    p3[3] = np.mean(x1 * np.conj(x2)) * np.mean(x2 * np.conj(x3)) * np.mean(x3 * np.conj(x1))
    p3[4] = np.mean(x1 * np.conj(x3)) * np.mean(x2 * np.conj(x2)) * np.mean(x3 * np.conj(x1))
    p3[5] = np.mean(x1 * np.conj(x3)) * np.mean(x2 * np.conj(x1)) * np.mean(x3 * np.conj(x2))

    # Inclusion–exclusion combination
    return float(np.real(p1 - np.sum(p2) + 2 * np.sum(p3)))


# ---------------------------------------------------------------------------
# Post-processing utility: row-wise outlier cleaning
# ---------------------------------------------------------------------------

def clean_rowwise_signed(matrix: np.ndarray, n_std: float = 2) -> np.ndarray:
    """Clean a 2-D matrix row by row by interpolating NaNs and removing outliers.

    Processing steps for each row:
        1. Interpolate NaN values using their valid neighbours.
        2. Detect outliers lying more than *n_std* standard deviations from
           the row mean (in either direction).
        3. Replace outliers using linear extrapolation from the preceding
           values (avoids artefacts at phase-diagram boundaries).

    This function is applied to TC−DTC and cumulant matrices before plotting
    the contour diagrams to suppress numerical artefacts near bifurcation
    boundaries.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_rows, n_cols)
        Input matrix (e.g., a (P × K) metric map).
    n_std : float, optional
        Number of standard deviations used as the outlier threshold
        (default 2).

    Returns
    -------
    cleaned : np.ndarray, shape (n_rows, n_cols)
        Cleaned matrix with NaNs interpolated and outliers replaced.
    """
    M = matrix.copy().astype(float)
    cleaned = M.copy()

    for i in range(M.shape[0]):
        row = M[i, :].copy()

        # --- Step 1: Interpolate NaN values ---
        nan_mask = np.isnan(row)
        if np.any(nan_mask):
            x = np.arange(len(row))
            valid = ~nan_mask
            if np.sum(valid) > 1:
                row[nan_mask] = np.interp(x[nan_mask], x[valid], row[valid])
            else:
                row[nan_mask] = 0.0  # entire row is NaN; fill with zero

        # --- Step 2: Detect signed outliers ---
        mean_val = np.mean(row)
        std_val = np.std(row)
        outlier_mask = (row > mean_val + n_std * std_val) | \
                       (row < mean_val - n_std * std_val)

        # --- Step 3: Replace outliers by linear trend extrapolation ---
        for idx in np.where(outlier_mask)[0]:
            if idx == 0:
                # First element: replace with row mean
                row[idx] = mean_val
            else:
                # Extrapolate: previous value + local slope
                delta = (row[idx - 1] - row[idx - 2]) if idx > 1 else 0.0
                row[idx] = row[idx - 1] + delta

        cleaned[i, :] = row

    return cleaned


# ---------------------------------------------------------------------------
# Non-Gaussian information-theoretic measures: KSG estimator
# ---------------------------------------------------------------------------
# The KSG (Kraskov–Stögbauer–Grassberger) estimator is fully non-parametric:
# it makes NO distributional assumptions whatsoever.  It estimates differential
# entropy via k-nearest-neighbour distances and from there derives TC, DTC,
# and O-information.
#
# Reference: Kraskov, Stögbauer & Grassberger (2004), Phys. Rev. E 69, 066138.
# ---------------------------------------------------------------------------

def _ksg_entropy(X: np.ndarray, k: int = 5) -> float:
    """Kozachenko–Leonenko (KSG) entropy estimator using k-nearest
    neighbours with the Chebyshev (L∞) norm.

    H(X) ≈ ψ(T) − ψ(k) + d·⟨log(2ε_i)⟩

    where ε_i is the L∞ distance to the k-th neighbour of point i,
    d is the dimensionality, and ψ is the digamma function.

    Parameters
    ----------
    X : np.ndarray, shape (T, d)  or (T,) for 1-D
    k : int
        Number of neighbours.

    Returns
    -------
    float   Entropy estimate in nats.

    References
    ----------
    Kraskov, Stögbauer & Grassberger (2004), Phys. Rev. E 69, 066138.
    """
    if X.ndim == 1:
        X = X[:, np.newaxis]
    T, d = X.shape

    # Build KD-tree with Chebyshev (max-norm) metric
    tree = KDTree(X)
    # query k+1 neighbours (the first is the point itself at distance 0)
    dists, _ = tree.query(X, k=k + 1, p=np.inf)
    # ε_i = distance to k-th neighbour (index k, since index 0 is self)
    eps = dists[:, k]

    # Avoid log(0) for duplicate points
    eps = np.maximum(eps, 1e-15)

    return float(digamma(T) - digamma(k) + d * np.mean(np.log(2.0 * eps)))


def TC_ksg(X: np.ndarray, k: int = 5) -> float:
    """Total Correlation via KSG entropy estimation.

    TC(X) = Σ_i H(X_i) − H(X_1, …, X_n)

    Parameters
    ----------
    X : np.ndarray, shape (T, n)
    k : int
        Number of neighbours for entropy estimation.

    Returns
    -------
    float   TC in nats.
    """
    n = X.shape[1]
    H_marginals = sum(_ksg_entropy(X[:, i], k=k) for i in range(n))
    H_joint = _ksg_entropy(X, k=k)
    return H_marginals - H_joint


def DTC_ksg(X: np.ndarray, k: int = 5) -> float:
    """Dual Total Correlation via KSG entropy estimation.

    DTC(X) = Σ_i H(X_{−i}) − (n−1) H(X_1, …, X_n)

    Parameters
    ----------
    X : np.ndarray, shape (T, n)
    k : int

    Returns
    -------
    float   DTC in nats.
    """
    n = X.shape[1]
    indices = np.arange(n)
    H_leave_one_out = sum(
        _ksg_entropy(X[:, np.delete(indices, i)], k=k)
        for i in range(n)
    )
    H_joint = _ksg_entropy(X, k=k)
    return H_leave_one_out - (n - 1) * H_joint


def Oinfo_ksg(X: np.ndarray, k: int = 5) -> float:
    """O-information (TC − DTC) via KSG."""
    return TC_ksg(X, k=k) - DTC_ksg(X, k=k)
