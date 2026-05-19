"""Recurrence Quantification Analysis (RQA) for market regime detection.

Computes recurrence plots from price return time-delay embeddings
and extracts nonlinear dynamical measures as regime-change factors.

References
----------
- Zbilut, J.P. & Webber, C.L. (1992). Embeddings and delays as derived
  from quantification of recurrence plots. Physics Letters A, 171(3-4).
- Marwan, N. et al. (2007). Recurrence plots for the analysis of complex
  systems. Physics Reports, 438(5-6).
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def phase_space_embedding(s: np.ndarray, dim: int = 3, tau: int = 1) -> np.ndarray:
    """Time-delay embedding of a scalar time series.

    Parameters
    ----------
    s : ndarray, shape (T,)
        Scalar time series (e.g. daily returns).
    dim : int
        Embedding dimension (default 3).
    tau : int
        Time delay in samples (default 1).

    Returns
    -------
    X : ndarray, shape (T - (dim-1)*tau, dim)
        Embedded phase-space vectors.

    Notes
    -----
    Uses Takens' embedding theorem. dim=3, tau=1 is a sensible
    default for financial daily returns with ~252 data points per year.
    """
    T = len(s)
    if T < dim * tau:
        return np.empty((0, dim))
    N = T - (dim - 1) * tau
    X = np.column_stack([s[i * tau: i * tau + N] for i in range(dim)])
    return X


def recurrence_matrix(X: np.ndarray, eps: float | None = None,
                      pct: float = 10.0) -> np.ndarray:
    """Binary recurrence matrix from phase-space vectors.

    R[i,j] = 1 if ||x_i - x_j|| < eps, else 0.

    Parameters
    ----------
    X : ndarray, shape (N, dim)
        Phase-space embedded vectors.
    eps : float or None
        Fixed radius threshold. If None, uses percentile of distances.
    pct : float
        Percentile of pairwise distances to use as threshold
        (default 10.0, meaning 10th percentile). Only used when
        eps is None.

    Returns
    -------
    R : ndarray, shape (N, N)
        Binary recurrence matrix.
    """
    N = len(X)
    if N < 2:
        return np.empty((0, 0))
    dists = squareform(pdist(X, metric='euclidean'))
    if eps is None:
        upper = dists[np.triu_indices(N, k=1)]
        if len(upper) == 0:
            return np.zeros((N, N), dtype=bool)
        eps = np.percentile(upper, pct)
    return dists < eps


def recurrence_quantification(R: np.ndarray) -> dict[str, float]:
    """Extract RQA measures from a binary recurrence matrix.

    Parameters
    ----------
    R : ndarray, shape (N, N)
        Binary recurrence matrix.

    Returns
    -------
    metrics : dict
        - RR: Recurrence rate (fraction of recurrent points)
        - DET: Determinism (fraction of recurrent points in diagonal lines)
        - Lmax: Longest diagonal line length
        - ENTR: Shannon entropy of diagonal line length distribution
        - LAM: Laminarity (fraction in vertical lines)
        - TT: Trapping time (mean vertical line length)
    """
    N = R.shape[0]
    if N < 2:
        return {'rr': 0.0, 'det': 0.0, 'lmax': 1.0, 'entr': 0.0,
                'lam': 0.0, 'tt': 1.0}

    # Exclude main diagonal (self-recurrence)
    total = N * N - N
    if total <= 0:
        return {'rr': 0.0, 'det': 0.0, 'lmax': 1.0, 'entr': 0.0,
                'lam': 0.0, 'tt': 1.0}

    rr = float(np.sum(R) - N) / total

    min_len = 2
    # Diagonal line segments
    diag_lines = []
    for diag in range(-N + 1, N):
        if diag == 0:
            continue
        diag_elements = R.diagonal(offset=diag)
        seg_len = 0
        for val in diag_elements:
            if val:
                seg_len += 1
            else:
                if seg_len >= min_len:
                    diag_lines.append(seg_len)
                seg_len = 0
        if seg_len >= min_len:
            diag_lines.append(seg_len)

    total_diag = sum(diag_lines) if diag_lines else 0
    det = total_diag / (np.sum(R) - N) if (np.sum(R) - N) > 0 else 0.0
    lmax = max(diag_lines) if diag_lines else 1.0

    # Entropy of line length distribution
    if diag_lines:
        hist, _ = np.histogram(diag_lines, bins=range(min_len, max(diag_lines) + 2))
        probs = hist / np.sum(hist)
        entr = -np.sum(probs * np.log(probs + 1e-12))
    else:
        entr = 0.0

    # Vertical line segments (laminarity)
    vert_lines = []
    for col in range(N):
        col_elements = R[:, col]
        seg_len = 0
        for val in col_elements:
            if val:
                seg_len += 1
            else:
                if seg_len >= min_len:
                    vert_lines.append(seg_len)
                seg_len = 0
        if seg_len >= min_len:
            vert_lines.append(seg_len)

    total_vert = sum(vert_lines) if vert_lines else 0
    lam = total_vert / (np.sum(R) - N) if (np.sum(R) - N) > 0 else 0.0

    # Trapping time: mean vertical line length
    tt = np.mean(vert_lines) if vert_lines else 1.0

    return {
        'rr': float(rr),
        'det': float(det),
        'lmax': float(lmax),
        'entr': float(entr),
        'lam': float(lam),
        'tt': float(tt),
    }


def compute_rqa_factors(returns: np.ndarray, window: int = 63,
                        dim: int = 3, tau: int = 1,
                        pct: float = 10.0) -> dict[str, np.ndarray]:
    """Compute rolling RQA factors from return series.

    Parameters
    ----------
    returns : ndarray, shape (T, N)
        Daily returns, T timesteps x N assets.
    window : int
        Rolling window length (default 63 ~ 3 months).
    dim : int
        Embedding dimension (default 3).
    tau : int
        Embedding delay (default 1).
    pct : float
        Distance percentile threshold (default 10.0).

    Returns
    -------
    factors : dict of ndarray, shape (T-window+1,)
        Time series of each RQA measure, aligned at window end.
    """
    T, N = returns.shape
    n_out = T - window + 1
    if n_out < 1:
        raise ValueError(f"Time series length {T} < window {window}")

    # Average returns across assets → market return embedding
    out = {
        'rr': np.full(n_out, np.nan),
        'det': np.full(n_out, np.nan),
        'lmax': np.full(n_out, np.nan),
        'entr': np.full(n_out, np.nan),
        'lam': np.full(n_out, np.nan),
        'tt': np.full(n_out, np.nan),
    }

    for i in range(n_out):
        chunk = returns[i: i + window]
        mkt = np.mean(chunk, axis=1)  # equal-weight market return

        X = phase_space_embedding(mkt, dim=dim, tau=tau)
        if len(X) < 3:
            continue

        R = recurrence_matrix(X, pct=pct)
        if R.size == 0:
            continue

        metrics = recurrence_quantification(R)
        for k, v in metrics.items():
            out[k][i] = v

    return out
