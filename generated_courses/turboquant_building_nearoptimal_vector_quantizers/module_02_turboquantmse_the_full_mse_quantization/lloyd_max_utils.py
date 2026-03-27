"""
lloyd_max_utils.py — Completed Lloyd-Max Utilities (from Module 1)
==================================================================

This file contains a completed, importable version of the Lloyd-Max quantizer
you built in Module 1. It is provided here as a ready-to-use dependency for
Module 2's TurboQuant_mse pipeline.

The key function is `lloyd_max_codebook(b, d)`, which returns the 2^b optimal
centroid values for quantizing a single coordinate of a randomly-rotated unit
vector in R^d.

How it works (recap from Module 1):
    After random rotation, each coordinate follows:
        f_X(t) = C_d · (1 − t²)^{(d−3)/2}    t ∈ [−1, 1]
    (which converges to N(0, 1/d) for d ≥ 64).

    The Lloyd-Max algorithm finds centroids {c_l} minimising E[(X − Q(X))²]:
        Alternate:
            (1) Boundaries: t_l = (c_{l-1} + c_l) / 2   (Voronoi midpoints)
            (2) Centroids:  c_l = E[X | t_{l-1} < X ≤ t_l]  (conditional mean)

    Both steps use exact numerical integration over f_X.

Key results (d=128):
    b=1: centroids ≈ [−0.0627, +0.0627],   per-coord MSE ≈ 2.82×10⁻³  (total 0.361)
    b=2: centroids ≈ [−0.118, −0.0354, +0.0354, +0.118],  total MSE ≈ 0.117
    b=3: 8 centroids,   total MSE ≈ 0.030
    b=4: 16 centroids,  total MSE ≈ 0.009

Reference: Module 1, Exercise 01 — "The Continuous k-means Problem"
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Beta coordinate PDF (same as Module 0 & 1 definition)
# ---------------------------------------------------------------------------

def _beta_coord_pdf(x: float, d: int) -> float:
    """
    PDF of a single coordinate of a uniformly random unit vector in R^d.

        f_X(t) = Γ(d/2) / (√π · Γ((d−1)/2)) · (1 − t²)^{(d−3)/2}

    Returns 0.0 outside the support (−1, 1).
    """
    x = float(x)
    if abs(x) >= 1.0:
        return 0.0
    C = gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
    return C * (1.0 - x ** 2) ** ((d - 3) / 2)


# ---------------------------------------------------------------------------
# Lloyd-Max Quantizer (completed implementation)
# ---------------------------------------------------------------------------

class LloydMaxQuantizer:
    """
    Optimal scalar quantizer for the Beta coordinate distribution.

    Finds k = 2^b centroids minimising E[(X − Q(X))²] via Lloyd-Max iteration.
    Uses exact numerical integration — no sampling, essentially exact for d ≥ 3.

    Usage
    -----
    >>> q = LloydMaxQuantizer(d=128, n_bits=2)
    >>> centroids, mse = q.fit()   # centroids: shape (4,), mse: per-coordinate MSE
    """

    def __init__(self, d: int, n_bits: int, n_iter: int = 300, tol: float = 1e-12):
        self.d = d
        self.n_bits = n_bits
        self.n_centroids = 2 ** n_bits
        self.n_iter = n_iter
        self.tol = tol
        self.centroids_ = None
        self.boundaries_ = None

    def pdf(self, x: float) -> float:
        return _beta_coord_pdf(x, self.d)

    def _initialize_centroids(self) -> np.ndarray:
        """Symmetric initial centroids, evenly spaced in (−1, 1)."""
        k = self.n_centroids
        return np.array([-1.0 + (2 * (i + 1)) / (k + 1) for i in range(k)])

    def _compute_boundaries(self, centroids: np.ndarray) -> np.ndarray:
        """Voronoi boundaries: midpoints between centroids, with ±1 at the edges."""
        midpoints = (centroids[:-1] + centroids[1:]) / 2.0
        return np.concatenate([[-1.0], midpoints, [1.0]])

    def _update_centroids(self, boundaries: np.ndarray) -> np.ndarray:
        """
        Update centroid l to E[X | boundaries[l] < X ≤ boundaries[l+1]].

            c_l = ∫_{t_{l-1}}^{t_l} x · f_X(x) dx
                  ─────────────────────────────────
                  ∫_{t_{l-1}}^{t_l}   f_X(x) dx

        Falls back to bin midpoint when the denominator is numerically zero
        (only occurs for outer bins at very high bit-widths where the tails of
        f_X have negligible probability mass).
        """
        new_centroids = np.empty(self.n_centroids)
        for l in range(self.n_centroids):
            lo, hi = boundaries[l], boundaries[l + 1]
            numerator, _   = quad(lambda x: x * self.pdf(x), lo, hi, limit=100)
            denominator, _ = quad(lambda x:     self.pdf(x), lo, hi, limit=100)
            if abs(denominator) < 1e-15:
                new_centroids[l] = (lo + hi) / 2.0  # fallback: bin midpoint
            else:
                new_centroids[l] = numerator / denominator
        return new_centroids

    def _mse_cost(self, centroids: np.ndarray, boundaries: np.ndarray) -> float:
        """Per-coordinate MSE = E[(X − Q(X))²]."""
        total = 0.0
        for l in range(self.n_centroids):
            lo, hi = boundaries[l], boundaries[l + 1]
            c_l = centroids[l]
            cost, _ = quad(
                lambda x, c=c_l: (x - c) ** 2 * self.pdf(x),
                lo, hi, limit=100,
            )
            total += cost
        return total

    def fit(self):
        """
        Run Lloyd-Max to convergence.

        Returns
        -------
        centroids : np.ndarray, shape (2^b,), sorted ascending
        mse       : float, per-coordinate MSE achieved
        """
        centroids = self._initialize_centroids()
        for _ in range(self.n_iter):
            boundaries = self._compute_boundaries(centroids)
            new_centroids = self._update_centroids(boundaries)
            shift = np.max(np.abs(new_centroids - centroids))
            centroids = new_centroids
            if shift < self.tol:
                break
        self.centroids_ = centroids
        self.boundaries_ = self._compute_boundaries(centroids)
        mse = self._mse_cost(self.centroids_, self.boundaries_)
        return self.centroids_, mse


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

_CODEBOOK_CACHE: dict = {}


def lloyd_max_codebook(b: int, d: int = 128) -> np.ndarray:
    """
    Return the 2^b Lloyd-Max centroids for the Beta coordinate distribution
    at dimension d.  Results are cached after the first call.

    Parameters
    ----------
    b : int, bits per coordinate (1–4 recommended; 5+ is slow)
    d : int, vector dimension (default 128)

    Returns
    -------
    centroids : np.ndarray, shape (2^b,), sorted ascending
    """
    key = (b, d)
    if key not in _CODEBOOK_CACHE:
        q = LloydMaxQuantizer(d=d, n_bits=b)
        centroids, _ = q.fit()
        _CODEBOOK_CACHE[key] = centroids
    return _CODEBOOK_CACHE[key].copy()


def lloyd_max_mse(b: int, d: int = 128) -> float:
    """
    Return the per-coordinate MSE of the b-bit Lloyd-Max codebook for dimension d.
    Total MSE for a unit vector = d × per_coord_mse.
    """
    key = (b, d)
    if key not in _CODEBOOK_CACHE:
        lloyd_max_codebook(b, d)  # populates cache
    q = LloydMaxQuantizer(d=d, n_bits=b)
    centroids = _CODEBOOK_CACHE[key]
    boundaries = q._compute_boundaries(centroids)
    return q._mse_cost(centroids, boundaries)
