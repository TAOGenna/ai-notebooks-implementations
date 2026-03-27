"""
turboquant_core.py — Complete TurboQuant implementations for Module 5
======================================================================

This module provides fully-working implementations of all TurboQuant classes
from Modules 1–4 so that Module 5 exercises can import them without depending
on any prior module's exercise files (which contain incomplete TODOs).

Classes provided:
  - TurboQuantMSE      (Module 2): MSE-optimal quantizer via random rotation + Lloyd-Max
  - QJL                (Module 3): 1-bit Quantized Johnson-Lindenstrauss transform
  - ProdCode                     : Container for TurboQuantProd encoded vector
  - TurboQuantProd     (Module 4): Two-stage unbiased inner product quantizer

Utility functions:
  - sample_unit_vectors(N, d, rng)   → (N, d) unit-norm vectors
  - softmax(x)                        → probability vector
  - kl_divergence(p, q)               → float

Algorithm references:
  - TurboQuantMSE: Algorithm 1 from the TurboQuant paper
  - TurboQuantProd: Algorithm 2 from the TurboQuant paper (Section 4)
  - QJL: Theorem 3.1 from the companion QJL paper
"""

import numpy as np
from functools import lru_cache
from typing import List


# ============================================================================
# Lloyd-Max codebook computation
# ============================================================================

def _lloyd_max_centroids(d: int, b: int, n_samples: int = 200_000) -> np.ndarray:
    """
    Compute the 2^b Lloyd-Max centroids for the Beta coordinate distribution.

    After a random rotation, each coordinate of a unit-norm vector in R^d
    follows a symmetric Beta distribution:

        f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^{(d-3)/2}

    which converges to N(0, 1/d) in high dimensions. We draw samples from
    this distribution and run Lloyd's algorithm (continuous k-means) to find
    the 2^b optimal centroids that minimise expected squared quantization error.

    Parameters
    ----------
    d : int
        Vector dimension (determines the Beta distribution shape).
    b : int
        Bits per coordinate (codebook has 2^b entries).
    n_samples : int
        Monte Carlo samples for the distribution (more = more accurate).

    Returns
    -------
    centroids : np.ndarray, shape (2^b,), sorted ascending
    """
    rng = np.random.default_rng(42)
    k = 2 ** b

    # Correctly sample the coordinate distribution for a uniform unit vector on S^{d-1}.
    #
    # For x ~ Uniform(S^{d-1}), the i-th coordinate satisfies:
    #     x_i²  ~ Beta(1/2, (d-1)/2)     [squared coordinate distribution]
    #     x_i   = ±sqrt(x_i²)            [sign is independent uniform ±1]
    #
    # This gives E[x_i] = 0, E[x_i²] = 1/d (as expected for a unit vector).
    # For large d, x_i → N(0, 1/d) — the Gaussian approximation used in the paper.
    #
    # NOTE: A common mistake is to linearly map Beta(0.5, (d-1)/2) to [-1,1] via
    # `2u - 1`.  This is WRONG: it maps the distribution of x_i² (support [0,1])
    # linearly to [-1,1] instead of computing x_i = ±sqrt(x_i²).  The resulting
    # "coordinates" concentrate near ±1 rather than near 0, producing a codebook
    # with completely wrong scale (centroids ~1 for data with std ~1/√d).
    u = rng.beta(0.5, max((d - 1) / 2, 0.5), size=n_samples)  # u = x_i² in [0,1]
    samples = np.sqrt(u) * rng.choice([-1.0, 1.0], size=n_samples)  # x_i = ±√(x_i²)

    # Initialize centroids at equally-spaced quantiles
    percs = np.linspace(2, 98, k)
    cents = np.percentile(samples, percs)

    # Lloyd's algorithm
    for _ in range(300):
        diffs = np.abs(samples[:, None] - cents[None, :])   # (n_samples, k)
        labels = diffs.argmin(axis=1)                         # (n_samples,)
        new_cents = np.array(
            [samples[labels == j].mean() if (labels == j).any() else cents[j]
             for j in range(k)]
        )
        if np.max(np.abs(new_cents - cents)) < 1e-10:
            break
        cents = new_cents

    return np.sort(new_cents)


# Cache results: same (d, b) pair always gives the same codebook
_codebook_cache: dict = {}


def get_codebook(d: int, b: int) -> np.ndarray:
    """Return (possibly cached) Lloyd-Max centroids for dimension d, bits b."""
    key = (d, b)
    if key not in _codebook_cache:
        _codebook_cache[key] = _lloyd_max_centroids(d, b)
    return _codebook_cache[key]


# ============================================================================
# TurboQuantMSE — MSE-optimal quantizer (Algorithm 1)
# ============================================================================

class TurboQuantMSE:
    """
    TurboQuant MSE quantizer (Algorithm 1).

    Pipeline:
      Quant:   Pi * x  →  find nearest centroid per coordinate  →  indices
      DeQuant: centroids[indices]  →  Pi^T * y_tilde  →  reconstruction

    Distortion bound (Proposition 3.1 of the paper):
        D_mse ≤ (sqrt(3) * pi / 2) / 4^b

    Parameters
    ----------
    d    : int  — vector dimension
    b    : int  — bits per coordinate
    seed : int  — random seed for the rotation matrix
    """

    def __init__(self, d: int, b: int, seed: int = 0):
        self.d = d
        self.b = b
        # Random rotation matrix via QR decomposition (uniformly random orthogonal)
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(A)
        self.rotation = Q            # shape: (d, d), satisfies Pi @ Pi.T = I
        self.centroids = get_codebook(d, b)  # shape: (2^b,)

    # -- Batch interface (N vectors at once) ----------------------------------

    def quantize(self, X: np.ndarray) -> np.ndarray:
        """
        Encode N vectors to centroid indices.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)

        Returns
        -------
        indices : np.ndarray, shape (N, d), dtype int32
        """
        Xr = X @ self.rotation.T                                 # (N, d)
        diffs = np.abs(Xr[:, :, None] - self.centroids[None, None, :])  # (N, d, 2^b)
        return diffs.argmin(axis=-1).astype(np.int32)

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """
        Decode centroid indices back to approximate vectors.

        Parameters
        ----------
        indices : np.ndarray, shape (N, d)

        Returns
        -------
        X_hat : np.ndarray, shape (N, d)
        """
        Yr = self.centroids[indices]    # (N, d) — centroid values in rotated space
        return Yr @ self.rotation       # rotate back: Pi^T @ y_tilde  [Pi^T = Pi.T]

    # -- Single-vector interface ---------------------------------------------

    def quantize_single(self, x: np.ndarray) -> np.ndarray:
        """Encode a single vector (d,) → indices (d,)."""
        xr = self.rotation @ x                                # (d,)
        diffs = np.abs(xr[:, None] - self.centroids[None, :])  # (d, 2^b)
        return diffs.argmin(axis=-1).astype(np.int32)

    def dequantize_single(self, indices: np.ndarray) -> np.ndarray:
        """Decode a single vector's indices (d,) → reconstruction (d,)."""
        yr = self.centroids[indices]    # (d,)
        return self.rotation.T @ yr    # Pi^T * yr


# ============================================================================
# QJL — 1-bit Quantized Johnson-Lindenstrauss (Theorem 3.1)
# ============================================================================

class QJL:
    """
    Quantized Johnson-Lindenstrauss 1-bit quantizer.

    Encoding:  Q_qjl(x)   = sign(S @ x)            where S ~ N(0,1)^{d×d}
    Decoding:  Q_qjl^{-1}(z) = sqrt(pi/2) / d  *  S^T @ z

    Properties (Theorem 3.1 of the QJL paper):
      • Unbiased:   E[<y, Q^{-1}(Q(x))>] = <y, x>  for all unit-norm y
      • Variance:   Var ≤ (pi / (2d)) * ||y||^2 * ||x||^2

    Parameters
    ----------
    d    : int  — vector dimension
    seed : int  — random seed for S
    """

    def __init__(self, d: int, seed: int = 1):
        self.d = d
        rng = np.random.default_rng(seed)
        self.S = rng.standard_normal((d, d))   # random projection matrix

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Encode vector x (d,) → sign bits (d,) in {-1, +1}."""
        return np.sign(self.S @ x)

    def quantize_batch(self, X: np.ndarray) -> np.ndarray:
        """Encode N vectors (N, d) → sign bits (N, d) in {-1, +1}."""
        return np.sign(X @ self.S.T)

    def dequantize(self, z: np.ndarray) -> np.ndarray:
        """Decode sign bits (d,) → approximate vector (d,)."""
        return (np.sqrt(np.pi / 2) / self.d) * (self.S.T @ z)

    def dequantize_batch(self, Z: np.ndarray) -> np.ndarray:
        """Decode N sign-bit vectors (N, d) → approximate vectors (N, d).

        Derivation: for each row z_i, dequantize(z_i) = sqrt(pi/2)/d * S.T @ z_i
        Batched:    sqrt(pi/2)/d * (Z @ S)   [since (Z @ S)[n,i] = Z[n] · S[:,i] = S.T[i] · Z[n]]
        """
        return (np.sqrt(np.pi / 2) / self.d) * (Z @ self.S)


# ============================================================================
# TurboQuantProd — Two-stage unbiased inner product quantizer (Algorithm 2)
# ============================================================================

class ProdCode:
    """Compressed representation of one vector under TurboQuant_prod."""

    __slots__ = ("mse_indices", "qjl_bits", "residual_norm")

    def __init__(self, mse_indices: np.ndarray,
                 qjl_bits: np.ndarray,
                 residual_norm: float):
        """
        Parameters
        ----------
        mse_indices   : np.ndarray, shape (d,), dtype int32
            Codebook indices from the (b-1)-bit MSE quantizer.
        qjl_bits      : np.ndarray, shape (d,), values in {-1, +1}
            Sign bits from QJL applied to the residual.
        residual_norm : float
            ||r||₂ = ||x - DeQuant_mse(mse_indices)||.
            Used to rescale the QJL reconstruction at decode time.
        """
        self.mse_indices = mse_indices
        self.qjl_bits = qjl_bits
        self.residual_norm = float(residual_norm)


class TurboQuantProd:
    """
    TurboQuant_prod: Two-stage unbiased inner product quantizer (Algorithm 2).

    Stage 1 (coarse): Apply (b-1)-bit TurboQuantMSE → good reconstruction x̂
    Stage 2 (fine):   Compute residual r = x - x̂, apply 1-bit QJL → unbiased

    Decode: x_approx = x̂ + ||r|| * DeQuant_qjl(sign_bits)

    Inner product estimate (asymmetric, query y is full-precision):
        <y, x> ≈ <y, x_approx>

    Distortion bound (Theorem 4.1):
        D_prod ≤ (sqrt(3) * pi² * ||y||²) / (d * 4^b)

    Parameters
    ----------
    d    : int  — vector dimension
    b    : int  — total bits per coordinate (≥ 2 recommended)
    seed : int  — random seed
    """

    def __init__(self, d: int, b: int, seed: int = 0):
        self.d = d
        self.b = b
        self.mse_bits = max(b - 1, 0)
        self.use_mse = (self.mse_bits > 0)

        if self.use_mse:
            self.mse_quant = TurboQuantMSE(d=d, b=self.mse_bits, seed=seed)

        self.qjl = QJL(d=d, seed=seed + 1)

    # -------------------------------------------------------------------------
    # Single-vector interface
    # -------------------------------------------------------------------------

    def quantize(self, x: np.ndarray) -> ProdCode:
        """
        Encode a single vector x (d,) into a ProdCode.

        Steps:
          1. MSE coarse stage:  mse_indices = Quant_mse(x),  x̂ = DeQuant_mse
          2. Residual:          r = x - x̂
          3. QJL fine stage:    sign_bits = sign(S @ r)
          4. Store:             ||r||₂ for later rescaling
        """
        if self.use_mse:
            mse_idx = self.mse_quant.quantize_single(x)
            x_hat = self.mse_quant.dequantize_single(mse_idx)
        else:
            mse_idx = np.zeros(self.d, dtype=np.int32)
            x_hat = np.zeros(self.d)

        r = x - x_hat
        sign_bits = self.qjl.quantize(r)
        r_norm = float(np.linalg.norm(r))
        return ProdCode(mse_idx, sign_bits, r_norm)

    def dequantize(self, code: ProdCode) -> np.ndarray:
        """
        Decode a ProdCode back to an approximate vector (d,).

        Steps:
          1. MSE reconstruction: x̂ = DeQuant_mse(code.mse_indices)
          2. QJL residual:       r̂_unit = DeQuant_qjl(code.qjl_bits)
          3. Combine:            x_approx = x̂ + ||r|| * r̂_unit
        """
        if self.use_mse:
            x_hat = self.mse_quant.dequantize_single(code.mse_indices)
        else:
            x_hat = np.zeros(self.d)

        r_hat_unit = self.qjl.dequantize(code.qjl_bits)
        return x_hat + code.residual_norm * r_hat_unit

    def inner_product(self, code: ProdCode, y: np.ndarray) -> float:
        """
        Estimate <y, x> from compressed code and full-precision query y (d,).

        This is the asymmetric estimator: key is quantized, query is exact.
        Returns float.
        """
        return float(np.dot(y, self.dequantize(code)))

    # -------------------------------------------------------------------------
    # Batch interface (more efficient for large datasets)
    # -------------------------------------------------------------------------

    def quantize_batch(self, X: np.ndarray) -> List[ProdCode]:
        """
        Encode N vectors (N, d) efficiently using vectorised MSE and QJL.

        Returns
        -------
        codes : list[ProdCode], length N
        """
        N = X.shape[0]

        if self.use_mse:
            mse_indices = self.mse_quant.quantize(X)    # (N, d)
            X_hat = self.mse_quant.dequantize(mse_indices)  # (N, d)
        else:
            mse_indices = np.zeros((N, self.d), dtype=np.int32)
            X_hat = np.zeros((N, self.d))

        R = X - X_hat                                    # (N, d) residuals
        sign_bits = self.qjl.quantize_batch(R)           # (N, d)
        r_norms = np.linalg.norm(R, axis=1)             # (N,)

        return [ProdCode(mse_indices[i], sign_bits[i], r_norms[i])
                for i in range(N)]

    def reconstruct_batch(self, codes: List[ProdCode]) -> np.ndarray:
        """
        Decode a list of ProdCodes back to an (N, d) matrix of approximate vectors.

        Efficiently batches MSE decode (one matrix multiply) and QJL decode.
        """
        N = len(codes)

        if self.use_mse:
            mse_indices = np.array([c.mse_indices for c in codes])  # (N, d)
            X_hat = self.mse_quant.dequantize(mse_indices)          # (N, d)
        else:
            X_hat = np.zeros((N, self.d))

        sign_bits = np.array([c.qjl_bits for c in codes])           # (N, d)
        r_norms = np.array([c.residual_norm for c in codes])        # (N,)

        R_hat_unit = self.qjl.dequantize_batch(sign_bits)           # (N, d)
        R_hat = R_hat_unit * r_norms[:, None]                        # (N, d)

        return X_hat + R_hat  # (N, d)

    def batch_inner_products(self, codes: List[ProdCode],
                             y: np.ndarray) -> np.ndarray:
        """
        Compute <y, x_i> estimates for all N codes at once.

        Parameters
        ----------
        codes : list[ProdCode]
        y     : np.ndarray, shape (d,) — full-precision query

        Returns
        -------
        scores : np.ndarray, shape (N,)
        """
        X_approx = self.reconstruct_batch(codes)   # (N, d)
        return X_approx @ y                         # (N,)


# ============================================================================
# Utility functions
# ============================================================================

def sample_unit_vectors(N: int, d: int,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Sample N vectors uniformly from the unit hypersphere S^{d-1}.

    Method: draw z ~ N(0, I_d), then normalise: x = z / ||z||.
    Correctness: Gaussian is rotationally symmetric, so z/||z|| is uniform on S^{d-1}.

    Parameters
    ----------
    N   : int
    d   : int
    rng : np.random.Generator

    Returns
    -------
    X : np.ndarray, shape (N, d), each row has unit norm
    """
    Z = rng.standard_normal((N, d))
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / norms


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.

    Parameters
    ----------
    x : np.ndarray, shape (n,)

    Returns
    -------
    p : np.ndarray, shape (n,), sums to 1.0
    """
    x_shifted = x - x.max()    # subtract max for numerical stability
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    KL divergence KL(p || q) = sum_i p_i * log(p_i / q_i).

    Clipped to [eps, 1] to avoid log(0).

    Parameters
    ----------
    p   : np.ndarray — reference distribution (e.g., exact attention weights)
    q   : np.ndarray — approximate distribution (e.g., quantized attention weights)
    eps : float      — clipping floor for numerical stability

    Returns
    -------
    kl : float ≥ 0   (0 iff p == q)
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))
