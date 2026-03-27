"""
Exercise 1: The Residual is Small — Measuring What MSE Quantization Leaves Behind
==================================================================================

Before building TurboQuant_prod, we must understand *why* it works. The key insight:
after MSE quantization, the residual r = x - DeQuant_mse(Quant_mse(x)) is tiny.

QJL's inner product variance is proportional to ||r||^2 (Section 3.2 of the paper):

    Var[<y, DeQuant_qjl(Quant_qjl(r))>] = pi/(2d) * ||y||^2 * ||r||^2

So by first compressing x with (b-1)-bit MSE quantization, we shrink r dramatically —
and QJL inherits that shrinkage. This is the mathematical engine behind TurboQuant_prod.

In this exercise you will:
  1. Import TurboQuantMSE from Module 2 (ex01_assemble_turboquant_mse.py)
  2. Compute residuals r = x - dequant(quant(x)) at each bit-width
  3. Measure E[||r||^2] over many random vectors and compare to D_mse from Module 2
  4. Compute the resulting QJL variance bound to appreciate the shrinkage

Prerequisite: Module 2 (ex01_assemble_turboquant_mse.py)
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Import TurboQuantMSE from Module 2.
# We try several relative paths so this exercise works when run from:
#   • the module_04_... directory
#   • the course root
# ---------------------------------------------------------------------------
_SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "module_02_turboquantmse_the_full_mse_quantization"),
    os.path.join(os.path.dirname(__file__), "..", "module_02_turboquant_mse"),
]
for _p in _SEARCH_PATHS:
    if os.path.isdir(_p):
        sys.path.insert(0, os.path.abspath(_p))
        break

try:
    from ex01_assemble_turboquant_mse import TurboQuantMSE
except ImportError:
    # Inline minimal implementation so this exercise still runs standalone.
    # If you have module_02 available, the real import above takes precedence.
    from scipy.stats import beta as _beta_dist
    import warnings

    class TurboQuantMSE:
        """Minimal standalone TurboQuantMSE for use when Module 2 is absent."""

        def __init__(self, d: int, b: int, seed: int = 0):
            self.d = d
            self.b = b
            rng = np.random.default_rng(seed)
            # Random rotation via QR decomposition
            A = rng.standard_normal((d, d))
            Q, _ = np.linalg.qr(A)
            self.rotation = Q
            self.centroids = self._lloyd_max(b)

        def _lloyd_max(self, b: int, n_samples: int = 200_000) -> np.ndarray:
            rng = np.random.default_rng(42)
            d = self.d
            # Beta distribution samples for one coordinate of a uniform unit vector
            samples = rng.beta(0.5, (d - 1) / 2, size=n_samples)
            samples = samples * 2 - 1  # shift to [-1, 1]
            # signs
            samples *= rng.choice([-1, 1], size=n_samples)
            k = 2 ** b
            # k-means style Lloyd-Max
            cents = np.linspace(samples.min(), samples.max(), k)
            for _ in range(200):
                dists = np.abs(samples[:, None] - cents[None, :])
                idx = dists.argmin(axis=1)
                new_cents = np.array([samples[idx == j].mean() if (idx == j).any()
                                      else cents[j] for j in range(k)])
                if np.max(np.abs(new_cents - cents)) < 1e-9:
                    break
                cents = new_cents
            return np.sort(cents)

        def quantize(self, x: np.ndarray) -> np.ndarray:
            xr = x @ self.rotation.T
            dists = np.abs(xr[:, :, None] - self.centroids[None, None, :])
            return dists.argmin(axis=-1)

        def dequantize(self, indices: np.ndarray) -> np.ndarray:
            yr = self.centroids[indices]
            return yr @ self.rotation

        def quantize_single(self, x: np.ndarray) -> np.ndarray:
            xr = self.rotation @ x
            dists = np.abs(xr[:, None] - self.centroids[None, :])
            return dists.argmin(axis=-1)

        def dequantize_single(self, indices: np.ndarray) -> np.ndarray:
            yr = self.centroids[indices]
            return self.rotation.T @ yr


# ---------------------------------------------------------------------------
# D_mse reference values from Module 2 (Table 1 of the paper, d=128)
# D_mse ≈ C / 4^b  where C = sqrt(3)*pi/2
# ---------------------------------------------------------------------------
D_MSE_THEORY = {
    1: np.sqrt(3) * np.pi / 2 / 4**1,   # ≈ 0.680
    2: np.sqrt(3) * np.pi / 2 / 4**2,   # ≈ 0.170 ... wait, these are per-coord.
    3: np.sqrt(3) * np.pi / 2 / 4**3,
    4: np.sqrt(3) * np.pi / 2 / 4**4,
}
# Note: D_mse is the *total* squared error = d * (per-coord cost).
# Per-coordinate values observed empirically: 0.361, 0.117, 0.030, 0.009
# The theory gives C/4^b per unit vector; at d=128 it is C*d/4^b total.
# We store the per-unit-vector (total) values here.
D_MSE_EMPIRICAL_REF = {1: 0.361, 2: 0.117, 3: 0.030, 4: 0.009}


def compute_residuals(quantizer: TurboQuantMSE,
                      vectors: np.ndarray) -> np.ndarray:
    """
    Compute the quantization residual for each vector.

    The residual r_i = x_i - DeQuant_mse(Quant_mse(x_i)) captures every bit
    of information that the MSE quantizer *threw away*. This is exactly what
    TurboQuant_prod's QJL stage will recover (approximately) using just 1 bit
    per coordinate.

    Parameters
    ----------
    quantizer : TurboQuantMSE
        An already-initialised quantizer for a given (d, b).
    vectors : np.ndarray, shape (N, d)
        N unit-norm vectors drawn uniformly from S^{d-1}.

    Returns
    -------
    residuals : np.ndarray, shape (N, d)
        r_i = x_i - DeQuant_mse(Quant_mse(x_i))

    Hint
    ----
    Use quantizer.quantize_single and quantizer.dequantize_single inside a
    loop, OR batch the operation: both TurboQuantMSE interfaces are available.
    The residual is simply the original minus the reconstruction.
    """
    # =========================================================================
    # TODO: Compute the residual for every vector in `vectors`.  (~4 lines)
    #
    # Steps:
    #   1. For each vector x_i (shape: (d,)), call quantizer.quantize_single(x_i)
    #      to get integer codebook indices.
    #   2. Call quantizer.dequantize_single(indices) to get the reconstruction.
    #   3. Compute the residual r_i = x_i - reconstruction.
    #
    # Store all residuals as an (N, d) array and return it.
    # =========================================================================
    raise NotImplementedError("Implement compute_residuals")
    # =========================================================================


def measure_residual_norm_squared(residuals: np.ndarray) -> float:
    """
    Compute E[||r||^2] averaged over all residual vectors.

    This is the *empirical* MSE: the average squared Euclidean distance
    between original vectors and their quantized reconstructions.
    By the law of large numbers this converges to D_mse as N → ∞.

    Parameters
    ----------
    residuals : np.ndarray, shape (N, d)

    Returns
    -------
    mean_sq_norm : float
        Average of ||r_i||^2 over all N vectors.
    """
    # =========================================================================
    # TODO: Compute the mean squared norm of the residuals.  (~2 lines)
    #
    # Hint: ||r||^2 = np.dot(r, r). Vectorised: (residuals ** 2).sum(axis=1)
    # =========================================================================
    raise NotImplementedError("Implement measure_residual_norm_squared")
    # =========================================================================


def qjl_variance_bound(mean_residual_sq: float, d: int,
                        y_norm_sq: float = 1.0) -> float:
    """
    Compute the QJL variance bound when applied to the residual.

    From the QJL paper (Theorem 3.1):
        Var[<y, DeQuant_qjl(Quant_qjl(r))>] ≤ (pi / (2d)) * ||y||^2 * ||r||^2

    Since QJL is applied per-vector and we averaged ||r||^2, this gives:
        E_r[ Var ] ≤ (pi / (2d)) * ||y||^2 * E[||r||^2]

    Parameters
    ----------
    mean_residual_sq : float
        E[||r||^2], the empirical average residual norm squared.
    d : int
        Dimension of the vectors.
    y_norm_sq : float
        ||y||^2 for the query vector. Default 1.0 (unit-norm queries).

    Returns
    -------
    variance_bound : float
        Upper bound on QJL variance when applied to the residual.
    """
    # =========================================================================
    # TODO: Implement the variance bound formula above.  (~2 lines)
    #
    # Hint: variance_bound = (pi / (2*d)) * y_norm_sq * mean_residual_sq
    # =========================================================================
    raise NotImplementedError("Implement qjl_variance_bound")
    # =========================================================================


def sample_unit_vectors(N: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample N vectors uniformly from the unit hypersphere S^{d-1}.

    Sampling strategy: draw x ~ N(0, I), then divide by ||x||.
    This gives a uniform distribution on S^{d-1} (by rotational symmetry
    of the Gaussian distribution).

    Parameters
    ----------
    N : int
        Number of vectors to sample.
    d : int
        Dimension.
    rng : np.random.Generator

    Returns
    -------
    vectors : np.ndarray, shape (N, d)
        Unit-norm vectors.
    """
    raw = rng.standard_normal((N, d))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


# ---------------------------------------------------------------------------
# Main milestone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    print("=" * 70)
    print("Exercise 1: The Residual is Small")
    print("=" * 70)

    D = 128
    N = 2000   # vectors
    BIT_WIDTHS = [1, 2, 3, 4]
    rng = np.random.default_rng(42)

    vectors = sample_unit_vectors(N, D, rng)

    print(f"\nVectors: {N} unit-norm samples from S^{{{D-1}}}")
    print(f"{'b':>4} | {'E[||r||^2]':>12} | {'D_mse ref':>10} | "
          f"{'match':>6} | {'QJL var bound (||y||=1)':>24}")
    print("-" * 72)

    for b in BIT_WIDTHS:
        quant = TurboQuantMSE(d=D, b=b, seed=0)
        residuals = compute_residuals(quant, vectors)
        mean_r2 = measure_residual_norm_squared(residuals)
        var_bound = qjl_variance_bound(mean_r2, D, y_norm_sq=1.0)
        ref = D_MSE_EMPIRICAL_REF[b]
        match = "✓" if abs(mean_r2 - ref) / ref < 0.10 else "✗"
        print(f"{b:>4} | {mean_r2:>12.4f} | {ref:>10.3f} | {match:>6} | {var_bound:>24.6f}")

    print()
    print("Key insight:")
    print("  E[||r||^2] ≈ D_mse at each bit-width — the residual IS the MSE loss.")
    print("  QJL's variance scales with ||r||^2, so shrinking r shrinks variance.")
    print()
    print("QJL variance bound progression:")
    print("  b=1 → large residual → high QJL variance on residual")
    print("  b=4 → tiny residual → tiny QJL variance — nearly perfect inner products!")
    print()
    print("  This is why TurboQuant_prod works: MSE quantization does the heavy")
    print("  lifting, and QJL corrects the bias on what remains.")
