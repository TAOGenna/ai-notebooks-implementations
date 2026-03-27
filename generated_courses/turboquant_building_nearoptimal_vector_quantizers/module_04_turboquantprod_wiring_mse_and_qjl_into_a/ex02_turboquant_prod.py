"""
Exercise 2: Building TurboQuant_prod — The Two-Stage Unbiased Quantizer
========================================================================

This is Algorithm 2 from the TurboQuant paper. The insight:

  • TurboQuant_mse at (b-1) bits gives a good reconstruction x̂, but with a
    multiplicative bias in inner products (2/pi at b=1, improving slowly).
  • The residual r = x - x̂ is small (Exercise 1 proved this).
  • Apply 1-bit QJL to r: this is unbiased but introduces variance ∝ ||r||^2.
  • Combine: <y, x> ≈ <y, x̂> + <y, DeQuant_qjl(Quant_qjl(r))>

Because the MSE stage shrinks r, QJL's variance is tiny. We get:
  - Unbiasedness from QJL on the residual
  - Low variance because the residual is small

Full distortion bound (Theorem 4.1):
  D_prod ≤ (sqrt(3) * pi^2 * ||y||^2) / (d * 4^b)

Bit allocation per vector (b bits per coordinate total):
  • (b-1) bits → TurboQuant_mse (rotation matrix + codebook shared/known)
  • 1 bit      → QJL sign bits  (the JL matrix S is also shared/known)
  • Also store: ||r|| (a scalar) — needed for dequantization scaling

Prerequisite: Module 2 (ex01_assemble_turboquant_mse.py) and Module 3 (ex02_qjl_implementation.py)
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — locate Module 2 and Module 3 source files
# ---------------------------------------------------------------------------
_COURSE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_MOD2_CANDIDATES = [
    "module_02_turboquantmse_the_full_mse_quantization",
    "module_02_turboquant_mse",
]
_MOD3_CANDIDATES = [
    "module_03_qjl_and_the_signbit_trick_unbiased_1bit",
    "module_03_qjl",
]

for _cand in _MOD2_CANDIDATES:
    _p = os.path.join(_COURSE_ROOT, _cand)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break

for _cand in _MOD3_CANDIDATES:
    _p = os.path.join(_COURSE_ROOT, _cand)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break

# Also add module_04 directory itself (for shared helpers like ex01)
sys.path.insert(0, os.path.dirname(__file__))

try:
    from ex01_assemble_turboquant_mse import TurboQuantMSE
except ImportError:
    # Fallback: pull the standalone version from ex01 of this module
    from ex01_residual_analysis import TurboQuantMSE

try:
    from ex02_qjl_implementation import QJL
except ImportError:
    # Minimal QJL implementation so this file runs standalone
    class QJL:
        """
        Quantized Johnson-Lindenstrauss 1-bit quantizer.

        Q_qjl(x) = sign(S @ x)       where S ~ N(0,1)^{d×d}
        Q_qjl^{-1}(z) = sqrt(pi/2)/d * S.T @ z

        For any unit-norm y:
          E[<y, Q_qjl^{-1}(Q_qjl(x))>] = <y, x>        (unbiased)
          Var  ≤ (pi/(2d)) * ||y||^2 * ||x||^2
        """

        def __init__(self, d: int, seed: int = 1):
            self.d = d
            rng = np.random.default_rng(seed)
            self.S = rng.standard_normal((d, d))

        def quantize(self, x: np.ndarray) -> np.ndarray:
            """Returns sign bits: +1 or -1, shape (d,)."""
            return np.sign(self.S @ x)

        def dequantize(self, sign_bits: np.ndarray) -> np.ndarray:
            """
            Reconstruct from sign bits.
            Returns sqrt(pi/2)/d * S^T @ sign_bits, shape (d,).
            """
            scale = np.sqrt(np.pi / 2) / self.d
            return scale * (self.S.T @ sign_bits)


# ---------------------------------------------------------------------------
# Dataclass-style container for TurboQuant_prod encoded data
# ---------------------------------------------------------------------------
class ProdCode:
    """Holds the compressed representation of one vector under TurboQuant_prod."""

    def __init__(self, mse_indices: np.ndarray,
                 qjl_bits: np.ndarray,
                 residual_norm: float):
        """
        Parameters
        ----------
        mse_indices : np.ndarray, shape (d,), dtype int
            Codebook indices from the (b-1)-bit MSE quantizer.
        qjl_bits : np.ndarray, shape (d,), values in {-1, +1}
            Sign bits from applying QJL to the residual.
        residual_norm : float
            ||r|| = ||x - DeQuant_mse(mse_indices)||.
            Needed to scale the QJL reconstruction during dequantization.
        """
        self.mse_indices = mse_indices
        self.qjl_bits = qjl_bits
        self.residual_norm = residual_norm


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class TurboQuantProd:
    """
    TurboQuant_prod: Two-stage unbiased inner product quantizer (Algorithm 2).

    Bit allocation: given b bits per coordinate,
      • (b-1) bits  →  TurboQuantMSE for the coarse reconstruction
      •  1 bit      →  QJL applied to the residual r = x - x̂

    Encoding stores: (codebook indices, QJL sign bits, ||r||).
    The rotation matrix and JL matrix are shared state (not stored per vector).

    Distortion bound (Theorem 4.1 of the paper):
        D_prod ≤ (sqrt(3) * pi^2 * ||y||^2) / (d * 4^b)

    Usage
    -----
    >>> tq = TurboQuantProd(d=128, b=3, seed=0)
    >>> code = tq.quantize(x)       # returns a ProdCode
    >>> x_hat = tq.dequantize(code) # returns np.ndarray, shape (d,)
    >>> ip_est = tq.inner_product(code, y)  # <y, x_hat>

    Parameters
    ----------
    d : int
        Vector dimension.
    b : int
        Total bits per coordinate (≥ 2; at b=1 only QJL is used).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, d: int, b: int, seed: int = 0):
        self.d = d
        self.b = b
        self.mse_bits = max(b - 1, 0)   # bits allocated to MSE stage
        self.use_mse = (self.mse_bits > 0)

        # MSE quantizer at (b-1) bits — shares its rotation with this instance
        if self.use_mse:
            self.mse_quant = TurboQuantMSE(d=d, b=self.mse_bits, seed=seed)

        # QJL quantizer — independent random projection
        self.qjl = QJL(d=d, seed=seed + 1)

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------
    def quantize(self, x: np.ndarray) -> ProdCode:
        """
        Encode vector x with TurboQuant_prod (Algorithm 2, encode step).

        Algorithm steps:
          1. Coarse MSE quantization at (b-1) bits:
               mse_indices = Quant_mse(x)          # integer indices, shape (d,)
               x_hat       = DeQuant_mse(mse_indices)  # reconstruction, shape (d,)
          2. Compute residual:
               r = x - x_hat                        # shape (d,)
          3. Apply QJL to residual:
               sign_bits = Quant_qjl(r)             # ±1 per coordinate, shape (d,)
          4. Store residual norm (for scaling at decode time):
               r_norm = ||r||_2                      # scalar float

        At b=1, the MSE stage is skipped (mse_bits=0), so x_hat = 0 and r = x.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            A single vector to quantize (need not be unit-norm, but typically is).

        Returns
        -------
        code : ProdCode
            Compressed representation: (mse_indices, qjl_bits, residual_norm).
        """
        # =====================================================================
        # TODO: Implement the encode step above.  (~6 lines)
        #
        # If self.use_mse is True:
        #   - Call self.mse_quant.quantize_single(x) → mse_indices (shape d,)
        #   - Call self.mse_quant.dequantize_single(mse_indices) → x_hat
        # If self.use_mse is False (b=1):
        #   - mse_indices = np.zeros(self.d, dtype=int)
        #   - x_hat = np.zeros(self.d)
        #
        # Then:
        #   - Compute r = x - x_hat
        #   - Apply self.qjl.quantize(r) → sign_bits
        #   - Compute r_norm = np.linalg.norm(r)
        #   - Return ProdCode(mse_indices, sign_bits, r_norm)
        # =====================================================================
        raise NotImplementedError("Implement quantize in TurboQuantProd")
        # =====================================================================

    # ------------------------------------------------------------------
    # Dequantization
    # ------------------------------------------------------------------
    def dequantize(self, code: ProdCode) -> np.ndarray:
        """
        Decode a ProdCode back to an approximate vector (Algorithm 2, decode step).

        Algorithm steps:
          1. MSE reconstruction:
               x_hat = DeQuant_mse(code.mse_indices)      # shape (d,)
               (At b=1 this is zeros.)
          2. QJL reconstruction of the residual:
               r_hat_unit = DeQuant_qjl(code.qjl_bits)    # shape (d,), unit-scale
          3. Scale QJL output by the stored residual norm:
               r_hat = code.residual_norm * r_hat_unit
          4. Sum the two stages:
               x_approx = x_hat + r_hat                   # shape (d,)

        Why multiply by residual_norm?
        --------------------------------
        QJL dequantization (sqrt(pi/2)/d * S^T @ z) produces a vector whose
        *expected* inner product with y equals <y, r / ||r||> (unit-normalised
        residual direction). Multiplying by ||r|| restores the correct scale so
        that E[<y, r_hat>] = <y, r>.

        Parameters
        ----------
        code : ProdCode

        Returns
        -------
        x_approx : np.ndarray, shape (d,)
        """
        # =====================================================================
        # TODO: Implement the decode step above.  (~6 lines)
        #
        # Steps:
        #   1. MSE reconstruction:
        #        if self.use_mse: x_hat = self.mse_quant.dequantize_single(code.mse_indices)
        #        else:            x_hat = np.zeros(self.d)
        #   2. QJL reconstruction (unit scale):
        #        r_hat_unit = self.qjl.dequantize(code.qjl_bits)
        #   3. Scale:
        #        r_hat = code.residual_norm * r_hat_unit
        #   4. Return x_hat + r_hat
        # =====================================================================
        raise NotImplementedError("Implement dequantize in TurboQuantProd")
        # =====================================================================

    # ------------------------------------------------------------------
    # Inner product estimator
    # ------------------------------------------------------------------
    def inner_product(self, code: ProdCode, y: np.ndarray) -> float:
        """
        Estimate <y, x> from the compressed code and the (full-precision) query y.

        This corresponds to the asymmetric estimator used in KV cache attention:
        keys are quantized (stored as ProdCode), queries stay in full precision.

        Formula:
            ip_est = <y, DeQuant_prod(code)>
                   = <y, x_hat> + code.residual_norm * <y, DeQuant_qjl(code.qjl_bits)>

        Both terms are inner products between y and a d-dimensional vector.

        Parameters
        ----------
        code : ProdCode
            Compressed key vector.
        y : np.ndarray, shape (d,)
            Full-precision query vector.

        Returns
        -------
        estimate : float
        """
        # =====================================================================
        # TODO: Compute the inner product estimate.  (~4 lines)
        #
        # Option A (simple): dequantize then dot:
        #   x_approx = self.dequantize(code)
        #   return float(np.dot(y, x_approx))
        #
        # Option B (explicit, matches formula in docstring):
        #   ip_mse  = np.dot(y, x_hat) if self.use_mse else 0.0
        #   ip_qjl  = code.residual_norm * np.dot(y, self.qjl.dequantize(code.qjl_bits))
        #   return ip_mse + ip_qjl
        # =====================================================================
        raise NotImplementedError("Implement inner_product in TurboQuantProd")
        # =====================================================================


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def sample_unit_vectors(N: int, d: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Sample N unit-norm vectors uniformly from S^{d-1}."""
    raw = rng.standard_normal((N, d))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def measure_bias_and_variance(tq: TurboQuantProd, xs: np.ndarray,
                               ys: np.ndarray) -> tuple:
    """
    Estimate bias and variance of the inner product estimator.

    For each pair (x_i, y_i) compute the estimated inner product
    ip_hat_i = tq.inner_product(tq.quantize(x_i), y_i).
    Compare to the true inner product ip_true_i = <y_i, x_i>.

    Parameters
    ----------
    tq : TurboQuantProd
    xs : np.ndarray, shape (N, d)   — key vectors (to be quantized)
    ys : np.ndarray, shape (N, d)   — query vectors (full precision)

    Returns
    -------
    bias : float
        Mean of (ip_hat_i - ip_true_i)
    variance : float
        Variance of (ip_hat_i - ip_true_i)
    """
    N = xs.shape[0]
    errors = np.zeros(N)
    for i in range(N):
        code = tq.quantize(xs[i])
        ip_hat = tq.inner_product(code, ys[i])
        ip_true = float(np.dot(ys[i], xs[i]))
        errors[i] = ip_hat - ip_true
    return float(errors.mean()), float(errors.var())


def theory_distortion_bound(d: int, b: int, y_norm_sq: float = 1.0) -> float:
    """
    Theoretical upper bound on D_prod (Theorem 4.1):
        D_prod ≤ (sqrt(3) * pi^2 * ||y||^2) / (d * 4^b)
    """
    return (np.sqrt(3) * np.pi**2 * y_norm_sq) / (d * 4**b)


# ---------------------------------------------------------------------------
# Main milestone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 2: TurboQuant_prod — Two-Stage Unbiased Inner Products")
    print("=" * 70)

    D = 128
    N = 1000   # vector pairs
    BIT_WIDTHS = [2, 3, 4]
    rng = np.random.default_rng(7)

    xs = sample_unit_vectors(N, D, rng)
    ys = sample_unit_vectors(N, D, rng)

    print(f"\nVectors: {N} pairs of unit-norm samples from S^{{{D-1}}}")
    print(f"\n{'b':>3} | {'Bias':>10} | {'Variance':>12} | "
          f"{'Theory bound':>14} | {'Unbiased?':>10} | {'Within bound?':>14}")
    print("-" * 72)

    all_unbiased = True
    all_bounded = True

    for b in BIT_WIDTHS:
        tq = TurboQuantProd(d=D, b=b, seed=0)
        bias, var = measure_bias_and_variance(tq, xs, ys)
        theory = theory_distortion_bound(D, b)

        unbiased = abs(bias) < 0.01
        bounded = var <= theory * 1.1   # allow 10% slack for finite samples
        all_unbiased &= unbiased
        all_bounded &= bounded

        print(f"{b:>3} | {bias:>+10.4f} | {var:>12.6f} | "
              f"{theory:>14.6f} | {'✓' if unbiased else '✗':>10} | "
              f"{'✓' if bounded else '✗':>14}")

    print()
    if all_unbiased:
        print("UNBIASED at all bit-widths ✓")
    else:
        print("WARNING: bias detected — check your dequantize implementation.")

    if all_bounded:
        print("All within theoretical distortion bounds ✓")
    else:
        print("WARNING: variance exceeds theory bound — possible bug in QJL scaling.")

    print()
    print("Recall (Exercise 1): TurboQuant_mse at same b had large bias at b=2.")
    print("TurboQuant_prod fixes that bias completely by using QJL on the residual.")
    print()
    print("Distortion bound formula: D_prod ≤ sqrt(3)·π²·||y||² / (d·4^b)")
    print(f"  At b=2: {theory_distortion_bound(D,2):.6f}")
    print(f"  At b=3: {theory_distortion_bound(D,3):.6f}")
    print(f"  At b=4: {theory_distortion_bound(D,4):.6f}")
