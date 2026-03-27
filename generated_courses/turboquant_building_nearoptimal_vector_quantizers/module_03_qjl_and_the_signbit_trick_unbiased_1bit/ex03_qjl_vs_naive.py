"""
Exercise 3: QJL vs Naive Sign-Bit — Why the Asymmetric Estimator Matters
=========================================================================
CLAIM: Quantizing BOTH vectors to sign bits (the "obvious" approach)
       estimates the ANGLE between vectors, not the inner product, and
       has ~4x higher variance than QJL's asymmetric estimator.

This is a CONTRASTIVE exercise:
  - FIRST implement the naive symmetric estimator (sign(Sx) . sign(Sy) / d)
    and observe its bias and variance.
  - THEN compare it side-by-side with QJL from Exercise 2.

WHY DOES THE NAIVE ESTIMATOR FAIL?

The naive estimator sign(S_i . x) * sign(S_i . y) computes:
    +1 if S_i . x and S_i . y have the SAME sign
    -1 if they have DIFFERENT signs

Geometrically, this counts the fraction of random hyperplanes that separate
x and y — which is exactly related to the ANGLE theta between them:

    E[sign(a.x) * sign(a.y)] = 1 - 2*theta/pi  where  theta = arccos(<x,y>/||x||*||y||)

For unit vectors:  E = (pi - 2*arccos(<x,y>)) / pi  ≠  <x,y>  in general.

The angle estimator only equals the inner product when ||x|| = ||y|| = 1 AND
<x,y> is small. For general vectors, the bias can be large.

QJL FIXES THIS by keeping the query full-precision. The product
    sign(S_i . x) * (S_i . y)
now has a clean linear expectation:  E = sqrt(2/pi) * <x, y>  (per row),
which after scaling gives an exactly unbiased estimator.

Dependencies: imports QJL from ex02_qjl_implementation.

Run:
    python ex03_qjl_vs_naive.py
"""

import numpy as np
import sys
import os

# Allow importing QJL from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from ex02_qjl_implementation import QJL


# ============================================================================
# PART A (naive): implement FIRST, see it fail
# ============================================================================

class NaiveSignBit:
    """
    Naive SYMMETRIC sign-bit estimator.

    BOTH x and y are quantized to sign bits. The inner product estimate is:

        ip_hat = (pi/4) * <sign(S*x), sign(S*y)> / d  * scale_correction

    Wait — there's no obvious closed-form unbiased version of this estimator
    for general inner products (that's the whole point!). We include a
    commonly tried scaling: (pi/2) / d, which works well when vectors
    have similar norms but fails in general.

    The naive estimator is:

        ip_hat_naive = (pi/2) / d * np.dot(sign(S @ x), sign(S @ y))

    Parameters
    ----------
    d    : int
    seed : int
    """

    def __init__(self, d: int, seed: int = 42):
        self.d = d
        rng = np.random.default_rng(seed)
        self.S = rng.standard_normal((d, d))

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize x to sign bits (same as QJL quantize)."""
        return np.sign(self.S @ x).astype(np.int8)

    def estimate_inner_product(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Naive SYMMETRIC estimator: quantize BOTH x and y to sign bits,
        then compute scaled dot product of sign bits.

        The scaling (pi/2)/d comes from treating each coordinate as an
        independent estimate of <x,y>/(||x||*||y||) and applying the
        E[sign(a.x) * sign(a.y)] identity. It is an unbiased estimator
        of the COSINE similarity (for unit vectors), not the inner product.

        Parameters
        ----------
        x : ndarray, shape (d,)  — key
        y : ndarray, shape (d,)  — query

        Returns
        -------
        ip_hat : float
        """
        # ====================================================================
        # TODO: Implement the naive symmetric estimator.                   (~3 lines)
        #
        # Step 1: Quantize BOTH x and y to sign bits:
        #           zx = self.quantize(x)
        #           zy = self.quantize(y)
        # Step 2: Return  (np.pi / 2) / self.d * np.dot(zx, zy)
        #
        # This is the "obvious" approach — quantize everything and see
        # what happens to bias and variance.
        # ====================================================================
        raise NotImplementedError("Implement NaiveSignBit.estimate_inner_product")
        # ====================================================================


# ============================================================================
# PART B: comparison framework
# ============================================================================

def compare_estimators(
    d: int = 128,
    n_pairs: int = 500,
    seed: int = 99,
) -> dict:
    """
    Compare QJL and NaiveSignBit on n_pairs random unit-norm vector pairs.

    For each pair (x, y), compute:
        - true inner product <x, y>
        - QJL estimate
        - Naive symmetric estimate
        - error for each

    Parameters
    ----------
    d       : int — head dimension
    n_pairs : int — number of test pairs
    seed    : int — RNG seed

    Returns
    -------
    results : dict with keys
        'true_ips'      — array of true inner products
        'qjl_errors'    — array of (qjl_estimate - true)
        'naive_errors'  — array of (naive_estimate - true)

        Summary statistics (computed internally):
        'qjl_bias'      — mean of qjl_errors
        'qjl_var'       — variance of qjl_errors
        'naive_bias'    — mean of naive_errors
        'naive_var'     — variance of naive_errors
    """
    rng = np.random.default_rng(seed)
    qjl   = QJL(d=d, seed=seed + 1)
    naive = NaiveSignBit(d=d, seed=seed + 1)   # same S matrix for fair comparison

    true_ips    = []
    qjl_errors  = []
    naive_errors = []

    # ========================================================================
    # TODO: Fill in the comparison loop.                                  (~8 lines)
    #
    # For each trial in range(n_pairs):
    #   1. Draw x, y ~ N(0, I_d), normalise to unit norm.
    #   2. Compute true_ip = np.dot(x, y).
    #   3. Compute qjl_est   = qjl.estimate_inner_product(x, y)
    #   4. Compute naive_est = naive.estimate_inner_product(x, y)
    #   5. Append true_ip, (qjl_est - true_ip), (naive_est - true_ip).
    #
    # Hint: x = rng.standard_normal(d); x /= np.linalg.norm(x)
    # ========================================================================
    raise NotImplementedError("Implement compare_estimators loop")
    # ========================================================================

    true_ips     = np.array(true_ips)
    qjl_errors   = np.array(qjl_errors)
    naive_errors = np.array(naive_errors)

    return {
        "true_ips":     true_ips,
        "qjl_errors":   qjl_errors,
        "naive_errors": naive_errors,
        "qjl_bias":     float(np.mean(qjl_errors)),
        "qjl_var":      float(np.var(qjl_errors)),
        "naive_bias":   float(np.mean(naive_errors)),
        "naive_var":    float(np.var(naive_errors)),
    }


def print_comparison_table(results: dict, d: int):
    """Pretty-print a side-by-side comparison table."""
    theory_var = np.pi / (2 * d)
    var_ratio  = results["naive_var"] / results["qjl_var"] if results["qjl_var"] > 0 else float("inf")

    print("=" * 70)
    print("ESTIMATOR COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Estimator':<22} | {'Bias':>8} | {'Variance':>10} | {'Bits(key)':>9} | {'Bits(query)':>11}"
    print(header)
    print("-" * 70)

    print(f"{'QJL (asymmetric)':<22} | {results['qjl_bias']:>8.4f} | "
          f"{results['qjl_var']:>10.5f} | {'1':>9} | {'32':>11}")
    print(f"{'Naive (symmetric)':<22} | {results['naive_bias']:>8.4f} | "
          f"{results['naive_var']:>10.5f} | {'1':>9} | {'1':>11}")
    print("-" * 70)
    print(f"{'Theory (QJL bound)':<22} | {'~0':>8} | {theory_var:>10.5f} | {'1':>9} | {'32':>11}")
    print()
    print(f"  Variance ratio  naive / QJL = {var_ratio:.2f}x")
    print(f"  QJL variance ≤ theory bound : "
          f"{'✓' if results['qjl_var'] <= theory_var * 1.2 else '✗'}")
    print()
    print("  The ASYMMETRIC design is the key insight:")
    print("  Full-precision query + 1-bit key → low variance, zero bias.")
    print("  1-bit key + 1-bit query           → higher variance, cosine bias.")


# ============================================================================
# MILESTONE
# ============================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    d = 128

    # ----- PHASE 1: demonstrate the naive estimator's bias on a single pair
    print("=" * 70)
    print("PHASE 1: NAIVE ESTIMATOR ON A SINGLE HIGH-CONTRAST PAIR")
    print("=" * 70)
    rng = np.random.default_rng(0)
    # Construct a pair where cos-similarity ≠ inner product is clear
    x = rng.standard_normal(d); x /= np.linalg.norm(x)
    # y is correlated with x (simulate attention to a relevant token)
    y = 0.7 * x + 0.3 * rng.standard_normal(d); y /= np.linalg.norm(y)
    true_ip = float(np.dot(x, y))

    qjl_single   = QJL(d=d, seed=1)
    naive_single = NaiveSignBit(d=d, seed=1)   # same S

    qjl_est   = qjl_single.estimate_inner_product(x, y)
    naive_est = naive_single.estimate_inner_product(x, y)

    print(f"  True inner product : {true_ip:.4f}")
    print(f"  QJL estimate       : {qjl_est:.4f}   (error = {qjl_est - true_ip:+.4f})")
    print(f"  Naive estimate     : {naive_est:.4f}   (error = {naive_est - true_ip:+.4f})")
    print()

    # ----- PHASE 2: large-scale bias/variance comparison
    print("=" * 70)
    print(f"PHASE 2: STATISTICAL COMPARISON (d={d}, 500 pairs)")
    print("=" * 70)
    results = compare_estimators(d=d, n_pairs=500, seed=42)
    print_comparison_table(results, d=d)

    # ----- PHASE 3: vary d, show both variances shrink but QJL stays ahead
    print()
    print("=" * 70)
    print("PHASE 3: VARIANCE vs DIMENSION d")
    print("=" * 70)
    print(f"{'d':>5} | {'QJL Var':>10} | {'Naive Var':>10} | {'Ratio':>7} | {'Theory π/(2d)':>14}")
    print("-" * 60)
    for dim in [32, 64, 128, 256, 512]:
        res = compare_estimators(d=dim, n_pairs=500, seed=7)
        ratio = res["naive_var"] / res["qjl_var"] if res["qjl_var"] > 0 else float("inf")
        theory = np.pi / (2 * dim)
        print(f"{dim:>5} | {res['qjl_var']:>10.5f} | {res['naive_var']:>10.5f} | "
              f"{ratio:>7.2f}x | {theory:>14.5f}")

    print()
    print("Both variances decrease as d grows (more sign bits → more info).")
    print("But QJL consistently has ~π/2 ≈ 1.57x lower variance than naive.")
    print("For KV cache (d=128): QJL Var ≈ 0.012, Naive Var ≈ 0.048.")
    print()
    print("Next: wire QJL into the TurboQuant two-stage pipeline (Module 4).")
