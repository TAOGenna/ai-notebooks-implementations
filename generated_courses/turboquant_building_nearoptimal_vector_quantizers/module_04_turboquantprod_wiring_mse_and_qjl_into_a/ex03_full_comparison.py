"""
Exercise 3: The Full Picture — TurboQuant_mse vs TurboQuant_prod vs Lower Bounds
=================================================================================

This is a **comparative** exercise. You will run both quantizers, compute their
inner product distortions at bit-widths 1–5, plot all four curves:

  1. TurboQuant_prod   — our new unbiased quantizer (blue)
  2. TurboQuant_mse    — the MSE quantizer from Module 2 (orange)
  3. Upper bound       — D_prod ≤ sqrt(3)·π²/(d·4^b)  (red dashed)
  4. Lower bound       — D ≥ 1/(d·4^b)                (green dashed)

The gap between TurboQuant_prod and the lower bound is the key result:
at most ~2.7x, confirming near-optimality.

Note on TurboQuant_mse distortion as an INNER PRODUCT estimator:
  We measure E[(ip_hat - ip_true)^2], which includes both bias^2 AND variance.
  At low b, the multiplicative bias (2/pi ≈ 0.637 at b=1) dominates; at high b,
  the quantization variance dominates. TurboQuant_prod removes the bias term.

Prerequisite: Exercises 1 and 2 of this module.
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_COURSE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_HERE = os.path.dirname(__file__)

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

sys.path.insert(0, _HERE)

try:
    from ex01_assemble_turboquant_mse import TurboQuantMSE
except ImportError:
    from ex01_residual_analysis import TurboQuantMSE

from ex02_turboquant_prod import TurboQuantProd, ProdCode, sample_unit_vectors


# ---------------------------------------------------------------------------
# Helper: measure inner product distortion for TurboQuant_mse
# ---------------------------------------------------------------------------
def mse_inner_product_distortion(d: int, b: int,
                                  xs: np.ndarray,
                                  ys: np.ndarray,
                                  seed: int = 0) -> float:
    """
    Measure E[(ip_hat - ip_true)^2] for TurboQuantMSE as an inner product estimator.

    TurboQuant_mse was designed to minimise ||x - x̂||^2, not inner product error.
    At 1-bit, it applies a multiplicative factor of 2/pi ≈ 0.637 to every inner
    product (Section 3.1 of the paper). This function measures the total inner
    product mean-squared error, which includes both bias^2 and variance.

    Parameters
    ----------
    d, b : int       — dimension and bit-width
    xs : (N, d)      — key vectors to quantize
    ys : (N, d)      — query vectors (full precision)
    seed : int

    Returns
    -------
    distortion : float
        Mean squared inner product error averaged over all (x, y) pairs.
    """
    # =========================================================================
    # TODO: Implement MSE inner product distortion measurement.  (~8 lines)
    #
    # Steps:
    #   1. Create a TurboQuantMSE(d=d, b=b, seed=seed) instance.
    #   2. For each pair (xs[i], ys[i]):
    #        a. quantize xs[i]:   indices = tq.quantize_single(xs[i])
    #        b. reconstruct:      x_hat   = tq.dequantize_single(indices)
    #        c. estimate ip:      ip_hat  = np.dot(ys[i], x_hat)
    #        d. true ip:          ip_true = np.dot(ys[i], xs[i])
    #        e. squared error:    (ip_hat - ip_true)^2
    #   3. Return the mean of the squared errors.
    #
    # Hint: use a Python list to collect squared errors, then np.mean().
    # =========================================================================
    raise NotImplementedError("Implement mse_inner_product_distortion")
    # =========================================================================


def prod_inner_product_distortion(d: int, b: int,
                                   xs: np.ndarray,
                                   ys: np.ndarray,
                                   seed: int = 0) -> float:
    """
    Measure E[(ip_hat - ip_true)^2] for TurboQuantProd.

    Uses TurboQuantProd.inner_product() from Exercise 2.

    Parameters
    ----------
    d, b : int       — dimension and bit-width
    xs : (N, d)      — key vectors
    ys : (N, d)      — query vectors (full precision)
    seed : int

    Returns
    -------
    distortion : float
    """
    # =========================================================================
    # TODO: Implement TurboQuant_prod inner product distortion.  (~8 lines)
    #
    # Steps:
    #   1. Create TurboQuantProd(d=d, b=b, seed=seed).
    #   2. For each pair (xs[i], ys[i]):
    #        a. code     = tq.quantize(xs[i])
    #        b. ip_hat   = tq.inner_product(code, ys[i])
    #        c. ip_true  = np.dot(ys[i], xs[i])
    #        d. squared error: (ip_hat - ip_true)^2
    #   3. Return the mean squared error.
    # =========================================================================
    raise NotImplementedError("Implement prod_inner_product_distortion")
    # =========================================================================


def upper_bound(d: int, b: int, y_norm_sq: float = 1.0) -> float:
    """
    TurboQuant_prod theoretical upper bound (Theorem 4.1):
        D_prod ≤ (sqrt(3) * pi^2 * ||y||^2) / (d * 4^b)

    Parameters
    ----------
    d : int
    b : int
    y_norm_sq : float    — ||y||^2, default 1.0 (unit-norm queries)
    """
    # =========================================================================
    # TODO: Implement the upper bound formula.  (~1 line)
    # =========================================================================
    raise NotImplementedError("Implement upper_bound")
    # =========================================================================


def lower_bound(d: int, b: int, y_norm_sq: float = 1.0) -> float:
    """
    Information-theoretic lower bound (Shannon Lower Bound + Yao minimax):
        D ≥ ||y||^2 / (d * 4^b)

    This is the best ANY quantizer can do in the worst case.
    TurboQuant_prod achieves at most sqrt(3)*pi^2 ≈ 17.3 times this —
    but empirically the ratio is only ~2.7x.

    Parameters
    ----------
    d : int
    b : int
    y_norm_sq : float
    """
    # =========================================================================
    # TODO: Implement the lower bound formula.  (~1 line)
    # =========================================================================
    raise NotImplementedError("Implement lower_bound")
    # =========================================================================


def make_comparison_plot(bit_widths, mse_dists, prod_dists,
                          upper_bounds, lower_bounds, d: int,
                          save_path: str) -> None:
    """
    Create a publication-quality log-scale comparison plot.

    Four curves:
      • TurboQuant_prod (blue, solid, circles)      — unbiased
      • TurboQuant_mse (orange, solid, squares)     — biased at low b
      • Upper bound (red, dashed, no markers)        — theory guarantee
      • Lower bound (green, dashed, no markers)      — info-theory limit

    Parameters
    ----------
    bit_widths : list[int]
    mse_dists : list[float]    — empirical D for TurboQuant_mse
    prod_dists : list[float]   — empirical D for TurboQuant_prod
    upper_bounds : list[float]
    lower_bounds : list[float]
    d : int                    — for title annotation
    save_path : str            — file path to save the PNG
    """
    # =========================================================================
    # TODO: Implement the comparison plot.  (~18 lines)
    #
    # Steps:
    #   1. import matplotlib.pyplot as plt
    #   2. fig, ax = plt.subplots(figsize=(8, 5))
    #   3. Plot each of the 4 curves using ax.semilogy() (log-scale on y-axis):
    #        ax.semilogy(bit_widths, prod_dists,   'b-o',  label='TurboQuant_prod (unbiased)')
    #        ax.semilogy(bit_widths, mse_dists,    'o-',   color='orange', label='TurboQuant_mse')
    #        ax.semilogy(bit_widths, upper_bounds, 'r--',  label='Upper bound: sqrt(3)π²/(d·4^b)')
    #        ax.semilogy(bit_widths, lower_bounds, 'g--',  label='Lower bound: 1/(d·4^b)')
    #   4. Labels:
    #        ax.set_xlabel('Bit-width b (bits per coordinate)')
    #        ax.set_ylabel('Inner product distortion E[(ip_hat - ip_true)²]')
    #        ax.set_title(f'TurboQuant Inner Product Distortion vs Bounds (d={d})')
    #   5. ax.legend(), ax.grid(True, which='both', alpha=0.3)
    #   6. plt.tight_layout()
    #   7. plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #   8. plt.close()
    # =========================================================================
    raise NotImplementedError("Implement make_comparison_plot")
    # =========================================================================


# ---------------------------------------------------------------------------
# Main milestone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 3: TurboQuant_mse vs TurboQuant_prod vs Bounds")
    print("=" * 70)

    D = 128
    N = 800     # vector pairs per bit-width
    BIT_WIDTHS = [1, 2, 3, 4, 5]
    rng = np.random.default_rng(99)

    xs = sample_unit_vectors(N, D, rng)
    ys = sample_unit_vectors(N, D, rng)

    print(f"\nVectors: {N} unit-norm pairs, d={D}")
    print("Running both quantizers at each bit-width …\n")

    mse_dists = []
    prod_dists = []
    ubs = []
    lbs = []

    print(f"{'b':>3} | {'TQ_mse':>10} | {'TQ_prod':>10} | "
          f"{'Upper bnd':>12} | {'Lower bnd':>12} | {'prod/lb':>8}")
    print("-" * 68)

    for b in BIT_WIDTHS:
        d_mse = mse_inner_product_distortion(D, b, xs, ys)
        d_prod = prod_inner_product_distortion(D, b, xs, ys)
        ub = upper_bound(D, b)
        lb = lower_bound(D, b)

        mse_dists.append(d_mse)
        prod_dists.append(d_prod)
        ubs.append(ub)
        lbs.append(lb)

        ratio = d_prod / lb if lb > 0 else float('nan')
        print(f"{b:>3} | {d_mse:>10.6f} | {d_prod:>10.6f} | "
              f"{ub:>12.6f} | {lb:>12.6f} | {ratio:>8.2f}x")

    print()
    # Key ratios highlighted in the paper
    if len(prod_dists) >= 2 and len(lbs) >= 2:
        ratio_b2 = prod_dists[1] / lbs[1]   # b=2
        ratio_b4 = prod_dists[3] / lbs[3]   # b=4 (index 3 since b starts at 1)
        print(f"At b=2: TurboQuant_prod distortion / lower_bound = {ratio_b2:.2f}x")
        print(f"At b=4: TurboQuant_prod distortion / lower_bound = {ratio_b4:.2f}x")
        print(f"Paper claims ≤ 2.7x asymptotically — {'confirmed ✓' if ratio_b4 <= 3.0 else 'check implementation'}")

    # Save plot
    plot_path = os.path.join(os.path.dirname(__file__),
                             "milestone_03_full_comparison.png")
    print(f"\nSaving plot → {plot_path}")
    try:
        make_comparison_plot(BIT_WIDTHS, mse_dists, prod_dists,
                             ubs, lbs, D, plot_path)
        print("Plot saved ✓")
    except Exception as e:
        print(f"Plot failed: {e}")

    print()
    print("Summary of findings:")
    print("  • TurboQuant_prod tracks the lower bound within ~2-4x at all b.")
    print("  • TurboQuant_mse is biased at low b → higher distortion despite")
    print("    lower quantization noise (bias^2 dominates).")
    print("  • Both decay as 4^{-b} per additional bit — the theoretical rate.")
    print("  • Adding just 1 extra bit of MSE reduces distortion ~4x — impressive!")
