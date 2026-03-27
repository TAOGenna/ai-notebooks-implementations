"""
Exercise 1: Random Projections Preserve Inner Products — The JL Foundation
==========================================================================
CLAIM: A random Gaussian projection gives an UNBIASED estimate of the inner
       product with variance that shrinks as 1/d.

Before we can understand QJL (the 1-bit trick), we need to understand why
random Gaussian projections are special. The Johnson-Lindenstrauss Lemma says:

  If S is a (d x d) matrix with i.i.d. entries ~ N(0,1), then

      E[ <S*x, S*y> / d ]  =  <x, y>            (unbiased!)

      Var[ <S*x, S*y> / d ] = (||x||^2 * ||y||^2 + <x,y>^2) / d

This module verifies these properties empirically, setting the stage for QJL:
"If even 1 bit per dimension can approximately preserve these properties, we
 can compress KV cache keys to just 1 bit/coord with near-zero distortion."

Dependencies: None (this is the foundation — Module 0 concepts only)

Run:
    python ex01_jl_foundation.py
"""

import numpy as np


class JLTransform:
    """
    The standard (dense) Johnson-Lindenstrauss transform.

    Maps x in R^d  ->  S*x in R^m, where S ~ N(0,1)^{m x d}.

    The key property:
        <S*x, S*y> / m  is an unbiased estimator of  <x, y>
    with variance (||x||^2 * ||y||^2 + <x,y>^2) / m.

    When m = d (square), we project into the same dimension — no compression yet.
    The point here is to SEE the unbiasedness, then ask: can we compress to 1 bit?
    """

    def __init__(self, d: int, m: int, rng: np.random.Generator = None):
        """
        Parameters
        ----------
        d : int
            Input dimension.
        m : int
            Output (projection) dimension. When m == d: same dimension.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        """
        self.d = d
        self.m = m
        self.rng = rng or np.random.default_rng(42)
        self.S = None  # will be set in _draw_matrix()

    def _draw_matrix(self) -> np.ndarray:
        """
        Draw a fresh random Gaussian projection matrix.

        Each entry is i.i.d. N(0, 1). We do NOT normalise here;
        the 1/m scaling is applied during inner product estimation.

        Returns
        -------
        S : ndarray, shape (m, d)
        """
        # ====================================================================
        # TODO: Generate a random Gaussian matrix S of shape (m, d).      (~1 line)
        #
        # Hint: use self.rng.standard_normal(shape) for reproducibility.
        # ====================================================================
        raise NotImplementedError("Implement _draw_matrix")
        # ====================================================================

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project vector x using the current random matrix S.

        Parameters
        ----------
        x : ndarray, shape (d,)

        Returns
        -------
        Sx : ndarray, shape (m,)   — the projected vector S * x
        """
        # ====================================================================
        # TODO: Compute S @ x.                                              (~1 line)
        #
        # Make sure self.S is set (call _draw_matrix if needed).
        # ====================================================================
        raise NotImplementedError("Implement project")
        # ====================================================================

    def estimate_inner_product(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Estimate <x, y> using a FRESH random projection.

        The estimator is:
            ip_hat = <S*x, S*y> / m

        Expected value: <x, y>  (unbiased)
        Variance: (||x||^2 * ||y||^2 + <x,y>^2) / m

        Parameters
        ----------
        x, y : ndarray, shape (d,)

        Returns
        -------
        ip_hat : float
        """
        # ====================================================================
        # TODO: Draw a fresh S, compute S*x and S*y, return their dot       (~3 lines)
        #       product divided by m.
        #
        # Hint: call self._draw_matrix() to refresh S, then use self.project().
        # ====================================================================
        raise NotImplementedError("Implement estimate_inner_product")
        # ====================================================================


def theoretical_variance(x: np.ndarray, y: np.ndarray, m: int) -> float:
    """
    Theoretical variance of the JL inner product estimator.

        Var = (||x||^2 * ||y||^2 + <x,y>^2) / m

    Parameters
    ----------
    x, y : ndarray, shape (d,)
    m : int  — projection dimension

    Returns
    -------
    variance : float
    """
    # ========================================================================
    # TODO: Compute and return the theoretical variance.                   (~3 lines)
    #
    # Recall: ||v||^2 = np.dot(v, v)
    # ========================================================================
    raise NotImplementedError("Implement theoretical_variance")
    # ========================================================================


def run_jl_experiment(
    x: np.ndarray,
    y: np.ndarray,
    m: int,
    n_trials: int = 1000,
    seed: int = 0,
) -> dict:
    """
    Run n_trials independent JL estimates and collect statistics.

    For each trial, a NEW random matrix S is drawn (so each estimate
    is independent).

    Parameters
    ----------
    x, y    : ndarray, shape (d,)
    m       : int   — projection dimension
    n_trials: int   — number of independent trials
    seed    : int   — base random seed

    Returns
    -------
    results : dict with keys
        'estimates'   — array of n_trials estimates
        'mean'        — empirical mean of estimates
        'std'         — empirical std of estimates
        'true_ip'     — ground-truth <x, y>
        'theory_std'  — sqrt of theoretical variance
    """
    rng = np.random.default_rng(seed)
    jl = JLTransform(d=len(x), m=m, rng=rng)

    estimates = np.array([
        jl.estimate_inner_product(x, y) for _ in range(n_trials)
    ])

    true_ip = float(np.dot(x, y))
    theory_var = theoretical_variance(x, y, m)

    return {
        "estimates":   estimates,
        "mean":        float(np.mean(estimates)),
        "std":         float(np.std(estimates)),
        "true_ip":     true_ip,
        "theory_std":  float(np.sqrt(theory_var)),
    }


# ============================================================================
# MILESTONE — run this file to see the JL unbiasedness property
# ============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # ----- set up two realistic vectors (mimicking attention head embeddings)
    d = 128
    x = rng.standard_normal(d);  x /= np.linalg.norm(x)   # unit-norm key
    y = rng.standard_normal(d);  y /= np.linalg.norm(y)   # unit-norm query

    true_ip = float(np.dot(x, y))
    print(f"True inner product <x, y> = {true_ip:.4f}")
    print()

    # ----- experiment 1: vary projection dimension m
    print("=" * 60)
    print("JL INNER PRODUCT ESTIMATES vs PROJECTION DIMENSION")
    print("=" * 60)
    print(f"{'m':>6} | {'Mean':>8} | {'Emp Std':>8} | {'Theory Std':>10} | {'Bias':>7}")
    print("-" * 60)

    for m in [8, 16, 32, 64, 128]:
        res = run_jl_experiment(x, y, m=m, n_trials=2000, seed=1)
        bias = res["mean"] - res["true_ip"]
        print(f"{m:>6} | {res['mean']:>8.4f} | {res['std']:>8.4f} | "
              f"{res['theory_std']:>10.4f} | {bias:>7.4f}")

    print()
    print("Observation: empirical std ≈ theoretical std at all m.")
    print(f"             mean ≈ true ({true_ip:.4f}) regardless of m  → UNBIASED ✓")
    print()

    # ----- experiment 2: show unbiasedness at m=d (full dimension)
    m = d
    res = run_jl_experiment(x, y, m=m, n_trials=1000, seed=2)
    unbiased = abs(res["mean"] - res["true_ip"]) < 3 * res["theory_std"] / np.sqrt(1000)

    print("=" * 60)
    print(f"UNBIASEDNESS CHECK (m=d={d}, 1000 trials)")
    print("=" * 60)
    print(f"  True inner product : {res['true_ip']:.4f}")
    print(f"  JL mean estimate   : {res['mean']:.4f}")
    print(f"  Empirical std      : {res['std']:.4f}")
    print(f"  Theoretical std    : {res['theory_std']:.4f}")
    print(f"  Unbiased?          : {'✓' if unbiased else '✗'}")
    print()
    print("Key question: can we take this projection and reduce each")
    print("coordinate to just 1 BIT while keeping the estimator unbiased?")
    print("→ That is exactly what QJL does (Exercise 2).")
