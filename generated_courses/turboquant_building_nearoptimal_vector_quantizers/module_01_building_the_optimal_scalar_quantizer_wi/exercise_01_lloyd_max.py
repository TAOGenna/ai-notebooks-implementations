"""
Exercise 1: The Continuous k-means Problem — Finding Optimal Centroids
=======================================================================

CLAIM: The Lloyd-Max algorithm finds the optimal quantizer for any known distribution.

Background
----------
In TurboQuant, before quantizing a vector x ∈ R^d, we multiply it by a random
orthogonal matrix Π. This "random rotation" places the result uniformly on the unit
hypersphere S^{d-1}. A key consequence (proved in Module 0):

    Each coordinate y_i = (Πx)_i follows the Beta-like distribution:

        f_X(t) = C_d · (1 - t²)^{(d−3)/2},    t ∈ [−1, 1]

    where  C_d = Γ(d/2) / (√π · Γ((d−1)/2)).

Because we KNOW this distribution analytically, we don't need data to design the
quantizer. We just need to solve: given f_X and k = 2^b desired levels, find the
k centroids {c_l} that minimize E[(X − Q(X))²].

This is exactly the continuous 1D k-means problem, solved by Lloyd-Max iteration:
    Repeat until convergence:
        (1) Boundaries:   t_l = (c_{l−1} + c_l) / 2  (midpoints — Voronoi condition)
        (2) Centroids:    c_l = E[X | t_{l−1} < X ≤ t_l]  (conditional mean)

Your task: implement step (2) — the centroid update — using numerical integration.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma
from scipy.stats import norm as scipy_norm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Beta coordinate distribution
# (Reproduced from Module 0; see module_00 for derivation details)
# ---------------------------------------------------------------------------

def beta_pdf(x, d):
    """
    PDF of a single coordinate of a uniformly random unit vector in R^d.

    After multiplying any unit-norm vector by a random orthogonal matrix,
    each coordinate independently (approximately) follows:

        f_X(t) = C_d · (1 − t²)^{(d−3)/2},   t ∈ (−1, 1)

    with normalisation constant:

        C_d = Γ(d/2) / ( √π · Γ((d−1)/2) )

    For large d this converges to N(0, 1/d). The coordinate variance is
    exactly Var[X] = 1/d for all d.

    Args:
        x  : float or np.ndarray — the coordinate value(s)
        d  : int — ambient dimension (must be ≥ 3)

    Returns:
        float or np.ndarray — pdf value(s) at x; 0 outside (−1, 1)
    """
    x = np.asarray(x, dtype=float)
    # Support is the open interval (−1, 1)
    out = np.where(
        np.abs(x) < 1.0,
        gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
        * (1 - x ** 2) ** ((d - 3) / 2),
        0.0,
    )
    return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# Lloyd-Max Quantizer
# ---------------------------------------------------------------------------

class LloydMaxQuantizer:
    """
    Optimal scalar quantizer via the Lloyd-Max algorithm (continuous 1D k-means).

    Given the Beta coordinate distribution f_X for dimension d, finds the
    k = 2^b centroids {c_0, …, c_{k−1}} that minimise

        MSE = E_X[ min_l (X − c_l)² ]

    The algorithm alternates two steps until convergence:
        (1) Voronoi boundaries: t_l = midpoint(c_{l−1}, c_l)
        (2) Centroid update:    c_l = E[X | t_{l−1} < X ≤ t_l]  ← you implement this

    Because f_X is known analytically, both steps use exact numerical integration
    (no sampling). This makes the solution essentially exact for any bit-width b.

    Usage
    -----
    >>> q = LloydMaxQuantizer(d=128, n_bits=2)
    >>> centroids, mse = q.fit()
    """

    def __init__(self, d: int, n_bits: int, n_iter: int = 300, tol: float = 1e-12):
        """
        Args:
            d       : ambient dimension (sets the coordinate distribution)
            n_bits  : bits per coordinate b; number of centroids k = 2^b
            n_iter  : maximum Lloyd-Max iterations (200–300 is ample for d ≥ 16)
            tol     : convergence threshold — stop when max centroid shift < tol
        """
        self.d = d
        self.n_bits = n_bits
        self.n_centroids = 2 ** n_bits
        self.n_iter = n_iter
        self.tol = tol
        self.centroids = None   # set by fit()
        self.boundaries = None  # set by fit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def pdf(self, x: float) -> float:
        """Evaluate the Beta coordinate distribution pdf at scalar x."""
        return beta_pdf(x, self.d)

    def _initialize_centroids(self) -> np.ndarray:
        """
        Symmetric initial centroids at quantiles of N(0, 1/d).

        The Beta distribution for d ≥ 32 is very close to N(0, 1/d).
        Initialising at equal-probability quantiles of this Gaussian gives:
          (a) Symmetric placement (preserving the global-optimum guarantee)
          (b) Centroids concentrated where f_X has mass, avoiding empty cells
              at high bit-widths (which plague naive uniform [-1,1] spacing)

        Without this, naive uniform initialisation places centroids at ±6σ+
        for b=5,6, where f_X ≈ 0, causing the outer cells to degenerate and
        Lloyd-Max to converge to a suboptimal local solution.
        """
        k = self.n_centroids
        sigma = 1.0 / np.sqrt(self.d)
        # Equal-probability quantiles of N(0, sigma^2), clipped to the support
        probs = np.linspace(1.0 / (k + 1), k / (k + 1), k)
        return np.clip(sigma * scipy_norm.ppf(probs), -0.9999, 0.9999)

    def _compute_boundaries(self, centroids: np.ndarray) -> np.ndarray:
        """
        Voronoi boundaries: midpoints between consecutive centroids.

        The optimal 1D decision boundary between centroids c_l and c_{l+1}
        is simply their arithmetic mean. The leftmost boundary is −1 and
        the rightmost is +1 (the full support of f_X).

        Args:
            centroids : shape (k,), sorted centroid positions

        Returns:
            boundaries : shape (k+1,)
                         boundaries[0] = −1  (left edge of support)
                         boundaries[l] = (centroids[l−1] + centroids[l]) / 2
                         boundaries[k] = +1  (right edge of support)
        """
        midpoints = (centroids[:-1] + centroids[1:]) / 2.0
        return np.concatenate([[-1.0], midpoints, [1.0]])

    def _update_centroids(self, boundaries: np.ndarray) -> np.ndarray:
        """
        Update each centroid to the conditional mean within its Voronoi region.

        The optimal centroid for Voronoi region l = [t_{l−1}, t_l] is:

            c_l  =  E[ X | t_{l−1} < X ≤ t_l ]

                      ∫_{t_{l−1}}^{t_l}  x · f_X(x) dx
                 =   ─────────────────────────────────────
                      ∫_{t_{l−1}}^{t_l}      f_X(x) dx

        Both integrals are computed numerically with scipy.integrate.quad.

        Symbol guide (matches the math above):
            t_{l−1}    = boundaries[l]       left edge of region l
            t_l        = boundaries[l + 1]   right edge of region l
            f_X        = self.pdf            Beta coordinate pdf
            c_l        = new_centroids[l]    updated centroid

        Args:
            boundaries : shape (k+1,) — from _compute_boundaries.
                         boundaries[0] = −1,  boundaries[k] = +1.

        Returns:
            new_centroids : shape (k,), updated centroid positions.

        Hint
        ----
        scipy.integrate.quad(func, a, b) → (value, error_estimate)

        Edge case: if the denominator is nearly zero (the region has negligible
        probability mass), fall back to the region midpoint (lo + hi) / 2 to
        avoid NaN. This can happen at high bit-widths near the tails of f_X.
        """
        # ====================================================================
        # TODO: Compute the conditional mean for each Voronoi region (~8 lines)
        #
        # For l in range(self.n_centroids):
        #   lo  = boundaries[l]
        #   hi  = boundaries[l + 1]
        #   numerator   = ∫_{lo}^{hi}  x · self.pdf(x) dx   (use quad)
        #   denominator = ∫_{lo}^{hi}      self.pdf(x) dx   (use quad)
        #   new_centroids[l] = numerator / denominator
        #                      (or (lo + hi) / 2 if denominator ≈ 0)
        # ====================================================================
        raise NotImplementedError("Implement _update_centroids — see TODO above")
        # ====================================================================

    def _compute_mse_cost(
        self,
        centroids: np.ndarray,
        boundaries: np.ndarray,
    ) -> float:
        """
        Expected squared quantisation error (MSE per coordinate).

        MSE = E[(X − Q(X))²]
            = Σ_l  ∫_{t_{l−1}}^{t_l} (x − c_l)² · f_X(x) dx

        Args:
            centroids  : shape (k,)
            boundaries : shape (k+1,)

        Returns:
            float, per-coordinate MSE
        """
        total = 0.0
        for l in range(self.n_centroids):
            lo, hi = boundaries[l], boundaries[l + 1]
            c_l = centroids[l]
            region_mse, _ = quad(
                lambda x, c=c_l: (x - c) ** 2 * self.pdf(x),
                lo, hi,
                limit=100,
            )
            total += region_mse
        return total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self):
        """
        Run Lloyd-Max to convergence and return the optimal codebook.

        Returns
        -------
        centroids : np.ndarray, shape (2^b,)
            Optimal centroid positions, sorted ascending.
        mse : float
            Per-coordinate MSE achieved by this codebook.
        """
        centroids = self._initialize_centroids()

        for iteration in range(self.n_iter):
            boundaries = self._compute_boundaries(centroids)
            new_centroids = self._update_centroids(boundaries)

            max_shift = np.max(np.abs(new_centroids - centroids))
            centroids = new_centroids

            if max_shift < self.tol:
                # Converged
                break

        self.centroids = centroids
        self.boundaries = self._compute_boundaries(centroids)
        mse = self._compute_mse_cost(self.centroids, self.boundaries)
        return self.centroids, mse


# ---------------------------------------------------------------------------
# Milestone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Milestone: Optimal codebooks and MSE costs for b = 1, 2, 3, 4 at d = 128.

    Expected output (once you implement _update_centroids):
        b=1  centroids: [-0.0707,  0.0707]
        b=2  centroids: [-0.1174, -0.0351,  0.0351,  0.1174]
        ...
        b   per-coord MSE      total (×d)    ratio b-1→b
        1   2.82e-03           0.361         —
        2   9.06e-04           0.116         3.11×
        3   2.65e-04           0.034         3.41×
        4   7.28e-05           0.009         3.74×

    These values match the exact Beta(d=128) coordinate distribution.
    Note: the Gaussian approximation N(0,1/d) gives nearly identical results
    (within 2%), confirming that d=128 is well into the high-dimension regime.

    Notice that each extra bit reduces MSE by ~3.1–3.7×, converging toward the
    theoretical factor of 4× as b increases — the hallmark of the 1/4^b scaling
    law you will verify and explain in Exercise 2.
    """
    import sys

    d = 128
    print("=" * 64)
    print(f"  Lloyd-Max Optimal Codebook  |  d = {d}")
    print("=" * 64)

    results = {}
    for b in [1, 2, 3, 4]:
        q = LloydMaxQuantizer(d=d, n_bits=b)
        try:
            centroids, mse = q.fit()
        except NotImplementedError:
            print(f"\n[b={b}] *** Implement _update_centroids first! ***")
            sys.exit(1)

        results[b] = (centroids, mse)

        # Format centroids compactly
        c_str = "  ".join(f"{c:+.4f}" for c in centroids)
        print(f"\n  b={b}  ({2**b:2d} centroids):")
        print(f"    centroids: [{c_str}]")

    print()
    print(f"  {'b':>4}  {'per-coord MSE':>16}  {'total MSE (×d)':>16}  {'ratio to b-1':>14}")
    print("  " + "-" * 58)
    prev_total = None
    for b in [1, 2, 3, 4]:
        _, mse = results[b]
        total = d * mse
        ratio_str = f"{prev_total / total:.2f}×" if prev_total is not None else "—"
        print(f"  {b:>4}  {mse:>16.4e}  {total:>16.4f}  {ratio_str:>14}")
        prev_total = total

    print()
    print("  Insight: Each extra bit roughly divides the MSE by ~4 (1/4^b law).")
    print("  You will derive and verify this scaling in Exercise 2.")
    print("=" * 64)
