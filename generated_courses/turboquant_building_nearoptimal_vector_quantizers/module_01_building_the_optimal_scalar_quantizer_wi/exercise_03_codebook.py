"""
Exercise 3: From Centroids to Codebook — Building the Reusable Quantisation Table
===================================================================================

CLAIM: A precomputed codebook maps any coordinate to its nearest centroid in O(log k)
       time, achieving exactly the Lloyd-Max-optimal distortion on real data.

Background
----------
In Exercises 1 and 2 we solved for optimal centroids analytically.  Now we package
them into a practical data structure — the Codebook — that the full TurboQuant
pipeline (Module 3) will call millions of times per batch.

A Codebook stores:
  • centroids   — the k = 2^b representative values, sorted ascending
  • boundaries  — the k+1 Voronoi boundaries (midpoints between centroids)

It exposes two operations:
  • quantize(x)    : x ∈ ℝ  →  index ∈ {0, …, k−1}   (encode)
  • dequantize(idx): index  →  centroid value ∈ ℝ      (decode)

Key design choice in TurboQuant: the codebook is computed ONCE offline from the
known Beta distribution, then REUSED for every vector and every dataset.
This is the data-oblivious property — contrast with Product Quantisation, which
requires fitting a new codebook per dataset.

Your task
---------
Implement the four methods:
  1. quantize_scalar   — map one value to its nearest centroid index  (~4 lines)
  2. dequantize_scalar — map one index back to its centroid value      (~2 lines)
  3. quantize_array    — vectorised version for a NumPy array          (~4 lines)
  4. dequantize_array  — vectorised lookup for an array of indices     (~2 lines)

Then the milestone runs a round-trip test: quantize 10 000 Beta samples, dequantize,
and check that the average squared error matches the Lloyd-Max cost from Exercise 1.

Dependencies: numpy, scipy
Run from the module directory:  python exercise_03_codebook.py
"""

import numpy as np
from scipy.special import gamma
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from exercise_01_lloyd_max import LloydMaxQuantizer, beta_pdf


# ---------------------------------------------------------------------------
# Codebook
# ---------------------------------------------------------------------------

class Codebook:
    """
    Precomputed optimal scalar codebook for TurboQuant.

    Wraps the Lloyd-Max centroid solution into a fast encode/decode interface.
    The codebook is distribution-specific (Beta with parameter d) but completely
    data-oblivious — no dataset is needed to construct it.

    Attributes
    ----------
    d          : int  — ambient dimension
    n_bits     : int  — bits per coordinate (b)
    n_centroids: int  — k = 2^b
    centroids  : np.ndarray, shape (k,)   — sorted centroid values
    boundaries : np.ndarray, shape (k+1,) — Voronoi boundaries including ±1
    mse        : float — per-coordinate MSE achieved by this codebook
    """

    def __init__(self, d: int, n_bits: int):
        """
        Build the optimal codebook by running Lloyd-Max to convergence.

        Args:
            d      : ambient dimension (e.g. 128)
            n_bits : bits per coordinate (e.g. 1, 2, 3, or 4)
        """
        self.d = d
        self.n_bits = n_bits
        self.n_centroids = 2 ** n_bits

        # Solve the continuous k-means problem using Lloyd-Max (Exercise 1)
        q = LloydMaxQuantizer(d=d, n_bits=n_bits)
        self.centroids, self.mse = q.fit()          # centroids are sorted ascending
        self.boundaries = q.boundaries              # shape (k+1,)

    # ------------------------------------------------------------------
    # Scalar encode / decode
    # ------------------------------------------------------------------

    def quantize_scalar(self, x: float) -> int:
        """
        Map a single real coordinate to the index of its nearest centroid.

        The Voronoi region for centroid l is the interval
            [boundaries[l],  boundaries[l+1])
        so we binary-search x into the boundaries array.

        Args:
            x : float — a coordinate value (ideally in (−1, 1) for unit vectors)

        Returns:
            int — centroid index in {0, 1, …, k−1}

        Hint: np.searchsorted(sorted_array, value, side='right') returns the
        insertion point of value in sorted_array.  The centroid index is then
        that position minus 1, clipped to [0, k−1].
        """
        # ====================================================================
        # TODO: Binary-search x into self.boundaries to find its Voronoi region.
        #       Return the centroid index, clipped to valid range. (~4 lines)
        #
        # idx = np.searchsorted(self.boundaries, x, side='right') - 1
        # idx = int(np.clip(idx, 0, self.n_centroids - 1))
        # return idx
        # ====================================================================
        raise NotImplementedError("Implement quantize_scalar — see TODO above")
        # ====================================================================

    def dequantize_scalar(self, idx: int) -> float:
        """
        Map a centroid index back to its representative value.

        Args:
            idx : int — centroid index in {0, …, k−1}

        Returns:
            float — centroid value (the reconstruction)

        Hint: it's a single array lookup.
        """
        # ====================================================================
        # TODO: Return the centroid at position idx. (~2 lines)
        # ====================================================================
        raise NotImplementedError("Implement dequantize_scalar — see TODO above")
        # ====================================================================

    # ------------------------------------------------------------------
    # Vectorised encode / decode  (used in the full TurboQuant pipeline)
    # ------------------------------------------------------------------

    def quantize_array(self, xs: np.ndarray) -> np.ndarray:
        """
        Vectorised quantisation: map an array of coordinates to centroid indices.

        This is called once per vector in TurboQuant (on the d rotated coordinates),
        so it must be efficient. Use np.searchsorted with the full array at once.

        Args:
            xs : np.ndarray, any shape — coordinate values

        Returns:
            np.ndarray of int32, same shape as xs — centroid indices

        Hint: np.searchsorted works on entire arrays in one call.
              Remember to subtract 1 and clip to [0, k−1].
        """
        # ====================================================================
        # TODO: Vectorised version — same logic as quantize_scalar. (~4 lines)
        #
        # indices = np.searchsorted(self.boundaries, xs, side='right') - 1
        # indices = np.clip(indices, 0, self.n_centroids - 1)
        # return indices.astype(np.int32)
        # ====================================================================
        raise NotImplementedError("Implement quantize_array — see TODO above")
        # ====================================================================

    def dequantize_array(self, indices: np.ndarray) -> np.ndarray:
        """
        Vectorised dequantisation: map an array of centroid indices to values.

        Args:
            indices : np.ndarray of int, any shape — centroid indices

        Returns:
            np.ndarray of float64, same shape — reconstructed coordinate values

        Hint: fancy indexing — self.centroids[indices].
        """
        # ====================================================================
        # TODO: Return reconstructed values via array indexing. (~2 lines)
        # ====================================================================
        raise NotImplementedError("Implement dequantize_array — see TODO above")
        # ====================================================================

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def centroid_probabilities(self) -> np.ndarray:
        """
        Compute p_l = Pr[X in Voronoi region l] for each centroid.

        These probabilities are needed for entropy calculation (see inline
        question 2 in the README) and tell us how "unequal" the centroid
        usage is — a measure of potential gains from entropy coding.

        Returns:
            np.ndarray, shape (k,) — probabilities summing to ~1
        """
        from scipy.integrate import quad
        probs = np.zeros(self.n_centroids)
        for l in range(self.n_centroids):
            lo, hi = self.boundaries[l], self.boundaries[l + 1]
            p, _ = quad(lambda x: beta_pdf(x, self.d), lo, hi)
            probs[l] = p
        return probs

    def __repr__(self) -> str:
        return (
            f"Codebook(d={self.d}, b={self.n_bits}, "
            f"k={self.n_centroids}, mse={self.mse:.4e})"
        )


# ---------------------------------------------------------------------------
# Helper: draw samples from the Beta coordinate distribution
# ---------------------------------------------------------------------------

def sample_beta_coordinates(d: int, n_samples: int, rng=None) -> np.ndarray:
    """
    Draw i.i.d. samples from the Beta coordinate distribution for dimension d.

    Equivalent to generating a random unit vector in R^d and taking its first
    coordinate, repeated n_samples times.

    Method: sample a d-dimensional Gaussian, normalise to unit norm,
    take the first coordinate.  This is the same construction as Module 0.

    Args:
        d        : int — ambient dimension
        n_samples: int — number of coordinate samples
        rng      : np.random.Generator or None

    Returns:
        np.ndarray, shape (n_samples,) — samples in (−1, 1)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    # Sample d-dimensional Gaussian vectors, normalise, take coordinate 0
    G = rng.standard_normal((n_samples, d))
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    unit_vecs = G / norms
    return unit_vecs[:, 0]


# ---------------------------------------------------------------------------
# Milestone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Milestone: Round-trip accuracy test on 10 000 Beta-distributed samples.

    Expected output (once all TODOs are implemented):

        Codebook round-trip accuracy test
        d=128, n_samples=10,000
        ────────────────────────────────────────────────────────────
        b    │  empirical MSE     │  Lloyd-Max MSE   │  match?
        ────────────────────────────────────────────────────────────
        1    │  2.83e-03          │  2.82e-03        │  ✓  (< 5%)
        2    │  9.11e-04          │  9.06e-04        │  ✓  (< 5%)
        3    │  2.66e-04          │  2.65e-04        │  ✓  (< 5%)
        4    │  7.32e-05          │  7.28e-05        │  ✓  (< 5%)
        ──────────────────────────────────────────────────────────────
        All b: empirical ≈ theoretical (within Monte Carlo noise).
        The codebook is correct!

    The empirical MSE is computed by quantizing actual Beta-distributed samples
    and measuring the average squared error |x − dequant(quant(x))|².
    It should match the analytical Lloyd-Max cost within a few percent
    (Monte Carlo noise at n=10,000).
    """
    import sys

    d = 128
    N_SAMPLES = 10_000
    rng = np.random.default_rng(seed=2024)

    print(f"\nCodebook round-trip accuracy test")
    print(f"d={d}, n_samples={N_SAMPLES:,}")
    print("─" * 60)
    print(f"{'b':<5}│  {'empirical MSE':<18}│  {'Lloyd-Max MSE':<18}│  match?")
    print("─" * 60)

    all_ok = True
    for b in [1, 2, 3, 4]:
        try:
            cb = Codebook(d=d, n_bits=b)
        except NotImplementedError:
            print("*** Implement _update_centroids in exercise_01_lloyd_max.py first! ***")
            sys.exit(1)

        # Draw Beta-distributed coordinate samples
        samples = sample_beta_coordinates(d, N_SAMPLES, rng=rng)

        # Round-trip: quantise then dequantise
        try:
            indices = cb.quantize_array(samples)
            reconstructed = cb.dequantize_array(indices)
        except NotImplementedError:
            print(f"*** Implement quantize_array / dequantize_array for b={b}! ***")
            sys.exit(1)

        empirical_mse = float(np.mean((samples - reconstructed) ** 2))
        theoretical_mse = cb.mse
        rel_error = abs(empirical_mse - theoretical_mse) / theoretical_mse

        ok = rel_error < 0.05  # within 5% is fine for n=10,000
        all_ok = all_ok and ok
        status = "✓" if ok else f"✗ ({rel_error*100:.1f}% off)"
        print(
            f"{b:<5}│  {empirical_mse:<18.4e}│  {theoretical_mse:<18.4e}│  {status}"
        )

    print("─" * 60)
    if all_ok:
        print("All b: empirical ≈ theoretical (within Monte Carlo noise).")
        print("The codebook is correct!")
    else:
        print("Some bit-widths deviated > 5% — check your implementation.")

    # ------------------------------------------------------------------
    # Bonus: entropy of each codebook
    # ------------------------------------------------------------------
    print("\nBonus — Entropy of codebook indices (bits of entropy vs. uniform coding):")
    print(f"  {'b':<4}  {'entropy H':<12}  {'uniform bits':<14}  {'savings':<10}")
    print("  " + "-" * 46)
    for b in [1, 2, 3, 4]:
        cb = Codebook(d=d, n_bits=b)
        probs = cb.centroid_probabilities()
        # Shannon entropy H = −Σ p_l · log2(p_l)
        probs_nonzero = probs[probs > 1e-15]
        H = -np.sum(probs_nonzero * np.log2(probs_nonzero))
        savings_pct = (b - H) / b * 100
        print(f"  {b:<4}  {H:<12.4f}  {b:<14}  {savings_pct:.1f}%")
    print("\n  Insight: at b=4 the non-uniform centroid distribution allows ~5%")
    print("  entropy coding savings (paper Section 4.3), but TurboQuant skips")
    print("  entropy coding for implementation simplicity.")
