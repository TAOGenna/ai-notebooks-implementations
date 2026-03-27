"""
Exercise 01: Assembling TurboQuant_mse — Rotate, Quantize, Reconstruct
========================================================================

TYPE: fill_blank — implement the three marked methods, then run the milestone.

CLAIM: By wiring together the random rotation (Module 0) and the Lloyd-Max
       codebook (Module 1) into a single class, we get a complete quantizer
       that achieves MSE ≤ (√3·π/2) / 4^b for any unit vector.

Algorithm 1 from the TurboQuant paper (MSE version):
─────────────────────────────────────────────────────
  Setup:
    Π  ← random d×d rotation matrix  (Section 2.1, "data-oblivious preconditioning")
    {c_l} ← Lloyd-Max codebook for Beta(1/2, (d-1)/2)  (Section 2.2)

  Quant(x):
    1.  x_rot  = Π · x                      (rotate into "isotropic" basis)
    2.  for each coordinate i:
            idx[i] = argmin_l |x_rot[i] − c_l|²   (nearest centroid)
    3.  return idx                           (B = b·d bits total)

  DeQuant(idx):
    1.  y_tilde[i] = c_{idx[i]}             (map index → centroid value)
    2.  x_hat      = Π^T · y_tilde          (undo rotation)
    3.  return x_hat

Key insight: the rotation makes coordinates independent and identically
distributed — so independent scalar quantization per coordinate is optimal.
Without rotation, correlations between coordinates waste codebook capacity.

YOUR TASKS
──────────
  1. `quantize(x)`         — ~5 lines: rotate, find nearest centroid per coord
  2. `dequantize(indices)` — ~4 lines: centroid lookup, inverse rotation
  3. `compute_mse(x)`      — ~3 lines: quantize, dequantize, squared norm of error

Dependencies: rotation_utils.py, lloyd_max_utils.py (both in this module folder)
"""

import numpy as np
import sys
import os

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rotation_utils import random_rotation_matrix
from lloyd_max_utils import lloyd_max_codebook, lloyd_max_mse


# ═══════════════════════════════════════════════════════════════════════════════
# TurboQuantMSE — the main class you will complete
# ═══════════════════════════════════════════════════════════════════════════════

class TurboQuantMSE:
    """
    TurboQuant_mse: data-oblivious MSE-optimal vector quantizer.

    Quantises any unit-norm vector x ∈ R^d to b·d bits by:
        (a) applying a random rotation Π (makes coordinates i.i.d.)
        (b) applying the optimal scalar quantizer per rotated coordinate
        (c) storing the centroid INDICES (b bits each)

    Dequantisation reverses the process:
        (a) map indices → centroid values
        (b) apply Π^T to reconstruct x̂

    Usage
    -----
    >>> tq = TurboQuantMSE(d=128, b=2, seed=42)
    >>> idx    = tq.quantize(x)          # x: shape (d,), idx: shape (d,) int
    >>> x_hat  = tq.dequantize(idx)      # x_hat: shape (d,)
    >>> mse    = tq.compute_mse(x)       # scalar float
    """

    def __init__(self, d: int, b: int, seed: int = 42):
        """
        Initialise a TurboQuant_mse quantizer.

        Parameters
        ----------
        d    : int, vector dimension (e.g. 128 for typical KV cache embeddings)
        b    : int, bits per coordinate (1 = 32× compression, 2 = 16×, …)
        seed : int, random seed for the rotation matrix Π

        After __init__:
          self.Pi         — random d×d rotation matrix, shape (d, d)
          self.codebook   — Lloyd-Max centroids, shape (2^b,), sorted ascending
          self.boundaries — Voronoi boundaries, shape (2^b − 1,)
                            boundaries[k] = (codebook[k] + codebook[k+1]) / 2
        """
        self.d = d
        self.b = b
        self.n_centroids = 2 ** b

        # ── Step 1: generate the rotation matrix (Module 0) ─────────────────
        self.Pi = random_rotation_matrix(d, seed=seed)

        # ── Step 2: compute the codebook (Module 1) ─────────────────────────
        # lloyd_max_codebook runs the Lloyd-Max algorithm on the Beta coordinate
        # distribution; returns sorted centroid values in ascending order.
        self.codebook = lloyd_max_codebook(b=b, d=d)  # shape (2^b,)

        # Voronoi decision boundaries: midpoints between consecutive centroids.
        # np.searchsorted(self.boundaries, value) returns the centroid index
        # that is nearest to `value` (when boundaries are midpoints).
        self.boundaries = (self.codebook[:-1] + self.codebook[1:]) / 2.0  # shape (2^b − 1,)

    # ─────────────────────────────────────────────────────────────────────────
    # Task 1: Quantize
    # ─────────────────────────────────────────────────────────────────────────

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """
        Quantize a vector x ∈ R^d to an array of centroid indices.

        This implements lines 1-3 of Algorithm 1 (Quant step):
            x_rot  = Π · x                        (rotate into isotropic basis)
            idx[i] = nearest centroid index for x_rot[i]

        The key: after rotation, each coordinate x_rot[i] follows the Beta
        distribution for which self.codebook was designed — so the quantizer
        is optimal for EVERY coordinate simultaneously.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Input vector. Should be unit-norm for the theoretical MSE bounds
            to apply, but the method works for any vector.

        Returns
        -------
        indices : np.ndarray, shape (d,), dtype int
            Centroid indices in [0, 2^b − 1]. Quantized storage uses b bits
            per index, for a total of b·d bits to represent the whole vector.

        Hints
        -----
        • self.Pi has shape (d, d); use @ for matrix-vector multiplication.
        • np.searchsorted(self.boundaries, x_rot) maps each element of x_rot
          to its Voronoi region index. Since boundaries are sorted midpoints,
          searchsorted returns the index of the nearest centroid.
          Example: boundaries = [0.0] (1-bit)
                   searchsorted([0.0], -0.05) → 0  (left centroid)
                   searchsorted([0.0], +0.03) → 1  (right centroid)
        """
        # ====================================================================
        # TODO: Implement quantize (~5 lines)
        #
        # Step 1: Rotate x using the stored rotation matrix self.Pi
        #         x_rot = ???          shape: (d,)
        #
        # Step 2: For each coordinate of x_rot, find the nearest centroid.
        #         Hint: use np.searchsorted(self.boundaries, x_rot)
        #         This returns an integer array of indices into self.codebook.
        #         No for-loop needed — numpy does this for the whole vector at once.
        #
        # Step 3: Return the index array as an int array.
        #         Hint: cast with .astype(int) or the result of searchsorted
        #               is already integer-typed.
        # ====================================================================
        raise NotImplementedError("Implement quantize — see TODO above")
        # ====================================================================

    # ─────────────────────────────────────────────────────────────────────────
    # Task 2: Dequantize
    # ─────────────────────────────────────────────────────────────────────────

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """
        Reconstruct a vector from its quantized representation (centroid indices).

        This implements Algorithm 1 (DeQuant step):
            y_tilde[i] = codebook[indices[i]]     (index → centroid value)
            x_hat      = Π^T · y_tilde            (undo rotation)

        Why Π^T? Because Π is orthogonal (Π^T = Π^{-1}), so applying Π^T
        rotates y_tilde back into the original coordinate system.

        Parameters
        ----------
        indices : np.ndarray, shape (d,), dtype int
            Centroid indices in [0, 2^b − 1], as returned by quantize().

        Returns
        -------
        x_hat : np.ndarray, shape (d,)
            Reconstructed vector in the original space.

        Hints
        -----
        • self.codebook[indices] performs vectorised index lookup:
          if indices = [0, 2, 1], returns [codebook[0], codebook[2], codebook[1]].
        • Inverse rotation: self.Pi.T @ y_tilde
          (equivalently: self.Pi.T is the transpose of a (d,d) matrix)
        """
        # ====================================================================
        # TODO: Implement dequantize (~4 lines)
        #
        # Step 1: Look up centroid values for each index.
        #         y_tilde = self.codebook[???]      shape: (d,)
        #
        # Step 2: Apply the INVERSE rotation (Π^T) to y_tilde.
        #         x_hat = ???                       shape: (d,)
        #
        # Step 3: Return x_hat.
        # ====================================================================
        raise NotImplementedError("Implement dequantize — see TODO above")
        # ====================================================================

    # ─────────────────────────────────────────────────────────────────────────
    # Task 3: Compute MSE
    # ─────────────────────────────────────────────────────────────────────────

    def compute_mse(self, x: np.ndarray) -> float:
        """
        Compute the total MSE distortion for a single vector x.

        MSE = ‖x − x̂‖²   where  x̂ = DeQuant(Quant(x))

        Note: this is the TOTAL squared error across all d dimensions,
        not the per-coordinate average. For unit vectors ‖x‖ = 1, this equals
        the squared relative error: MSE = ‖x − x̂‖² / ‖x‖² = ‖x − x̂‖².

        The paper's bound (Theorem 1) guarantees:
            E[MSE] ≤ (√3 · π/2) / 4^b    for unit vectors x

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Input vector (ideally unit norm).

        Returns
        -------
        mse : float
            ‖x − DeQuant(Quant(x))‖²
        """
        # ====================================================================
        # TODO: Implement compute_mse (~3 lines)
        #
        # Step 1: Quantize x to get indices.
        # Step 2: Dequantize to get the reconstructed vector x_hat.
        # Step 3: Return the squared Euclidean distance: np.sum((x - x_hat)**2)
        #         or equivalently: np.linalg.norm(x - x_hat)**2
        # ====================================================================
        raise NotImplementedError("Implement compute_mse — see TODO above")
        # ====================================================================

    # ─────────────────────────────────────────────────────────────────────────
    # Provided helper — no implementation needed
    # ─────────────────────────────────────────────────────────────────────────

    def compress_and_reconstruct(self, x: np.ndarray):
        """
        Run the full pipeline: quantize x, then dequantize.

        Returns
        -------
        x_hat   : np.ndarray, shape (d,), reconstructed vector
        indices : np.ndarray, shape (d,), integer centroid indices (the compressed form)
        """
        indices = self.quantize(x)
        x_hat   = self.dequantize(indices)
        return x_hat, indices

    def bits_per_vector(self) -> int:
        """Total bits needed to store one quantized vector: b bits × d coordinates."""
        return self.b * self.d

    def compression_ratio(self) -> float:
        """Compression ratio vs 32-bit floats: 32·d / (b·d) = 32 / b."""
        return 32.0 / self.b


# ═══════════════════════════════════════════════════════════════════════════════
# Observable Milestone
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Milestone: Verify TurboQuant_mse MSE against the theoretical upper bound.

    Expected output pattern (once all three TODOs are implemented):

        ┌──────────────────────────────────────────────────────────────────────┐
        │  TurboQuant_mse  |  d=128, n=1000 random unit vectors               │
        ├────┬──────────────┬──────────────┬──────────┬─────────────────────── │
        │  b │  Empirical   │  Upper Bound │ % of UB  │  Compression Ratio     │
        ├────┼──────────────┼──────────────┼──────────┼─────────────────────── │
        │  1 │      0.361   │      0.680   │  53.1%   │  32.0×  (32→1 bit)     │
        │  2 │      0.117   │      0.170   │  68.7%   │  16.0×  (32→2 bit)     │
        │  3 │      0.030   │      0.0425  │  70.7%   │  10.7×  (32→3 bit)     │
        │  4 │      0.009   │      0.0106  │  85.1%   │   8.0×  (32→4 bit)     │
        └────┴──────────────┴──────────────┴──────────┴────────────────────────┘

    Insight: the empirical MSE is ≈50-85% of the upper bound — we are within
    a factor of ~2 of the information-theoretic optimum. Compare to naive uniform
    quantization (Module 0, Exercise 01) which was 5-10× worse!

    The 1/4^b scaling is unmistakable: each added bit cuts MSE by ~4×.
    This is the hallmark of near-optimal quantization.
    """
    np.random.seed(0)

    d = 128
    N = 1000   # number of test vectors

    # Generate random unit vectors for testing
    X = np.random.randn(N, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    # Theoretical upper bound: D_mse ≤ (√3·π/2) / 4^b  (Theorem 1, TurboQuant paper)
    # This is the bound for the TOTAL MSE over all d coordinates of a unit vector.
    upper_bound_const = np.sqrt(3.0) * np.pi / 2.0   # ≈ 2.720

    print()
    print("=" * 72)
    print(f"  TurboQuant_mse  |  d={d}, n={N} random unit vectors")
    print("=" * 72)
    print(f"  {'b':>3}  {'Empirical MSE':>14}  {'Upper Bound':>12}  "
          f"{'% of UB':>8}  {'Compress':>10}")
    print("  " + "-" * 60)

    prev_mse = None
    for b in [1, 2, 3, 4]:
        # Build quantizer (codebook computation may take ~1-3s for first call)
        tq = TurboQuantMSE(d=d, b=b, seed=42)

        # Measure MSE on all N test vectors
        mse_values = np.array([tq.compute_mse(x) for x in X])
        avg_mse    = float(np.mean(mse_values))

        upper_bound = upper_bound_const / (4.0 ** b)
        pct_of_ub   = 100.0 * avg_mse / upper_bound
        compress_ratio = tq.compression_ratio()

        # Check 4× reduction per bit
        ratio_str = ""
        if prev_mse is not None:
            ratio = prev_mse / avg_mse
            ratio_str = f"  (÷{ratio:.1f} vs b-1)"

        bound_check = "✓" if avg_mse <= upper_bound else "✗"

        print(f"  {b:>3}  {avg_mse:>14.4f}  {upper_bound:>12.4f}  "
              f"{pct_of_ub:>7.1f}%  {compress_ratio:>6.1f}×  {bound_check}{ratio_str}")

        prev_mse = avg_mse

    print()

    # Verify all within bound
    all_within = True
    for b in [1, 2, 3, 4]:
        tq = TurboQuantMSE(d=d, b=b, seed=42)
        mses = [tq.compute_mse(x) for x in X[:100]]   # quick re-check on subset
        ub   = upper_bound_const / (4.0 ** b)
        if np.mean(mses) > ub:
            all_within = False

    status = "All within theoretical upper bound ✓" if all_within else "WARNING: some exceed bound ✗"
    print(f"  {status}")
    print()

    # Compression summary
    print("  Compression summary (32-bit float baseline):")
    for b in [1, 2, 3, 4]:
        tq = TurboQuantMSE(d=d, b=b, seed=42)
        orig_bits     = 32 * d
        quant_bits    = tq.bits_per_vector()
        compress      = tq.compression_ratio()
        print(f"    b={b}: Original {orig_bits:5d} bits → Quantized {quant_bits:4d} bits  "
              f"({compress:.1f}× compression)")
    print()
    print("  KEY INSIGHT: MSE drops by ~4× per added bit (the 1/4^b law).")
    print("  Each extra bit is 'worth' a 4× distortion reduction.")
    print("  But wait — how well does this quantizer preserve INNER PRODUCTS?")
    print("  Run Exercise 02 to find out (spoiler: it doesn't, at 1-bit).")
    print("=" * 72)
