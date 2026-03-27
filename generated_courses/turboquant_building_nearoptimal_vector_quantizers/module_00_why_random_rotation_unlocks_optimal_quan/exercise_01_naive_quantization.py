"""
Exercise 01: What Does Quantization Lose? Measuring Distortion on Raw Vectors
==============================================================================

TYPE: explore — all code is provided. Your job is to RUN it, vary the parameters
      marked with `# TRY:`, observe the output, and build intuition.

CLAIM: Naive uniform quantization wastes bits because the coordinate distribution
       of real-world vectors is NOT uniform — most probability mass is concentrated
       away from the grid extremes.

CONTEXT (Vector Quantization 101)
──────────────────────────────────
A vector quantizer maps a high-dimensional vector x ∈ R^d to a binary string of
B bits, then reconstructs an approximation x̃ from those bits. The "bit-width"
b = B/d measures how many bits we spend per coordinate.

Two distortion metrics matter throughout this course:

  MSE distortion:  D_mse = E[‖x - x̃‖²]
  IP  distortion:  D_ip  = E[|⟨y, x⟩ - ⟨y, x̃⟩|²]   (inner product error)

The INFORMATION-THEORETIC LOWER BOUND for any randomized b-bit quantizer is:
  D_mse ≥ 1 / 4^b    (Shannon lower bound, Yao's minimax principle)

For b=1 → 1/4 = 0.25, b=2 → 1/16 = 0.0625, b=4 → 1/256 ≈ 0.0039

The naive approach below falls far short. Notice by how much.

HOW TO EXPLORE
──────────────
1. Run as-is:         python exercise_01_naive_quantization.py
2. Vary parameters using the TRY: comments below (lines ~120–130).
3. Observe how the gap between actual MSE and the lower bound changes.
4. Keep notes — you'll reference these numbers in Exercise 02.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Quantizer Implementation (provided — read carefully, nothing to implement)
# ─────────────────────────────────────────────────────────────────────────────

class NaiveUniformQuantizer:
    """
    The simplest possible quantizer: divide [-1, 1] into 2^b equal-width bins,
    snap each coordinate to the nearest bin center.

    This ignores the actual distribution of coordinate values entirely.
    No rotation, no statistics — just a uniform grid.

    Parameters
    ----------
    num_bits : int
        Bits per coordinate (b). Uses 2^b levels uniformly spaced in [-1, 1].
    clip_range : float
        Clip coordinates to [-clip_range, clip_range] before quantizing.
        Unit vectors have all coordinates in roughly [-1, 1] but extreme values
        can appear; clipping prevents grid overflow.
    """

    def __init__(self, num_bits: int, clip_range: float = 1.0):
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits          # K = 2^b quantization levels
        self.clip_range = clip_range

        # Build the codebook: K uniformly spaced values in [-clip_range, clip_range]
        # These are the "reconstruction values" (centroids) the grid uses.
        self.codebook = np.linspace(-clip_range, clip_range, self.num_levels)
        # Grid spacing: distance between adjacent bin centers
        self.step = 2.0 * clip_range / (self.num_levels - 1)

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """
        Quantize vector x by snapping each coordinate to the nearest grid point.

        Parameters
        ----------
        x : np.ndarray, shape (d,) or (N, d)
            Input vectors to quantize.

        Returns
        -------
        indices : np.ndarray of int, same shape as x
            Index into self.codebook for each coordinate. Stores B = b*d bits total.
        """
        x_clipped = np.clip(x, -self.clip_range, self.clip_range)
        # Shift to [0, 2*clip_range], scale to [0, num_levels-1], round to integer
        scaled = (x_clipped + self.clip_range) / self.step
        indices = np.round(scaled).astype(int)
        indices = np.clip(indices, 0, self.num_levels - 1)
        return indices

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """
        Reconstruct approximate vectors from quantization indices.

        Parameters
        ----------
        indices : np.ndarray of int
            Quantization indices produced by self.quantize().

        Returns
        -------
        x_hat : np.ndarray of float, same shape as indices
            Reconstructed (dequantized) coordinate values.
        """
        return self.codebook[indices]

    def quantize_and_reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Round-trip: quantize then immediately dequantize. Returns x̃."""
        return self.dequantize(self.quantize(x))


# ─────────────────────────────────────────────────────────────────────────────
# Distortion Metrics
# ─────────────────────────────────────────────────────────────────────────────

def mse_distortion(x: np.ndarray, x_hat: np.ndarray) -> tuple:
    """
    Mean squared error between original and reconstructed vectors.

    Returns BOTH total-vector MSE and per-coordinate MSE for comparison.

    Total-vector MSE:    D_total  = (1/N) * Σ_i ‖x_i - x̃_i‖²
    Per-coordinate MSE: D_coord  = D_total / d

    For unit vectors with ‖x‖=1, the information-theoretic lower bounds are:
        D_total  ≥ 1/4^b   (total-vector, independent of d)
        D_coord  ≥ 1/(d * 4^b)   (per coordinate)

    Key relationship: D_total = d × D_coord.
    The gap ratio (actual/lower_bound) is the same for both.

    Parameters
    ----------
    x     : np.ndarray, shape (N, d), original unit vectors
    x_hat : np.ndarray, shape (N, d), reconstructed vectors

    Returns
    -------
    total_mse : float, average total squared error per vector
    coord_mse : float, average squared error per coordinate (= total / d)
    """
    diff = x - x_hat
    d = x.shape[1]
    total_mse = float(np.mean(np.sum(diff ** 2, axis=1)))
    coord_mse = total_mse / d
    return total_mse, coord_mse


def inner_product_distortion(x: np.ndarray, x_hat: np.ndarray,
                             query: np.ndarray) -> float:
    """
    Inner product distortion: how much does quantization corrupt dot products?

    D_ip = (1/N) * sum_i (⟨y, x_i⟩ - ⟨y, x̃_i⟩)²

    where y is a fixed query vector (normalized to unit norm here).

    Parameters
    ----------
    x     : np.ndarray, shape (N, d)
    x_hat : np.ndarray, shape (N, d)
    query : np.ndarray, shape (d,)  — the "y" in ⟨y, x⟩

    Returns
    -------
    float : average squared IP error across N vectors
    """
    y = query / (np.linalg.norm(query) + 1e-12)
    true_ip  = x     @ y   # shape (N,)
    approx_ip = x_hat @ y  # shape (N,)
    return float(np.mean((true_ip - approx_ip) ** 2))


def theoretical_lower_bound(num_bits: int, d: int) -> float:
    """
    Shannon lower bound on per-coordinate MSE distortion for b-bit quantizers.

    For a unit vector with ‖x‖=1 in R^d, each coordinate has variance 1/d.
    The rate-distortion lower bound for a b-bit scalar quantizer is:

        D_mse_per_coord ≥ (1/d) / 4^b

    Equivalently, the TOTAL-VECTOR MSE lower bound is 1/4^b (independent of d).

    TurboQuant achieves (total) D_mse ≈ 0.36, 0.117, 0.03, 0.009 for b=1..4,
    all within ~2.7× of the total lower bound 1/4^b.

    Parameters
    ----------
    num_bits : int
    d        : int, vector dimension

    Returns
    -------
    float : per-coordinate lower bound = 1/(d * 4^b)
    """
    return 1.0 / (d * (4 ** num_bits))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────

def generate_unit_vectors(N: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate N random unit vectors in R^d (uniformly on S^{d-1}).

    Each vector is drawn from a standard Gaussian distribution (isotropic),
    then L2-normalized. This is the standard way to sample uniformly from
    the unit sphere.

    Parameters
    ----------
    N   : int, number of vectors
    d   : int, dimension
    rng : np.random.Generator

    Returns
    -------
    X : np.ndarray, shape (N, d), each row has ‖x_i‖ = 1
    """
    X = rng.standard_normal((N, d))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def run_distortion_experiment(d: int, N: int, bit_widths: list,
                               rng: np.random.Generator) -> dict:
    """
    Measure naive uniform quantizer distortion across multiple bit-widths.

    Parameters
    ----------
    d          : int, vector dimension
    N          : int, number of test vectors
    bit_widths : list of int, e.g. [1, 2, 3, 4]
    rng        : np.random.Generator

    Returns
    -------
    results : dict mapping bit_width ->
        {
          'coord_mse':        float  (per-coordinate MSE = total/d),
          'total_mse':        float  (total-vector MSE),
          'ip':               float  (IP distortion),
          'coord_lb':         float  (per-coord lower bound = 1/(d*4^b)),
          'total_lb':         float  (total-vector lower bound = 1/4^b),
          'gap_ratio':        float  (coord_mse / coord_lb — same as total/total_lb),
        }
    """
    X = generate_unit_vectors(N, d, rng)
    query = rng.standard_normal(d)

    results = {}
    for b in bit_widths:
        q = NaiveUniformQuantizer(num_bits=b, clip_range=1.0)
        X_hat = q.quantize_and_reconstruct(X)

        total_mse, coord_mse = mse_distortion(X, X_hat)
        ip  = inner_product_distortion(X, X_hat, query)
        coord_lb  = theoretical_lower_bound(b, d)
        total_lb  = 1.0 / (4 ** b)

        results[b] = {
            "coord_mse":  coord_mse,
            "total_mse":  total_mse,
            "ip":         ip,
            "coord_lb":   coord_lb,
            "total_lb":   total_lb,
            "gap_ratio":  coord_mse / coord_lb,  # same ratio as total_mse/total_lb
        }
    return results


def print_distortion_table(results: dict, d: int, N: int):
    """Pretty-print a comparison table of naive vs optimal distortion."""
    print(f"\n{'='*78}")
    print(f"  NAIVE UNIFORM QUANTIZER — Per-Coordinate Distortion Table")
    print(f"  d={d} dimensions, N={N} test vectors, clip_range=1.0")
    print(f"{'='*78}")
    print(f"  {'Bits':>4}  {'MSE/coord':>11}  {'LB/coord':>11}  "
          f"{'Gap (×)':>8}  {'Total MSE':>11}  {'Total LB':>10}")
    print(f"  {'-'*4}  {'-'*11}  {'-'*11}  {'-'*8}  {'-'*11}  {'-'*10}")

    for b, r in sorted(results.items()):
        print(f"  {b:>4}  {r['coord_mse']:>11.5f}  {r['coord_lb']:>11.7f}  "
              f"{r['gap_ratio']:>8.1f}×  {r['total_mse']:>11.4f}  {r['total_lb']:>10.4f}")

    print(f"{'='*78}")
    print()
    print("  Interpretation:")
    print("  ─ 'MSE/coord'  = per-coordinate MSE  (total MSE / d)")
    print("  ─ 'LB/coord'   = per-coordinate lower bound = 1/(d × 4^b)")
    print("  ─ 'Total LB'   = 1/4^b  (information-theoretic lower bound on total MSE)")
    print("  ─ 'Gap (×)'    = how many times worse than the optimal bound")
    print("  ─  A gap of 1.0× would be information-theoretically optimal.")
    print("  ─  TurboQuant reaches ~1.4–2.3× — far better than naive's 50–450×.")
    print()


def print_coordinate_stats(d: int, N: int, rng: np.random.Generator):
    """
    Show the empirical distribution of coordinate values for unit vectors.
    This reveals WHY uniform grids are wasteful.
    """
    X = generate_unit_vectors(N, d, rng)
    coords = X.flatten()

    print(f"  Coordinate distribution for unit vectors (d={d}, N={N}):")
    print(f"  ─ Mean:     {coords.mean():+.4f}  (expected: 0)")
    print(f"  ─ Std:      {coords.std():.4f}  (expected: 1/√d = {1/d**0.5:.4f})")
    print(f"  ─ Min/Max:  [{coords.min():.4f}, {coords.max():.4f}]")

    # Fraction of coordinates in the "outer" 20% of [-1, 1]
    outer_frac = np.mean(np.abs(coords) > 0.8)
    print(f"  ─ Fraction |coord| > 0.8:  {outer_frac:.4f}")
    print(f"    → Uniform grids allocate bins uniformly, but most coords are near 0!")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main (Observable Milestone)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # =========================================================================
    # TRY: Change d to see how distortion behaves in different dimensions.
    #      Try d = 16, 64, 128, 256, 1024
    # =========================================================================
    d = 128

    # =========================================================================
    # TRY: Change N (more vectors = more accurate distortion estimates).
    #      Try N = 100, 1000, 10000
    # =========================================================================
    N = 2000

    # =========================================================================
    # TRY: Add more bit-widths, e.g. bit_widths = [1, 2, 3, 4, 5, 6, 8]
    # =========================================================================
    bit_widths = [1, 2, 3, 4]

    rng = np.random.default_rng(seed=42)

    print("\n" + "="*70)
    print("  EXERCISE 01: What Does Quantization Lose?")
    print("  Claim: Naive uniform quantization wastes bits on empty grid regions.")
    print("="*70)

    # Show coordinate concentration first
    print()
    print_coordinate_stats(d, N, rng)

    # Run distortion experiment
    rng2 = np.random.default_rng(seed=42)
    results = run_distortion_experiment(d, N, bit_widths, rng2)
    print_distortion_table(results, d, N)

    # Highlight the key gap at b=1
    b1 = results[1]
    print(f"  KEY INSIGHT at b=1 bit (d={d}):")
    print(f"  ─ Per-coord MSE (naive):  {b1['coord_mse']:.4f}")
    print(f"  ─ Per-coord lower bound:  {b1['coord_lb']:.6f}  (= 1/(d×4^1) = 1/{d*4})")
    print(f"  ─ Total MSE (naive):      {b1['total_mse']:.4f}")
    print(f"  ─ Total lower bound:      {b1['total_lb']:.4f}  (= 1/4^1 = 0.25)")
    print(f"  ─ Gap factor:             {b1['gap_ratio']:.0f}×  "
          f"(same ratio for per-coord and total)")
    print()
    print(f"  Why is per-coord MSE ≈ {b1['coord_mse']:.2f} at b=1?")
    print(f"  ─ Coordinates of unit vectors have std ≈ 1/√{d} ≈ {1/d**0.5:.3f}")
    print(f"  ─ 1-bit quantizer snaps everything to ±1.0 (the grid extremes)")
    print(f"  ─ Error per coord ≈ (0 - 1)² = 1 — the grid is far from the data!")
    print()

    # Show how the gap changes with bits (log scale bar chart)
    print("  Gap ratio vs. bit-width (log₁₀ scale, smaller = closer to optimal):")
    for b, r in sorted(results.items()):
        log_gap = np.log10(r["gap_ratio"])
        bar = "█" * int(log_gap * 8)
        print(f"    b={b}: {r['gap_ratio']:6.1f}×  (10^{log_gap:.1f})  {bar}")
    print()
    print("  Note: TurboQuant (Module 02) achieves gap ≈ 1.4–2.3×")
    print("        vs naive's 50–450× shown here.")
    print()
    print("  Root cause: The coordinate histogram of a unit vector is")
    print("  concentrated near 0 (std ≈ 1/√d). The uniform grid wastes")
    print("  bins on rarely-visited extremes. Solution: design the grid")
    print("  specifically for the Beta distribution — after rotation.")
    print()
    print("  NEXT: Exercise 02 — Random Rotation creates a universal,")
    print("        analytically known distribution to quantize optimally.")
    print()
