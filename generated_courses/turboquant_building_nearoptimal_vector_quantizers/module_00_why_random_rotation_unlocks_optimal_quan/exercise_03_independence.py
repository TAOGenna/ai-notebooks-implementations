"""
Exercise 03: Why Near-Independence Makes Per-Coordinate Quantization Work
=========================================================================

TYPE: fill_blank — implement the two marked functions, then run the milestone.

CLAIM: After random rotation, distinct coordinates are nearly independent in
       high dimensions. This is the crucial justification for quantizing each
       coordinate separately, with no coupling between dimensions.

WHY THIS MATTERS
────────────────
Exercise 02 showed that each coordinate individually follows a known Beta
distribution. But that says nothing about relationships BETWEEN coordinates.

If coordinate j and coordinate k were strongly correlated (or otherwise
dependent), then quantizing them independently would throw away information
about their relationship — we'd need a joint codebook over (j, k) pairs.

The good news: after random rotation, pairwise correlations are near zero
AND — more importantly — pairwise mutual information is near zero. This
near-independence property means per-coordinate scalar quantization achieves
nearly the same distortion as the best possible joint vector quantization.

A critical subtlety: UNCORRELATED ≠ INDEPENDENT. Low correlation means
E[X_j · X_k] ≈ 0, but two variables can be uncorrelated yet strongly dependent
(e.g., X and X²). Mutual information captures ALL statistical dependencies,
not just linear ones. See the inline question at the end of this module's README.

YOUR TASKS (fill in two functions)
────────────────────────────────────
  1. compute_correlation_matrix(X_rot) — ~6 lines using NumPy
  2. estimate_mutual_information(u, v)  — ~4 lines using 2D histogram binning

WHAT YOU'LL SEE
────────────────
  • Correlation matrix: near-identity with tiny off-diagonal entries
  • Max |correlation| ≈ 0.03 for d=128, N=10,000 (shrinks as N/d grows)
  • Mutual information ≈ 0 for all coordinate pairs
  • The heatmap shows a bright diagonal, near-zero everywhere else
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_fn

# Re-use rotation code from Exercise 02 — run that exercise first!
# (We reproduce the needed functions here to keep this file self-contained.)


# ─────────────────────────────────────────────────────────────────────────────
# Rotation utilities (reproduced from Exercise 02)
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a uniformly random d×d rotation matrix via QR decomposition.
    (Reproduced from Exercise 02 — see that file for full documentation.)

    Parameters
    ----------
    d   : int
    rng : np.random.Generator

    Returns
    -------
    Pi : np.ndarray, shape (d, d)
    """
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    Q = Q * np.sign(np.diag(R))
    return Q


def rotate_vectors(Pi: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Apply rotation Π to a batch of vectors.

    Parameters
    ----------
    Pi : np.ndarray, shape (d, d)
    X  : np.ndarray, shape (N, d)

    Returns
    -------
    X_rot : np.ndarray, shape (N, d)
    """
    return X @ Pi.T


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Compute Sample Correlation Matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlation_matrix(X_rot: np.ndarray) -> np.ndarray:
    """
    Compute the sample Pearson correlation matrix for rotated coordinate vectors.

    Given N rotated vectors of dimension d (rows of X_rot), compute the d×d
    matrix C where:

        C[j, k] = Corr(X_j, X_k) = Cov(X_j, X_k) / (std(X_j) · std(X_k))

    For independent coordinates:  C = I  (identity matrix)
    In practice after rotation:   C ≈ I  (near-identity)

    The diagonal entries are exactly 1.0 by definition.
    Off-diagonal entries |C[j,k]| < 0.05 indicates near-independence for our
    purposes.

    Parameters
    ----------
    X_rot : np.ndarray, shape (N, d)
        N rotated unit vectors, each of dimension d.
        Rows = vectors, columns = coordinates.

    Returns
    -------
    C : np.ndarray, shape (d, d)
        Pearson correlation matrix. Diagonal is 1.0, off-diagonal near 0.0.

    Notes
    -----
    Algorithm:
      1. Center each coordinate: X_centered = X_rot - X_rot.mean(axis=0)
      2. Compute covariance matrix: Cov = X_centered.T @ X_centered / (N-1)
         (shape: d×d)
      3. Compute standard deviations per coordinate: stds = sqrt(diag(Cov))
         (shape: d,)
      4. Normalize: C = Cov / (stds[:, None] * stds[None, :])
         This divides each entry Cov[j,k] by std_j * std_k.
      5. Clip C to [-1, 1] for numerical stability.

    Hint: np.outer(stds, stds) gives the d×d matrix of std_j * std_k products.
    """
    # =========================================================================
    # TODO: Implement sample correlation matrix (~6 lines)
    #
    # Step 1: X_centered = X_rot - X_rot.mean(axis=0)
    # Step 2: cov = X_centered.T @ X_centered / (N - 1)    [N = X_rot.shape[0]]
    # Step 3: stds = np.sqrt(np.diag(cov))
    # Step 4: outer = np.outer(stds, stds)
    # Step 5: C = cov / outer
    # Step 6: return np.clip(C, -1.0, 1.0)
    # =========================================================================
    raise NotImplementedError("Implement compute_correlation_matrix")
    # =========================================================================


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Estimate Mutual Information via 2D Histogram
# ─────────────────────────────────────────────────────────────────────────────

def estimate_mutual_information(u: np.ndarray, v: np.ndarray,
                                n_bins: int = 20) -> float:
    """
    Estimate mutual information I(U; V) between two coordinate sequences.

    Mutual information measures ALL statistical dependency, not just linear:
        I(U; V) = Σ_{u,v} p(u,v) · log2(p(u,v) / (p(u) · p(v)))

    If U and V are independent: p(u,v) = p(u)·p(v)  →  I(U;V) = 0.
    Independence ⟹ I=0, and I=0 ⟹ independence.

    This is stronger than correlation: Corr=0 does NOT imply I=0.
    (But I≈0 is the property TurboQuant relies on for per-coordinate design.)

    We estimate I via histogram binning (plug-in estimator):
      1. Build a 2D histogram of (u, v) with n_bins×n_bins bins.
      2. Normalize to get joint probability matrix P_joint  (sum to 1).
      3. Marginals: P_u = P_joint.sum(axis=1), P_v = P_joint.sum(axis=0).
      4. Expected product: P_product[i,j] = P_u[i] * P_v[j].
      5. I = Σ_{i,j} P_joint[i,j] * log2(P_joint[i,j] / P_product[i,j])
         (only sum over bins where P_joint[i,j] > 0)

    Parameters
    ----------
    u      : np.ndarray, shape (N,), coordinate j values across N vectors
    v      : np.ndarray, shape (N,), coordinate k values across N vectors
    n_bins : int, number of histogram bins per axis (default=20)
             More bins = finer resolution but noisier estimates for small N.

    Returns
    -------
    mi : float, mutual information in bits (nats if using log instead of log2)
                Returns 0.0 for completely independent variables.

    Notes
    -----
    Use np.histogram2d to build the joint histogram.
    Be careful: add a small epsilon (1e-10) before taking log to avoid log(0).

    Algorithm:
      hist2d, _, _ = np.histogram2d(u, v, bins=n_bins, range=[[-1,1],[-1,1]])
      Then normalize, compute marginals, compute MI sum.
    """
    # =========================================================================
    # TODO: Implement mutual information estimation (~4–6 lines)
    #
    # Step 1: hist2d, _, _ = np.histogram2d(u, v, bins=n_bins, range=[[-1,1],[-1,1]])
    # Step 2: P_joint = hist2d / hist2d.sum()          (normalize to probability)
    # Step 3: P_u = P_joint.sum(axis=1, keepdims=True) (marginal over v)
    #         P_v = P_joint.sum(axis=0, keepdims=True) (marginal over u)
    # Step 4: P_prod = P_u * P_v                        (outer product, shape (n_bins, n_bins))
    # Step 5: mask = P_joint > 0                        (avoid log(0))
    #         mi = np.sum(P_joint[mask] * np.log2(P_joint[mask] / P_prod[mask]))
    # Step 6: return max(float(mi), 0.0)               (clip numerical negatives)
    #
    # Hint: For independent variables, P_joint ≈ P_u * P_v everywhere, so
    #       each term is p * log(p/p) = p * log(1) = 0 → total I = 0.
    # =========================================================================
    raise NotImplementedError("Implement estimate_mutual_information")
    # =========================================================================


# ─────────────────────────────────────────────────────────────────────────────
# Supporting Analysis Functions (provided — no implementation needed)
# ─────────────────────────────────────────────────────────────────────────────

def summarize_correlation_matrix(C: np.ndarray) -> dict:
    """
    Extract summary statistics from the correlation matrix.

    Parameters
    ----------
    C : np.ndarray, shape (d, d)

    Returns
    -------
    stats : dict with keys:
        'max_off_diag'   : max |C[j,k]| for j≠k
        'mean_off_diag'  : mean |C[j,k]| for j≠k
        'frac_above_005' : fraction of off-diagonal |C[j,k]| > 0.05
    """
    d = C.shape[0]
    mask = ~np.eye(d, dtype=bool)   # True where j ≠ k
    off_diag = np.abs(C[mask])

    return {
        "max_off_diag":   float(off_diag.max()),
        "mean_off_diag":  float(off_diag.mean()),
        "frac_above_005": float((off_diag > 0.05).mean()),
    }


def compute_pairwise_mi_sample(X_rot: np.ndarray, n_pairs: int = 20,
                                rng: np.random.Generator = None,
                                n_bins: int = 20) -> list:
    """
    Estimate mutual information for a random sample of coordinate pairs.

    Computing all d*(d-1)/2 pairs is expensive for large d. Instead, we
    sample n_pairs random (j, k) pairs and estimate their MI.

    Parameters
    ----------
    X_rot   : np.ndarray, shape (N, d)
    n_pairs : int, number of pairs to sample
    rng     : np.random.Generator
    n_bins  : int

    Returns
    -------
    mi_values : list of floats, MI estimates for each sampled pair
    """
    N, d = X_rot.shape
    if rng is None:
        rng = np.random.default_rng(0)

    mi_values = []
    seen = set()
    attempts = 0
    while len(mi_values) < n_pairs and attempts < n_pairs * 10:
        j, k = sorted(rng.integers(0, d, size=2))
        if j == k or (j, k) in seen:
            attempts += 1
            continue
        seen.add((j, k))
        mi = estimate_mutual_information(X_rot[:, j], X_rot[:, k], n_bins=n_bins)
        mi_values.append(mi)
        attempts += 1

    return mi_values


def plot_correlation_heatmap(C: np.ndarray, d: int, N: int, save_path: str):
    """
    Visualize the correlation matrix as a color heatmap.

    A near-identity matrix appears as a bright diagonal with a dark (near-zero)
    background — the visual signature of independent coordinates.

    Parameters
    ----------
    C         : np.ndarray, shape (d, d)
    d         : int
    N         : int
    save_path : str
    """
    # For readability, cap display at 32×32 (top-left corner of C)
    display_size = min(d, 32)
    C_display = C[:display_size, :display_size]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Pairwise Coordinate Correlations After Random Rotation  (d={d}, N={N:,})\n"
        "Near-identity matrix → coordinates are nearly independent",
        fontsize=11
    )

    # Full heatmap (possibly truncated)
    im0 = axes[0].imshow(C_display, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    axes[0].set_title(f"Correlation matrix (first {display_size}×{display_size} coords)", fontsize=10)
    axes[0].set_xlabel("Coordinate index", fontsize=9)
    axes[0].set_ylabel("Coordinate index", fontsize=9)
    plt.colorbar(im0, ax=axes[0])

    # Zoom in: only off-diagonal entries, tighter color scale
    C_offdiag = C_display.copy()
    np.fill_diagonal(C_offdiag, 0.0)    # zero out diagonal to see off-diag range
    max_val = max(np.abs(C_offdiag).max(), 1e-5)

    im1 = axes[1].imshow(C_offdiag, vmin=-max_val, vmax=max_val,
                          cmap="RdBu_r", aspect="auto")
    axes[1].set_title("Off-diagonal only (diagonal zeroed)\nBright = correlated, Dark = independent",
                       fontsize=10)
    axes[1].set_xlabel("Coordinate index", fontsize=9)
    axes[1].set_ylabel("Coordinate index", fontsize=9)
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [Saved plot → {save_path}]")


# ─────────────────────────────────────────────────────────────────────────────
# Main (Observable Milestone)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # =========================================================================
    # TRY: Change d — larger d makes coordinates more independent.
    #      At d=16, max correlation ≈ 0.10. At d=128, ≈ 0.03. At d=512, ≈ 0.01.
    # =========================================================================
    d = 128

    # =========================================================================
    # TRY: Increase N — more vectors give a better estimate of the correlation.
    #      Theoretical bound: E[|C_offdiag|] ~ 1/sqrt(N) for random matrices.
    #      Try N = 1000, 5000, 10000, 50000.
    # =========================================================================
    N = 10_000

    # =========================================================================
    # TRY: Change n_mi_pairs to estimate MI for more coordinate pairs.
    #      Note: each MI estimate takes O(N * n_bins^2) time.
    # =========================================================================
    n_mi_pairs = 30

    rng = np.random.default_rng(seed=31415)

    print("\n" + "="*68)
    print("  EXERCISE 03: Why Near-Independence Makes Per-Coordinate Quantization Work")
    print("  Claim: Rotated coordinates are nearly independent — pairwise")
    print("         correlations AND mutual information are near zero.")
    print("="*68)
    print(f"\n  Setting: d={d}, N={N:,} vectors\n")

    # Generate unit vectors and rotate them
    X_raw  = rng.standard_normal((N, d))
    norms  = np.linalg.norm(X_raw, axis=1, keepdims=True)
    X_unit = X_raw / norms

    Pi     = generate_random_rotation(d, rng)
    X_rot  = rotate_vectors(Pi, X_unit)

    # ── Task 1: Correlation Matrix ────────────────────────────────────────────
    print("  [Step 1/3] Computing pairwise correlation matrix...")
    C = compute_correlation_matrix(X_rot)
    stats_dict = summarize_correlation_matrix(C)

    print(f"\n  Correlation Matrix Statistics:")
    print(f"  ─ Max  |off-diagonal correlation|: {stats_dict['max_off_diag']:.4f}")
    print(f"  ─ Mean |off-diagonal correlation|: {stats_dict['mean_off_diag']:.4f}")
    print(f"  ─ Fraction with |corr| > 0.05:     {stats_dict['frac_above_005']*100:.1f}%")

    threshold = 0.05
    if stats_dict["max_off_diag"] < threshold:
        print(f"\n  ✓ Max |correlation| = {stats_dict['max_off_diag']:.4f} < {threshold}")
        print(f"    Coordinates are nearly independent!")
        print(f"    This means per-coordinate scalar quantization loses almost")
        print(f"    nothing vs joint vector quantization.")
    else:
        print(f"\n  ⚠ Max |correlation| = {stats_dict['max_off_diag']:.4f} ≥ {threshold}")
        print(f"    Try increasing N or d to see stronger independence.")

    # ── Task 2: Mutual Information ────────────────────────────────────────────
    print(f"\n  [Step 2/3] Estimating mutual information for {n_mi_pairs} random coordinate pairs...")
    mi_rng = np.random.default_rng(seed=2718)
    mi_values = compute_pairwise_mi_sample(X_rot, n_pairs=n_mi_pairs, rng=mi_rng, n_bins=20)

    print(f"\n  Mutual Information Statistics (over {len(mi_values)} sampled pairs):")
    print(f"  ─ Max  MI:  {max(mi_values):.4f} bits")
    print(f"  ─ Mean MI:  {np.mean(mi_values):.4f} bits")
    print(f"  ─ Min  MI:  {min(mi_values):.4f} bits")
    print()
    print("  Reference: Independent Gaussian variables would show MI ≈ 0.000 bits.")
    print("  High MI (e.g., 0.5+ bits) would mean coordinates share information")
    print("  that per-coordinate quantization cannot exploit.")

    # Sample histogram of MI values
    print()
    print("  MI distribution (each '.' = 1 pair):")
    for threshold_mi in [0.005, 0.01, 0.02, 0.05, 0.10]:
        count = sum(1 for mi in mi_values if mi < threshold_mi)
        bar = "." * count + " " * (len(mi_values) - count)
        print(f"    MI < {threshold_mi:.3f} bits: [{bar}] {count}/{len(mi_values)}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print(f"\n  [Step 3/3] Saving correlation heatmap...")
    heatmap_path = "milestone_03_independence.png"
    plot_correlation_heatmap(C, d=d, N=N, save_path=heatmap_path)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print()
    print("  " + "─"*64)
    print(f"  MILESTONE SUMMARY:")
    print(f"  ─ Max |correlation|:  {stats_dict['max_off_diag']:.4f}"
          f"  (want < 0.05 for d={d}, N={N:,})")
    print(f"  ─ Mean MI:            {np.mean(mi_values):.4f} bits"
          f"  (want ≈ 0.00 for independence)")
    print()
    print("  CONCLUSION: Coordinates of rotated vectors are nearly independent.")
    print("  This justifies the key simplification in TurboQuant:")
    print("  → Design ONE 1D scalar quantizer for the Beta distribution,")
    print("    apply it independently to each coordinate.")
    print("    Near-independence guarantees this is near-optimal joint VQ.")
    print()
    print("  NEXT MODULE: Implement the Lloyd-Max algorithm to find the optimal")
    print("               scalar quantizer for the Beta/Gaussian distribution,")
    print("               and verify it hits the 1/4^b distortion bound.")
    print()
