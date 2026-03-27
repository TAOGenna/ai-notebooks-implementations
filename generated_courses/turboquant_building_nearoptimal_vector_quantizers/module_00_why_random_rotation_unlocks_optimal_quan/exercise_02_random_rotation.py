"""
Exercise 02: How Random Rotation Creates a Universal Distribution
=================================================================

TYPE: fill_blank — implement the three marked functions, then run the milestone.

CLAIM: Multiplying ANY unit vector by a random rotation matrix Π places it
       uniformly on the unit hypersphere. Each coordinate then follows a known
       Beta distribution — converging to N(0, 1/d) as d grows. This means we
       can design a quantizer WITHOUT seeing any data.

WHY THIS MATTERS
────────────────
Exercise 01 showed naive uniform quantization leaves a large gap vs. the
information-theoretic lower bound. The gap exists because the coordinate
distribution of raw unit vectors is NOT uniform — the quantizer wastes bins.

After rotation, we know EXACTLY what distribution each coordinate follows:
  f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^{(d-3)/2}

where Γ is the gamma function. This is the Beta distribution on [-1, 1].

Key facts:
  • E[X]   = 0          (symmetric, zero-centered)
  • Var[X] = 1/d        (variance concentrates as dimension grows)
  • As d→∞: Beta → N(0, 1/d)  (central limit theorem effect)

Knowing this distribution analytically is the unlock for optimal quantization.

YOUR TASKS (fill in three functions)
─────────────────────────────────────
  1. generate_random_rotation(d) — ~5 lines using QR decomposition
  2. rotate_vector(Pi, x)        — ~2 lines, simple matrix-vector product
  3. theoretical_beta_pdf(t, d)  — ~4 lines using scipy.special.gamma

VERIFICATION STRATEGY
──────────────────────
After implementing, the milestone:
  • Runs a Kolmogorov-Smirnov test: p-value > 0.05 means empirical distribution
    is statistically indistinguishable from the theoretical Beta pdf.
  • Saves a histogram overlaid with the theoretical Beta and Gaussian curves.
  • At d=128, the Beta and Gaussian overlap nearly perfectly — visual confirmation
    that we can use the Gaussian approximation for codebook design.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (saves to file)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.special import gamma as gamma_fn
from scipy.special import gammaln   # log-Gamma: avoids overflow for large d


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Generate a Random Rotation Matrix
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a uniformly random d×d rotation (orthogonal) matrix Π.

    A rotation matrix satisfies:  Π^T Π = I   and   det(Π) = +1
    Multiplying any unit vector x by Π places Π·x uniformly on S^{d-1}.

    ALGORITHM (Haar measure via QR decomposition):
    ─────────────────────────────────────────────
    1. Sample a d×d matrix G with i.i.d. N(0,1) entries.
    2. Compute the QR decomposition:  G = Q · R
    3. Q is uniformly distributed on the orthogonal group O(d), but may
       have det(Q) = ±1. To ensure a proper rotation (det = +1), multiply
       each column of Q by the sign of the corresponding diagonal of R:
         Q_adjusted = Q * sign(diag(R))
    4. Return Q_adjusted.

    This is the standard algorithm for sampling from the Haar measure on SO(d).
    See: "How to Generate Random Matrices from the Classical Compact Groups",
    Mezzadri (2006).

    Parameters
    ----------
    d   : int, dimension (rotation matrix will be d×d)
    rng : np.random.Generator, random number generator for reproducibility

    Returns
    -------
    Pi : np.ndarray, shape (d, d), orthogonal matrix with det ≈ +1
    """
    # =========================================================================
    # TODO: Implement random rotation matrix generation (~5 lines)
    #
    # Step 1: Sample G = rng.standard_normal((d, d))
    # Step 2: Compute Q, R = np.linalg.qr(G)
    # Step 3: Extract the diagonal of R: d_diag = np.diag(R)
    # Step 4: Multiply columns of Q by sign of diagonal: Q * np.sign(d_diag)
    # Step 5: Return the adjusted Q
    #
    # Hint: np.sign(0) returns 0 — use np.where(d_diag >= 0, 1, -1) if needed,
    #       but in practice d_diag is almost never exactly 0 for continuous G.
    # =========================================================================
    raise NotImplementedError("Implement generate_random_rotation")
    # =========================================================================


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Apply Rotation to a Vector (or Batch)
# ─────────────────────────────────────────────────────────────────────────────

def rotate_vector(Pi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Apply rotation matrix Π to one or more vectors.

    For a single vector x of shape (d,):     returns Π @ x, shape (d,)
    For a batch X of shape (N, d):           returns X @ Π^T, shape (N, d)

    Note: the batch formula is X @ Π^T (not Π @ X^T) because our convention
    stores vectors as ROWS. Rotating each row x_i gives x_i @ Π^T = (Π x_i)^T.

    After rotation, the L2 norm is preserved exactly:
        ‖Π · x‖ = ‖x‖  for all x   (orthogonal transformations are isometries)

    Parameters
    ----------
    Pi : np.ndarray, shape (d, d), orthogonal rotation matrix
    x  : np.ndarray, shape (d,) or (N, d)

    Returns
    -------
    x_rot : np.ndarray, same shape as x
    """
    # =========================================================================
    # TODO: Implement rotation (~2 lines)
    #
    # If x is 1D (shape (d,)):   return Pi @ x
    # If x is 2D (shape (N, d)): return x @ Pi.T
    #
    # Hint: use x.ndim to check dimensionality.
    # =========================================================================
    raise NotImplementedError("Implement rotate_vector")
    # =========================================================================


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Theoretical Beta PDF for a Single Rotated Coordinate
# ─────────────────────────────────────────────────────────────────────────────

def theoretical_beta_pdf(t: np.ndarray, d: int) -> np.ndarray:
    """
    Probability density of a single coordinate of a uniform random unit vector.

    If x is uniform on S^{d-1} (equivalently, after random rotation), each
    individual coordinate X_j has density:

        f(t) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - t²)^{(d-3)/2},  t ∈ [-1, 1]

    Symbols:
        Γ(·) = gamma function = generalization of factorial to real numbers
               e.g. Γ(n) = (n-1)! for integer n, Γ(1/2) = √π
        d    = vector dimension
        t    = coordinate value ∈ [-1, 1]

    This is a scaled Beta distribution:
        if T ~ Beta(α, α) with α = (d-1)/2, then 2T-1 has this density.

    For large d: f(t) ≈ N(0, 1/d) = √(d/(2π)) · exp(-d·t²/2)

    Parameters
    ----------
    t : np.ndarray, coordinate values in [-1, 1]
    d : int, vector dimension

    Returns
    -------
    pdf : np.ndarray, same shape as t, density values (non-negative)

    Notes
    -----
    Use gammaln (log-Gamma) for numerical stability at large d.
    gamma_fn(d/2) overflows for d≳340 (e.g., d=512). Working in log space:

        log_C = gammaln(d/2) - 0.5*np.log(np.pi) - gammaln((d-1)/2)
        C     = np.exp(log_C)

    Clamp values to avoid (1-t²)^{(d-3)/2} going negative due to floating point:
        safe_base = np.maximum(1.0 - t**2, 0.0)
    """
    # =========================================================================
    # TODO: Implement theoretical Beta pdf (~5 lines)
    #
    # Step 1: Compute log normalization constant in log space (avoids overflow):
    #             log_C = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
    #             C = np.exp(log_C)
    # Step 2: Compute the power term:
    #             base  = np.maximum(1.0 - t**2, 0.0)
    #             power = (d - 3) / 2
    # Step 3: pdf = C * base ** power
    # Step 4: Return pdf
    #
    # Hint: For d=3, power=0 → f(t) = C (uniform on [-1,1]).
    #       For d≥4, the density is bell-shaped, unimodal at 0, tails at ±1.
    #       Do NOT use gamma_fn(d/2) directly — it overflows at d≳340.
    # =========================================================================
    raise NotImplementedError("Implement theoretical_beta_pdf")
    # =========================================================================


# ─────────────────────────────────────────────────────────────────────────────
# Supporting Functions (provided — no implementation needed)
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_approximation_pdf(t: np.ndarray, d: int) -> np.ndarray:
    """
    Gaussian approximation to the coordinate distribution: N(0, 1/d).

    The central limit theorem argument: each coordinate of a uniform unit vector
    is a sum-like quantity over d random signs, so it converges to N(0, 1/d).

    Useful for codebook design in high dimensions (e.g., d ≥ 64).

    Parameters
    ----------
    t : np.ndarray
    d : int

    Returns
    -------
    pdf : np.ndarray
    """
    variance = 1.0 / d
    std = np.sqrt(variance)
    return stats.norm.pdf(t, loc=0, scale=std)


def run_ks_test(rotated_coords: np.ndarray, d: int) -> tuple:
    """
    Kolmogorov-Smirnov test: does the empirical coordinate distribution match
    the theoretical Beta distribution?

    We test the null hypothesis H0: "the data follows the theoretical Beta pdf."
    A p-value > 0.05 means we CANNOT reject H0 — distributions match.
    A p-value < 0.05 means the distributions are statistically different.

    Parameters
    ----------
    rotated_coords : np.ndarray, shape (M,), flattened coordinate samples
    d              : int, vector dimension

    Returns
    -------
    ks_stat : float, KS statistic (smaller = better fit)
    p_value : float, p-value (larger = better fit, want > 0.05)
    """
    # We can't pass a custom pdf directly to scipy.stats.kstest for non-standard
    # distributions, so we use the CDF numerically.
    # Use the Beta distribution equivalence: X_coord = 2*Beta((d-1)/2,(d-1)/2) - 1
    alpha = (d - 1) / 2.0

    # Map samples from [-1,1] to [0,1] for the standard Beta CDF
    samples_01 = (rotated_coords + 1.0) / 2.0
    samples_01 = np.clip(samples_01, 0, 1)

    ks_stat, p_value = stats.kstest(
        samples_01,
        lambda x: stats.beta.cdf(x, alpha, alpha)
    )
    return float(ks_stat), float(p_value)


def plot_coordinate_distributions(results_by_d: dict, save_path: str):
    """
    Overlay empirical histogram, theoretical Beta pdf, and Gaussian approximation.

    Creates a 2×2 grid of subplots, one per dimension value in results_by_d.

    Parameters
    ----------
    results_by_d : dict  { d: {'coords': np.ndarray, 'Pi': np.ndarray} }
    save_path    : str, output PNG file path
    """
    dims = sorted(results_by_d.keys())
    n_plots = len(dims)
    ncols = 2
    nrows = (n_plots + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    fig.suptitle(
        "Rotated Coordinate Distribution: Beta pdf (blue) vs N(0,1/d) (orange)\n"
        "Empirical histogram (grey) should match both curves — they converge as d grows",
        fontsize=11, y=1.01
    )

    t_grid = np.linspace(-1, 1, 400)

    for ax, d in zip(axes, dims):
        coords = results_by_d[d]["coords"]

        # Empirical histogram
        ax.hist(coords, bins=60, density=True, color="lightgrey",
                edgecolor="white", linewidth=0.5, label="Empirical", alpha=0.9)

        # Theoretical Beta pdf
        beta_pdf = theoretical_beta_pdf(t_grid, d)
        ax.plot(t_grid, beta_pdf, color="steelblue", linewidth=2,
                label=f"Beta pdf (exact)")

        # Gaussian approximation
        gauss_pdf = gaussian_approximation_pdf(t_grid, d)
        ax.plot(t_grid, gauss_pdf, color="darkorange", linewidth=2,
                linestyle="--", label=f"N(0, 1/{d})")

        ks_stat, p_val = run_ks_test(coords, d)
        ax.set_title(f"d={d}   KS p-value={p_val:.4f}  {'✓ PASS' if p_val > 0.05 else '✗ FAIL'}",
                     fontsize=10)
        ax.set_xlabel("Coordinate value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(-1, 1)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [Saved plot → {save_path}]")


# ─────────────────────────────────────────────────────────────────────────────
# Main (Observable Milestone)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # =========================================================================
    # TRY: Change the dimensions to see the Beta→Gaussian convergence.
    #      At d=16, the Beta and Gaussian curves look clearly different.
    #      At d=128, they are nearly indistinguishable.
    #      At d=512, a histogram can't distinguish them at all.
    # =========================================================================
    dimensions = [16, 64, 128, 512]

    # =========================================================================
    # TRY: Increase N_vectors for a smoother histogram.
    #      The KS p-value stabilizes around N=1000 per dimension.
    # =========================================================================
    N_vectors = 2000      # vectors per dimension
    rng = np.random.default_rng(seed=7)

    print("\n" + "="*68)
    print("  EXERCISE 02: How Random Rotation Creates a Universal Distribution")
    print("  Claim: After rotation, each coordinate follows the Beta pdf.")
    print("="*68)

    results_by_d = {}

    print(f"\n  {'d':>6}  {'N coords':>10}  {'KS stat':>10}  {'p-value':>10}  {'Result':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    for d in dimensions:
        # Generate unit vectors (arbitrary — does NOT need to be uniform)
        # TRY: Change this to non-unit vectors and observe that results are
        #      identical after L2 normalization.
        X_raw = rng.standard_normal((N_vectors, d))
        norms  = np.linalg.norm(X_raw, axis=1, keepdims=True)
        X_unit = X_raw / norms                                  # shape (N_vectors, d)

        # Generate ONE random rotation matrix (shared across all vectors)
        Pi = generate_random_rotation(d, rng)                   # shape (d, d)

        # Rotate all vectors
        X_rot = rotate_vector(Pi, X_unit)                       # shape (N_vectors, d)

        # Collect ALL coordinates (d coordinates per vector)
        all_coords = X_rot.flatten()                            # shape (N_vectors * d,)

        # Statistical test
        ks_stat, p_val = run_ks_test(all_coords, d)
        result_str = "PASS ✓" if p_val > 0.05 else "FAIL ✗"

        print(f"  {d:>6}  {len(all_coords):>10,}  {ks_stat:>10.4f}  {p_val:>10.4f}  {result_str:>8}")

        results_by_d[d] = {"coords": all_coords, "Pi": Pi}

    print()
    print("  KS test: p-value > 0.05 → cannot reject that data follows Beta pdf.")
    print("  All dimensions should PASS — this is the mathematical guarantee.")

    # Inspect convergence at d=128 (or largest available)
    d_focus = max(d for d in dimensions if d <= 128) if any(d <= 128 for d in dimensions) else dimensions[-1]
    coords_focus = results_by_d[d_focus]["coords"]
    empirical_var = np.var(coords_focus)
    theoretical_var = 1.0 / d_focus
    print(f"\n  At d={d_focus}:")
    print(f"  ─ Empirical variance:    {empirical_var:.6f}")
    print(f"  ─ Theoretical (1/d):     {theoretical_var:.6f}")
    print(f"  ─ Relative error:        {abs(empirical_var - theoretical_var)/theoretical_var * 100:.2f}%")
    print()
    print("  KEY INSIGHT: At d=128, the Beta pdf and Gaussian N(0,1/128) are")
    print("  nearly identical. We can use the Gaussian approximation to design")
    print("  an optimal scalar quantizer — without ever seeing real data!")

    # Save plot
    plot_path = "milestone_02_rotation_distribution.png"
    plot_coordinate_distributions(results_by_d, save_path=plot_path)

    print()
    print("  NEXT: Exercise 03 — Are distinct coordinates actually independent?")
    print("        If yes, per-coordinate quantization loses nothing vs joint VQ.")
    print()
