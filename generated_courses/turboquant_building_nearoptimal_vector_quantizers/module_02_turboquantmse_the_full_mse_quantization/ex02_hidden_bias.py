"""
Exercise 02: The Hidden Bias — Why MSE-Optimal Quantizers Distort Inner Products
=================================================================================

TYPE: contrastive — you implement the NAIVE inner product estimator first,
      run it and observe a systematic distortion, then measure the bias precisely.

CLAIM: TurboQuant_mse achieves near-optimal MSE. But MSE-optimal ≠ inner-
       product-optimal. At 1-bit, the estimated inner product is only 2/π ≈ 63.7%
       of the true value. This bias is exact, not just noise — it is the central
       limitation that motivates the entire second half of the TurboQuant paper.

CONTEXT
───────
Transformer inference requires computing ⟨q, k⟩ between a query vector q and
thousands of cached key vectors k. If we quantize k → k̂ and compute ⟨q, k̂⟩
instead of ⟨q, k⟩, what do we lose?

Exercise 01 showed that the reconstruction error ‖k − k̂‖² is small (0.117 at
b=2). But reconstruction quality ≠ inner product fidelity. The question is:
    does  ⟨q, k̂⟩  ≈  ⟨q, k⟩  ?

This exercise answers that question empirically — and the answer is surprising.

THE CONTRASTIVE STRUCTURE
─────────────────────────
  PART 1 (NAIVE): Estimate inner products directly from dequantized vectors.
      → One fixed rotation matrix, one estimate per pair.
      → Plot estimated vs true: scatter cloud, hard to see systematic bias.

  PART 2 (BETTER): Average over K random rotations per pair to reduce variance.
      → With low variance, the systematic BIAS (slope ≠ 1) becomes unmistakable.
      → The slope is a multiplicative constant that depends only on b, not on
         the specific vectors or rotation.

  PART 3 (MEASURE): Fit a linear regression to quantify the bias slope.
      → At b=1: slope ≈ 2/π ≈ 0.637.  This is EXACT (we will prove it in Module 3).
      → At b=2,3,4: slope increases toward 1.0 but is still less than 1.

YOUR TASKS
──────────
  1. `estimate_ip_single_rotation`  — ~4 lines: quantize x, dequantize, dot with y
  2. `estimate_ip_multi_rotation`   — ~8 lines: average IPs over K random Π matrices
  3. `measure_bias`                 — ~12 lines: generate pairs, compute estimated
                                      and true IPs, fit linear regression
  4. Fill in the scatter plot section — ~4 lines using the provided plot skeleton
"""

import numpy as np
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ex01_assemble_turboquant_mse import TurboQuantMSE


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 (NAIVE): Single-Rotation Inner Product Estimator
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_ip_single_rotation(tq: TurboQuantMSE,
                                x: np.ndarray,
                                y: np.ndarray) -> float:
    """
    NAIVE: Estimate ⟨x, y⟩ by quantizing x and computing ⟨y, x̂⟩.

    This is the most obvious thing to do after quantization:
        1. Quantize x → indices using the TurboQuantMSE instance.
        2. Dequantize indices → x̂.
        3. Return ⟨y, x̂⟩ = y^T · x̂.

    There is NO averaging here — just one rotation matrix (baked into `tq`).
    The estimate is cheap but noisy AND biased. We will discover both
    properties in the milestone.

    Parameters
    ----------
    tq : TurboQuantMSE, a fully initialised quantizer (has a fixed Π inside)
    x  : np.ndarray, shape (d,), unit-norm query vector to be quantized
    y  : np.ndarray, shape (d,), unit-norm "key" vector (not quantized)

    Returns
    -------
    ip_estimate : float, estimated ⟨x, y⟩
    """
    # ========================================================================
    # TODO: Implement estimate_ip_single_rotation (~4 lines)
    #
    # Step 1: Quantize x  →  indices  using tq.quantize(x)
    # Step 2: Dequantize  →  x_hat   using tq.dequantize(indices)
    # Step 3: Return np.dot(y, x_hat)
    # ========================================================================
    raise NotImplementedError("Implement estimate_ip_single_rotation")
    # ========================================================================


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 (BETTER): Multi-Rotation Inner Product Estimator
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_ip_multi_rotation(x: np.ndarray,
                               y: np.ndarray,
                               b: int,
                               n_rotations: int = 30,
                               base_seed: int = 0) -> float:
    """
    Estimate ⟨x, y⟩ by averaging over K independent random rotation matrices.

    Why average? A single Π gives a noisy estimate. Different Π matrices
    introduce different quantization errors. Averaging over many Π matrices:
        (a) reduces variance (noise averages out)
        (b) reveals the SYSTEMATIC BIAS (which does NOT average out)

    Mathematics:
        E_Π[⟨y, DeQuant(Quant_Π(x))⟩] = bias_b · ⟨y, x⟩

    where bias_b is a constant that depends only on b, not on x, y, or Π.
    At b=1, bias_1 = 2/π exactly (we will derive this in Module 3).

    Parameters
    ----------
    x          : np.ndarray, shape (d,), unit-norm vector to be quantized
    y          : np.ndarray, shape (d,), unit-norm vector for the inner product
    b          : int, bits per coordinate
    n_rotations: int, number of independent rotation matrices to average over
    base_seed  : int, starting seed (seed k = base_seed + k)

    Returns
    -------
    ip_estimate : float, averaged inner product estimate
    """
    d = len(x)
    # ========================================================================
    # TODO: Implement estimate_ip_multi_rotation (~8 lines)
    #
    # Create n_rotations TurboQuantMSE instances with different seeds.
    # For each instance:
    #   - Compute the single-rotation IP estimate using estimate_ip_single_rotation
    # Return the mean of all estimates.
    #
    # Pseudocode:
    #   estimates = []
    #   for k in range(n_rotations):
    #       tq = TurboQuantMSE(d=d, b=b, seed=base_seed + k)
    #       estimates.append( estimate_ip_single_rotation(tq, x, y) )
    #   return np.mean(estimates)
    #
    # Hint: creating TurboQuantMSE calls lloyd_max once but generates a NEW
    #       rotation matrix Π for each seed. The codebook is recomputed each
    #       time (slow!). For efficiency, you can cache the codebook by creating
    #       ONE reference TurboQuantMSE and reusing its .codebook and .boundaries.
    #       But the simple loop above is correct and readable — use it first.
    # ========================================================================
    raise NotImplementedError("Implement estimate_ip_multi_rotation")
    # ========================================================================


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 (MEASURE): Fit Bias Slope via Linear Regression
# ═══════════════════════════════════════════════════════════════════════════════

def measure_bias(b: int,
                 d: int = 128,
                 n_pairs: int = 400,
                 n_rotations: int = 30,
                 seed: int = 7) -> tuple:
    """
    Measure the multiplicative bias of TurboQuant_mse inner product estimates.

    For `n_pairs` random unit vector pairs (x_k, y_k):
        - Compute the true inner product:     true_k  = ⟨x_k, y_k⟩
        - Compute the multi-rotation estimate: est_k  = estimate_ip_multi_rotation(x_k, y_k, b)
    Then fit a linear model:  est = bias · true + ε   (no intercept, by symmetry)
    and return the bias slope.

    The slope should be:
        b=1 → exactly 2/π ≈ 0.637 (proven in Module 3)
        b=2 → approaching 1.0 (verify empirically)
        b=3 → closer to 1.0
        b=4 → very close to 1.0

    Parameters
    ----------
    b          : int, bits per coordinate
    d          : int, vector dimension
    n_pairs    : int, number of random (x, y) pairs
    n_rotations: int, rotations to average per pair (reduces variance)
    seed       : int, random seed for generating unit vector pairs

    Returns
    -------
    true_ips      : np.ndarray, shape (n_pairs,), true inner products ⟨x_k, y_k⟩
    estimated_ips : np.ndarray, shape (n_pairs,), multi-rotation IP estimates
    bias_slope    : float, slope from linear regression (no intercept)
    """
    rng = np.random.default_rng(seed)

    # Generate n_pairs random unit vector pairs
    X = rng.standard_normal((n_pairs, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.standard_normal((n_pairs, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)

    # ========================================================================
    # TODO: Implement measure_bias (~12 lines)
    #
    # Step 1: Compute true inner products for all pairs.
    #   true_ips = np.array([np.dot(X[k], Y[k]) for k in range(n_pairs)])
    #
    # Step 2: Compute estimated inner products for all pairs using
    #         estimate_ip_multi_rotation.
    #   estimated_ips = np.array([
    #       estimate_ip_multi_rotation(X[k], Y[k], b=b, n_rotations=n_rotations,
    #                                  base_seed=k*n_rotations)
    #       for k in range(n_pairs)
    #   ])
    #   Note: using base_seed=k*n_rotations ensures non-overlapping seeds
    #         across pairs, so different pairs use genuinely different rotations.
    #
    # Step 3: Fit a no-intercept linear model: est = slope * true
    #   The least-squares no-intercept slope is:
    #       bias_slope = sum(true_ips * estimated_ips) / sum(true_ips ** 2)
    #   This is equivalent to np.polyfit(true_ips, estimated_ips, 1)[0] but
    #   for a model that passes through the origin (which is correct here
    #   because E[est] = 0 when E[true] = 0, by symmetry).
    #
    # Step 4: Return (true_ips, estimated_ips, bias_slope)
    #
    # Tip: this will take a few minutes for n_pairs=400, n_rotations=30 at b=1.
    #      Print progress every 50 pairs to stay sane.
    #      e.g. if k % 50 == 0: print(f"  pair {k}/{n_pairs}...", flush=True)
    # ========================================================================
    raise NotImplementedError("Implement measure_bias")
    # ========================================================================


# ═══════════════════════════════════════════════════════════════════════════════
# Scatter Plot Helper (provided)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_bias_discovery(results: dict, save_path: str):
    """
    Create a 2×2 scatter plot showing estimated vs true inner products for b=1,2,3,4.

    For each bit-width:
        x-axis: true ⟨x, y⟩
        y-axis: estimated ⟨y, x̂⟩ (averaged over rotations)
        red line: y = bias_slope · x   (fitted line)
        dashed grey: y = x             (ideal, unbiased line)

    Parameters
    ----------
    results   : dict  { b: (true_ips, estimated_ips, bias_slope) }
    save_path : str, output PNG path
    """
    fig = plt.figure(figsize=(10, 9))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    color_cycle = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

    for idx, b in enumerate([1, 2, 3, 4]):
        if b not in results:
            continue
        true_ips, estimated_ips, slope = results[b]

        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        # scatter
        ax.scatter(true_ips, estimated_ips,
                   alpha=0.35, s=12, color=color_cycle[idx], label="estimates")

        # ideal (unbiased) reference line
        lim = max(abs(true_ips).max(), abs(estimated_ips).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim],
                color="grey", linestyle="--", linewidth=1.0, label="ideal (slope=1)")

        # fitted bias line
        ax.plot([-lim, lim], [-lim * slope, lim * slope],
                color="black", linewidth=1.8, label=f"fit: slope={slope:.3f}")

        # reference: 2/pi line for b=1
        if b == 1:
            two_over_pi = 2.0 / np.pi
            ax.plot([-lim, lim], [-lim * two_over_pi, lim * two_over_pi],
                    color="orange", linestyle=":", linewidth=1.6,
                    label=f"2/π = {two_over_pi:.3f}")

        ax.set_xlabel("True ⟨x, y⟩", fontsize=9)
        ax.set_ylabel("Estimated ⟨y, x̂⟩", fontsize=9)
        ax.set_title(f"b={b} bit  |  bias slope = {slope:.4f}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper left")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim * 1.3, lim * 1.3)
        ax.axhline(0, color="grey", linewidth=0.4)
        ax.axvline(0, color="grey", linewidth=0.4)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Inner Product Bias of TurboQuant_mse\n"
        "Each plot: estimated vs true ⟨x, y⟩ for 400 random unit vector pairs\n"
        "Slope < 1.0 means systematic UNDERESTIMATION of the true inner product",
        fontsize=10, y=1.01
    )

    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [Saved scatter plot → {save_path}]")


# ═══════════════════════════════════════════════════════════════════════════════
# Observable Milestone
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Milestone: Discover and measure the multiplicative bias of TurboQuant_mse.

    The key observable:
      - At b=1, the regression slope ≈ 2/π ≈ 0.637 — the quantizer
        systematically underestimates inner products by 36.3%.
      - This is NOT noise; it is a deterministic property of MSE quantizers.
      - Higher bit-widths reduce the bias, but it never fully disappears.
      - The plot makes this unmistakably clear.

    Output: saves milestone_02_bias_discovery.png
    """
    import time

    d         = 128
    n_pairs   = 400     # TRY: increase to 1000 for a cleaner scatter
    n_rotations = 30    # TRY: increase to 100 for lower variance per point

    print()
    print("=" * 72)
    print("  Exercise 02: Discovering the Inner Product Bias of TurboQuant_mse")
    print("=" * 72)
    print(f"  Setup: d={d}, {n_pairs} random unit-vector pairs, "
          f"{n_rotations} rotations per pair")
    print()

    # ── PART 1: Show the bias visually with a single rotation ────────────────
    print("  PART 1 — Single-rotation estimates (one fixed Π):")
    tq1 = TurboQuantMSE(d=d, b=1, seed=99)
    rng_demo = np.random.default_rng(42)
    x_demo = rng_demo.standard_normal(d); x_demo /= np.linalg.norm(x_demo)
    y_demo = rng_demo.standard_normal(d); y_demo /= np.linalg.norm(y_demo)
    true_demo = float(np.dot(x_demo, y_demo))
    est_demo  = estimate_ip_single_rotation(tq1, x_demo, y_demo)
    print(f"    True ⟨x, y⟩ = {true_demo:.4f}")
    print(f"    Estimated   = {est_demo:.4f}  (single rotation, b=1)")
    print(f"    Ratio       = {est_demo / true_demo:.4f}  (2/π ≈ {2/np.pi:.4f})")
    print()

    # ── PART 2 & 3: Full bias measurement across b=1,2,3,4 ──────────────────
    results = {}
    print("  PART 2 & 3 — Multi-rotation estimates, bias measurement:")
    print(f"  (Computing {n_pairs} pairs × {n_rotations} rotations each — ~2-4 min)")
    print()

    for b in [1, 2, 3, 4]:
        t0 = time.time()
        print(f"  b={b}: running...", end=" ", flush=True)
        true_ips, estimated_ips, slope = measure_bias(
            b=b, d=d, n_pairs=n_pairs, n_rotations=n_rotations, seed=7
        )
        elapsed = time.time() - t0
        results[b] = (true_ips, estimated_ips, slope)
        print(f"slope = {slope:.4f}   ({elapsed:.1f}s)")

    print()
    print("  ┌─────┬─────────────────────────────────────────────────────────┐")
    print("  │  b  │  Bias slope  │  Reference  │  Bias (underestimate by)  │")
    print("  ├─────┼──────────────┼─────────────┼───────────────────────────┤")
    for b in [1, 2, 3, 4]:
        _, _, slope = results[b]
        ref = "2/π" if b == 1 else "  — "
        bias_pct = (1.0 - slope) * 100.0
        if b == 1:
            diff = abs(slope - 2.0 / np.pi)
            note = f"  (|diff from 2/π| = {diff:.5f})"
        else:
            note = ""
        print(f"  │  {b}  │   {slope:.4f}     │  {ref:>7}    │  "
              f"underestimates by {bias_pct:.1f}%{note}")
    print("  └─────┴──────────────┴─────────────┴───────────────────────────┘")
    print()
    print("  KEY INSIGHT:")
    print(f"  • The 1-bit bias = 2/π ≈ {2/np.pi:.4f} is exact (not just empirical).")
    print("  • Proof: Q_1bit(z) = sign(z)·c where c = E[|z|] for z ~ N(0,1/d).")
    print("           E[⟨y, Π^T·sign(Π·x)·c⟩] = (2/π)·⟨y, x⟩  [proven in Module 3]")
    print("  • Higher bits reduce but don't eliminate bias —")
    print("    MSE-optimal quantization is NOT inner-product-optimal.")
    print("  • This motivates TurboQuant_prod (Module 4): a two-stage quantizer")
    print("    that achieves UNBIASED inner product estimation.")
    print()

    # ── Save scatter plot ────────────────────────────────────────────────────
    plot_path = "milestone_02_bias_discovery.png"
    plot_bias_discovery(results, save_path=plot_path)
    print()
    print("  The scatter plot tells the whole story:")
    print("  • b=1: data points lie on a slope-0.637 line, not slope-1 (ideal).")
    print("  • b=4: points cluster near the slope-1 line — but still not perfect.")
    print()
    print("  NEXT: Exercise 03 — How does TurboQuant_mse perform on realistic")
    print("        KV cache vectors that are NOT unit-norm?")
    print("=" * 72)
