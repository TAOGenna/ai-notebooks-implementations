"""
Exercise 03: Quantizing Real Embeddings — MSE on Synthetic KV Cache Vectors
============================================================================

TYPE: fill_blank — implement the three marked functions, then run the milestone.

CLAIM: TurboQuant_mse handles arbitrary-norm vectors by storing ‖x‖ separately
       at 16-bit precision. The norm storage overhead is 16/(b·d) bits per
       coordinate — shrinking with dimension d, unlike traditional block-wise
       quantizers that need a scale per block of 32-128 elements.

CONTEXT: KV Cache Vectors
──────────────────────────
In autoregressive transformers (GPT, LLaMA, etc.), each attention layer stores
key and value embeddings for all prior tokens — the "KV cache". During decoding,
the model computes ⟨query, key_i⟩ for every cached key to determine attention.

The memory bottleneck: the KV cache grows as:
    n_layers × n_heads × d_head × seq_len × 2 × sizeof(float)
    e.g. 32 layers × 32 heads × 128 dim × 4096 tokens × 2 × 2 bytes = 1 GB

Compressing keys from 32 bits → 2 bits = 16× reduction in KV cache memory.

KV cache vectors are NOT pure unit vectors:
  1. They have non-uniform norms (different tokens have different magnitudes).
  2. Some "outlier channels" have systematically larger magnitudes due to
     LayerNorm saturation (a known phenomenon in LLMs).

TurboQuant's solution: normalise → quantize → store norm separately.

The pipeline for arbitrary-norm vectors:
  Store:
    1. Compute norm n = ‖x‖.
    2. Normalise: x_unit = x / n.
    3. Quantize the unit vector: indices = Quant(x_unit).
    4. Store (indices, n) — indices at b bits/coord, norm at 16 bits.

  Retrieve:
    1. Dequantize: x_hat_unit = DeQuant(indices).
    2. Rescale: x_hat = n · x_hat_unit.

YOUR TASKS
──────────
  1. `store_kv_vector`     — ~4 lines: normalise x, quantize, return (indices, norm)
  2. `retrieve_kv_vector`  — ~4 lines: dequantize, rescale by stored norm
  3. `compute_relative_error` — ~3 lines: ‖x − x_hat‖ / ‖x‖
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ex01_assemble_turboquant_mse import TurboQuantMSE


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic KV Cache Vector Generator (provided — no changes needed)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_kv_cache_vectors(n: int,
                               d: int = 128,
                               n_outlier_channels: int = 16,
                               outlier_scale: float = 6.0,
                               seed: int = 42) -> np.ndarray:
    """
    Generate synthetic key/value cache vectors that mimic realistic LLM KV cache
    embeddings.

    Statistical properties (matching empirical observations in LLM KV caches):
        • Most channels:     entries ~ N(0, 0.5²)   (small, regular channels)
        • Outlier channels:  entries ~ N(0, (0.5 · outlier_scale)²)
          Outlier channels arise from LayerNorm saturation — a few channels
          systematically carry much larger activations.
        • Vector norms:      ‖x‖ ~ Gamma(shape=2, scale=1.0) (non-unit, variable)
          After LayerNorm, the norms are distributed, not fixed at 1.

    The outlier channels are selected as the first `n_outlier_channels` channels
    (indices 0..n_outlier_channels-1) for reproducibility.

    Parameters
    ----------
    n                  : int, number of vectors to generate
    d                  : int, vector dimension (default 128)
    n_outlier_channels : int, how many "outlier" channels have large magnitude
    outlier_scale      : float, magnification of outlier channel std dev
    seed               : int, random seed

    Returns
    -------
    X : np.ndarray, shape (n, d), KV-like embedding vectors (NOT unit-norm)
    """
    rng  = np.random.default_rng(seed)
    base_std = 0.5

    # Base: all channels small-magnitude
    X = rng.normal(0.0, base_std, size=(n, d))

    # Outlier channels: larger magnitude (multiply the existing entries)
    X[:, :n_outlier_channels] *= outlier_scale

    # Variable norms: scale each vector by a sample from Gamma distribution
    # (Gamma(2, 1.0) has mean=2, std=√2, range=(0, ∞) — mimics positive norms)
    norms = rng.gamma(shape=2.0, scale=1.0, size=n)   # shape (n,)
    current_norms = np.linalg.norm(X, axis=1)          # shape (n,)
    # Rescale so the target norm is drawn from the Gamma distribution
    scale_factors = norms / current_norms              # shape (n,)
    X *= scale_factors[:, np.newaxis]

    return X


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1: Store (Compress) a KV Vector
# ═══════════════════════════════════════════════════════════════════════════════

def store_kv_vector(tq: TurboQuantMSE,
                    x: np.ndarray) -> tuple:
    """
    Compress a single KV cache vector for storage.

    Pipeline (one-way):
        x  →  normalise  →  x_unit  →  Quant  →  indices
               ↓ save                              ↓ save
             norm (16 bits)                    indices (b·d bits)

    The norm is stored at 16-bit float precision (np.float16), which costs
    16 bits per vector regardless of dimension d. At d=128 and b=2 bits:
        - Quantized representation: 2 × 128 = 256 bits
        - Norm storage:              16 bits
        - Total:                    272 bits
        - Original:                 32 × 128 = 4096 bits
        - Compression:              4096 / 272 ≈ 15.1× (vs. ideal 16× without norm)

    The norm overhead amortizes to 16 / (b × d) extra bits per coordinate:
        d=128, b=2: 16 / 256 = 0.0625 extra bits/coord → 2.0625 effective bits

    Parameters
    ----------
    tq : TurboQuantMSE, the quantizer (holds Π and the codebook)
    x  : np.ndarray, shape (d,), input KV vector (arbitrary norm, non-zero)

    Returns
    -------
    indices : np.ndarray, shape (d,), int centroid indices
    norm    : np.float16, stored norm ‖x‖ in 16-bit precision

    Hints
    -----
    • Compute norm: norm_val = np.linalg.norm(x)
    • Normalise: x_unit = x / norm_val
    • Quantize: indices = tq.quantize(x_unit)
    • Store norm as 16-bit float: norm = np.float16(norm_val)
    """
    # ========================================================================
    # TODO: Implement store_kv_vector (~4 lines)
    #
    # Step 1: Compute the L2 norm of x.
    #         norm_val = np.linalg.norm(x)
    #
    # Step 2: Normalise x to unit norm.
    #         x_unit = x / norm_val
    #
    # Step 3: Quantize the unit-norm vector.
    #         indices = tq.quantize(x_unit)
    #
    # Step 4: Return (indices, np.float16(norm_val))
    # ========================================================================
    raise NotImplementedError("Implement store_kv_vector")
    # ========================================================================


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2: Retrieve (Decompress) a KV Vector
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_kv_vector(tq: TurboQuantMSE,
                       indices: np.ndarray,
                       norm: np.float16) -> np.ndarray:
    """
    Reconstruct a KV cache vector from its compressed representation.

    Pipeline (reverse):
        indices  →  DeQuant  →  x_hat_unit  →  rescale by norm  →  x_hat
              ↑                                        ↑
        (b·d bits)                              (16-bit float)

    Parameters
    ----------
    tq      : TurboQuantMSE, the same quantizer used during storage
    indices : np.ndarray, shape (d,), int centroid indices from store_kv_vector
    norm    : np.float16, stored norm from store_kv_vector

    Returns
    -------
    x_hat : np.ndarray, shape (d,), reconstructed KV vector with original norm scale

    Hints
    -----
    • Dequantize: x_hat_unit = tq.dequantize(indices)
    • Rescale: x_hat = float(norm) * x_hat_unit
      (cast norm back to float64 to avoid accumulation of float16 rounding)
    """
    # ========================================================================
    # TODO: Implement retrieve_kv_vector (~4 lines)
    #
    # Step 1: Dequantize indices to get the unit-norm reconstruction.
    #         x_hat_unit = tq.dequantize(indices)
    #
    # Step 2: Rescale by the stored norm (cast to float64 first).
    #         x_hat = float(norm) * x_hat_unit
    #
    # Step 3: Return x_hat.
    # ========================================================================
    raise NotImplementedError("Implement retrieve_kv_vector")
    # ========================================================================


# ═══════════════════════════════════════════════════════════════════════════════
# Task 3: Compute Relative Reconstruction Error
# ═══════════════════════════════════════════════════════════════════════════════

def compute_relative_error(x: np.ndarray, x_hat: np.ndarray) -> float:
    """
    Compute the normalised reconstruction error ‖x − x̂‖ / ‖x‖.

    This is the relative error — it measures how large the reconstruction
    error is compared to the original vector's magnitude.

    Unlike the absolute MSE ‖x − x̂‖², the relative error is scale-invariant:
    if you double the norm of x (and x̂ proportionally), the relative error
    stays the same. This makes it a better quality metric when comparing
    vectors with varying norms (as in the KV cache).

    For unit vectors (‖x‖ = 1), relative error = ‖x − x̂‖ = sqrt(MSE).

    Parameters
    ----------
    x     : np.ndarray, shape (d,), original vector
    x_hat : np.ndarray, shape (d,), reconstructed vector

    Returns
    -------
    rel_error : float, ‖x − x̂‖ / ‖x‖  (a value in [0, ∞); 0 = perfect)
    """
    # ========================================================================
    # TODO: Implement compute_relative_error (~3 lines)
    #
    # Step 1: Compute ‖x − x̂‖ using np.linalg.norm.
    # Step 2: Compute ‖x‖ using np.linalg.norm.
    # Step 3: Return error / norm_x
    # ========================================================================
    raise NotImplementedError("Implement compute_relative_error")
    # ========================================================================


# ═══════════════════════════════════════════════════════════════════════════════
# Provided analysis helper
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_kv_compression(tq: TurboQuantMSE,
                            X: np.ndarray) -> dict:
    """
    Compress and reconstruct all vectors in X; compute error statistics.

    Parameters
    ----------
    tq : TurboQuantMSE
    X  : np.ndarray, shape (n, d), KV cache vectors

    Returns
    -------
    stats : dict with keys:
        'abs_mse'       : mean ‖x − x̂‖²             (absolute MSE)
        'rel_error_mean': mean ‖x − x̂‖ / ‖x‖         (mean relative error)
        'rel_error_std' : std  ‖x − x̂‖ / ‖x‖
        'norm_mean'     : mean ‖x‖ of original vectors
        'norm_std'      : std  ‖x‖
        'norm16_error'  : mean |‖x‖ − float16(‖x‖)| / ‖x‖  (norm quantisation error)
    """
    n = len(X)
    abs_mses     = np.zeros(n)
    rel_errors   = np.zeros(n)
    orig_norms   = np.zeros(n)
    norm16_errs  = np.zeros(n)

    for i, x in enumerate(X):
        indices, norm16 = store_kv_vector(tq, x)
        x_hat = retrieve_kv_vector(tq, indices, norm16)

        abs_mses[i]    = np.sum((x - x_hat) ** 2)
        rel_errors[i]  = compute_relative_error(x, x_hat)
        orig_norms[i]  = np.linalg.norm(x)
        norm16_errs[i] = abs(float(np.linalg.norm(x)) - float(norm16)) / float(np.linalg.norm(x))

    return {
        "abs_mse":        float(np.mean(abs_mses)),
        "rel_error_mean": float(np.mean(rel_errors)),
        "rel_error_std":  float(np.std(rel_errors)),
        "norm_mean":      float(np.mean(orig_norms)),
        "norm_std":       float(np.std(orig_norms)),
        "norm16_error":   float(np.mean(norm16_errs)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Observable Milestone
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Milestone: Compare TurboQuant_mse performance on unit vectors vs realistic
    KV cache vectors with non-uniform norms and outlier channels.

    Expected output pattern (after implementing all three TODOs):

      Unit vectors (b=2):
        MSE: 0.117   |  Relative error: 3.4%

      KV-like vectors (b=2, norm storage enabled):
        Absolute MSE: 0.47   (= unit MSE × avg_norm²)
        Relative error: 3.4%  ← SAME as unit vectors (norm-invariant)
        Norm storage overhead: 16 bits / (2 × 128 bits) = 0.0625 bits/coord
        Effective bit-width: 2.0625 bits for 3.4% relative error

    KEY INSIGHT: Storing the norm separately costs almost nothing (16/128d),
    and the relative error is the same whether the input is unit-norm or not.
    Traditional block quantizers (e.g. KIVI, INT8 with scale/zero-point) add
    1-2 bits/coord overhead that does NOT shrink with d — TurboQuant wins at
    high dimension.
    """
    d = 128
    n = 1000
    b = 2   # TRY: change to 1, 3, 4 and see how relative error scales

    print()
    print("=" * 72)
    print(f"  Exercise 03: TurboQuant_mse on KV Cache Vectors  |  d={d}, n={n}")
    print("=" * 72)

    # ── Build quantizer ───────────────────────────────────────────────────────
    tq = TurboQuantMSE(d=d, b=b, seed=42)

    # ── BASELINE: unit vectors ───────────────────────────────────────────────
    print(f"\n  [1] Baseline — random unit vectors (b={b}):")
    rng = np.random.default_rng(0)
    X_unit = rng.standard_normal((n, d))
    X_unit /= np.linalg.norm(X_unit, axis=1, keepdims=True)

    stats_unit = analyze_kv_compression(tq, X_unit)
    print(f"      Absolute MSE:     {stats_unit['abs_mse']:.4f}")
    print(f"      Relative error:   {stats_unit['rel_error_mean']*100:.2f}% "
          f"± {stats_unit['rel_error_std']*100:.2f}%")
    print(f"      Avg ‖x‖:          {stats_unit['norm_mean']:.4f} "
          f"(expected: 1.0000 for unit vectors)")
    print(f"      Norm quant error: {stats_unit['norm16_error']*100:.4f}% "
          f"(float16 vs float32 norm)")

    # ── KV CACHE VECTORS ──────────────────────────────────────────────────────
    print(f"\n  [2] Realistic KV cache vectors (b={b}, with outlier channels):")
    X_kv = generate_kv_cache_vectors(n, d=d, n_outlier_channels=16,
                                     outlier_scale=6.0, seed=42)
    avg_norm  = float(np.mean(np.linalg.norm(X_kv, axis=1)))
    norm_std  = float(np.std(np.linalg.norm(X_kv, axis=1)))

    stats_kv = analyze_kv_compression(tq, X_kv)
    print(f"      Avg ‖x‖:          {stats_kv['norm_mean']:.3f} "
          f"± {stats_kv['norm_std']:.3f}  (non-unit norms)")
    print(f"      Absolute MSE:     {stats_kv['abs_mse']:.4f}  "
          f"(= unit MSE × avg_norm² ≈ "
          f"{stats_unit['abs_mse'] * stats_kv['norm_mean']**2:.4f})")
    print(f"      Relative error:   {stats_kv['rel_error_mean']*100:.2f}% "
          f"± {stats_kv['rel_error_std']*100:.2f}%")
    print(f"      Norm quant error: {stats_kv['norm16_error']*100:.4f}% "
          f"(float16 norm storage)")

    # ── OVERHEAD ANALYSIS ─────────────────────────────────────────────────────
    print(f"\n  [3] Storage overhead analysis (b={b}, d={d}):")
    quant_bits   = b * d                      # bits for indices
    norm_bits    = 16                         # bits for float16 norm
    total_bits   = quant_bits + norm_bits     # total compressed
    orig_bits    = 32 * d                     # original float32
    effective_bpc = total_bits / d            # effective bits per coordinate

    print(f"      Quantized indices:  {b} × {d} = {quant_bits} bits")
    print(f"      Norm (float16):           {norm_bits} bits")
    print(f"      Total compressed:         {total_bits} bits")
    print(f"      Original (float32):       {orig_bits} bits")
    print(f"      Effective compression:    {orig_bits / total_bits:.1f}×")
    print(f"      Norm overhead per coord:  {norm_bits}/{quant_bits} = "
          f"{norm_bits/quant_bits:.4f} bits/coord")
    print(f"      Effective bit-width:      {effective_bpc:.4f} bits/coord "
          f"(vs ideal {b} bits/coord)")

    # ── COMPARISON ACROSS BIT-WIDTHS ─────────────────────────────────────────
    print(f"\n  [4] Relative error vs bit-width (KV-like vectors, n={n}):")
    print(f"  {'b':>4}  {'Rel Error':>12}  {'Eff Compression':>17}  "
          f"{'Overhead (bits/coord)':>22}")
    print("  " + "-" * 64)
    for b_test in [1, 2, 3, 4]:
        tq_test  = TurboQuantMSE(d=d, b=b_test, seed=42)
        stats_t  = analyze_kv_compression(tq_test, X_kv[:300])  # subset for speed
        eff_bpc  = (b_test * d + 16) / d
        eff_comp = (32 * d) / (b_test * d + 16)
        overhead = 16 / (b_test * d)
        print(f"  {b_test:>4}  {stats_t['rel_error_mean']*100:>10.2f}%  "
              f"{eff_comp:>14.1f}×  "
              f"{overhead:>21.4f}")
    print()

    # ── KEY INSIGHT ───────────────────────────────────────────────────────────
    print("  KEY INSIGHTS:")
    print(f"  1. Relative error ≈ {stats_unit['rel_error_mean']*100:.1f}% (b=2) regardless of ‖x‖.")
    print("     Absolute MSE scales with ‖x‖² — it's NOT a good quality metric for")
    print("     variable-norm vectors. Always report RELATIVE error for KV caches.")
    print()
    print(f"  2. Norm overhead shrinks with d: {16}/{b*d} = "
          f"{16/(b*d):.4f} bits/coord at d={d}, b={b}.")
    print("     Traditional block quantizers need 1-2 bits/coord for scale+zero-point,")
    print("     regardless of block size. TurboQuant wins as d grows.")
    print()
    print("  3. The 16-bit norm introduces < 0.01% additional error —")
    print("     completely negligible vs the quantization error itself.")
    print()
    print("  NEXT: Module 3 — The QJL Sign-Bit Trick: achieving UNBIASED")
    print("        inner product estimation with just 1 bit per coordinate.")
    print("=" * 72)
