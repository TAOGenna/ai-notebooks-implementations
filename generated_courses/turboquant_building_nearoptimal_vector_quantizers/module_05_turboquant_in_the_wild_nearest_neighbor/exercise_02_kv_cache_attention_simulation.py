"""
Exercise 2: Simulating KV Cache Attention — Does 3-Bit Quantization Preserve Softmax?
=======================================================================================

CLAIM: At 3 bits per coordinate, TurboQuant-quantized key vectors produce attention
weight distributions that are virtually indistinguishable from full-precision attention
(KL divergence < 0.002), confirming the paper's key result.

This is a *fill-in-the-blank* exercise. You will simulate a single-head attention
computation where the Key cache is compressed with TurboQuant at various bit-widths:

    Attention weights  = softmax( Q · K^T / sqrt(d) )
    Approx weights     = softmax( Q · K̂^T / sqrt(d) )   where K̂ = DeQuant(Quant(K))

You will measure KL(exact || approx) at b = 1, 2, 3, 4 bits and see that 3 bits
is the "quality-neutral" threshold from the paper.

Prerequisite: turboquant_core.py (in this module directory).

Setting: n = 512 cached tokens, d = 128 per head (standard for 7B-class LLMs).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from turboquant_core import TurboQuantProd, softmax, kl_divergence, sample_unit_vectors

# ─── Attention simulation parameters ─────────────────────────────────────────
N_TOKENS = 512     # sequence length (KV cache size)
D_HEAD   = 128     # dimension per attention head
RNG      = np.random.default_rng(42)


# =============================================================================
# PROVIDED: KV cache generation and exact attention
# =============================================================================

def generate_kv_cache(n: int, d: int, rng: np.random.Generator,
                       scale: float = 0.1) -> tuple:
    """
    Generate a synthetic KV cache and a single query vector.

    In a real transformer, keys are output of a linear projection of token
    embeddings. We mimic the scale and distribution with Gaussian vectors,
    normalised to have expected unit norm (the softmax temperature accounts
    for the 1/sqrt(d) scaling).

    Parameters
    ----------
    n     : int   — number of cached tokens (sequence length so far)
    d     : int   — head dimension
    rng   : Generator
    scale : float — key vector scale (keeps attention logits manageable)

    Returns
    -------
    keys   : np.ndarray, shape (n, d)  — key matrix
    values : np.ndarray, shape (n, d)  — value matrix (not quantized here)
    query  : np.ndarray, shape (d,)    — single query vector
    """
    # Draw unit vectors, then scale down (to avoid extreme softmax concentrations)
    keys   = sample_unit_vectors(n, d, rng) * scale   # (n, d)
    values = sample_unit_vectors(n, d, rng)            # (n, d) — full precision
    query  = sample_unit_vectors(1, d, rng)[0]        # (d,)
    return keys, values, query


def compute_exact_attention(query: np.ndarray,
                             keys: np.ndarray) -> np.ndarray:
    """
    Compute full-precision attention weights via softmax(Q · K^T / sqrt(d)).

    Parameters
    ----------
    query : np.ndarray, shape (d,)
    keys  : np.ndarray, shape (n, d)

    Returns
    -------
    weights : np.ndarray, shape (n,), sums to 1.0
    """
    d = query.shape[0]
    logits = keys @ query / np.sqrt(d)     # (n,) — inner product with each key
    return softmax(logits)


# =============================================================================
# STUDENT TODO — Exercise 2a: Quantize the key cache  (~6 lines)
# =============================================================================

def quantize_key_cache(keys: np.ndarray,
                        tq: TurboQuantProd) -> list:
    """
    Quantize all n key vectors using TurboQuantProd.

    In a real KV cache deployment, you would run this once when keys are written
    to cache, then keep only the compressed representation in GPU memory.

    Parameters
    ----------
    keys : np.ndarray, shape (n, d)  — full-precision key matrix
    tq   : TurboQuantProd

    Returns
    -------
    key_codes : list[ProdCode], length n
        One ProdCode per token. Each ProdCode stores the (b-1)-bit MSE indices,
        the 1-bit QJL sign bits, and the residual norm ||r||.

    Hint
    ----
    Use tq.quantize_batch(keys) — this is vectorized and much faster than
    calling tq.quantize(keys[i]) in a loop.
    """
    # =========================================================================
    # TODO: Quantize all n key vectors.  (~2 lines)
    #
    # key_codes = tq.quantize_batch(keys)
    # return key_codes
    # =========================================================================
    raise NotImplementedError("Implement quantize_key_cache")
    # =========================================================================


# =============================================================================
# STUDENT TODO — Exercise 2b: Compute approximate attention logits  (~6 lines)
# =============================================================================

def compute_quantized_attention(query: np.ndarray,
                                 key_codes: list,
                                 tq: TurboQuantProd) -> np.ndarray:
    """
    Compute approximate attention weights using quantized keys.

    Instead of the exact <query, key_i> inner product, we compute:
        logit_i = <query, DeQuant(key_codes[i])> / sqrt(d)

    This is the asymmetric estimator: the query stays in full precision (it is
    computed on-the-fly, not cached), while each key is reconstructed from its
    compressed code.

    Parameters
    ----------
    query     : np.ndarray, shape (d,)
    key_codes : list[ProdCode], length n
    tq        : TurboQuantProd

    Returns
    -------
    approx_weights : np.ndarray, shape (n,), sums to 1.0

    Hint
    ----
    Step 1 — Reconstruct all keys into an (n, d) matrix:
        K_hat = tq.reconstruct_batch(key_codes)          # (n, d)

    Step 2 — Compute approximate logits (same formula as exact attention):
        logits = K_hat @ query / np.sqrt(d)              # (n,)

    Step 3 — Apply softmax:
        approx_weights = softmax(logits)
        (Import softmax from turboquant_core — it handles numerical stability.)
    """
    # =========================================================================
    # TODO: Compute approximate attention weights.  (~4 lines)
    #
    # Step 1: Reconstruct quantized keys into a float matrix.
    # Step 2: Compute inner products with the query, scaled by 1/sqrt(d).
    # Step 3: Apply softmax to obtain a probability distribution.
    # =========================================================================
    raise NotImplementedError("Implement compute_quantized_attention")
    # =========================================================================


# =============================================================================
# STUDENT TODO — Exercise 2c: Measure KL divergence  (~4 lines)
# =============================================================================

def compare_attention_distributions(exact_weights: np.ndarray,
                                     approx_weights: np.ndarray) -> dict:
    """
    Quantify how different the quantized attention distribution is from exact.

    Parameters
    ----------
    exact_weights  : np.ndarray, shape (n,) — exact softmax attention weights
    approx_weights : np.ndarray, shape (n,) — approximate weights from quantized keys

    Returns
    -------
    metrics : dict with keys:
        "kl_divergence"    : float — KL(exact || approx)  (lower is better)
        "max_error"        : float — max |exact_i - approx_i|  (L∞ error)

    Hint
    ----
    KL divergence: use kl_divergence(exact_weights, approx_weights) from
    turboquant_core. Note the direction: KL(p || q) where p=exact, q=approx.

    Max error: np.max(np.abs(exact_weights - approx_weights))
    """
    # =========================================================================
    # TODO: Compute the two metrics above.  (~4 lines)
    #
    # kl = kl_divergence(exact_weights, approx_weights)
    # max_err = np.max(np.abs(exact_weights - approx_weights))
    # return {"kl_divergence": kl, "max_error": max_err}
    # =========================================================================
    raise NotImplementedError("Implement compare_attention_distributions")
    # =========================================================================


# =============================================================================
# Milestone — run this to reproduce the paper's KV cache quality claim
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 2: KV Cache Attention Simulation")
    print("=" * 70)
    print(f"\nSetup: n={N_TOKENS} cached tokens, d={D_HEAD} per head (single query)")
    print("Generating synthetic KV cache...")

    keys, values, query = generate_kv_cache(N_TOKENS, D_HEAD, RNG, scale=0.1)

    print("Computing exact attention weights (full precision)...")
    exact_weights = compute_exact_attention(query, keys)

    print()
    print("─" * 70)
    print(f" {'Bit-width':<10} | {'KL divergence':>14} | {'Max attn error':>14} | "
          f"{'Mem reduction':>14}")
    print("─" * 70)

    bit_widths = [1, 2, 3, 4]
    all_metrics = {}

    for b in bit_widths:
        # Build a TurboQuantProd for this bit-width
        tq = TurboQuantProd(d=D_HEAD, b=b, seed=0)

        # Quantize the key cache
        key_codes = quantize_key_cache(keys, tq)

        # Compute approximate attention
        approx_weights = compute_quantized_attention(query, key_codes, tq)

        # Measure divergence
        metrics = compare_attention_distributions(exact_weights, approx_weights)

        # Memory reduction: full-precision is 32 bits/coord, quantized is b bits/coord
        # (ignoring shared rotation/JL matrix overhead — amortised over all tokens)
        mem_reduction = 32.0 / b

        kl  = metrics["kl_divergence"]
        mae = metrics["max_error"]
        print(f" b={b:<9} | {kl:>14.4f} | {mae:>14.4f} | {mem_reduction:>13.1f}x")
        all_metrics[b] = metrics

    print("─" * 70)

    # ── Key findings ──────────────────────────────────────────────────────────
    print()
    if 3 in all_metrics:
        kl_3bit = all_metrics[3]["kl_divergence"]
        if kl_3bit < 0.002:
            print(f"At 3 bits: KL divergence = {kl_3bit:.4f} < 0.002")
            print("→ Attention is virtually identical to full precision! ✓")
        else:
            print(f"At 3 bits: KL divergence = {kl_3bit:.4f}")
            print("→ Close to full precision — check your softmax and QJL scaling.")

    if 4 in all_metrics and 2 in all_metrics:
        kl_4 = all_metrics[4]["kl_divergence"]
        kl_2 = all_metrics[2]["kl_divergence"]
        ratio = kl_2 / kl_4 if kl_4 > 0 else float("inf")
        print(f"\nKL divergence ratio b=2 vs b=4: {ratio:.1f}x more error at lower bits.")

    print()
    print("Connections to the paper:")
    print("  • At 3.5 bits, TurboQuant achieves 'zero quality loss' on LLM benchmarks.")
    print("  • The KL divergence threshold < 0.002 corresponds to this neutrality claim.")
    print("  • Below 3 bits, the softmax amplifies quantization error for 'important' tokens")
    print("    (high-attention tokens), which is why needle-in-a-haystack degrades at 2-bit.")
    print()
    print("Memory reduction table:")
    for b in bit_widths:
        mem = 32.0 / b
        print(f"  b={b}: {mem:.1f}x compression (32-bit float → {b}-bit TurboQuant)")
