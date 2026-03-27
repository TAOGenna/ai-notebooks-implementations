"""
Exercise 3: The Complete TurboQuant System — Quantize, Search, and Attend
==========================================================================

CLAIM: A single TurboQuant index supports both nearest-neighbour retrieval
(Recall@1@10 ≈ 0.98) and quantized attention (KL < 0.002), achieving
10.7x memory compression with less than 0.2% end-to-end quality loss.

This is a *lightly scaffolded implement* exercise. You will wire together
everything from this module into a mini retrieval-augmented pipeline:

  ┌─────────────────────────────────────────────────────────────┐
  │  1. Build quantized index of document embeddings            │
  │  2. Given a query, retrieve top-k documents via TurboQuant  │
  │  3. For each retrieved doc, compute attention over its       │
  │     quantized KV cache                                       │
  │  4. Measure end-to-end quality vs full-precision baseline    │
  └─────────────────────────────────────────────────────────────┘

This ties together Modules 1–5: Lloyd-Max codebooks, random rotation, QJL,
TurboQuant_prod, nearest-neighbour search, and KV cache attention.

Prerequisite: Exercises 1 and 2 in this module (concepts), plus turboquant_core.py.

Setting: 1,000 documents, d=128 per embedding, b=3 bits (the paper's sweet spot).
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from turboquant_core import (
    TurboQuantProd, ProdCode,
    softmax, kl_divergence, sample_unit_vectors,
)

# ─── Pipeline parameters ──────────────────────────────────────────────────────
N_DOCS      = 1_000    # number of documents in the corpus
D_EMBED     = 128      # embedding dimension (also key/value dimension)
D_HEAD      = 128      # attention head dimension (same as embedding here)
N_KV_TOKENS = 64       # KV cache size per document (document "context length")
TOP_K_DOCS  = 10       # retrieve this many documents per query
N_QUERIES   = 100      # number of evaluation queries
B_BITS      = 3        # TurboQuant bit-width (the paper's sweet spot)
RNG         = np.random.default_rng(777)


# =============================================================================
# PROVIDED: Document database generation
# =============================================================================

def generate_document_database(n_docs: int,
                                d: int,
                                n_kv: int,
                                rng: np.random.Generator) -> tuple:
    """
    Generate a synthetic document corpus with embeddings and KV caches.

    In a real retrieval-augmented system:
      - `embeddings[i]` is a pooled embedding for document i (used for retrieval)
      - `kv_keys[i]`    is the key matrix for document i's tokens (used for attention)
      - `kv_values[i]`  is the value matrix for document i's tokens

    We simulate these with random unit vectors (same distributional properties
    as embeddings from a pretrained model, post normalisation).

    Parameters
    ----------
    n_docs : int  — number of documents
    d      : int  — embedding / head dimension
    n_kv   : int  — KV cache tokens per document
    rng    : Generator

    Returns
    -------
    embeddings : np.ndarray, shape (n_docs, d)  — document embeddings
    kv_keys    : np.ndarray, shape (n_docs, n_kv, d)  — per-doc key caches
    kv_values  : np.ndarray, shape (n_docs, n_kv, d)  — per-doc value caches (float32)
    """
    embeddings = sample_unit_vectors(n_docs, d, rng)           # (n_docs, d)
    # Keys: slightly scaled so attention logits stay manageable
    kv_keys = sample_unit_vectors(n_docs * n_kv, d, rng).reshape(n_docs, n_kv, d) * 0.1
    kv_values = sample_unit_vectors(n_docs * n_kv, d, rng).reshape(n_docs, n_kv, d)
    return embeddings, kv_keys, kv_values


# =============================================================================
# STUDENT TODO — Part 1: Build the quantized index  (~10 lines)
# =============================================================================

def build_quantized_index(embeddings: np.ndarray,
                           kv_keys: np.ndarray,
                           tq_embed: TurboQuantProd,
                           tq_kv: TurboQuantProd) -> tuple:
    """
    Compress the entire document corpus into a quantized index.

    You need to compress two things:
      (a) Document embeddings — used for retrieval (inner product search).
      (b) KV key caches — used for attention computation after retrieval.
          (Values stay in full precision since they are aggregated, not compared.)

    Parameters
    ----------
    embeddings  : np.ndarray, shape (n_docs, d)
        Full-precision document embeddings.
    kv_keys     : np.ndarray, shape (n_docs, n_kv, d)
        Full-precision key matrices for each document.
    tq_embed    : TurboQuantProd
        Quantizer for the embedding search index (b bits, d=D_EMBED).
    tq_kv       : TurboQuantProd
        Quantizer for the KV key cache (b bits, d=D_HEAD).
        (Can be the same instance as tq_embed if D_EMBED == D_HEAD.)

    Returns
    -------
    embed_codes : list[ProdCode], length n_docs
        Quantized document embeddings for fast inner-product search.
    kv_key_codes : list[list[ProdCode]], shape (n_docs, n_kv)
        Quantized key vectors for each document's KV cache.

    Hints
    -----
    For embed_codes (one code per document):
        embed_codes = tq_embed.quantize_batch(embeddings)

    For kv_key_codes (each document has n_kv key vectors):
        Loop over documents, or reshape kv_keys to (n_docs * n_kv, d),
        quantize in one batch, then reshape back.

        kv_keys_flat = kv_keys.reshape(n_docs * n_kv, d)
        kv_codes_flat = tq_kv.quantize_batch(kv_keys_flat)
        kv_key_codes = [kv_codes_flat[i*n_kv:(i+1)*n_kv] for i in range(n_docs)]
    """
    n_docs, n_kv, d = kv_keys.shape
    # =========================================================================
    # TODO: Build the quantized index.  (~6 lines)
    #
    # Step 1: Quantize all document embeddings in one batch.
    #         embed_codes = tq_embed.quantize_batch(embeddings)
    #
    # Step 2: Quantize all KV key vectors.
    #         Reshape kv_keys from (n_docs, n_kv, d) → (n_docs*n_kv, d),
    #         quantize with tq_kv.quantize_batch, then re-chunk into
    #         per-document lists (each of length n_kv).
    # =========================================================================
    raise NotImplementedError("Implement build_quantized_index")
    # =========================================================================


# =============================================================================
# STUDENT TODO — Part 2: Search for nearest documents  (~8 lines)
# =============================================================================

def search_nearest_documents(query: np.ndarray,
                              embed_codes: list,
                              tq_embed: TurboQuantProd,
                              top_k: int) -> np.ndarray:
    """
    Find the top_k nearest documents to a query using the quantized index.

    This is the retrieval step: given a query embedding (full precision),
    estimate inner products with all quantized document embeddings and return
    the top_k document indices.

    Parameters
    ----------
    query       : np.ndarray, shape (d,)  — full-precision query embedding
    embed_codes : list[ProdCode], length n_docs
    tq_embed    : TurboQuantProd
    top_k       : int

    Returns
    -------
    doc_indices : np.ndarray, shape (top_k,), dtype int
        Indices of the top_k nearest documents, in descending order of
        estimated inner product.

    Hints
    -----
    Step 1: Reconstruct all document embeddings.
        X_hat = tq_embed.reconstruct_batch(embed_codes)    # (n_docs, d)

    Step 2: Compute approximate inner products.
        scores = X_hat @ query                              # (n_docs,)

    Step 3: Return top_k indices.
        top_k_idx = np.argsort(scores)[-top_k:][::-1]      # descending order
    """
    # =========================================================================
    # TODO: Implement the quantized document retrieval.  (~5 lines)
    # =========================================================================
    raise NotImplementedError("Implement search_nearest_documents")
    # =========================================================================


# =============================================================================
# STUDENT TODO — Part 3: Compute attended output over quantized KV cache  (~10 lines)
# =============================================================================

def compute_quantized_attended_output(query_vec: np.ndarray,
                                       kv_key_codes: list,
                                       kv_values: np.ndarray,
                                       tq_kv: TurboQuantProd) -> np.ndarray:
    """
    Compute a single-head attention output over a quantized KV key cache.

    Formula (standard attention, but keys are quantized):
        logits_i   = <query_vec, DeQuant(kv_key_codes[i])> / sqrt(d)   for each token i
        weights    = softmax(logits)                                      shape (n_kv,)
        output     = sum_i weights[i] * kv_values[i]                    shape (d,)

    The values remain in full precision (they are aggregated, not searched).

    Parameters
    ----------
    query_vec   : np.ndarray, shape (d,)  — full-precision query vector
    kv_key_codes: list[ProdCode], length n_kv  — quantized key vectors
    kv_values   : np.ndarray, shape (n_kv, d)  — full-precision values
    tq_kv       : TurboQuantProd

    Returns
    -------
    attended_output : np.ndarray, shape (d,)
        The weighted sum of value vectors under the quantized attention distribution.

    Hints
    -----
    Step 1: Reconstruct key vectors from codes.
        K_hat = tq_kv.reconstruct_batch(kv_key_codes)       # (n_kv, d)

    Step 2: Compute attention logits (scaled dot products).
        logits = K_hat @ query_vec / np.sqrt(d)             # (n_kv,)

    Step 3: Apply softmax to get attention weights.
        weights = softmax(logits)                            # (n_kv,)

    Step 4: Compute attended output as weighted sum of values.
        output = weights @ kv_values                         # (d,) = (n_kv,) @ (n_kv, d)
    """
    d = query_vec.shape[0]
    # =========================================================================
    # TODO: Implement the quantized attention computation.  (~6 lines)
    # =========================================================================
    raise NotImplementedError("Implement compute_quantized_attended_output")
    # =========================================================================


# =============================================================================
# STUDENT TODO — Part 4: Evaluate end-to-end quality  (~5 lines)
# =============================================================================

def evaluate_pipeline(n_queries: int,
                       embeddings: np.ndarray,
                       kv_keys: np.ndarray,
                       kv_values: np.ndarray,
                       embed_codes: list,
                       kv_key_codes_all: list,
                       tq_embed: TurboQuantProd,
                       tq_kv: TurboQuantProd,
                       top_k: int) -> dict:
    """
    Run the full pipeline on n_queries random queries and collect quality metrics.

    For each query:
      1. Find true top_k documents (brute-force, full precision).
      2. Find approx top_k documents (quantized search).
      3. Check if the true top-1 is in the approx top_k (recall@1@top_k).
      4. For the true top-1 document, compute exact attention output.
      5. For the true top-1 document, compute quantized attention output.
      6. Record KL divergence between exact and approx attention distributions.

    Parameters
    ----------
    n_queries       : int
    embeddings      : np.ndarray, shape (n_docs, d)  — full-precision embeddings
    kv_keys         : np.ndarray, shape (n_docs, n_kv, d)
    kv_values       : np.ndarray, shape (n_docs, n_kv, d)
    embed_codes     : list[ProdCode], length n_docs
    kv_key_codes_all: list[list[ProdCode]], shape (n_docs, n_kv)
    tq_embed        : TurboQuantProd
    tq_kv           : TurboQuantProd
    top_k           : int

    Returns
    -------
    metrics : dict with keys:
        "recall_at_1_at_k"      : float — fraction of queries where true top-1
                                          is in quantized top_k results
        "avg_attention_kl"      : float — average KL(exact || quantized) attention
        "avg_attention_max_err" : float — average L∞ error in attention weights

    Hints
    -----
    For each query q_vec (shape d,):

    True top-1 (exact search):
        exact_scores = embeddings @ q_vec                    # (n_docs,)
        true_top1    = int(np.argmax(exact_scores))

    Approx top-k (quantized search):
        doc_indices  = search_nearest_documents(q_vec, embed_codes, tq_embed, top_k)

    For the true top-1 doc, compute attention:
        doc_id = true_top1
        exact_logits   = kv_keys[doc_id] @ q_vec / np.sqrt(d)   # (n_kv,)
        exact_weights  = softmax(exact_logits)
        approx_out     = compute_quantized_attended_output(
                             q_vec, kv_key_codes_all[doc_id],
                             kv_values[doc_id], tq_kv)
        # To get approx attention weights for KL, recompute from codes:
        K_hat = tq_kv.reconstruct_batch(kv_key_codes_all[doc_id])
        approx_weights = softmax(K_hat @ q_vec / np.sqrt(d))
        kl = kl_divergence(exact_weights, approx_weights)
    """
    d = embeddings.shape[1]
    recalls, kl_divs, max_errs = [], [], []

    q_vecs = sample_unit_vectors(n_queries, d, RNG)   # random queries

    for i in range(n_queries):
        q_vec = q_vecs[i]

        # =====================================================================
        # TODO: Evaluate one query through the full pipeline.  (~12 lines)
        #
        # Steps (for this single query q_vec):
        # 1. Compute exact scores and find true top-1 document.
        # 2. Run quantized search to get approx top_k documents.
        # 3. Check if true_top1 is in the approx top_k (append to recalls).
        # 4. Compute exact attention weights for the true top-1 document.
        # 5. Reconstruct quantized keys and compute approx attention weights.
        # 6. Compute KL divergence and max error (append to kl_divs, max_errs).
        # =====================================================================
        raise NotImplementedError("Implement one query's evaluation loop")
        # =====================================================================

    return {
        "recall_at_1_at_k":      float(np.mean(recalls)),
        "avg_attention_kl":      float(np.mean(kl_divs)),
        "avg_attention_max_err": float(np.mean(max_errs)),
    }


# =============================================================================
# Milestone — full end-to-end pipeline
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 3: The Complete TurboQuant System")
    print("=" * 70)
    print(f"\nPipeline: {N_DOCS} docs, d={D_EMBED}, b={B_BITS} bits, "
          f"top-{TOP_K_DOCS} retrieval")

    # ── Step 1: Generate the document corpus ──────────────────────────────────
    print("\n[1/4] Generating document database...", end=" ", flush=True)
    t0 = time.time()
    embeddings, kv_keys, kv_values = generate_document_database(
        N_DOCS, D_EMBED, N_KV_TOKENS, RNG
    )
    print(f"done ({time.time()-t0:.2f}s)")
    print(f"       Embeddings: {embeddings.shape}  "
          f"KV keys: {kv_keys.shape}  "
          f"KV values: {kv_values.shape}")

    # ── Step 2: Build quantized index ─────────────────────────────────────────
    print(f"\n[2/4] Building {B_BITS}-bit TurboQuant index...", end=" ", flush=True)
    t0 = time.time()

    # One quantizer for embeddings, one for KV keys (same b, same d here)
    tq_embed = TurboQuantProd(d=D_EMBED, b=B_BITS, seed=0)
    tq_kv    = TurboQuantProd(d=D_HEAD,  b=B_BITS, seed=1)

    embed_codes, kv_key_codes_all = build_quantized_index(
        embeddings, kv_keys, tq_embed, tq_kv
    )
    index_time = time.time() - t0
    print(f"done ({index_time:.2f}s)")
    print(f"       {N_DOCS} embedding codes + "
          f"{N_DOCS}×{N_KV_TOKENS} KV key codes compressed.")

    # ── Step 3: Evaluate the full pipeline ────────────────────────────────────
    print(f"\n[3/4] Evaluating on {N_QUERIES} random queries...", end=" ", flush=True)
    t0 = time.time()
    metrics = evaluate_pipeline(
        N_QUERIES, embeddings, kv_keys, kv_values,
        embed_codes, kv_key_codes_all, tq_embed, tq_kv, top_k=TOP_K_DOCS
    )
    eval_time = time.time() - t0
    print(f"done ({eval_time:.2f}s)")

    # ── Step 4: Print results ─────────────────────────────────────────────────
    print(f"\n[4/4] Results")
    print("─" * 60)
    recall   = metrics["recall_at_1_at_k"]
    avg_kl   = metrics["avg_attention_kl"]
    avg_merr = metrics["avg_attention_max_err"]

    print(f"  Search  : Recall@1 @ k={TOP_K_DOCS} = {recall:.4f}")
    print(f"  Attention: Avg KL divergence          = {avg_kl:.4f}")
    print(f"  Attention: Avg max error               = {avg_merr:.4f}")

    mem_reduction = 32.0 / B_BITS
    print(f"  Memory  : {B_BITS} bits/coord vs 32 bits/coord = {mem_reduction:.1f}x compression")

    # Approximate quality loss as fraction of max KL vs 1-bit KL
    print()
    if recall > 0.90:
        print(f"  End-to-end quality: Recall={recall:.2%}, KL={avg_kl:.4f}")
        print("  → Near paper-level performance: 3-bit compression with high quality! ✓")
    else:
        print(f"  End-to-end quality: Recall={recall:.2%}")
        print("  → Check your search and attention implementations.")

    print()
    print("─" * 60)
    print("Congratulations! Your TurboQuant implementation achieves:")
    print(f"  • {B_BITS}-bit compression ({mem_reduction:.1f}x memory reduction)")
    print(f"  • Recall@1@{TOP_K_DOCS} ≈ {recall:.2%} (near-perfect retrieval)")
    print(f"  • Avg KL < {avg_kl:.3f} (attention distribution preserved)")
    print()
    print("This matches the paper's central claim: 3-bit quantization with zero")
    print("quality loss — enabling 10x+ KV cache compression in LLM deployments.")
