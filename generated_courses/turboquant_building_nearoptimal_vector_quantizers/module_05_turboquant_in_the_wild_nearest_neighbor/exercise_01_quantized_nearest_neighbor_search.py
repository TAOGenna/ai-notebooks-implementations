"""
Exercise 1: Quantized Nearest Neighbor Search — TurboQuant Beats Product Quantization
=======================================================================================

CLAIM: TurboQuant achieves higher recall than scalar Product Quantization at the same
bit-width, despite knowing nothing about the data — and indexes 1000x faster.

This is a *comparative* exercise. You will:
  1. Implement a nearest-neighbor search pipeline using TurboQuant_prod
     (quantize the database, compute approximate inner products, return top-k).
  2. Implement the same pipeline using scalar Product Quantization
     (train per-coordinate k-means, encode database, decode to reconstruct, search).
  3. Compute Recall@k for k = 1, 2, 4, 8, 16, 32, 64 for BOTH methods at
     bit-widths b = 2 and b = 4.

Milestone output:
  A recall table comparing TurboQuant vs PQ, plus the indexing time comparison.
  TurboQuant should achieve higher Recall@1 at both bit-widths.

Prerequisite: turboquant_core.py (in this module directory).

Setting: 10,000 database vectors, d = 200 (mimicking GloVe 200d word embeddings).
         100 query vectors. True nearest neighbor = brute-force inner product.
"""

import sys
import os
import time
import numpy as np

# ─── Import TurboQuantProd and helpers ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from turboquant_core import TurboQuantProd, sample_unit_vectors

# Optional: matplotlib for the recall curve plot
try:
    import matplotlib
    matplotlib.use("Agg")            # headless backend — no display needed
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─── Dataset parameters ───────────────────────────────────────────────────────
N_DB    = 10_000   # database size
N_QUERY = 100      # number of queries
D       = 200      # dimension (mimicking GloVe-200)
KS      = [1, 2, 4, 8, 16, 32, 64]   # values of k for Recall@k evaluation
RNG     = np.random.default_rng(2024)


# =============================================================================
# PROVIDED: Scalar Product Quantization baseline
# =============================================================================

class ProductQuantization:
    """
    Scalar Product Quantization (PQ) baseline.

    Treats each coordinate as an independent 1D subspace and trains a
    2^n_bits-level k-means codebook on the data per coordinate.

    This is the data-adaptive counterpart to TurboQuant's data-oblivious
    approach. PQ learns optimal levels from the training data; TurboQuant
    rotates to make the distribution predictable and uses a pre-computed codebook.

    Total bits per vector = d × n_bits  (same as TurboQuant at same bit-width).

    Parameters
    ----------
    d      : int  — vector dimension
    n_bits : int  — bits per coordinate (2^n_bits levels per coordinate)
    """

    def __init__(self, d: int, n_bits: int):
        self.d = d
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits
        self.codebooks: np.ndarray | None = None   # shape: (d, n_levels)

    def fit(self, data: np.ndarray) -> None:
        """
        Train per-coordinate codebooks by running 1D Lloyd's algorithm.

        For each coordinate i, runs k-means to find the n_levels centroids
        that minimise the expected squared quantization error for that dimension.

        Parameters
        ----------
        data : np.ndarray, shape (N, d)
        """
        N, d = data.shape
        assert d == self.d, f"Expected d={self.d}, got {d}"
        self.codebooks = np.zeros((d, self.n_levels))

        for i in range(d):
            col = data[:, i]                                          # (N,)
            # Initialise centroids at equally-spaced percentiles
            percs = np.linspace(2, 98, self.n_levels)
            cents = np.percentile(col, percs)

            # Lloyd's algorithm (fast for 1D data)
            for _ in range(80):
                diffs = np.abs(col[:, None] - cents[None, :])        # (N, k)
                labels = diffs.argmin(axis=1)                         # (N,)
                new_cents = np.array(
                    [col[labels == j].mean() if (labels == j).any() else cents[j]
                     for j in range(self.n_levels)]
                )
                if np.max(np.abs(new_cents - cents)) < 1e-9:
                    break
                cents = new_cents

            self.codebooks[i] = np.sort(cents)

    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode vectors to per-coordinate centroid indices.

        Parameters
        ----------
        data : np.ndarray, shape (N, d)

        Returns
        -------
        indices : np.ndarray, shape (N, d), dtype int32
        """
        assert self.codebooks is not None, "Call fit() first."
        N, d = data.shape
        # Chunked vectorised assignment to keep memory bounded
        chunk = 1000
        indices = np.zeros((N, d), dtype=np.int32)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            block = data[start:end]                                   # (cs, d)
            # (cs, d, n_levels) distance tensor — feasible for small chunk
            diffs = np.abs(block[:, :, None] - self.codebooks[None, :, :])
            indices[start:end] = diffs.argmin(axis=2)
        return indices

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Reconstruct vectors from codebook indices.

        Parameters
        ----------
        indices : np.ndarray, shape (N, d), dtype int

        Returns
        -------
        reconstructed : np.ndarray, shape (N, d)
        """
        # Fancy indexing: codebooks[dim, index] for all (N, d)
        # codebooks: (d, n_levels); indices: (N, d)
        # output[n, i] = codebooks[i, indices[n, i]]
        return self.codebooks[
            np.arange(self.d)[None, :],   # (1, d)
            indices                        # (N, d)
        ]                                  # → (N, d)


# =============================================================================
# PROVIDED: Ground truth + evaluation utilities
# =============================================================================

def compute_exact_top1(queries: np.ndarray,
                        database: np.ndarray) -> np.ndarray:
    """
    Brute-force exact nearest neighbour (by inner product).

    Parameters
    ----------
    queries  : np.ndarray, shape (Q, d)
    database : np.ndarray, shape (N, d)

    Returns
    -------
    true_top1 : np.ndarray, shape (Q,), dtype int
        Index of the true nearest database vector for each query.
    """
    scores = queries @ database.T    # (Q, N) — all pairwise inner products
    return scores.argmax(axis=1)


def compute_recall_at_k(true_top1: np.ndarray,
                         approx_top_k: np.ndarray) -> float:
    """
    Recall@1@k — fraction of queries where true top-1 appears in approx top-k.

    Parameters
    ----------
    true_top1   : np.ndarray, shape (Q,) — index of true nearest neighbour per query
    approx_top_k: np.ndarray, shape (Q, k) — top-k indices from the approx method

    Returns
    -------
    recall : float in [0, 1]
    """
    hits = 0
    Q = len(true_top1)
    for i in range(Q):
        if true_top1[i] in approx_top_k[i]:
            hits += 1
    return hits / Q


def plot_recall_curves(results: dict,
                        ks: list,
                        save_path: str) -> None:
    """
    Save a recall-curve plot comparing TurboQuant and PQ at each bit-width.

    Parameters
    ----------
    results   : dict mapping label (str) → list of recall values for each k
    ks        : list of int — x-axis values
    save_path : str — file path to save the PNG
    """
    if not HAS_MPL:
        print("  [matplotlib not installed — skipping plot]")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "TQ 4-bit": dict(color="#1f77b4", linestyle="-",  marker="o"),
        "PQ 4-bit": dict(color="#ff7f0e", linestyle="--", marker="s"),
        "TQ 2-bit": dict(color="#2ca02c", linestyle="-",  marker="^"),
        "PQ 2-bit": dict(color="#d62728", linestyle="--", marker="D"),
    }
    for label, recalls in results.items():
        style = styles.get(label, {})
        ax.plot(ks, recalls, label=label, **style)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("k  (number of returned candidates)", fontsize=12)
    ax.set_ylabel("Recall@1@k", fontsize=12)
    ax.set_title(f"TurboQuant vs Product Quantization\n"
                 f"N={N_DB} vectors, d={D}", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# =============================================================================
# STUDENT TODO — Exercise 1a: TurboQuant search pipeline (~8 lines)
# =============================================================================

def turboquant_search(queries: np.ndarray,
                       db_codes: list,
                       tq: TurboQuantProd,
                       top_k: int) -> np.ndarray:
    """
    Approximate nearest-neighbour search using pre-quantized TurboQuant codes.

    The database has already been encoded: db_codes[i] is the ProdCode for
    database vector i. Your task is to:
      1. Reconstruct all N database vectors from their codes into one (N, d) matrix.
      2. For each query, compute the inner product with every database vector.
      3. Return the top_k indices with the highest inner product for each query.

    Parameters
    ----------
    queries  : np.ndarray, shape (Q, d) — full-precision query vectors
    db_codes : list[ProdCode], length N — pre-computed TurboQuant codes
    tq       : TurboQuantProd           — quantizer used for encoding
    top_k    : int                      — number of candidates to return

    Returns
    -------
    top_k_indices : np.ndarray, shape (Q, top_k)
        For each query, the indices of the top_k approximate nearest neighbours,
        in descending order of estimated inner product.

    Hints
    -----
    • Use tq.reconstruct_batch(db_codes) to get the (N, d) reconstruction matrix.
    • Inner products: queries @ reconstructed.T  → shape (Q, N)
    • np.argpartition(scores, -top_k)[-top_k:] gives top_k indices efficiently
      (or np.argsort(...)[-top_k:] for simplicity).
    """
    # =========================================================================
    # TODO: Implement the TurboQuant search pipeline.  (~8 lines)
    #
    # Step 1: Decode all N database codes into a matrix.
    #         X_hat = tq.reconstruct_batch(db_codes)     # (N, d)
    #
    # Step 2: Compute inner products between every query and every database vector.
    #         scores = queries @ X_hat.T                  # (Q, N)
    #
    # Step 3: For each query (each row of scores), find the top_k indices
    #         with the highest scores.  Return them in a (Q, top_k) array.
    #
    # Hint for step 3:
    #   Using argpartition (faster):
    #     for each row, np.argpartition(row, -top_k)[-top_k:]
    #   Or using argsort (simpler, slightly slower for large k):
    #     np.argsort(row)[-top_k:]
    # =========================================================================
    raise NotImplementedError("Implement turboquant_search")
    # =========================================================================


# =============================================================================
# STUDENT TODO — Exercise 1b: Product Quantization search pipeline (~8 lines)
# =============================================================================

def pq_search(queries: np.ndarray,
               pq: ProductQuantization,
               db_encoded: np.ndarray,
               top_k: int) -> np.ndarray:
    """
    Approximate nearest-neighbour search using pre-trained Product Quantization.

    The database has already been encoded: db_encoded[i] contains the
    per-coordinate centroid indices for database vector i. Your task is to:
      1. Decode all N encoded vectors back to their approximate float vectors.
      2. Compute inner products between every query and every decoded vector.
      3. Return the top_k indices with the highest inner product for each query.

    Parameters
    ----------
    queries    : np.ndarray, shape (Q, d)
    pq         : ProductQuantization  — trained PQ object (contains codebooks)
    db_encoded : np.ndarray, shape (N, d), dtype int — pre-encoded database
    top_k      : int

    Returns
    -------
    top_k_indices : np.ndarray, shape (Q, top_k)

    Hints
    -----
    • Use pq.decode(db_encoded) to reconstruct the (N, d) float matrix.
    • Then proceed exactly as in turboquant_search.
    """
    # =========================================================================
    # TODO: Implement the PQ search pipeline.  (~8 lines)
    #
    # Step 1: Decode database codes to float vectors.
    #         X_hat = pq.decode(db_encoded)              # (N, d)
    #
    # Step 2: Compute inner products.
    #         scores = queries @ X_hat.T                  # (Q, N)
    #
    # Step 3: Find top_k indices for each query (same as turboquant_search).
    # =========================================================================
    raise NotImplementedError("Implement pq_search")
    # =========================================================================


# =============================================================================
# STUDENT TODO — Exercise 1c: Recall comparison loop (~5 lines)
# =============================================================================

def run_recall_comparison(queries: np.ndarray,
                           database: np.ndarray,
                           true_top1: np.ndarray,
                           bit_widths: list) -> dict:
    """
    Run the full recall comparison between TurboQuant and PQ at all bit-widths
    and all k values in KS.

    For each bit-width b:
      • Index the database with TurboQuant_prod (b bits/coord)
      • Index the database with ProductQuantization (b bits/coord)
      • For each k in KS, compute Recall@1@k for both methods

    Parameters
    ----------
    queries    : np.ndarray, shape (Q, d)
    database   : np.ndarray, shape (N, d)
    true_top1  : np.ndarray, shape (Q,)   — ground-truth top-1 indices
    bit_widths : list[int]                 — e.g. [2, 4]

    Returns
    -------
    results : dict
        Keys are strings like "TQ 4-bit", "PQ 4-bit", "TQ 2-bit", "PQ 2-bit".
        Values are lists of recall values, one per k in KS.
    timing  : dict
        Keys are "TQ 4-bit", "PQ 4-bit", etc.
        Values are indexing time in seconds.

    Hints
    -----
    For TurboQuant indexing:
      t0 = time.time()
      tq = TurboQuantProd(d=D, b=b, seed=0)
      db_codes = tq.quantize_batch(database)
      tq_time = time.time() - t0

    For PQ indexing:
      t0 = time.time()
      pq = ProductQuantization(d=D, n_bits=b)
      pq.fit(database)                     # <-- the slow step (k-means training)
      db_encoded = pq.encode(database)
      pq_time = time.time() - t0

    For recall at each k:
      top_k_tq = turboquant_search(queries, db_codes, tq, top_k=k)
      top_k_pq = pq_search(queries, pq, db_encoded, top_k=k)
      recall_tq = compute_recall_at_k(true_top1, top_k_tq)
      recall_pq = compute_recall_at_k(true_top1, top_k_pq)
    """
    results = {}
    timing = {}

    for b in bit_widths:
        print(f"\n  [b={b}] Indexing with TurboQuant... ", end="", flush=True)
        # =====================================================================
        # TODO: Build the TurboQuant index and measure indexing time.  (~4 lines)
        #
        # 1. Record start time: t0 = time.time()
        # 2. Create TurboQuantProd(d=D, b=b, seed=0)
        # 3. Quantize the database: tq.quantize_batch(database)
        # 4. Record elapsed: tq_time = time.time() - t0
        # =====================================================================
        raise NotImplementedError("Build TurboQuant index here")
        # =====================================================================
        print(f"{tq_time:.3f}s")
        timing[f"TQ {b}-bit"] = tq_time

        print(f"  [b={b}] Training PQ codebooks... ", end="", flush=True)
        # NOTE: Fill BOTH TODO blocks (TurboQuant and PQ) before running —
        # they both define variables used by the recall loop below.
        # =====================================================================
        # TODO: Build the PQ index and measure training time.  (~5 lines)
        #
        # 1. Record start time.
        # 2. Create ProductQuantization(d=D, n_bits=b)
        # 3. Call pq.fit(database)     — this runs Lloyd's algorithm per coordinate
        # 4. Encode the database: pq.encode(database)
        # 5. Record elapsed.
        # =====================================================================
        raise NotImplementedError("Build PQ index here")
        # =====================================================================
        print(f"{pq_time:.3f}s")
        timing[f"PQ {b}-bit"] = pq_time

        tq_recalls, pq_recalls = [], []
        for k in KS:
            top_k_tq = turboquant_search(queries, db_codes, tq, top_k=k)
            top_k_pq = pq_search(queries, pq, db_encoded, top_k=k)
            tq_recalls.append(compute_recall_at_k(true_top1, top_k_tq))
            pq_recalls.append(compute_recall_at_k(true_top1, top_k_pq))

        results[f"TQ {b}-bit"] = tq_recalls
        results[f"PQ {b}-bit"] = pq_recalls

    return results, timing


# =============================================================================
# Milestone — run this to see TurboQuant vs PQ in action
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: TurboQuant Beats Product Quantization in Recall")
    print("=" * 70)
    print(f"\nDataset: {N_DB} database vectors, {N_QUERY} queries, d={D}")
    print("Generating synthetic data (mimicking GloVe-200)...")

    database = sample_unit_vectors(N_DB, D, RNG)
    queries  = sample_unit_vectors(N_QUERY, D, RNG)

    print("Computing exact top-1 neighbours (brute force)...")
    true_top1 = compute_exact_top1(queries, database)

    print("\nRunning comparison at b=2 and b=4 bits/coordinate...")
    results, timing = run_recall_comparison(
        queries, database, true_top1, bit_widths=[2, 4]
    )

    # ── Print recall table ──────────────────────────────────────────────────
    print()
    print("─" * 70)
    k_header = " | ".join(f"k={k:>3}" for k in KS)
    print(f" {'Method':<14} | {k_header}")
    print("─" * 70)
    for label in ["TQ 4-bit", "PQ 4-bit", "TQ 2-bit", "PQ 2-bit"]:
        if label not in results:
            continue
        recalls = " | ".join(f"{r:>6.3f}" for r in results[label])
        print(f" {label:<14} | {recalls}")
    print("─" * 70)

    # ── Print timing comparison ──────────────────────────────────────────────
    print("\nIndexing time (time to make the database searchable):")
    for label, t in timing.items():
        print(f"  {label:<12}: {t:.3f}s")

    if "TQ 4-bit" in timing and "PQ 4-bit" in timing:
        tq_t = timing["TQ 4-bit"]
        pq_t = timing["PQ 4-bit"]
        if tq_t > 0:
            speedup = pq_t / tq_t
            print(f"\n  TurboQuant is ~{speedup:.0f}x faster to index!")
            print(f"  Why? TurboQuant is data-oblivious: no k-means training needed.")
            print(f"  PQ must fit 200 separate 1D k-means models before it can answer queries.")

    # ── Key insight ──────────────────────────────────────────────────────────
    print()
    if "TQ 4-bit" in results and "PQ 4-bit" in results:
        tq_r1 = results["TQ 4-bit"][0]     # Recall@1 (k=1)
        pq_r1 = results["PQ 4-bit"][0]
        print(f"Recall@1 at 4 bits: TurboQuant={tq_r1:.3f}, PQ={pq_r1:.3f}")
        if tq_r1 > pq_r1:
            print("→ TurboQuant wins despite having NO data-specific tuning!")
        else:
            print("→ Results may vary with dataset; try b=4 with more queries.")

    print()
    print("Key lesson: The random rotation is more valuable than data-adapted codebooks.")
    print("By equalising coordinate variances, TurboQuant turns a hard problem (uneven")
    print("distributions) into a solved one (Beta/Gaussian distribution, known optimal codebook).")

    # ── Plot ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__),
                            "milestone_01_recall_comparison.png")
    plot_recall_curves(results, KS, out_path)
    if HAS_MPL:
        print(f"\nRecall curve saved to: {out_path}")
