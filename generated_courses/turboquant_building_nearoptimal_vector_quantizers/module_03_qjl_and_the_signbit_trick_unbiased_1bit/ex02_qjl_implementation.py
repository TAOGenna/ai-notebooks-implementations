"""
Exercise 2: From Full Precision to Sign Bits — Implementing QJL
===============================================================
CLAIM: Quantizing to a SINGLE sign bit per coordinate, after a random
       Gaussian projection, still yields an UNBIASED inner product
       estimator — and the variance bound is exactly pi/(2d) * ||y||^2.

The QJL transform (Quantized Johnson-Lindenstrauss) is defined as:

    Quantize:   Q(x)  =  sign(S * x)          ← 1 bit per dimension
    Dequantize: Q^{-1}(z) = sqrt(pi/2) / d * S^T * z

The asymmetric inner product estimator uses the FULL-PRECISION query y
with the SIGN-BIT quantized key Q(x):

    <y, Q^{-1}(Q(x))>  =  sqrt(pi/2)/d * sum_i [ (S_i . y) * sign(S_i . x) ]

where S_i is the i-th row of S.

Why does this work?
    E[ sign(S_i . x) * (S_i . y) ]
        = E[ sign(N_x) * N_y ]     where (N_x, N_y) are jointly Gaussian
        = sqrt(2/pi) * <x, y>      (a standard identity)
    Multiply by sqrt(pi/2)/d and sum d terms → E = <x, y>  (unbiased!)

The variance bound is:
    Var ≤ pi/(2d) * ||y||^2 * ||x||^2        (from the paper, Thm 2)
    (when x, y are unit-norm: Var ≤ pi/(2d))

Dependencies: Builds on ex01 concepts. Reuses the same vector construction.

Run:
    python ex02_qjl_implementation.py
"""

import numpy as np


class QJL:
    """
    Quantized Johnson-Lindenstrauss (QJL) transform.

    Memory layout for the key x:
        - d sign bits  (1 bit/coord → d/8 bytes total, vs 4d bytes for float32)
        - The random matrix S is SHARED between encoder and decoder;
          no quantization constants are stored.  That is the "zero overhead."

    The design is ASYMMETRIC on purpose:
        Keys are quantized to 1 bit.
        Queries remain full-precision at inference time.
    This asymmetry is what keeps the estimator unbiased.

    Parameters
    ----------
    d : int
        Input (and output) dimension.
    seed : int
        Seed used to generate the FIXED random matrix S.
        Both encoder and decoder must use the SAME seed — this is the
        "shared randomness" that makes QJL work with zero storage overhead.
    """

    def __init__(self, d: int, seed: int = 42):
        self.d = d
        rng = np.random.default_rng(seed)
        # S is the shared (d x d) Gaussian matrix.
        # Rows S_i will be used as the "random directions."
        self.S = rng.standard_normal((d, d))   # shape: (d, d)

    # ------------------------------------------------------------------
    def quantize(self, x: np.ndarray) -> np.ndarray:
        """
        Quantize vector x to d sign bits.

        Step 1: project  z_raw = S @ x             shape: (d,)
        Step 2: take sign  z = sign(z_raw)         values in {-1, +1}

        In practice you'd pack these into uint8 bits; here we store {-1,+1}
        as int8 for clarity.

        Parameters
        ----------
        x : ndarray, shape (d,)  — the KEY vector (e.g., a KV-cache key)

        Returns
        -------
        z : ndarray of int8, shape (d,)  — 1 bit per coord, stored as ±1
        """
        # ====================================================================
        # TODO: Implement quantize.                                        (~2 lines)
        #
        # 1. Compute the projection: z_raw = self.S @ x
        # 2. Return np.sign(z_raw).astype(np.int8)
        #
        # Edge case: np.sign(0) = 0; in practice this almost never happens
        # with Gaussian projections and continuous x, but you may replace
        # zeros with +1 if you want strictly ±1 outputs.
        # ====================================================================
        raise NotImplementedError("Implement quantize")
        # ====================================================================

    # ------------------------------------------------------------------
    def dequantize(self, z: np.ndarray) -> np.ndarray:
        """
        Reconstruct an approximate version of x from its sign bits.

            Q^{-1}(z) = sqrt(pi/2) / d * S^T @ z

        This reconstruction is NOT used in the inner product estimator
        (see estimate_inner_product below); it is provided here to show
        the dequantization formula from the paper.

        Parameters
        ----------
        z : ndarray, shape (d,)  — sign bits in {-1, +1}

        Returns
        -------
        x_hat : ndarray, shape (d,)  — reconstructed vector
        """
        # ====================================================================
        # TODO: Implement dequantize.                                      (~1 line)
        #
        # Hint: (sqrt(pi/2) / d) * self.S.T @ z
        #       np.sqrt(np.pi / 2) gives you the constant.
        # ====================================================================
        raise NotImplementedError("Implement dequantize")
        # ====================================================================

    # ------------------------------------------------------------------
    def estimate_inner_product(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the ASYMMETRIC inner product estimate.

            <y, Q^{-1}(Q(x))>
                = sqrt(pi/2)/d * (S^T @ sign(S @ x)) . y
                = sqrt(pi/2)/d * sum_i [ (S_i . y) * sign(S_i . x) ]

        This is the KEY formula in QJL:
          - x is the KEY   (compressed to sign bits)
          - y is the QUERY (kept at full precision)

        The expected value equals <x, y> exactly (unbiased), and the
        variance is at most pi/(2d) * ||x||^2 * ||y||^2.

        Parameters
        ----------
        x : ndarray, shape (d,)  — the key (will be quantized internally)
        y : ndarray, shape (d,)  — the query (full precision)

        Returns
        -------
        ip_hat : float
        """
        # ====================================================================
        # TODO: Implement the asymmetric estimator.                        (~3 lines)
        #
        # Step 1: quantize x  → z = self.quantize(x)
        # Step 2: dequantize  → x_hat = self.dequantize(z)   (reuse your method)
        # Step 3: return np.dot(y, x_hat)
        #
        # Note: this is equivalent to:
        #   np.sqrt(np.pi/2) / self.d * np.dot(self.S.T @ z, y)
        # Feel free to use either form.
        # ====================================================================
        raise NotImplementedError("Implement estimate_inner_product")
        # ====================================================================


# ============================================================================
# Helper: run a bias/variance experiment over many DIFFERENT vector pairs
# ============================================================================

def run_qjl_experiment(
    d: int = 128,
    n_pairs: int = 100,
    seed: int = 7,
) -> dict:
    """
    For n_pairs different (x, y) pairs of unit-norm vectors, compute the
    QJL inner product estimate and record the error.

    Returns statistics that let us verify:
        (a) bias  ≈ 0
        (b) variance ≈ pi/(2d)  (for unit-norm vectors)

    Parameters
    ----------
    d       : int — head dimension
    n_pairs : int — number of (x, y) pairs to test
    seed    : int — RNG seed

    Returns
    -------
    results : dict
        'errors'           — array of (estimate - true) for each pair
        'mean_error'       — mean of errors (should ≈ 0)
        'mean_sq_error'    — average squared error (≈ variance when bias≈0)
        'theory_var_bound' — pi / (2 * d)
    """
    rng = np.random.default_rng(seed)
    qjl = QJL(d=d, seed=seed + 1)  # different seed from pair generation

    errors = []
    for _ in range(n_pairs):
        x = rng.standard_normal(d); x /= np.linalg.norm(x)
        y = rng.standard_normal(d); y /= np.linalg.norm(y)
        true_ip = float(np.dot(x, y))
        est = qjl.estimate_inner_product(x, y)
        errors.append(est - true_ip)

    errors = np.array(errors)
    theory_var = np.pi / (2 * d)

    return {
        "errors":           errors,
        "mean_error":       float(np.mean(errors)),
        "mean_sq_error":    float(np.mean(errors ** 2)),
        "theory_var_bound": theory_var,
    }


# ============================================================================
# MILESTONE
# ============================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    d = 128
    rng = np.random.default_rng(42)

    # Reference vectors (unit norm, same as ex01)
    x = rng.standard_normal(d); x /= np.linalg.norm(x)
    y = rng.standard_normal(d); y /= np.linalg.norm(y)
    true_ip = float(np.dot(x, y))

    qjl = QJL(d=d, seed=0)

    # ------ 1. sanity-check a single estimate
    z = qjl.quantize(x)
    x_hat = qjl.dequantize(z)
    ip_est = qjl.estimate_inner_product(x, y)

    print("=" * 60)
    print("QJL SINGLE-SHOT SANITY CHECK")
    print("=" * 60)
    print(f"  True inner product    : {true_ip:.4f}")
    print(f"  QJL estimate          : {ip_est:.4f}")
    print(f"  |z| unique values     : {np.unique(z.astype(int))}  (should be ±1)")
    print(f"  Memory: sign bits use {d} bits = {d//8} bytes  "
          f"(vs {d*4} bytes for float32)  → {d*4/(d//8):.0f}x compression")
    print()

    # ------ 2. bias/variance experiment across many pairs
    print("=" * 60)
    print(f"QJL BIAS/VARIANCE CHECK  (d={d}, 500 vector pairs)")
    print("=" * 60)
    res = run_qjl_experiment(d=d, n_pairs=500, seed=3)
    theory_std = np.sqrt(res["theory_var_bound"])

    print(f"  Mean error (bias)     : {res['mean_error']:.5f}   (should ≈ 0)")
    print(f"  Mean squared error    : {res['mean_sq_error']:.5f}")
    print(f"  Theory variance bound : {res['theory_var_bound']:.5f}  (= π/(2d))")
    print(f"  Theory std bound      : {theory_std:.5f}  (= √(π/(2d)))")
    within_bound = res["mean_sq_error"] <= res["theory_var_bound"] * 1.2
    print(f"  MSE ≤ variance bound  : {'✓' if within_bound else '✗'}")
    print()

    # ------ 3. show compression savings
    bits_original  = d * 32          # float32
    bits_quantized = d * 1           # 1 bit per coord
    print("=" * 60)
    print("MEMORY SUMMARY")
    print("=" * 60)
    print(f"  Original vector  : {bits_original} bits  ({bits_original//8} bytes)")
    print(f"  QJL sign bits    : {bits_quantized} bits  ({bits_quantized//8} bytes)")
    print(f"  Compression ratio: {bits_original / bits_quantized:.0f}x")
    print(f"  Quantization consts needed: 0  (zero overhead ✓)")
    print()
    print("Key insight: 1 bit/coord, unbiased, variance π/(2d) — ready to")
    print("see how the ASYMMETRIC design beats the naive symmetric estimator (ex03).")
