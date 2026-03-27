# Module 3: QJL and the Sign-Bit Trick — Unbiased 1-Bit Inner Products

## The Core Claim

> A single sign bit per coordinate, applied after a random Gaussian projection,
> yields an **unbiased** inner product estimator with provably bounded variance —
> and requires **zero** stored quantization constants.

In Module 2 you discovered that MSE-optimal quantizers are **biased** for inner
products. A 1-bit MSE quantizer has a multiplicative bias of 2/π ≈ 0.637.
This module fixes that by introducing the **Quantized Johnson-Lindenstrauss (QJL)**
transform — a completely different 1-bit scheme built for inner products, not MSE.

---

## Prerequisites

- Module 0 (vector quantization fundamentals, distortion metrics)
- Module 2 (inner product bias of MSE quantizers) — conceptually helpful but not a code dependency

---

## What You Will Build

| Exercise | Claim | Key Output |
|---|---|---|
| ex01 | Random Gaussian projections are unbiased inner product estimators | Empirical mean ≈ true IP, variance ≈ theory for all m |
| ex02 | Quantizing to 1 sign bit preserves unbiasedness with variance π/(2d) | Bias ≈ 0, MSE ≤ π/(2·128) ≈ 0.0123 |
| ex03 | The asymmetric estimator (full-precision query) is essential | 4× lower variance than naive symmetric sign-bit estimator |

---

## How to Work Through This Module

1. **Read** the module-level docstring in each file before writing any code.
2. **Implement** the `TODO` blocks in order — each file builds on the previous.
3. **Run** the milestone at the bottom of each file before moving to the next.
4. Check that your numbers match the expected ranges shown in the milestone output.

```bash
python ex01_jl_foundation.py
python ex02_qjl_implementation.py
python ex03_qjl_vs_naive.py
```

---

## Notation

| Symbol | Meaning |
|--------|---------|
| **S** | Random Gaussian matrix, shape (d × d), entries ~ N(0, 1). Shared between encoder and decoder via a fixed seed. |
| **x**, **y** | Input vectors ∈ ℝᵈ (typically unit-norm). **x** is the vector to quantize (key), **y** is the query (full precision). |
| d | Vector dimension (e.g., 128 for a typical attention head). |
| Q(x) | QJL quantization: sign(S · x) ∈ {−1, +1}ᵈ — one sign bit per coordinate. |
| Q⁻¹(z) | QJL dequantization: √(π/2)/d · Sᵀ · z — maps sign bits back to ℝᵈ. |
| Sᵢ | The i-th row of S, a d-dimensional Gaussian vector. |
| ‖x‖ | Euclidean (L2) norm of x. |

---

## Concepts Covered

### The Johnson-Lindenstrauss Lemma (ex01)

If **S** is a (d × d) matrix with i.i.d. entries ~ N(0,1), then for any vectors
**x**, **y** ∈ ℝᵈ:

```
E[ <S·x, S·y> / d ]  =  <x, y>                         (unbiased)

Var[ <S·x, S·y> / d ] = (‖x‖²·‖y‖² + <x,y>²) / d
```

The variance shrinks as 1/d — more projection dimensions means tighter estimates.
This is the **foundation** of QJL.

### QJL: Quantized Johnson-Lindenstrauss Transform (ex02)

Replace the full-precision projected vector with just its **sign**:

```
Quantize:    Q(x)   = sign(S · x)           ← d bits, stored as {-1,+1}
Dequantize:  Q⁻¹(z) = sqrt(π/2) / d · Sᵀ · z
```

The **asymmetric** inner product estimator:

```
<y, Q⁻¹(Q(x))>  =  sqrt(π/2)/d · Σᵢ (Sᵢ · y) · sign(Sᵢ · x)
```

**Why is it unbiased?** For any row Sᵢ of S, the products (Sᵢ·x) and (Sᵢ·y)
are jointly Gaussian with:

```
E[sign(Sᵢ·x) · (Sᵢ·y)]  =  sqrt(2/π) · <x, y>
```

This is a standard result from Gaussian geometry. Multiplying by sqrt(π/2)/d
and summing d independent terms gives exactly E = <x,y>. ✓

**Variance bound** (Theorem 2 in the paper):

```
Var ≤ π/(2d) · ‖x‖² · ‖y‖²
```

For unit-norm vectors: Var ≤ π/(2d) ≈ 0.0123 when d=128.

**Zero overhead**: the matrix **S** is generated from a fixed seed shared
between encoder and decoder — no quantization constants are stored alongside
the compressed vector. Only d bits per vector.

### The Asymmetric Design (ex03)

The "obvious" approach quantizes **both** vectors to sign bits:

```
ip_hat_naive = (π/2)/d · <sign(S·x), sign(S·y)>
```

This estimates the **cosine similarity** (angle between vectors), not the
inner product. For unit vectors they're related by a nonlinear function of
the angle, introducing bias for general inner products. It also has
approximately **π/2 ≈ 1.57×** higher variance than QJL.

QJL avoids this by keeping the query at full precision — the key is 1 bit,
the query stays 32 bits. In KV-cache compression this is exactly the right
trade: keys are stored (and compressed), queries arrive at inference time
in full precision.

---

## Analytical Questions

Work through these after completing all three exercises.
Answers should reference specific numbers from your milestone outputs.

**Q1 (Analysis — memory/compute trade-off):**
QJL uses a dense random Gaussian matrix **S** of shape (d × d), requiring
O(d²) storage and O(d²) FLOPs per projection. For d=128 in a KV cache, S
is 128×128 × 4 bytes = 64 KB per attention head — negligible. But for
d=4096 (e.g., some LLM projection layers), S would require ~64 MB *per
head*. Propose a structured random matrix design (e.g., random Hadamard
transform + random diagonal sign flips) that reduces storage to O(d) and
compute to O(d log d). What theoretical property of QJL might be weakened?
Would the estimator remain unbiased?

**Q2 (Analysis — attention logit error under softmax):**
The QJL variance bound for a single inner product estimate is π/(2d) ·
‖y‖². For a transformer with d=128 and context length n=4096, a query
attends to n keys simultaneously. Each attention logit aᵢ = <q, kᵢ> has an
independent estimation error εᵢ with Var(εᵢ) ≤ π/(2·128) ≈ 0.0123.

(a) What is the expected maximum absolute error max_i |εᵢ| across all
n=4096 logits? (Hint: for independent sub-Gaussian random variables, the
max scales as σ · √(2 log n).)

(b) Softmax is computed as exp(aᵢ) / Σⱼ exp(aⱼ). The key concern is
whether a small logit error δ changes which token has the highest attention
weight. Given your answer from (a), under what conditions (temperature,
magnitude of true inner products) would softmax be *robust* to QJL noise,
and when might it degrade?

**Q3 (Synthesis — estimator design):**
The QJL dequantization formula Q⁻¹(z) = sqrt(π/2)/d · Sᵀ · z is derived
from the specific expectation identity E[sign(a·x)·(a·y)] = sqrt(2/π)·<x,y>
for Gaussian **a**. Suppose instead of Gaussian rows, you used rows drawn
uniformly from the unit hypersphere. Would the expectation identity change?
Derive (or argue intuitively) what the new scaling constant would be, and
whether the estimator would still be unbiased.

**Q4 (Analysis — 1-bit vs multi-bit QJL):**
In ex01 you saw that increasing the projection dimension m decreases variance
as 1/m. In QJL, m = d (same dimension, but only 1 bit per coordinate).
Suppose you modified QJL to use 2 bits per coordinate (e.g., quantize
S·x to 4 levels instead of 2). Would you expect variance to decrease?
By approximately what factor? How does the answer change if you increase m
(more projection dimensions) instead of adding bits per coordinate?

---

## Connection to TurboQuant

QJL is one of two building blocks in TurboQuant's final product quantizer
(Module 5). The recipe will be:

1. Quantize **x** with (b−1)-bit MSE quantizer → get reconstruction **x̂**
2. Compute residual **r** = **x** − **x̂**
3. Apply QJL to **r** → 1 sign bit per coordinate, unbiased correction
4. Store: MSE indices + sign bits + ‖**r**‖

The QJL component handles the bias that the MSE quantizer introduces,
yielding an **unbiased inner product estimator** at b total bits with
variance ≤ (√3·π²·‖y‖²) / (d · 4ᵇ).
