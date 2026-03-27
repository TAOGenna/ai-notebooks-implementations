# Module 02: TurboQuant_mse — The Full MSE Quantization Pipeline

> **Claim:** Rotating vectors before quantization enables near-optimal MSE compression.
> But MSE-optimal ≠ inner-product-optimal — and at 1-bit the bias is exactly 2/π.

## What This Module Covers

This module wires together the two tools you built in earlier modules — the random
rotation matrix (Module 0) and the Lloyd-Max scalar quantizer (Module 1) — into
**TurboQuant_mse**: a complete, end-to-end vector quantization pipeline.

By the end you will have:

1. A working `TurboQuantMSE` class implementing Algorithm 1 from the TurboQuant paper.
2. Empirical verification that the MSE bound `D ≤ (√3·π/2) / 4^b` holds on real data.
3. A scatter plot that makes the inner product bias **unmistakably visible**.
4. An understanding of why storing norms separately is a near-zero-cost operation.

## Prerequisites

- Module 0 complete (random rotation, Beta coordinate distribution)
- Module 1 complete (Lloyd-Max scalar quantizer, MSE scaling law)
- Python packages: `numpy`, `scipy`, `matplotlib` (see `requirements.txt`)

## How to Work Through This Module

Work through exercises in order. Each exercise builds on the previous one.

```bash
# Exercise 01: build TurboQuantMSE, verify MSE bounds
python ex01_assemble_turboquant_mse.py

# Exercise 02: discover the inner product bias (saves a plot)
python ex02_hidden_bias.py

# Exercise 03: quantize KV cache vectors with norm storage
python ex03_real_embeddings.py
```

> **Note on timing:** `lloyd_max_codebook` runs the Lloyd-Max algorithm once per
> `(b, d)` pair and caches the result. The first call for each `b` takes ~2-5 seconds.
> Exercise 02 is the slowest (~2-4 min total) because it runs hundreds of quantization
> trials. Outputs are printed to the terminal; Exercise 02 also saves a PNG file.

---

## Exercise 01 — Assembling TurboQuant_mse: Rotate, Quantize, Reconstruct

**Type:** fill-blank  |  **Difficulty:** ★★☆

Implement three methods of the `TurboQuantMSE` class:

| Method | Lines | What it does |
|--------|-------|--------------|
| `quantize(x)` | ~5 | Rotate x → find nearest centroid per coord |
| `dequantize(indices)` | ~4 | Centroid lookup → inverse rotation |
| `compute_mse(x)` | ~3 | `‖x − DeQuant(Quant(x))‖²` |

**Algorithm 1 (TurboQuant paper):**

```
Setup:  Π ← random d×d rotation     {c_l} ← Lloyd-Max codebook

Quant(x):
  x_rot = Π · x
  for each i: idx[i] = argmin_l |x_rot[i] − c_l|²

DeQuant(idx):
  y_tilde[i] = c_{idx[i]}
  x_hat = Π^T · y_tilde
```

**Milestone output** (once all TODOs are implemented):

```
TurboQuant_mse  |  d=128, n=1000 random unit vectors
  b    Empirical MSE   Upper Bound   % of UB   Compress
  1         0.3610        0.6802      53.1%      32.0×  ✓
  2         0.1171        0.1700      68.9%      16.0×  ✓
  3         0.0301        0.0425      70.7%      10.7×  ✓
  4         0.0090        0.0106      85.1%       8.0×  ✓
All within theoretical upper bound ✓
```

The empirical MSE is 50–85% of the upper bound. Notice the 4× drop per extra bit —
this is the 1/4^b scaling law you verified in Module 1, now confirmed end-to-end.

---

## Exercise 02 — The Hidden Bias: Why MSE-Optimal Quantizers Distort Inner Products

**Type:** contrastive  |  **Difficulty:** ★★★

This is the pivotal exercise of the module. You will discover — empirically —
that MSE-optimal quantization introduces a **systematic multiplicative bias**
in inner product estimation.

### The contrastive structure

| Stage | What you implement | What you see |
|-------|-------------------|--------------|
| **Naive** | `estimate_ip_single_rotation`: quantize x, dot with y | Noisy single estimate |
| **Better** | `estimate_ip_multi_rotation`: average over K=30 rotations | Bias visible, variance low |
| **Measure** | `measure_bias`: linear regression slope over 400 pairs | Exact bias factor |

### Milestone output

```
  b=1:  slope = 0.6366   (|diff from 2/π| ≈ 0.00004)
  b=2:  slope ≈ 0.85     (increases with bit-width)
  b=3:  slope ≈ 0.96
  b=4:  slope ≈ 0.99
```

And a scatter plot `milestone_02_bias_discovery.png` showing:
- **b=1:** all points lie on a line with slope ≈ 2/π — not the ideal slope-1 diagonal.
- **b=4:** points cluster near the ideal diagonal — but still slightly below.

**Why exactly 2/π at b=1?**
For b=1, the optimal codebook is `{−c, +c}` with `c = E[|z|]` for z ~ N(0, 1/d).
Dequantization gives `x̂ = Π^T · sign(Π·x) · c`.
Then (see Inline Question 1 below):

```
E[⟨y, x̂⟩] = E[⟨Π·y, c·sign(Π·x)⟩] = (2/π) · ⟨y, x⟩
```

This is not noise — it is a structural property of any MSE-optimal 1-bit quantizer.

---

## Exercise 03 — Quantizing Real Embeddings: MSE on Synthetic KV Cache Vectors

**Type:** fill-blank  |  **Difficulty:** ★★☆

Extend TurboQuant_mse to handle **arbitrary-norm vectors** by storing the norm
separately at float16 precision.

Implement:

| Function | Lines | What it does |
|----------|-------|--------------|
| `store_kv_vector(tq, x)` | ~4 | normalise → quantize, return `(indices, norm_float16)` |
| `retrieve_kv_vector(tq, indices, norm)` | ~4 | dequantize → rescale |
| `compute_relative_error(x, x_hat)` | ~3 | `‖x − x̂‖ / ‖x‖` |

**Milestone output** (b=2, d=128):

```
Unit vectors (b=2):
  Absolute MSE:   0.117
  Relative error: 3.4% ± 0.8%

KV-like vectors (b=2, with outlier channels):
  Avg ‖x‖: 2.85 ± 1.48
  Absolute MSE:   0.95  (= unit MSE × avg_norm²)
  Relative error: 3.4% ± 0.9%  ← SAME as unit vectors

Storage overhead (b=2, d=128):
  Norm overhead per coord:  16/256 = 0.0625 bits/coord
  Effective bit-width:      2.0625 bits for ~3.4% relative error
```

Key insight: relative error is **scale-invariant** — it stays ~3.4% whether the
vector has norm 1 or norm 100. Traditional block-wise quantizers (KIVI, AWQ)
must store a `scale` and `zero_point` per 32-128 element block, adding ~1 bit/coord
of overhead that does **not** decrease with dimension. TurboQuant's overhead
`16/(b·d)` shrinks toward zero as d grows.

---

## Analytical Questions

Answer these after completing all three exercises. Aim for depth — cite specific
numbers from your milestone outputs.

### Q1 (Analysis): Deriving the Exact 2/π Bias at 1-Bit

At b=1, the Lloyd-Max centroid values for z ~ N(0, 1/d) are `±c` where
`c = E[|z|] = sqrt(2/(π·d))`. The dequantized vector is:

```
x̂ = Π^T · (sign(Π·x) · c)
```

Use the following facts:
- Π·x and Π·y are jointly normal with Cov((Π·x)_i, (Π·y)_i) = ⟨x,y⟩/d
- For jointly normal (u, v) with E[u] = 0: E[v·f(u)] = Cov(u,v)·E[f'(u)]
  (Stein's lemma for Gaussian random variables)
- For N(0, σ²): E[|z|] = σ·√(2/π)

**Derive** E[⟨y, x̂⟩] = (2/π)·⟨y, x⟩ analytically. At what step does the
factor of 2/π emerge, and why does it persist regardless of the specific
vectors x, y, or rotation Π?

*(Target: 4-6 lines of algebra using the facts above.)*

---

### Q2 (Analysis): Why the Norm Storage Overhead Shrinks With Dimension

TurboQuant stores one float16 norm per vector (16 bits), adding `16/(b·d)`
bits per coordinate. For d=128, b=2: that's `16/256 = 0.0625 bits/coord`.

Traditional quantizers (KIVI, INT8 with per-block scaling) store a `scale`
and optionally a `zero_point` per **block of 32 elements**, adding `32/32 = 1`
bit/coord (or 64/128 = 0.5 bits/coord for larger blocks).

**Explain** why TurboQuant's overhead shrinks with d while block-wise
quantizers' overhead is independent of d. Your answer should reference:
1. What structural property of TurboQuant allows a **single** scale per vector.
2. Why block-wise methods cannot achieve the same (what prevents them from
   using one scale per vector instead of per block).
3. At what dimension d does TurboQuant's norm overhead become smaller than
   KIVI's per-block overhead (assuming KIVI uses 32-element blocks, 16-bit
   scale, and b=2)?

---

### Q3 (Synthesis): Designing an Unbiased Inner Product Estimator

Your Exercise 02 results show that the inner product bias at b=2 is
approximately 0.85 — the estimator systematically underestimates by ~15%.

**Design** a simple post-processing correction for the b=2 case:
1. Given only the estimated inner product `est_ip` and the known bias slope
   (measured empirically in your Exercise 02 output), how would you correct it?
2. After correction, the estimate is unbiased. But is it lower-variance than
   the original? Express the variance of the corrected estimator in terms of
   the original estimator's variance and the bias slope.
3. The TurboQuant paper does NOT use this simple correction — instead it
   proposes a two-stage TurboQuant_prod. What limitation of your correction
   approach motivates the more complex solution?

*(Hint: think about what happens when the bias slope is estimated from data
vs. when it is known analytically, and whether the correction works for
**all** pairs (x, y) or only on average.)*

---

### Q4 (Analysis): The 1/4^b Scaling Law — When Does It Break Down?

Your Exercise 01 milestone shows MSE dropping by roughly 4× per extra bit:
b=1→0.361, b=2→0.117, b=3→0.030, b=4→0.009.

1. The information-theoretic lower bound for any randomized quantizer is
   `D_mse ≥ 1/4^b` (Theorem 2 in the paper). At b=1, TurboQuant achieves
   D=0.361 vs. lower bound 0.25 — a factor of 1.44. Does this ratio grow
   or shrink with b? Compute it for b=2,3,4 from your milestone output.

2. The paper's upper bound is `D ≤ (√3·π/2)/4^b ≈ 2.72/4^b`. At what
   bit-width b does the **gap between upper and lower bound** become less
   than 10% of the lower bound value?
   *(Hint: the gap constant is (√3·π/2 − 1) ≈ 1.72; you need `1.72/4^b < 0.1 · 1/4^b`
   — is this even achievable? What does that tell you about the bound's tightness?)*

3. In practice, at b=4, the empirical MSE is already within 15% of the
   lower bound. What practical implication does this have for choosing b for
   LLM KV cache compression?

---

## Key Concepts (Reference)

| Concept | Module | File |
|---------|--------|------|
| Random rotation, Beta distribution | Module 0 | `exercise_02_random_rotation.py` |
| Lloyd-Max algorithm, 1/4^b scaling | Module 1 | `exercise_01_lloyd_max.py` |
| TurboQuant_mse, Algorithm 1 | **Module 2** | `ex01_assemble_turboquant_mse.py` |
| Inner product bias, 2/π factor | **Module 2** | `ex02_hidden_bias.py` |
| Norm storage, KV cache compression | **Module 2** | `ex03_real_embeddings.py` |
| QJL: unbiased 1-bit inner products | Module 3 | — |
| TurboQuant_prod: two-stage pipeline | Module 4 | — |

## What's Next

**Module 3: QJL and the Sign-Bit Trick — Unbiased 1-Bit Inner Product Estimation**

The inner product bias you measured in Exercise 02 (2/π at b=1) is a
structural property of MSE-optimal quantizers. Module 3 introduces QJL
(Quantized Johnson-Lindenstrauss), which achieves **zero bias** with just 1 bit
by using a different random matrix — random Gaussian S instead of orthogonal Π —
and never trying to minimise MSE at all.
