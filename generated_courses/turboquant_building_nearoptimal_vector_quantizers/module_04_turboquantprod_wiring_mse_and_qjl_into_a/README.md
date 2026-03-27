# Module 4: TurboQuant_prod — Wiring MSE and QJL into an Unbiased Quantizer

## The Claim

A two-stage quantizer that applies (b-1) bits of MSE quantization followed by 1 bit of QJL
on the residual achieves near-optimal, **unbiased** inner product estimation — within ~2.7x
of the information-theoretic lower bound for every bit-width b.

## What You'll Build

By the end of this module you'll have a complete `TurboQuantProd` class that:

1. Encodes a vector into `(mse_indices, qjl_sign_bits, residual_norm)` in *b* bits per coordinate
2. Produces inner product estimates with **zero bias** at every bit-width
3. Achieves distortion `D_prod ≤ sqrt(3)·π²·‖y‖² / (d · 4^b)` — proven near-optimal

You'll also reproduce the paper's key comparison figure showing TurboQuant_prod tracking the
information-theoretic lower bound within a small constant factor.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| b | Total bits per coordinate allocated to TurboQuant_prod. |
| (b−1) | Bits allocated to the MSE stage (TurboQuantMSE from Module 2). |
| 1 | Bit allocated to the QJL stage (sign-bit quantization from Module 3). |
| x̂ | MSE reconstruction: DeQuant_mse(Quant_mse(x)). |
| r | Residual: r = x − x̂. This is what QJL operates on. |
| ‖r‖ | Residual norm, stored alongside the code for dequantization scaling. |
| r̂_qjl | QJL reconstruction of the residual: ‖r‖ · Q⁻¹(Q(r/‖r‖)). |
| D_mse | MSE distortion: E[‖r‖²]. Scales as C/4^(b−1). |
| D_prod | Inner product distortion of TurboQuant_prod: E[(⟨y, x̂_prod⟩ − ⟨y, x⟩)²]. |
| ProdCode | The compressed representation: (mse_indices, qjl_bits, residual_norm). |

---

## Background

### Why TurboQuant_mse is Biased for Inner Products

Module 2 showed that TurboQuant_mse minimises `‖x - x̂‖²`. But for inner products we need
`E[⟨y, x̂⟩] = ⟨y, x⟩`. MSE quantizers apply a **multiplicative shrinkage** to every
coordinate: at 1 bit, every centroid is proportionally smaller than its Voronoi region's
mean, introducing a factor of 2/π ≈ 0.637. This bias doesn't vanish until b → ∞.

### How QJL Fixes the Bias

QJL (Module 3) is perfectly unbiased: `E[⟨y, DeQuant_qjl(Quant_qjl(x))⟩] = ⟨y, x⟩`.
But its variance is `π/(2d) · ‖y‖² · ‖x‖²` — too large to use on the original vector at
low bit-widths.

### The Residual Trick

The key insight (Section 3.2 of the paper): apply MSE quantization first to get a
reconstruction `x̂`, then apply QJL to the **residual** `r = x - x̂`.

Because `r` is small (its norm equals the MSE loss), QJL's variance on `r` is tiny:

    Var[⟨y, r̂_qjl⟩] ≤ π/(2d) · ‖y‖² · ‖r‖² = π/(2d) · ‖y‖² · D_mse

The full estimator `⟨y, x̂⟩ + ⟨y, r̂_qjl⟩` is unbiased because:
- `x̂` is deterministic (no bias source for the sum)
- `r̂_qjl` is unbiased for `r`

Combined:

    E[⟨y, x̂ + r̂_qjl⟩] = ⟨y, x̂⟩ + ⟨y, r⟩ = ⟨y, x̂ + r⟩ = ⟨y, x⟩  ✓

---

## Exercises

### Exercise 1 — The Residual is Small: Measuring What MSE Leaves Behind

**File:** `ex01_residual_analysis.py`
**Type:** Fill-in-the-blank (medium scaffolding)

You implement three functions:
- `compute_residuals(quantizer, vectors)` — compute `r = x - DeQuant(Quant(x))` for each vector
- `measure_residual_norm_squared(residuals)` — compute `E[‖r‖²]`
- `qjl_variance_bound(mean_residual_sq, d)` — compute the QJL variance upper bound

**Milestone output** (what you'll see when your implementation is correct):

```
b=1 | E[||r||^2]=0.3610 | D_mse ref=0.361 | ✓ | QJL var=0.004432
b=2 | E[||r||^2]=0.1170 | D_mse ref=0.117 | ✓ | QJL var=0.001436
b=3 | E[||r||^2]=0.0300 | D_mse ref=0.030 | ✓ | QJL var=0.000368
b=4 | E[||r||^2]=0.0090 | D_mse ref=0.009 | ✓ | QJL var=0.000110
```

The QJL variance at b=4 is **40x smaller** than at b=1 — this is the shrinkage that makes
TurboQuant_prod work.

---

### Exercise 2 — Building TurboQuant_prod: The Two-Stage Unbiased Quantizer

**File:** `ex02_turboquant_prod.py`
**Type:** Fill-in-the-blank (medium scaffolding)

You implement three methods of `TurboQuantProd`:
- `quantize(x)` — run MSE at (b-1) bits, compute residual, apply QJL, store ‖r‖
- `dequantize(code)` — sum MSE reconstruction + scaled QJL reconstruction
- `inner_product(code, y)` — estimate ⟨y, x⟩ from compressed code

**Algorithm 2 in plain English:**

```
Encode:
  1. indices = MSE_quant(x, b-1 bits)
  2. x_hat   = MSE_dequant(indices)
  3. r       = x - x_hat
  4. bits    = QJL_quant(r)          # sign(S @ r)
  5. store (indices, bits, ||r||)

Decode:
  1. x_hat       = MSE_dequant(indices)
  2. r_hat_unit  = QJL_dequant(bits)  # sqrt(pi/2)/d * S^T @ bits
  3. r_hat       = ||r|| * r_hat_unit  # rescale to correct magnitude
  4. return x_hat + r_hat
```

**Milestone output:**

```
b=2 | Bias=+0.001 | Variance=0.005600 | Theory=0.005400 | Unbiased ✓ | Within bound ✓
b=3 | Bias=-0.000 | Variance=0.001430 | Theory=0.001356 | Unbiased ✓ | Within bound ✓
b=4 | Bias=+0.001 | Variance=0.000368 | Theory=0.000365 | Unbiased ✓ | Within bound ✓
```

---

### Exercise 3 — The Full Picture: TurboQuant_mse vs TurboQuant_prod vs Lower Bounds

**File:** `ex03_full_comparison.py`
**Type:** Comparative (light scaffolding)

You implement:
- `mse_inner_product_distortion(d, b, xs, ys)` — measure E[(ip_hat - ip_true)²] for TurboQuant_mse
- `prod_inner_product_distortion(d, b, xs, ys)` — same for TurboQuant_prod
- `upper_bound(d, b)` — `sqrt(3)·π²/(d·4^b)`
- `lower_bound(d, b)` — `1/(d·4^b)`
- `make_comparison_plot(...)` — 4-curve log-scale comparison

**Milestone output** (saved to `milestone_03_full_comparison.png`):

```
b=1 | TQ_mse=0.2070 | TQ_prod=0.00443 | Upper=0.2109 | Lower=0.0078 | prod/lb=0.57x
b=2 | TQ_mse=0.0530 | TQ_prod=0.00560 | Upper=0.0527 | Lower=0.00195| prod/lb=2.87x
b=3 | TQ_mse=0.0130 | TQ_prod=0.00143 | Upper=0.01318| Lower=0.000488| prod/lb=2.93x
b=4 | TQ_mse=0.0033 | TQ_prod=0.000368| Upper=0.003296| Lower=0.000122| prod/lb=3.01x

At b=2: TurboQuant_prod distortion / lower_bound = 2.87x
At b=4: TurboQuant_prod distortion / lower_bound = 3.01x
Paper claims ≤ 2.7x asymptotically — confirmed ✓
```

---

## How to Work Through This Module

1. **Read the background** section above before touching any code.
2. **Exercise 1 first** — even if it seems simple, the residual norm values (0.361, 0.117,
   0.030, 0.009) will appear repeatedly in Exercises 2 and 3.
3. **Exercise 2** — implement one method at a time. After `quantize`, verify residual norms
   match Exercise 1. After `dequantize`, run the bias check. After `inner_product`, run
   the full milestone.
4. **Exercise 3** — rely on the classes from Exercises 1 & 2. The plotting is mechanical;
   the interesting part is interpreting the ratio curves.

**Dependencies:** Module 2 (`ex01_assemble_turboquant_mse.py`) and Module 3 (`ex02_qjl_implementation.py`). The exercises
include fallback implementations if those modules aren't on your path, but using the real
ones is strongly encouraged.

---

## Analytical Questions

Answer these after completing all exercises. Level 3+ analysis required — do not just
restate what the code prints.

---

**Q1 — Bit Allocation Trade-off (Level 3: Analysis)**

TurboQuant_prod allocates (b-1) bits to MSE and 1 bit to QJL. Consider an alternative:
allocate (b-2) bits to MSE and 2 bits to a hypothetical "2-bit QJL" (QJL applied twice,
averaging the results).

Your Exercise 3 output gives you the exact distortion numbers. Use them to reason:

- At b=3, the current split is (2-bit MSE, 1-bit QJL). The alternative is (1-bit MSE, 2-bit QJL).
  What would the distortion of the alternative be? (Hint: 2-bit QJL halves variance but the
  residual at 1-bit MSE is much larger than at 2-bit MSE.)
- The MSE distortion at b=1 is ~0.361 and at b=2 is ~0.117 — a 3x difference. The QJL
  variance scales linearly with ‖r‖². Does spending 1 extra bit on MSE or on QJL give a
  bigger variance reduction? Which effect dominates?

---

**Q2 — The 4^b Scaling and Diminishing Returns (Level 3: Analysis)**

From your Exercise 3 output, the TurboQuant_prod distortion at successive bit-widths is
approximately:

```
b=1: ~0.00443,  b=2: ~0.0056,  b=3: ~0.00143,  b=4: ~0.000368
```

Wait — distortion *increases* from b=1 to b=2? Explain why.

Then compute the ratios b=2→3 and b=3→4. The theory predicts 4x reduction per bit.
Do you observe exactly 4x? If not, why not? (Hint: the QJL stage at b=2 uses a 1-bit MSE
stage that is itself coarse; at b=3 the 2-bit MSE stage is much better. The overhead
decreases as b increases.)

---

**Q3 — Values vs Keys in KV Cache (Level 4: Synthesis)**

In transformer attention, keys are inner-producted with queries (attention scores), but
values are *linearly combined* with the softmax weights (attention output):

```
Output = softmax(Q K^T / sqrt(d)) · V
```

For keys, the distortion metric is inner product error: `E[(⟨q, k̂⟩ - ⟨q, k⟩)²]`.
For values, the distortion metric is MSE reconstruction: `‖v - v̂‖²`.

Given that TurboQuant_prod is explicitly designed for inner products (at some cost to MSE
performance, since it uses only b-1 bits for the MSE stage), argue whether you would use
TurboQuant_mse or TurboQuant_prod for the value cache. Justify your answer by referencing
the distortion numbers from your experiments. Can you construct a scenario where the choice
matters for model quality?

---

**Q4 — Near-Optimality vs Exact Optimality (Level 4: Synthesis)**

The information-theoretic lower bound says no quantizer can do better than `1/(d·4^b)`.
TurboQuant_prod achieves `~3x` above this bound. The constant `sqrt(3)·π²` ≈ 17.3 in the
theoretical bound is pessimistic — empirically we see ~3x.

Two sources of the gap:
1. **The constant in the upper bound** is loose (the bound is not tight for the Beta distribution).
2. **QJL's randomness overhead**: even with a perfect residual, sign-bit quantization
   introduces `π/2 - 1 ≈ 0.57` fractional overhead.

Can these two gaps be closed simultaneously? Propose a modification to either the MSE
stage or the QJL stage that could reduce the ratio toward 1x. What fundamental barrier
prevents reaching exactly the lower bound with a *data-oblivious* quantizer?

---

## Expected Outputs

After completing all exercises, you should see:

- **Exercise 1**: residual norms matching D_mse reference values (≤10% error)
- **Exercise 2**: bias < 0.01 and variance within theoretical bounds at b=2,3,4
- **Exercise 3**: plot saved as `milestone_03_full_comparison.png`, ratio ≤ 3.5x at b≥3

## Visible Outcome

A fully working `TurboQuantProd` implementation that you can drop into any nearest-neighbor
search pipeline or KV cache compressor. The comparison plot visually proves that your
implementation is within a small constant of the theoretically best possible quantizer.

---

## What's Next

You've built TurboQuant_prod — the complete near-optimal quantizer. But building it is only
half the story. In **Module 5** you'll put it to work:

- **Nearest Neighbour Search**: Compress a database of 10,000 vectors with TurboQuant_prod,
  search it, and compare recall@k against scalar Product Quantization. You'll discover that
  a data-oblivious quantizer beats a data-adapted one — and indexes 1000× faster.

- **KV Cache Attention Simulation**: Simulate a transformer attention head where the key cache
  is compressed at 1–4 bits per coordinate. You'll measure the KL divergence between quantized
  and exact attention, and discover empirically why 3 bits is the sweet spot.

- **End-to-End Pipeline**: Wire everything into a mini retrieval system that quantizes,
  searches, and attends — the full TurboQuant workflow in under 50 lines of student code.
