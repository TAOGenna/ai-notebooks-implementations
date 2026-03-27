# Module 00: Why Random Rotation Unlocks Optimal Quantization

> **Claim:** A single random rotation matrix converts ANY distribution of unit vectors
> into a universal one — the Beta distribution on each coordinate — whose shape is
> known analytically. This means you can design a near-optimal quantizer with zero
> knowledge of the input data.

---

## What This Module Covers

Before writing a single quantization algorithm, we need to understand *why* the
problem is tractable. The central insight of TurboQuant is:

1. **Quantization has a fundamental limit.** The Shannon lower bound says the best
   any randomized b-bit quantizer can do is MSE ≥ 1/4ᵇ. Naive approaches fall far
   short.

2. **Random rotation makes distributions universal.** Multiplying any unit vector by
   a random orthogonal matrix Π places the result uniformly on the unit hypersphere
   S^{d-1}. Each individual coordinate then follows a Beta distribution with a known,
   analytically tractable form.

3. **The Beta distribution converges to Gaussian in high dimensions.** For large d,
   each coordinate behaves like N(0, 1/d) — and this approximation becomes tight
   enough to use for quantizer design at d ≥ 64.

4. **Distinct coordinates become nearly independent.** Pairwise correlation and mutual
   information between different coordinates are near zero. This licenses per-coordinate
   scalar quantization — the algorithmic simplification that makes TurboQuant fast and
   practical.

---

## Prerequisites

- NumPy, SciPy, Matplotlib (`pip install numpy scipy matplotlib`)
- Python 3.9+
- No GPU required — all exercises run on CPU in seconds

---

## Exercises

Work through these **in order**. Each one builds intuition that the next exercise
makes rigorous.

---

### Exercise 01 — `exercise_01_naive_quantization.py`
**What Does Quantization Lose? Measuring Distortion on Raw Vectors**

*Type: Explore — all code is provided.*

Run this exercise first. It shows that naive uniform quantization leaves a large
gap vs. the theoretical lower bound — and gives you a concrete sense of what
"distortion" means numerically.

```bash
python exercise_01_naive_quantization.py
```

**What to observe:**
- At d=128, b=1: MSE ≈ 0.83 vs. lower bound 0.25 (gap ≈ 3.3×)
- At d=128, b=4: MSE ≈ 0.013 vs. lower bound 0.0039 (still 3.3×)
- The gap doesn't shrink as bits increase — naive quantization is structurally bad.

**TRY** varying `d` (16, 64, 128, 1024) and observe how distortion changes.
Notice that the *gap ratio* stays roughly constant — this is a property of the
quantizer design, not the dimension.

---

### Exercise 02 — `exercise_02_random_rotation.py`
**How Random Rotation Creates a Universal Distribution**

*Type: Fill-blank — implement 3 functions, then run.*

```bash
python exercise_02_random_rotation.py
```

**Your tasks:**
| Function | Lines | What you implement |
|---|---|---|
| `generate_random_rotation(d, rng)` | ~5 | QR decomposition of Gaussian matrix |
| `rotate_vector(Pi, x)` | ~2 | Matrix-vector (or matrix-matrix) product |
| `theoretical_beta_pdf(t, d)` | ~4 | Gamma function formula for marginal density |

**Milestone output:** KS test p-values > 0.05 at d = 16, 64, 128, 512, and a
saved plot `milestone_02_rotation_distribution.png` overlaying the empirical
coordinate histogram with the exact Beta pdf and the Gaussian approximation.

At d=128, you should see the Beta and Gaussian curves nearly overlap — this
visual discovery is the motivating insight for TurboQuant's codebook design.

---

### Exercise 03 — `exercise_03_independence.py`
**Why Near-Independence Makes Per-Coordinate Quantization Work**

*Type: Fill-blank — implement 2 functions, then run.*

```bash
python exercise_03_independence.py
```

**Your tasks:**
| Function | Lines | What you implement |
|---|---|---|
| `compute_correlation_matrix(X_rot)` | ~6 | Sample Pearson correlation matrix |
| `estimate_mutual_information(u, v)` | ~4–6 | Histogram-based MI estimation |

**Milestone output:** Max off-diagonal correlation < 0.05 for d=128, N=10,000.
A saved heatmap `milestone_03_independence.png` showing a bright diagonal and
near-zero off-diagonal entries.

Printed conclusion:
```
Max |correlation|: 0.03 — coordinates are nearly independent.
This means per-coordinate scalar quantization loses almost nothing
vs joint vector quantization.
```

---

## Running Order

```
exercise_01  →  exercise_02  →  exercise_03
   (explore)       (fill)           (fill)
```

Module 01 will build directly on the rotation from Exercise 02 and implement the
Lloyd-Max algorithm to optimally quantize the Beta/Gaussian distribution.

---

## Analytical Questions

Think about these *after* completing all three exercises. They are intentionally
hard and open-ended — the goal is to reason from the measurements you produced,
not to look up formulas.

---

### Q1 — The Convergence Rate (Analysis)

At d=16, the Beta distribution is visibly different from the Gaussian N(0, 1/d).
At d=128, they are nearly identical. At d=512, the KS test gives essentially
identical p-values for both.

> **Question:** At what dimension d would you expect the Gaussian approximation
> N(0, 1/d) to introduce less than 1% *relative* error in the MSE distortion
> bound? How would you measure this empirically using the tools from Exercise 02?

**Hints to guide your thinking:**
- The Beta distribution has kurtosis that decreases as d increases. At what
  kurtosis value does a Gaussian approximation produce < 1% error in the second
  moment of a quantization step?
- The convergence rate is governed by Berry-Esseen: the error in the CDF between
  the Beta and Gaussian is O(1/√d). If you need 1% relative error in MSE, what
  O(1/√d) threshold does that translate to?
- Could you design a numerical experiment using `theoretical_beta_pdf` and
  `gaussian_approximation_pdf` to find the crossover d without running any ML?

---

### Q2 — Hadamard Preconditioning vs. Random Rotation (Synthesis)

TurboQuant uses a random rotation matrix Π generated fresh for each inference
call. Papers like QuaRot and FlashAttention-3 instead use a fixed Hadamard
matrix H (where H_{ij} = ±1/√d) as a preconditioning step.

> **Question:** If you replaced `generate_random_rotation` with a fixed Hadamard
> matrix, what would change about the coordinate distribution? Would the KS test
> from Exercise 02 still pass? Would per-coordinate quantization still work?
> Under what conditions does Hadamard preconditioning give the same guarantee as
> random rotation — and when does it fail?

**Hints to guide your thinking:**
- A Hadamard matrix is orthogonal, so it preserves norms. But it is *not* drawn
  from the Haar measure — it is a fixed matrix.
- After applying a fixed H to a *random* unit vector x, the distribution of H·x
  depends on whether x itself has any structure aligned with the columns of H.
- TurboQuant's data-oblivious guarantee relies on the rotation being random
  relative to the data. What happens if you have input data that is adversarially
  chosen to be eigenvectors of H?
- In practice, LLM activations are not adversarial. Why might Hadamard work
  empirically even though the theoretical guarantee is weaker?

---

### Q3 — Correlation vs. Independence (Analysis)

Exercise 03 showed near-zero pairwise Pearson correlations between rotated
coordinates. But uncorrelated does NOT mean independent.

> **Question:** Construct a concrete example of two zero-mean random variables
> U and V that are uncorrelated (E[UV] = 0) but strongly dependent (I(U;V) > 0).
> Then explain: for the per-coordinate quantization argument to hold, why does
> TurboQuant need *near-independence* (I ≈ 0) rather than just *low correlation*
> (E[UV] ≈ 0)?
>
> More precisely: if coordinates had zero pairwise correlation but nonzero mutual
> information (say, I(X_j, X_k) = 0.1 bits for all j≠k in a d=128 vector), how
> much suboptimality would that introduce into the per-coordinate quantizer, and
> would it matter in practice for 4-bit quantization?

**Hints to guide your thinking:**
- A classic example: U ~ Uniform[-1, 1], V = U². Are they correlated? Are they
  independent?
- For per-coordinate quantization to be optimal, you need: the optimal quantizer
  for (X_j, X_k) jointly to decompose into two independent scalar quantizers.
  This decomposition holds *if and only if* X_j and X_k are independent — not
  merely uncorrelated.
- The actual near-independence of rotated coordinates comes from the geometry of
  the sphere: for a uniform point on S^{d-1}, the joint density of any k
  coordinates converges to a product of k Gaussian densities as d→∞. This is
  a concentration-of-measure argument, not just a second-moment statement.
- At 4 bits per coordinate, the distortion floor is ~0.009/d. How large would
  I(X_j, X_k) = 0.1 bits translate to in additive MSE distortion? Is it
  measurable at this precision level?

---

### Q4 — The Distortion Gap and Its Source (Analysis)

Exercise 01 showed that the gap between naive quantization and the lower bound
is roughly 3.3× across all bit-widths. TurboQuant closes this to ~2.7×.

> **Question:** Where does the remaining 2.7× gap come from in TurboQuant, even
> though it uses the *optimal* scalar quantizer for the known Beta distribution?
> Is this gap fundamental (a consequence of scalar vs. vector quantization), or
> is there a modified algorithm that could close it further — and at what cost?

**Hints to guide your thinking:**
- The Shannon lower bound applies to ALL quantizers, including ones that can
  jointly encode all d coordinates. Per-coordinate scalar quantization is a
  *constrained* form of vector quantization.
- The gap between optimal scalar quantization and optimal vector quantization is
  well-studied: for a Gaussian source, it is exactly π·e/6 ≈ 1.53× in MSE terms
  (the "granular gain" of high-rate quantization theory). How does this relate
  to the 2.7× gap you observe?
- Could entropy coding of the quantization indices close part of the gap? The
  paper notes that the quantized coordinate distribution is non-uniform (its
  entropy is ~3.8 bits at b=4, not 4.0 bits). What compression does this offer?

---

## Key Numbers to Remember

| Quantity | Value |
|---|---|
| Shannon lower bound (total-vector MSE, b bits/coord) | 1/4ᵇ |
| Naive uniform quantization gap (d=128, clip=1.0) | ~50–450× |
| Per-coordinate MSE of naive 1-bit quantizer (d=128) | ~0.87 |
| TurboQuant gap (optimal scalar, Beta distribution) | ~1.4–2.3× |
| Coordinate variance after rotation | 1/d |
| Max off-diagonal correlation (d=128, N=10k) | < 0.05 |
| Max pairwise mutual information (d=128, N=10k) | < 0.005 bits |
| Dimension where Beta ≈ Gaussian (visual) | d ≈ 64–128 |
| KS p-value threshold for distribution match | > 0.05 |

---

## What's Next

**Module 01: "Why the Beta Distribution Yields the Optimal Codebook"**

You'll implement the Lloyd-Max algorithm — iterative Voronoi optimization for
the Beta/Gaussian distribution — and verify that it achieves the per-coordinate
MSE values predicted by theory: 0.36/d (b=1), 0.117/d (b=2), 0.03/d (b=3),
0.009/d (b=4). This is the "codebook design" step that makes TurboQuant tick.
