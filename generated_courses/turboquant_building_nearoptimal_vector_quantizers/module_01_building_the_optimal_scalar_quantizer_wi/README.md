# Module 1: Building the Optimal Scalar Quantizer with Lloyd-Max

> **Claim:** Once we know the coordinate distribution (from Module 0's random rotation),
> we can analytically design the perfect quantizer — no data needed.

---

## What this module covers

In Module 0 you discovered that randomly rotating a unit-norm vector places its
coordinates on the Beta distribution:

```
f_X(t) = C_d · (1 − t²)^{(d−3)/2},    t ∈ (−1, 1)
```

This module exploits that fact. Because the distribution is **known in advance**,
we can solve the optimal quantizer design problem *analytically* — without any training
data, unlike Product Quantisation or other data-driven methods.

The key algorithm is **Lloyd-Max** — the continuous analogue of k-means. You will:

1. Implement Lloyd-Max using numerical integration to find optimal centroids
2. Verify that distortion obeys the `1/4^b` power law (each bit buys 4× reduction)
3. Package the result into a `Codebook` class ready for use in Module 3

By the end, you will have reproduced the TurboQuant paper's exact distortion numbers
and a publication-quality figure showing the algorithm is within **~2.7× of the
information-theoretic lower bound**.

---

## Prerequisites

- Completed Module 0 (or familiarity with the Beta coordinate distribution)
- Python packages: `numpy`, `scipy`, `matplotlib`

```bash
pip install numpy scipy matplotlib
```

---

## How to work through this module

Work through the exercises **in order** — each builds on the previous.

### Exercise 1 — `exercise_01_lloyd_max.py`

**The Continuous k-means Problem: Finding Optimal Centroids**

Fill in `_update_centroids` (~8 lines): for each Voronoi region, compute the
conditional mean `E[X | region]` using numerical integration (`scipy.integrate.quad`).

**Milestone:** Once implemented, run:
```bash
python exercise_01_lloyd_max.py
```
You will see the optimal centroids and per-coordinate MSE for `b = 1, 2, 3, 4` at `d = 128`.
The numbers match the TurboQuant paper's Table 1 exactly:

| bits b | per-coord MSE | total MSE (×d) |
|--------|---------------|----------------|
| 1      | 2.82e-03      | 0.361          |
| 2      | 9.14e-04      | 0.117          |
| 3      | 2.34e-04      | 0.030          |
| 4      | 7.03e-05      | 0.009          |

---

### Exercise 2 — `exercise_02_distortion_scaling.py`

**Why Distortion Drops Exponentially: Verifying the 1/4^b Scaling Law**

*Type: comparative.* You implement two things side-by-side:
- The empirical Lloyd-Max MSE for `b = 1 … 6`
- The Panter-Dite upper bound and Shannon lower bound

Fill in `compute_empirical_mse` (~10 lines), `compute_theoretical_bounds` (~5 lines),
and `plot_distortion_scaling` (~8 lines).

**Milestone:**
```bash
python exercise_02_distortion_scaling.py
```
Saves `milestone_02_distortion_scaling.png` — a log-scale plot with three curves
(empirical, upper bound, lower bound) that reproduces Figure 5 from the paper.

Expected ratios (empirical ÷ lower bound):

```
b=1 → 1.45×   b=2 → 1.87×   b=3 → 1.92×   b=4 → 2.30×
TurboQuant paper claims ≤ 2.72× — CONFIRMED!
```

---

### Exercise 3 — `exercise_03_codebook.py`

**From Centroids to Codebook: Building the Reusable Quantisation Table**

Implement `quantize_scalar` (~4 lines), `dequantize_scalar` (~2 lines),
`quantize_array` (~4 lines), and `dequantize_array` (~2 lines) in the `Codebook` class.

**Milestone:**
```bash
python exercise_03_codebook.py
```
Round-trips 10 000 Beta-distributed samples through the codebook and verifies that
the average squared error matches the analytical Lloyd-Max cost within statistical noise.

---

## The Core Math

### Lloyd-Max Iteration

Given distribution `f_X` and `k = 2^b` centroids, alternate until convergence:

```
(1) Boundaries:   t_l  = (c_{l−1} + c_l) / 2          [Voronoi midpoints]
(2) Centroids:    c_l  = E[X | t_{l−1} < X ≤ t_l]      [conditional mean]
                       = ∫_{t_{l−1}}^{t_l} x·f_X(x) dx
                         ─────────────────────────────
                         ∫_{t_{l−1}}^{t_l}   f_X(x) dx
```

### The 1/4^b Scaling Law (Panter-Dite Formula)

For any density `f_X` with variance `σ²`, the optimal scalar quantiser MSE scales as:

```
D_opt(b)  ≈  (√3·π/2) · σ² · 4^{−b}        [Panter-Dite upper bound]
D_opt(b)  ≥  (1/d) · 4^{−b}                 [Shannon lower bound, per-coord]
```

For the unit-sphere coordinate (`σ² = 1/d`), the total MSE is bounded by:

```
1/4^b  ≤  D_total(b)  ≤  (√3·π/2) · 4^{−b}  ≈  2.72/4^b
```

TurboQuant lives inside this band.

---

## Analytical Questions

*Answer these after completing all three exercises. Aim for depth — back up your
reasoning with numbers from the milestones.*

---

**Q1 (Analysis): Why does Lloyd-Max find the global optimum for the Beta distribution?**

Lloyd-Max is not guaranteed to find the global optimum in general — it can get stuck
in local minima. For the Beta coordinate distribution, however, we can be confident
it finds the globally optimal codebook.

*Reason through the following:*
The Beta distribution `f_X(t) = C_d · (1−t²)^{(d−3)/2}` is **symmetric and unimodal**
for `d ≥ 5`. Our initial centroids are placed symmetrically (at `±1/(k+1), ±3/(k+1), …`).
Why does this symmetry, combined with the unimodality of `f_X`, guarantee that no
asymmetric local minimum exists? Hint: consider the structure of the MSE objective
and whether a perturbation that breaks symmetry can ever lower the cost.

---

**Q2 (Analysis): Why does the ratio (empirical MSE / lower bound) increase with b?**

From your Exercise 2 output, the ratio grows from ~1.45× at `b=1` to ~2.30× at `b=4`,
and appears to approach the Panter-Dite limit of `√3·π/2 ≈ 2.72`.

*Give a precise explanation:* The Panter-Dite formula is a **high-resolution
approximation** — it assumes the quantisation error is locally uniform within each
cell. At `b=1` (only 2 centroids) this assumption fails badly, so the actual Lloyd-Max
solution is *better* than Panter-Dite predicts (i.e., closer to the lower bound).
As `b` grows, cells become smaller and the uniform-error approximation tightens.

Concretely: at `b=1` the two cells span the full distribution, so the error is
far from uniform within each cell. At `b=6` each of 64 cells spans a tiny range
where `f_X` is nearly flat. Quantify this by comparing the width of the widest
Voronoi cell at `b=1` vs. `b=4` for `d=128`. How does the cell width relate to
the standard deviation `σ = 1/√d ≈ 0.088`?

---

**Q3 (Synthesis): How would the codebook change if the coordinate distribution were
NOT the Beta/Gaussian, e.g. if outlier channels have variance 100× larger?**

In practice, some transformer attention weight channels are "outliers" with
significantly larger variance. The TurboQuant paper handles these separately
(Section 5.2, "outlier channel treatment").

*Design question:* Suppose you have `d=128` coordinates where 8 of them follow
`N(0, 100/d)` and the remaining 120 follow `N(0, 1/d)`.

(a) If you apply the standard Beta codebook to ALL coordinates equally, by how much
does the MSE increase for the outlier channels relative to the standard channels?
(Hint: the optimal Lloyd-Max MSE scales as `σ² · constant(b)`, so a `σ²` that is
100× larger gives a proportionally larger error.)

(b) The paper's fix is to allocate more bits to outlier channels: e.g., 3 bits for
outliers and 2 bits for standard channels (achieving 2.5 effective bits per coordinate).
Is it possible to maintain the 1/4^b distortion law per-channel in this mixed-bit
scheme? What is the total distortion in terms of the per-channel distortions?

(c) What role does the random rotation Π play when the INPUT distribution has outlier
channels? Does rotation help or hurt in this scenario?

---

**Q4 (Synthesis): The entropy of codebook indices is strictly less than b bits —
could you exploit this for "free" compression?**

From the Exercise 3 milestone's bonus section, the entropy `H` of centroid indices is
less than `b` bits for all `b`. The gap is ~5% at `b=4` (H ≈ 3.8 bits vs. 4 bits uniform).

(a) Using the centroid probabilities printed in the milestone, compute the entropy
`H = −Σ p_l·log₂(p_l)` for `b=2` and `b=4`. Why does the entropy deficit
*(b − H)* grow with `b`? Hint: think about the shape of `f_X` and how the
centroid probabilities become more non-uniform as `k = 2^b` grows.

(b) The paper claims entropy coding could reduce 4-bit quantisation to ~3.8 effective
bits. Why might TurboQuant's designers choose NOT to use entropy coding, even though
it is "free"? List at least two practical considerations (think about decoder
complexity, batching, and memory access patterns for LLM inference).

---

## Key Numbers to Remember

| Quantity | Value |
|---|---|
| Coordinate variance (d=128) | 1/128 ≈ 0.0078 |
| b=1 total MSE | 0.361 |
| b=4 total MSE | 0.009 |
| MSE ratio per bit added | ~4× (converges to exactly 4×) |
| Panter-Dite constant | √3·π/2 ≈ 2.718 |
| TurboQuant gap to lower bound | ≤ 2.72× (all b) |
| Entropy deficit at b=4 | ~5% (3.8 bits effective) |

---

## What's Next

Module 2 reveals a fatal flaw: **the MSE-optimal codebook is biased for inner products**.
A perfect MSE quantizer introduces a systematic multiplicative error in dot-product
estimates. You will measure this bias experimentally, trace it to the centroid
distribution, and understand why it makes the codebook useless for nearest-neighbour
search — motivating the QJL residual trick in Module 4.
