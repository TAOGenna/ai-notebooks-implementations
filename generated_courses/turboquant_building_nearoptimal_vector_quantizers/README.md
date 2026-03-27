# TurboQuant: Building Near-Optimal Vector Quantizers from Scratch

A progressive, hands-on course that builds the **TurboQuant** vector quantization algorithm from the ground up. By the end, you'll have implemented a near-optimal quantizer that compresses high-dimensional vectors to just 3 bits per coordinate with virtually zero quality loss — reproducing the key results from the [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026).

## What You'll Build

You'll implement the complete TurboQuant system in Python/NumPy:

- **Module 0**: Discover why random rotation is the key insight that makes data-oblivious quantization possible
- **Module 1**: Build an optimal scalar quantizer using the Lloyd-Max algorithm
- **Module 2**: Wire them into TurboQuant_mse — and discover its hidden bias on inner products
- **Module 3**: Implement QJL, the 1-bit sign-bit trick that provides unbiased inner product estimates
- **Module 4**: Combine MSE + QJL into TurboQuant_prod — the full near-optimal algorithm
- **Module 5**: Apply your quantizer to nearest neighbor search and KV cache attention simulation

## Prerequisites

- **Linear algebra**: inner products, norms, matrix multiplication, orthogonal transformations
- **Probability**: expected value, variance, distributions (Gaussian, Beta), unbiased estimators
- **Python + NumPy**: comfortable with array operations, broadcasting, basic matplotlib
- **ML basics**: know what attention, queries, keys, and values are (no transformer coding required)

## Setup

```bash
# Clone and navigate to the course directory
cd turboquant_building_nearoptimal_vector_quantizers

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

**Python 3.9+** is required. All exercises run on CPU — no GPU needed.

## Learning Path

The modules have the following dependency structure:

```
Module 0: Why Random Rotation Unlocks Optimal Quantization
    |                  \
    v                   v
Module 1:            Module 3:
Lloyd-Max Quantizer  QJL Sign-Bit Trick
    |                   |
    v                   |
Module 2:               |
TurboQuant_mse          |
    |                   |
    +------- + ---------+
             |
             v
Module 4: TurboQuant_prod (Two-Stage Unbiased Quantizer)
             |
             v
Module 5: Applications (NN Search + KV Cache)
```

**Modules 1 and 3 can be tackled in parallel** after completing Module 0. Module 1 builds the MSE quantizer path; Module 3 builds the QJL (inner product) path. They converge in Module 4.

### Recommended order for sequential study:
1. `module_00_why_random_rotation_unlocks_optimal_quan/` — Foundations: rotation, distribution, independence
2. `module_01_building_the_optimal_scalar_quantizer_wi/` — Optimal scalar quantization
3. `module_02_turboquantmse_the_full_mse_quantization/` — MSE quantizer + bias discovery
4. `module_03_qjl_and_the_signbit_trick_unbiased_1bit/` — Sign-bit quantization for inner products
5. `module_04_turboquantprod_wiring_mse_and_qjl_into_a/` — The full two-stage algorithm
6. `module_05_turboquant_in_the_wild_nearest_neighbor/` — Nearest neighbor search + KV cache simulation

## How to Work Through Each Module

1. **Read the module README** — it explains the concepts, notation, and what you'll discover
2. **Work through exercises in order** (ex01, ex02, ex03...) — each builds on the previous
3. **Fill in the TODO blocks** — line count estimates tell you how much code to write
4. **Run each exercise** — the `__main__` block prints results that validate your implementation
5. **Answer the analytical questions** in the module README — these build deeper intuition

## Key Numbers from the Paper

These are the quantitative targets your implementations should reproduce:

| Metric | b=1 | b=2 | b=3 | b=4 |
|--------|-----|-----|-----|-----|
| MSE distortion (D_mse) | 0.36 | 0.117 | 0.03 | 0.009 |
| Inner product distortion (D_prod × d) | 1.57 | 0.56 | 0.18 | 0.047 |
| IP bias of MSE quantizer | 2/π ≈ 0.637 | 0.912 | 0.978 | 0.995 |
| Theoretical lower bound (MSE) | 0.25 | 0.0625 | 0.0156 | 0.00391 |
| Ratio to lower bound | 1.45× | 1.87× | 1.92× | 2.30× |

Other key claims:
- TurboQuant is within a factor of **≈ 2.7** of the information-theoretic lower bound
- **3.5-bit** KV cache quantization achieves **zero quality degradation** on LongBench
- **4×+ compression** with **identical** needle-in-a-haystack performance to full precision
- Indexing time for nearest neighbor: **0.0007s** (TurboQuant) vs **37–597s** (PQ/RabitQ)
- 4-bit TurboQuant achieves **up to 8× speedup** over 32-bit unquantized keys on H100 GPUs

## What's Next

After completing this course, here are directions to explore — each bridges back to something you built:

- **PolarQuant and polar coordinate quantization**: Your TurboQuant uses Cartesian coordinates after rotation. PolarQuant (Module 0's companion paper) converts to polar coordinates instead, quantizing angles rather than coordinates. How would the distortion bounds change if you replaced the per-coordinate quantizer with a per-angle quantizer? → [PolarQuant paper](https://arxiv.org/abs/2502.02617)

- **Entropy encoding for further compression**: In Module 1, you computed codebook centroids with non-uniform usage probabilities. Applying Huffman or arithmetic coding to the centroid indices could reduce 4-bit quantization to ~3.8 effective bits. How would you integrate this into the pipeline without slowing down dequantization?

- **Outlier channel treatment**: Your Module 2 quantizer treats all coordinates equally. Real LLMs have outlier channels with 10–100× larger activations. The paper splits channels into outlier and non-outlier sets, allocating more bits to outliers (e.g., 32 channels at 3-bit + 96 channels at 2-bit = 2.5 effective bits). How would you detect outliers online?

- **Structured random matrices for efficiency**: Your QJL (Module 3) uses dense Gaussian matrices requiring O(d²) storage. Structured alternatives like random Hadamard transforms (used in QuaRot and FlashAttention-3) achieve O(d log d) computation. Could you replace the dense rotation in Module 0 with a fast Hadamard transform and maintain the same distortion guarantees?

- **Hardware-aware quantization kernels**: Your Python implementation demonstrates correctness but not speed. Real deployment requires custom CUDA kernels that pack b-bit integers into registers and use SIMD for parallel dequantization. The paper reports 8× speedup on H100 GPUs at 4-bit — how would you design the memory layout for cache-friendly access?

- **Extending to model weight quantization**: TurboQuant targets activations (KV cache), but model weights are static and can use offline methods. How would combining TurboQuant's data-oblivious approach with calibration-based methods (GPTQ, AWQ) improve weight quantization?

---
_Generated from [https://arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874) on 2026-03-26 by scaffoldly._
