# Arithmetic Transformers: A Comprehensive Literature Review

*Research report for the Minimal 10-Digit Transformer project*
*Date: 2026-02-25*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Academic Papers](#2-academic-papers)
   - 2.1 Teaching Arithmetic to Small Transformers (Lee et al., 2023)
   - 2.2 Transformers Can Do Arithmetic with the Right Embeddings (McLeish et al., 2024)
   - 2.3 Understanding Addition in Transformers (Quirke & Barez, 2024)
   - 2.4 Understanding Addition and Subtraction in Transformers (Quirke et al., 2024)
   - 2.5 Self-Improving Transformers (Lee et al., 2025)
3. [The Glove Box Challenge](#3-the-glove-box-challenge)
   - 3.1 Challenge Overview
   - 3.2 Trained Model Entries
   - 3.3 Hand-Coded Entries
4. [Gist Analysis](#4-gist-analysis)
   - 4.1 N8python - 343 Parameters
   - 4.2 xangma - 197 Parameters
   - 4.3 Cosmin Negruseri - 190 Parameters
   - 4.4 Cosmin Negruseri - 130 Parameters (Factorized)
5. [Related and Citing Work](#5-related-and-citing-work)
6. [Key Technical Insights](#6-key-technical-insights)
7. [Recommendations for Our Project](#7-recommendations-for-our-project)

---

## 1. Executive Summary

This report surveys the state of the art in transformer-based integer addition, with particular focus on minimal-parameter models. The field has made remarkable progress: from millions of parameters needed for basic arithmetic in 2023, to hand-coded solutions requiring as few as 130 parameters in 2026. The trained record stands at approximately 491 parameters, while hand-coded solutions have reached 130 parameters.

**Key finding for our project:** The gap between the trained record (~491 params) and hand-coded record (130 params) is approximately 3.8x. This gap represents the overhead of gradient descent finding an efficient solution versus a human directly encoding the algorithm. Closing this gap requires understanding exactly what computation is needed and finding architectural constraints that guide training toward the minimal solution.

**Critical technical elements identified across all successful approaches:**
- Reversed digit format (least-significant digit first)
- Sinusoidal or learned positional encodings that align digits by significance
- Small hidden dimensions (4-5)
- 1-2 transformer layers
- 2-3 attention heads
- Rank-1 factorizations for weight compression
- ReLU-based carry detection in MLPs

---

## 2. Academic Papers

### 2.1 Teaching Arithmetic to Small Transformers

**Authors:** Nayoung Lee, Kartik Sreenivasan, Jason D. Lee, Kangwook Lee, Dimitris Papailiopoulos
**Published:** ICLR 2024 (arXiv: 2307.03381, July 2023)
**Citations:** ~62 (as of early 2026)

**Architecture:**
- NanoGPT: 6 self-attention layers, 6 heads, embedding dimension 384, ~10.6M parameters
- Character-level tokenization
- Absolute positional encoding
- Also tested GPT-2 (124M parameters)

**Data Formatting Methods (key contribution):**

| Format | Example | Accuracy | Samples Needed |
|--------|---------|----------|----------------|
| Plain | `128+367=495` | ~85% (plateaus) | Never reaches 100% |
| Reverse | `$128+367=594$` | ~100% | ~2,500 (3-digit) |
| Simplified Scratchpad | `Input: 128+367 / A->5, C->1 / A->9, C->0 / ...` | ~100% | ~2,000 (3-digit) |
| Detailed Scratchpad | Full intermediate steps with text | ~100% | ~1,000 (3-digit) |

**Key Findings:**
1. **Reverse format is critical**: Reversing the output to start with the least significant digit dramatically improves performance because each output digit only depends on two input digits and the carry from the previous position -- a local function.
2. **Sharp phase transitions**: Accuracy jumps from near-0 to near-100% over a narrow range of training samples.
3. **Scratchpad/chain-of-thought helps**: Providing intermediate reasoning steps improves sample efficiency and accuracy simultaneously, even without pretraining.
4. **Length generalization fails**: Models trained on shorter additions do not generalize to longer ones. This limitation is a major focus of subsequent work.

**What worked:** Reverse format, scratchpad data augmentation, training from scratch on arithmetic-only data.
**What did not work:** Plain format for high accuracy, length generalization, zero-shot arithmetic in pretrained models.

**Source:** [arXiv:2307.03381](https://arxiv.org/abs/2307.03381), [GitHub](https://github.com/lee-ny/teaching_arithmetic)

---

### 2.2 Transformers Can Do Arithmetic with the Right Embeddings

**Authors:** Sean McLeish, Arpit Bansal, Alex Stein, Neel Jain, John Kirchenbauer, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Jonas Geiping, Avi Schwarzschild, Tom Goldstein
**Published:** NeurIPS 2024 (arXiv: 2405.17399, May 2024)

**Architecture:**
- Standard configuration: hidden=1024, intermediate=2048, 16 attention heads
- Parameter counts vary by recurrence:
  - 16x1 (standard): ~122M params
  - 8x2 (recurrent): ~64M params
  - 4x4: ~34M params
  - 1x16 (maximally recurrent): ~12M params
- Deepnorm initialization

**Core Innovation -- Abacus Embeddings:**
Abacus Embeddings assign learned positional embeddings to each digit based on its position *within a number* (not within the sequence). During training, indices start from a random offset beta sampled from U[1,100]; at test time beta=1. All digits of the same significance across both operands and the answer receive the same positional embedding.

This is fundamentally different from standard positional encodings (which track position in the sequence) and effectively tells the model which digits to align -- mimicking how humans line up columns in manual addition.

**Additional Techniques:**
1. **Input Injection**: Skip connections from the input embedding to every decoder layer. Reduces OOD generalization errors by 50%.
2. **Recurrent/Looped Layers**: Reusing the same layer weights multiple times. An 8x2 architecture halved errors versus a 16x1 on OOD problems.
3. **Progressive Loss Training**: Randomly varying recurrence count during training.

**Results:**
- Trained on 20-digit numbers, achieves up to 99% accuracy on 100-digit addition (5x generalization)
- This is 6x better generalization than prior work using FIRE embeddings (2.5x)
- In-distribution accuracy: 99-100%
- Extreme OOD (100-120 digits): 95%+

**Comparison of Positional Encodings:**

| Encoding | Length Generalization | In-Distribution |
|----------|---------------------|-----------------|
| NoPE | Poor | Baseline |
| RoPE | Limited | Good |
| FIRE | 2.5x | Good |
| Abacus | 6x | Excellent |
| Abacus + FIRE | Best | Excellent |

**Data Format:** Reversed (least-significant digit first). Uses loss masking on input tokens, only computing loss on answer digits. Training set: 20 million samples.

**Source:** [arXiv:2405.17399](https://arxiv.org/abs/2405.17399), [GitHub](https://github.com/mcleish7/arithmetic)

---

### 2.3 Understanding Addition in Transformers

**Authors:** Philip Quirke, Fazl Barez
**Published:** ICLR 2024 (arXiv: 2310.13121, October 2023)

**Architecture:**
- **1-layer transformer** with 3 attention heads
- Vocabulary: digits 0-9 plus "+", "="
- Tested with 5, 10, and 15-digit addition
- No explicit parameter count given, but very small

**Input Format:** `D4D3D2D1D0 + D'4D'3D'2D'1D'0 = A5A4A3A2A1A0` (left-to-right, most-significant first)

**Key Mechanistic Findings:**

The model decomposes addition into **parallel digit-specific streams**, each executing different algorithms:

1. **Base Add (BA)**: Sum two digits modulo 10
2. **Make Carry 1 (MC1)**: Determines if digit pair sum >= 10
3. **Use Carry 1 (UC1)**: Adds previous column's carry to current sum
4. **Use Sum 9 (US9)**: Propagates cascading carries when digits sum to exactly 9

**Attention Patterns:** A distinctive "double staircase" pattern. After the question tokens are processed, each of the 3 heads attends to different digit pairs in staggered rows, enabling access to data from 3 tokens per computation row.

**Critical Failure Mode:** The model cannot reliably handle US9 cascades (e.g., 99999 + 00001 = 100000) where carries must propagate across multiple columns. This is because the left-to-right generation order conflicts with the right-to-left nature of carry propagation.

**Training Details:**
- Dataset: 1.8 million of 10 billion possible questions
- ~5,000 training epochs, batch size 64
- US9 cascades artificially oversampled to increase their frequency
- Final loss: ~0.009 (5-digit)

**Source:** [arXiv:2310.13121](https://arxiv.org/abs/2310.13121), [GitHub](https://github.com/apartresearch/Integer_Addition)

---

### 2.4 Understanding Addition and Subtraction in Transformers

**Authors:** Philip Quirke, Clement Neo, Fazl Barez
**Published:** ICLR 2025 (arXiv: 2402.02619, February 2024, last revised October 2025)

**Architecture:**
- 2-3 layers, 3-4 attention heads
- Largest model: ~10M parameters (4 orders of magnitude smaller than GPT-3)
- Vocabulary: 14 tokens (digits 0-9 plus operators)
- Converges in under one hour on a single GPU

**Key Contributions:**

1. **Unified Mechanistic Account**: The paper identifies the exact circuits implementing arithmetic:

   **For addition (cascading carry):**
   - **ST (TriCase)**: Classifies each digit pair sum into three categories: definitely causes carry (e.g., 6+7), definitely no carry (e.g., 2+3), or ambiguous (sum = 9, e.g., 5+4)
   - **SV (multidigit cascade)**: Recursively combines ST values to resolve cascading carries
   
   **For subtraction (cascading borrow):**
   - **MB (borrow one)**: Identifies when borrowing is needed
   - **MV (cascading borrow)**: Mirrors SV logic for borrow propagation
   - **SGN**: Determines answer sign

2. **49 Models Trained**: Systematic ablation across different configurations, all validated against the proposed algorithm.

3. **99.999% Accuracy**: Fewer than 10 failures per 1 million test questions on most models.

4. **LLM Survey**: Only 7% of 180 surveyed LLMs (1B-405B parameters) can reliably perform addition. Model size shows no correlation with arithmetic capability.

**Key Insight for Our Project:** A 2-layer, 3-head transformer is sufficient for near-perfect n-digit addition. The TriCase classification (carry/no-carry/ambiguous) is the core computational primitive.

**Source:** [arXiv:2402.02619](https://arxiv.org/abs/2402.02619)

---

### 2.5 Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges

**Authors:** Nayoung Lee, Ziyang Cai, Avi Schwarzschild, Kangwook Lee, Dimitris Papailiopoulos
**Published:** ICML 2025 (arXiv: 2502.01612, February 2025)

**Architecture:** Standard transformer (NanoGPT-style), no architectural modifications or positional embedding changes. Uses the same 10.6M parameter NanoGPT from their 2023 paper.

**Method -- Self-Improvement Loop:**
1. Train model on easy problems (e.g., 10-digit addition)
2. Model generates solutions for harder problems (e.g., 11-digit)
3. Filter generated solutions for correctness (verification is easier than generation)
4. Train on correct self-generated solutions
5. Repeat, gradually increasing difficulty

**Results:**
- Generalizes from 10-digit to 100-digit addition "without apparent saturation"
- Exponential improvements in out-of-distribution performance across training rounds
- Works across arithmetic, string manipulation, and maze solving

**Key Insights:**
- Difficulty curriculum is critical -- models need structured progression
- Filtering out incorrect self-generated examples is essential
- Pretraining accelerates self-improvement
- No architectural changes needed -- the method is purely about training strategy

**Relevance to Our Project:** While this paper uses large models, the self-improvement approach could be valuable for training small models: train on easy cases first, then gradually extend.

**Source:** [arXiv:2502.01612](https://arxiv.org/abs/2502.01612)

---

## 3. The Glove Box Challenge

### 3.1 Challenge Overview

The "Glove Box" challenge (also referred to as the "Magic Box" challenge) was initiated by Dimitris Papailiopoulos in February 2026. The challenge originated from his quote about having "something close to a magic box where I throw in a question and a first answer comes back basically for free" -- referring to using AI coding assistants (Claude Code, Codex) to explore research questions.

Papailiopoulos asked Claude Code and Codex to each train the smallest possible transformer that can add two 10-digit numbers with high accuracy (>=99%). This sparked a community competition to push the parameter count as low as possible.

**Challenge Rules (reconstructed):**
- Task: Add two integers, each up to 10^10 - 1 (10-digit numbers)
- Model must be a legitimate transformer architecture (attention + MLP)
- Must achieve >=99% accuracy on random test cases
- Two categories: **trained** (weights found via gradient descent) and **hand-coded** (weights set manually)

### 3.2 Trained Model Entries (Chronological Progression)

| Entry | Parameters | Method | Notes |
|-------|-----------|--------|-------|
| Claude Code v1 | 6,080 | Standard training | Initial baseline |
| Codex v1 | 1,644 | Standard training | Initial baseline |
| Claude Code v2 | 980 | Creative architecture | After "try harder" prompt |
| Codex v2 | 970 | Creative architecture | After "try harder" prompt |
| Unknown | 777 | Low-rank techniques | Community entry |
| Unknown | 512 | Low-rank directions | Community entry |
| Unknown | 491 | Trained | Current trained record |

**Source:** [X/@DimitrisPapail](https://x.com/DimitrisPapail/status/2024596491474554902)

### 3.3 Hand-Coded Entries (Parameter Progression)

| Entry | Parameters | Author | Architecture | Key Technique |
|-------|-----------|--------|-------------|---------------|
| Baseline | 343 | N8python | Qwen3, 2-layer | Full weight specification |
| Rank-1 | 197 | xangma | Qwen3 + rank-1 | Factorized linear layers + embeddings |
| NanoGPT | 190 | Cosmin Negruseri | NanoGPT, 1-layer | Sinusoidal PE, parabolic decoding |
| Factorized NanoGPT | 130 | Cosmin Negruseri | NanoGPT, 1-layer, rank-1 | Full rank-1 decomposition |

---

## 4. Gist Analysis

### 4.1 N8python -- 343 Parameters (Qwen3 Hand-Coded)

**Gist:** [github.com/N8python/02e41d...](https://gist.github.com/N8python/02e41d156ec615328cde2e1e5c0e9d53)

**Architecture:**
- Framework: MLX (Apple)
- Model type: Qwen3 (with grouped query attention)
- Layers: 2
- Hidden size: 5
- Attention heads: 2 (1 KV head)
- Head dimension: 2
- MLP intermediate size: 3
- Vocabulary: 10 (digits 0-9)
- Output tokens: 11 (for sum up to 11 digits)

**Input Encoding:**
```
[0] + reversed_a_digits + [0, 0] + reversed_b_digits + [0]
```
Both numbers padded to 10 digits and reversed. Total sequence: 23 tokens.

**How It Works:**
1. **Embedding**: Maps digit i to a 5D vector where dim[0]=100, dim[1]=i (the digit value)
2. **Attention**: Routes information about corresponding digit pairs through the sequence
3. **MLP**: Uses ReLU gating to perform carry detection and modular arithmetic
4. **Output**: Autoregressively generates 11 output tokens representing the reversed sum

**Parameter Breakdown (approximate):**
- Embeddings: 50 (10 x 5)
- LM head: 55 (11 x 5)
- Attention (per layer): q_proj, k_proj, v_proj, o_proj
- MLP (per layer): gate, up, down projections
- Layer norms: weight vectors
- Total: 343

**Key Design Insight:** The digit value is encoded in one dimension while the "presence" flag is in another, allowing the attention mechanism to locate digit pairs and the MLP to perform the arithmetic.

---

### 4.2 xangma -- 197 Parameters (Rank-1 Qwen3)

**Gist:** [github.com/xangma/c538a7a...](https://gist.github.com/xangma/c538a7a9d415f16e61f7bb26ae5cf6b0)

This gist takes the N8python 343-parameter baseline and applies progressive compression:

**Compression Stages:**

| Stage | Technique | Params Saved | Running Total |
|-------|-----------|-------------|---------------|
| Baseline | Full Qwen3 | -- | 343 |
| Rank-1 Linear | Decompose W = u * v^T | -84 | 259 |
| Factorized Embedding | E[i] = A[i] @ B, where A is 10x2, B is 2x5 | -20 | 239 |
| Sparse Gate (Layer 0) | Reduce gate from 5->3 dims | -9 | 230 |
| Parameter-free Norms | Fixed scale RMSNorm (no learnable weights) | -33 | 197 |

**Rank-1 Factorization Detail:**
Instead of storing a full m x n weight matrix W (m*n parameters), store two vectors u (m params) and v (n params):
```
output = (x . v) * u   # dot product then scale
```
For a 4x5 projection: 20 params -> 9 params (55% reduction).

**Factorized Embedding Detail:**
```
E[token] = A[token] @ B
A: 10 x 2 matrix (digit -> [1.0, digit_value])
B: 2 x 5 matrix (route to embedding dims)
```
Reduces 50 params to 30, a 40% reduction. In practice the A matrix values are A[i] = [1.0, float(i)].

**RMSNormNoWeight:**
Replace learnable normalization weights with fixed constants (1.0 for layer norms, 16.0 for attention head norms). Saves all normalization parameters.

---

### 4.3 Cosmin Negruseri -- 190 Parameters (NanoGPT Hand-Coded)

**Gist:** [github.com/cosminscn/65a5fa...](https://gist.github.com/cosminscn/65a5fa5e20524495415f3cdd6bfdd7d2)

**Architecture:**
- Framework: PyTorch (NanoGPT-style)
- Layers: 1
- Attention heads: 2
- Embedding dimension: 4
- MLP hidden dimension: 4
- Vocabulary: 10 (digits 0-9; '+' and '=' mapped to token 0)
- Block size: 35 tokens
- Total: ~190 parameters

**Input Format:** `{10-digit_A}+{10-digit_B}={11-digit_sum}` where '+' and '=' are mapped to token 0.

**Positional Encoding Strategy:**
Uses sinusoidal embeddings with period 11 (theta = 2*pi/11):
- Dim 1: sin(p * theta) * amplitude
- Dim 2: cos(p * theta) * amplitude
- Amplitude = 100.0 for positions 0-21 (input region)
- Amplitude = 1.0 for positions 22-34 (output region)

The period of 11 is chosen because the two 10-digit numbers plus operators span positions that are exactly 11 apart -- corresponding digits in the two operands are separated by 11 positions.

**Attention Routing (Sinusoidal Resonance):**
- Head 0 queries 8 positions back (aligning with one operand's digits)
- Head 1 queries 9 positions back (aligning with the other operand's digits)
- Keys extracted from sin/cos dimensions
- Values preserve raw digit values from dim 0

**Carry Detection (MLP):**
Four ReLU neurons with high-magnitude weights (~1000) create sharp decision boundaries:
- Neurons 0-1: Detect whether the sum of two aligned digits >= 10 (carry required)
- Neurons 2-3: Handle wrap-around when carry causes sum to exceed 9 again

**Parabolic Output Decoding:**
The output head implements:
```
logit[v] = 2*v*x - v^2 = -(v - x)^2 + x^2
```
Taking argmax yields the nearest integer to x, effectively rounding the accumulated digit value.

**Parameter Breakdown:**
- Token embeddings: 40 (10 x 4)
- Attention c_attn: 48 (4 -> 12)
- Attention c_proj: 16 (4 -> 4)
- MLP c_fc: 16 (4 -> 4)
- MLP c_proj: 16 (4 -> 4)
- LM head: 40 + 10 bias = 50
- Positional: non-trainable (sinusoidal buffer)
- **Total: ~190**

---

### 4.4 Cosmin Negruseri -- 130 Parameters (Factorized NanoGPT)

**Gist:** [github.com/cosminscn/89c110...](https://gist.github.com/cosminscn/89c110dbae76ea0c873d67607e466f5b)

This builds on the 190-parameter model above and applies rank-1 factorization:

**Architecture:** Same as 190-parameter model (1 layer, 2 heads, embed=4, MLP=4)

**Compression Techniques:**

1. **Factorized Embeddings**: Decompose E = A @ B where A is 10x1 (digit values) and B is 1x4 (routing). Reduces 40 params to 14.

2. **Rank-1 MLP Projection**: The c_proj in MLP uses u (4x1) @ v (1x4) instead of full 4x4 matrix. Reduces 16 params to 8.

3. **Rank-1 LM Head**: The output layer uses rank-1 decomposition plus bias. Reduces 50 params to ~24.

**Parameter Breakdown:**
- Factorized embeddings: 14 (10 + 4)
- Attention c_attn: 48 (kept full -- this is the critical routing layer)
- Attention c_proj: 16 (kept full)
- MLP c_fc: 20 (weights + bias)
- MLP c_proj: 8 (rank-1: u=4, v=4)
- LM head: 14 + 10 bias = 24 (rank-1: u=10, v=4, bias=10)
- **Total: ~130**

**Design Philosophy:** The attention weights (c_attn, c_proj) are kept full because they implement the critical digit-routing logic that cannot be further compressed without losing the sinusoidal resonance patterns. The MLP and output layers, which perform more "smooth" numerical operations, tolerate rank-1 approximation well.

---

## 5. Related and Citing Work

### 5.1 Positional Description Matters for Transformers Arithmetic (Shen et al., 2023)

**arXiv:** [2311.14737](https://arxiv.org/abs/2311.14737) | **Venue:** ICLR 2024

Proposes modifying positional encodings or task representation to improve arithmetic. Key result: 15-digit multiplication with 100M parameters and 300k samples; addition extrapolation from 10 to 12 digits with 120k samples.

### 5.2 Position Coupling (Cho et al., 2024)

**arXiv:** [2405.20671](https://arxiv.org/abs/2405.20671) | **Venue:** NeurIPS 2024

Assigns the same position IDs to digits of the same significance across operands and answer. A 1-layer transformer with position coupling can solve addition with exponentially many digits. Models trained on 1-30 digit additions generalize to 200 digits (6.67x).

**Relevance:** This is conceptually similar to Abacus Embeddings but applied at the position ID level. Both demonstrate that telling the model which digits to align is the key insight.

### 5.3 FoNE: Fourier Number Embeddings (Zhou et al., 2025)

**arXiv:** [2502.09741](https://arxiv.org/abs/2502.09741)

Maps numbers into embedding space using Fourier features (sin/cos at different periods). Achieves 100% accuracy on arithmetic tasks with 64x less training data. Only 2 embedding dimensions per digit needed.

**Relevance:** Validates the hand-coded approaches' use of sinusoidal representations. The Cosmin Negruseri gists use exactly this type of Fourier-based encoding.

### 5.4 Extrapolation by Association (Cai et al., 2025)

**arXiv:** [2506.09251](https://arxiv.org/abs/2506.09251) | **Venue:** NeurIPS 2025 Spotlight

Demonstrates that length generalization transfers across related tasks. Training on a longer auxiliary task enables generalization on a shorter target task. Mechanistically, transfer correlates with re-use of the same attention heads between tasks.

### 5.5 Pre-trained LLMs Use Fourier Features for Addition (NeurIPS 2024)

Shows that even pretrained LLMs internally learn Fourier-like representations for number tokens, validating the theoretical basis for Fourier/sinusoidal approaches.

### 5.6 Reducing the Transformer Architecture to a Minimum (Bermeitinger et al., 2024)

**arXiv:** [2410.13732](https://arxiv.org/abs/2410.13732)

Proposes removing MLP layers, collapsing Q/K matrices, and using symmetric similarity measures. Saves up to 90% of parameters on classification tasks. While tested on vision (MNIST, CIFAR-10), the principles of collapsing attention matrices are relevant to our parameter minimization goal.

### 5.7 Arithmetic-Transformer (Thomas Ahle)

**GitHub:** [thomasahle/arithmetic-transformer](https://github.com/thomasahle/arithmetic-transformer)

Tests LSTM, Transformer (various PE), and hybrid architectures. Best result: LSTM+Transformer hybrid reaches 18-digit addition with ~130K parameters. Uses curriculum learning (progress from 1-digit to N-digit). Key insight: LSTM as first layer handles unbounded sequence lengths better than fixed positional encodings.

---

## 6. Key Technical Insights

### 6.1 Why Reverse Format Works

The reverse (least-significant digit first) format is universally used in successful approaches because:

1. **Locality of computation**: Each output digit depends only on the corresponding two input digits plus a carry bit from the previous position. This is a local function of just 3 variables.
2. **Causal alignment**: In autoregressive generation (left-to-right), the carry propagates in the same direction as generation -- from less significant to more significant digits.
3. **Error containment**: Even if the model makes an error on one digit, subsequent digits can still be correct because the carry is a single bit.

### 6.2 The Core Addition Algorithm

From the mechanistic interpretability work, addition in a transformer requires:

1. **Digit routing**: Attention mechanism identifies which two digits to add (via positional encoding)
2. **TriCase classification**: For each digit pair, determine if sum is <10 (no carry), >10 (carry), or =9 (ambiguous -- depends on lower digits)
3. **Carry cascade resolution**: Resolve the ambiguous cases by checking lower-significance digits
4. **Modular arithmetic**: Compute (digit_a + digit_b + carry) mod 10

This requires at minimum:
- 2 attention heads (to fetch the two operand digits)
- 1 MLP layer (to compute sum and carry)
- Some mechanism for carry propagation (either cascading circuits or the autoregressive carry-forward)

### 6.3 Parameter Budget Analysis

For a minimal 10-digit adder with vocab=10, hidden=d, and the above components:

| Component | Full | Rank-1 | Notes |
|-----------|------|--------|-------|
| Embedding | 10*d | 10+d | Factorized |
| Attention (2 heads) | ~4*d^2 | ~8*d | If factorizable |
| MLP | ~2*d^2 | ~4*d | Rank-1 projections |
| Output head | 10*d | 10+d | Factorized |
| Layer norms | ~4*d | 0 | Fixed norms |
| **Total (d=4)** | **~196** | **~90** | Theoretical minimum |

The 130-parameter hand-coded solution is close to this theoretical minimum with d=4, keeping attention full (64 params) but factorizing everything else.

### 6.4 What the Hand-Coded Solutions Teach Us

The progression from 343 to 130 parameters reveals which components carry the "essential information":

1. **Attention weights are most important**: The c_attn (48 params at d=4) and c_proj (16 params) are kept full even in the most compressed version. They implement the digit-routing logic via sinusoidal resonance.

2. **Embeddings are highly compressible**: A simple [1.0, digit_value] encoding suffices. The model only needs to know "this is a digit" and "what digit is it."

3. **MLP can be rank-1**: The carry detection and modular arithmetic operations, while nonlinear, can be expressed with very few parameters.

4. **Normalization parameters are unnecessary**: Fixed-scale RMSNorm works fine because the hand-coded weights are already properly scaled.

5. **Sinusoidal PE is both free and powerful**: The non-trainable sinusoidal positional encoding with carefully chosen period (11 in the NanoGPT approach) provides digit alignment at zero parameter cost.

---

## 7. Recommendations for Our Project

### 7.1 Goal: Beat 491 Trained Parameters

Based on this research, here are the most promising approaches ranked by potential:

#### Approach 1: Constrained Architecture Training (Most Promising)

Start with the 130-parameter hand-coded architecture but make it trainable:
- 1 layer, 2 heads, embed_dim=4, MLP_dim=4
- Fixed sinusoidal PE (period 11, non-trainable)
- Rank-1 factorizations on MLP and output head
- Factorized embeddings
- Fixed-scale normalization (no learnable norm weights)
- **Estimated: ~130-200 trainable parameters**

The key question: can gradient descent find the right weights with this architecture? The hand-coded solution proves the architecture is sufficient.

#### Approach 2: Slightly Larger Architecture with Aggressive Compression

- 1-2 layers, 2-3 heads, embed_dim=5
- Start with a conventional architecture (~343-500 params)
- Train normally, then apply post-training compression:
  - SVD/rank-1 factorization of weight matrices
  - Prune near-zero weights
  - Quantize remaining weights
- **Estimated: 200-400 trained parameters**

#### Approach 3: Knowledge Distillation from Hand-Coded Teacher

- Use the 130-parameter hand-coded model as a teacher
- Train a similar-sized student with distillation loss
- The student may learn the same algorithm but with trained weights
- **Estimated: ~130-200 parameters**

#### Approach 4: Curriculum Learning

Following the "Self-Improving Transformers" approach but with a tiny model:
- Train on 1-digit addition first, then 2-digit, etc.
- Use the reverse format
- Filter self-generated solutions for correctness
- This may help the model converge to the right algorithm despite few parameters

### 7.2 Critical Design Decisions

1. **Data Format**: Use reversed (LSB-first) format. This is universal across all successful approaches.

2. **Positional Encoding**: Use sinusoidal with period 11 (non-trainable). This is proven by the hand-coded solutions and validated by FoNE research. Zero parameter cost.

3. **Hidden Dimension**: 4 is the minimum proven to work (Cosmin Negruseri). 5 gives more headroom (N8python/xangma).

4. **Number of Layers**: 1 is sufficient for the hand-coded solutions. Training may benefit from 2 layers for additional capacity.

5. **Attention Heads**: 2 minimum (one per operand digit). 3 gives clearer streams per Quirke & Barez.

6. **Activation Function**: Use ReLU, not GELU. ReLU enables sharp decision boundaries needed for carry detection. This is explicit in the hand-coded solutions.

7. **Weight Initialization**: Initialize near the known hand-coded solution to help gradient descent converge to the right basin.

### 7.3 What To Avoid

- **Plain format**: Never reaches 100% accuracy even with large models
- **Standard positional encodings**: Absolute learned PE wastes parameters; use fixed sinusoidal
- **Large embedding dimensions**: >5 is wasteful for this task
- **GELU activation**: Not sharp enough for carry detection
- **Ignoring the TriCase structure**: The carry/no-carry/ambiguous classification is the core algorithm

### 7.4 Open Questions

1. Can gradient descent discover the sinusoidal-resonance attention pattern, or does it need to be hard-coded?
2. Is there a training curriculum that helps tiny models converge to the right algorithm?
3. Can we use the hand-coded solutions as initialization for fine-tuning?
4. What is the true theoretical minimum number of parameters for a trained 10-digit adder?
5. Does the Qwen3 architecture (with grouped query attention) offer advantages over standard multi-head attention for this task?

---

## Sources

### Academic Papers
- [Teaching Arithmetic to Small Transformers](https://arxiv.org/abs/2307.03381) - Lee et al., ICLR 2024
- [Transformers Can Do Arithmetic with the Right Embeddings](https://arxiv.org/abs/2405.17399) - McLeish et al., NeurIPS 2024
- [Understanding Addition in Transformers](https://arxiv.org/abs/2310.13121) - Quirke & Barez, ICLR 2024
- [Understanding Addition and Subtraction in Transformers](https://arxiv.org/abs/2402.02619) - Quirke et al., ICLR 2025
- [Self-Improving Transformers](https://arxiv.org/abs/2502.01612) - Lee et al., ICML 2025
- [Positional Description Matters for Transformers Arithmetic](https://arxiv.org/abs/2311.14737) - Shen et al., ICLR 2024
- [Position Coupling](https://arxiv.org/abs/2405.20671) - Cho et al., NeurIPS 2024
- [FoNE: Fourier Number Embeddings](https://arxiv.org/abs/2502.09741) - Zhou et al., 2025
- [Extrapolation by Association](https://arxiv.org/abs/2506.09251) - Cai et al., NeurIPS 2025
- [Reducing the Transformer Architecture to a Minimum](https://arxiv.org/abs/2410.13732) - Bermeitinger et al., 2024

### GitHub Repositories
- [lee-ny/teaching_arithmetic](https://github.com/lee-ny/teaching_arithmetic)
- [mcleish7/arithmetic](https://github.com/mcleish7/arithmetic)
- [apartresearch/Integer_Addition](https://github.com/apartresearch/Integer_Addition)
- [thomasahle/arithmetic-transformer](https://github.com/thomasahle/arithmetic-transformer)

### Gists (Glove Box Challenge Entries)
- [N8python - 343 params](https://gist.github.com/N8python/02e41d156ec615328cde2e1e5c0e9d53)
- [xangma - 197 params](https://gist.github.com/xangma/c538a7a9d415f16e61f7bb26ae5cf6b0)
- [cosminscn - 190 params](https://gist.github.com/cosminscn/65a5fa5e20524495415f3cdd6bfdd7d2)
- [cosminscn - 130 params (factorized)](https://gist.github.com/cosminscn/89c110dbae76ea0c873d67607e466f5b)

### Twitter/X Threads
- [@DimitrisPapail challenge thread](https://x.com/DimitrisPapail/status/2024596491474554902)
- [@DimitrisPapail on computational circuits](https://x.com/DimitrisPapail/status/1952165188683096089)
- [Simon Willison quote](https://simonwillison.net/2026/Feb/17/dimitris-papailiopoulos/)
