# Literature Review: Transformer Architectures, Attention Mechanisms, and Parameter-Efficient Designs

**Date:** 2026-02-25
**Purpose:** Research report for building a minimal-parameter transformer for 10-digit addition
**Context:** Current SOTA is ~491 params (trained) and ~130 params (hand-coded) for 10-digit addition with >=99% accuracy

---

## Table of Contents

1. [Modern Transformer Architectures](#1-modern-transformer-architectures)
2. [Attention Mechanisms](#2-attention-mechanisms)
3. [Sparsity and Matrix Decomposition](#3-sparsity-and-matrix-decomposition)
4. [Layer Reuse and Sharing](#4-layer-reuse-and-sharing)
5. [Causality and Masking](#5-causality-and-masking)
6. [Synthesis: Implications for Minimal Addition Transformers](#6-synthesis-implications-for-minimal-addition-transformers)
7. [References](#7-references)

---

## 1. Modern Transformer Architectures

### 1.1 DeepSeek-V2 and Multi-head Latent Attention (MLA)

**Paper:** "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (arXiv:2405.04434, May 2024)

DeepSeek-V2 is a 236B-parameter MoE model (21B activated per token) introducing **Multi-head Latent Attention (MLA)**. The key innovation is replacing the standard W_KV matrix with a low-rank factorized projection:

- **Compression**: KV states are compressed into a low-dimensional latent vector before caching
- **Decompression**: During inference, latent vectors are decompressed on-the-fly to recreate full K and V matrices
- **Impact**: 93.3% KV cache reduction, 5.76x maximum generation throughput increase, 42.5% training cost reduction

**Relevance to minimal models**: MLA's core idea -- compressing key/value representations through a latent bottleneck -- is directly applicable to parameter-efficient designs. For a minimal addition model, we could compress attention KV representations to a very small latent dimension (e.g., 2-4 dimensions) to dramatically reduce parameter count.

Source: [arXiv:2405.04434](https://arxiv.org/abs/2405.04434), [Explanation by Shirley Li](https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4/)

### 1.2 DeepSeek-V3

**Paper:** "DeepSeek-V3 Technical Report" (arXiv:2412.19437, December 2024)

DeepSeek-V3 is a 671B-parameter MoE model (37B activated) building on V2:

- **Architecture**: MLA with 128 heads, 256 routed experts (8 per token + 1 shared)
- **Auxiliary-loss-free load balancing**: Eliminates the performance-degrading auxiliary losses used in prior MoE work
- **Multi-Token Prediction (MTP)**: Predicts multiple future tokens per position during training, improving representation quality
- **DualPipe pipeline parallelism**: Overlaps forward/backward communication phases

**Relevance**: The multi-token prediction objective is interesting for addition -- rather than predicting one digit at a time, predicting multiple output digits simultaneously could improve accuracy and reduce sequential dependency.

Source: [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

### 1.3 DeepSeek-R1

**Paper:** "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (arXiv:2501.12948, January 2025)

DeepSeek-R1 builds on the V3 architecture but emphasizes training methodology over architecture:

- **Group Relative Policy Optimization (GRPO)**: RL framework using only correctness-based rewards
- **DeepSeek-R1-Zero**: Trained purely via RL without supervised fine-tuning, enabling reasoning without supervised datasets
- **Distilled models**: Smaller models (1.5B-70B) distilled from the full model retain strong reasoning

**Relevance**: The insight that reasoning can emerge from pure RL (without SFT) is relevant for our arithmetic task. We might explore training with only correctness rewards rather than standard cross-entropy loss.

Source: [arXiv:2501.12948](https://arxiv.org/pdf/2501.12948), [HiddenLayer Analysis](https://hiddenlayer.com/innovation-hub/analysing-deepseek-r1s-architecture/)

### 1.4 NVIDIA Nemotron

**Paper:** "NVIDIA Nemotron 3: Efficient and Open Intelligence" (arXiv:2512.20856, December 2025)

Nemotron 3 introduces a **hybrid Mamba-Transformer MoE** architecture:

- **Hybrid design**: Combines Mamba layers (efficient long-range dependencies) with Transformer attention layers (structural/logical relationships)
- **Latent MoE**: Novel hardware-aware expert design in the larger models (Super, Ultra)
- **MTP layers**: Multi-token prediction for better long-form generation
- **Native 1M-token context windows**
- **Sizes**: 30B (Nano), 100B (Super), ~500B (Ultra)

**Relevance**: The hybrid Mamba-Transformer concept is interesting but likely overkill for our 23-token sequences. However, the idea of combining different computational primitives (e.g., a simple recurrent/state-based mechanism with attention) could be parameter-efficient.

Source: [NVIDIA Nemotron](https://research.nvidia.com/labs/nemotron/Nemotron-3/), [arXiv:2512.20856](https://arxiv.org/pdf/2512.20856)

### 1.5 ModernBERT

**Paper:** "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder" (arXiv:2412.13663, December 2024)

ModernBERT modernizes the BERT architecture with several improvements:

- **Rotary Positional Embeddings (RoPE)**: Better position understanding, scalable to 8192 tokens
- **GeGLU layers**: Replacing standard MLP with gated linear units using GeLU activation
- **Alternating attention**: Every 3rd layer uses global attention (RoPE theta=160,000), others use local sliding window (128 tokens, theta=10,000)
- **Unpadding**: Eliminates wasted compute on padding tokens
- **Training**: 2T tokens, native 8192 sequence length

**Relevance**: Several techniques are directly applicable:
1. **GeGLU** can replace standard FFN with better expressiveness per parameter
2. **Alternating local/global attention** pattern could work for addition where local digit interactions dominate
3. **Unpadding** is useful if we use variable-length inputs

Source: [arXiv:2412.13663](https://arxiv.org/abs/2412.13663), [HuggingFace Blog](https://huggingface.co/blog/modernbert)

### 1.6 Ettin

**Paper:** Ettin Suite (Johns Hopkins University CLSP, 2025)

Ettin provides the **first controlled encoder vs. decoder comparison**:

- Identical training data (2T tokens across all models)
- Matched architectures (only attention patterns and objectives differ)
- Models from 17M to 1B parameters with 250+ checkpoints
- Uses "deep but thin" Transformer architecture across all variants

**Key Finding**: Attempts to repurpose one architecture for the other's domain via continued pretraining on the opposite objective produce subpar results. Decoders remain dominant in generation tasks regardless of adaptation.

**Relevance**: This suggests that for addition (a generation task), a decoder-only architecture is natural. However, the bidirectional context of encoders could help with understanding the full input before generating output. A prefix-LM approach (bidirectional on input, causal on output) might combine the best of both.

Source: [GitHub: JHU-CLSP/ettin-encoder-vs-decoder](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder)

### 1.7 NanoGPT and microGPT

**NanoGPT** (Andrej Karpathy, 2022): Minimal ~600-line GPT training implementation. Key efficiency features:
- Shared input/output embeddings (weight tying)
- Vocabulary padding to nearest multiple of 64 for hardware efficiency
- PyTorch 2.0 torch.compile() optimization

**microGPT** (Andrej Karpathy, February 2026): Extreme minimalism -- 200 lines of pure Python, no dependencies, 4,192 parameters:
- Character-level tokenizer
- RMS normalization (instead of LayerNorm)
- Square ReLU nonlinearity (instead of GeLU)
- No biases
- Positional and token embeddings with residual connections

**Relevance**: microGPT demonstrates that a functional transformer can operate at ~4K parameters. The design choices (RMSNorm, squared ReLU, no biases) are directly applicable to our sub-500-parameter target. Eliminating biases alone can save significant parameters in tiny models.

Source: [NanoGPT GitHub](https://github.com/karpathy/nanoGPT), [microGPT blog post](http://karpathy.github.io/2026/02/12/microgpt/), [microGPT gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

### 1.8 ByteDance Architecture Research

ByteDance has contributed several relevant works:

- **MegaScale** (NSDI 2024): Production system for training LLMs on 10,000+ GPUs, achieving 55.2% MFU on 175B models. Incorporates parallel transformer blocks, sliding window attention, and LAMB optimizer.
- **ByteTransformer**: High-performance transformer inference engine
- **1.58-bit FLUX**: Quantizing 99.5% of Vision Transformer parameters to 1.58 bits while maintaining performance

**Relevance**: The 1.58-bit quantization work is relevant for extreme parameter efficiency -- if weights can be constrained to ternary values {-1, 0, 1}, the effective information per parameter is reduced but the model might still function for simple tasks like addition.

Source: [MegaScale paper](https://arxiv.org/html/2402.15627v1), [1.58-bit FLUX](https://www.marktechpost.com/2024/12/30/bytedance-research-introduces-1-58-bit-flux/)

### 1.9 Latest Architecture Trends (2025-2026)

Key emerging trends:

- **Hybrid architectures**: Combining Mamba/SSM with Transformer attention (Nemotron-H, Jamba)
- **Sub-quadratic alternatives**: Mamba, RWKV, RetNet reaching parity with Transformers on many benchmarks
- **Hourglass FFN designs**: Wide-narrow-wide sub-MLPs connected by residuals
- **Pre-normalization standard**: RMSNorm before attention/FFN is now standard
- **Grouped-Query Attention**: Default in Llama 2/3, Mistral, etc.
- **Rotary Position Embeddings**: Near-universal adoption

**ICLR 2026 highlight**: "Can Transformers Really Do It All?" examines when Transformers are overkill vs. when simpler architectures suffice.

Source: [Rohit Bandaru's Transformer Design Guide](https://rohitbandaru.github.io/blog/Transformer-Design-Guide-Pt2/)

---

## 2. Attention Mechanisms

### 2.1 Standard Multi-Head Attention (MHA)

The original attention mechanism from "Attention Is All You Need" (Vaswani et al., 2017):

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

For h heads with model dimension d_model:
- Parameters per attention layer: 4 * d_model^2 (for Q, K, V projections + output)
- For small d_model (e.g., 16), this is only 4 * 256 = 1024 parameters per layer

**For addition**: Quirke & Barez (2024) showed that a single-layer, 3-head attention model is sufficient for >99.999% on 5-digit addition. The 3 heads develop distinct roles: attending to corresponding digit pairs, carry propagation, and position encoding.

### 2.2 Multi-Query Attention (MQA)

**Paper:** "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer, 2019)

All attention heads share a single set of keys and values:
- Queries remain per-head: h * d_k parameters
- Keys and values are shared: 2 * d_k parameters (instead of 2 * h * d_k)
- **Parameter savings**: Reduces KV parameters by factor of h

**Relevance**: For minimal models with very few heads (1-3), MQA offers diminishing returns since the head count is already low. However, for a single-head model, MQA and MHA are identical.

### 2.3 Grouped Query Attention (GQA)

**Paper:** "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., arXiv:2305.13245, EMNLP 2023)

GQA is a generalization between MHA and MQA:
- Groups of query heads share KV projections
- With G groups for H heads: reduces KV parameters by factor H/G
- Adopted by Llama 2/3, Mistral 7B

**Recent extensions** (2024): Key-Distributed GQA (KDGQA) and Dynamic GQA (DGQA) use key head norms to inform query allocation.

**Relevance**: In a tiny model (1-2 heads), GQA is equivalent to MHA or MQA. The concept of sharing projections across heads is more relevant in our context.

Source: [arXiv:2305.13245](https://arxiv.org/abs/2305.13245), [IBM Explanation](https://www.ibm.com/think/topics/grouped-query-attention)

### 2.4 Multi-head Latent Attention (MLA)

**Paper:** DeepSeek-V2 (arXiv:2405.04434)

MLA compresses KV into a latent space:
```
c_KV = W_DKV * x     (compress to latent, d_c << d)
K = W_UK * c_KV      (decompress keys)
V = W_UV * c_KV      (decompress values)
```

The effective weight matrix is W_UK * W_DKV -- a low-rank factorization of the key projection.

**Relevance**: This is highly relevant for minimal models. Instead of a d_model x d_model key projection, we can use d_model x r and r x d_model projections where r << d_model. For d_model=16 and r=2, this reduces from 256 to 64 parameters.

Source: [Understanding MLA](https://planetbanatt.net/articles/mla.html), [Sebastian Raschka's explanation](https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/)

### 2.5 Sparse Attention

#### 2.5.1 Longformer

**Paper:** "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)

Three sparse patterns:
1. **Sliding window**: Each token attends to w neighbors
2. **Dilated sliding window**: Gaps in the window for larger receptive field
3. **Global tokens**: Selected tokens attend to all positions

#### 2.5.2 BigBird

**Paper:** "Big Bird: Transformers for Longer Sequences" (Zaheer et al., NeurIPS 2020, arXiv:2007.14062)

Combines three attention patterns:
1. **Global tokens**: Attend to entire sequence
2. **Local sliding window**: Attend to neighbors
3. **Random connections**: Random token-to-token attention

BigBird proves this sparse attention is a **universal approximator** and is **Turing complete**, reducing complexity from O(n^2) to O(n).

**Relevance**: For addition with 23 tokens, full attention is computationally trivial. However, structured sparsity patterns that match the addition algorithm (e.g., each output digit attends only to its two input digits and the carry chain) could provide an inductive bias that reduces learning difficulty.

Source: [arXiv:2007.14062](https://arxiv.org/abs/2007.14062), [HuggingFace BigBird](https://huggingface.co/blog/big-bird)

### 2.6 Linear Attention Approximations

#### 2.6.1 Performer

**Paper:** "Rethinking Attention with Performers" (Choromanski et al., arXiv:2009.14794, 2020)

Uses FAVOR+ (Fast Attention Via positive Orthogonal Random features):
- Approximates softmax(QK^T) using random feature maps: phi(Q) * phi(K)^T
- Complexity: O(n * d^2) instead of O(n^2 * d)
- Unbiased estimation with convergence guarantees

#### 2.6.2 Linformer

**Paper:** "Linformer: Self-Attention with Linear Complexity" (Wang et al., arXiv:2006.04768, 2020)

Projects K and V to lower dimension k << n:
```
K' = E_K * K   (n x d -> k x d)
V' = E_V * V   (n x d -> k x d)
Attention = softmax(Q * K'^T / sqrt(d)) * V'
```

Key insight: Self-attention matrix is approximately low-rank.

#### 2.6.3 Gated Linear Attention (GLA)

**Paper:** "Gated Linear Attention Transformers with Hardware-Efficient Training" (Yang et al., ICML 2024, arXiv:2312.06635)

- Data-dependent gates for linear attention
- Can be formulated as RNN with matrix-valued hidden states
- Faster than FlashAttention-2 even on 1K sequences
- Excellent length generalization (2K training to 20K+ inference)

**Relevance**: For very short sequences (23 tokens for addition), linear attention approximations don't offer computational benefits. However, GLA's RNN dual formulation could be interesting for a recurrent approach to addition.

Source: [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)

### 2.7 SE(3)-Transformers and Geometric Attention

**Paper:** "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks" (Fuchs et al., NeurIPS 2020, arXiv:2006.10503)

- Generalizes self-attention to respect SE(3) symmetries (3D rotations and translations)
- Attention weights are invariant to global pose; value updates are equivariant
- Used primarily in molecular/protein modeling

**Relevance**: Not directly applicable to digit addition. However, the concept of building equivariances into attention is instructive. Addition has a natural equivariance: adding a constant to both operands shifts the result by twice that constant. Encoding such structural properties could help.

Source: [arXiv:2006.10503](https://arxiv.org/abs/2006.10503)

### 2.8 Minimum Attention for Addition

Based on mechanistic interpretability studies:

**1-layer, 3-head model** (Quirke & Barez, ICLR 2024): Achieves >99.999% on 5-15 digit addition. The three heads develop distinct roles:
- Head 1: Attends to digit pairs for direct addition
- Head 2: Carry propagation (attending to positions that might generate carries)
- Head 3: Position/format handling

**1-layer, 2-head model**: Works but with overlapping attention patterns (heads attend to multiple roles simultaneously), making interpretation harder.

**Theoretical result** (Cho et al., NeurIPS 2024): A 1-layer Transformer with position coupling can solve addition of exponentially long integers.

**Practical minimum for 10 digits**: Based on the literature, 1 layer with 2-3 attention heads appears sufficient, provided proper positional encoding (e.g., Abacus embeddings or position coupling).

Source: [arXiv:2310.13121](https://arxiv.org/abs/2310.13121), [arXiv:2402.02619](https://arxiv.org/abs/2402.02619), [arXiv:2405.20671](https://arxiv.org/abs/2405.20671)

### 2.9 FlashAttention (Implementation, Not Architecture)

**Papers:** FlashAttention (Dao et al., 2022), FlashAttention-2 (2023), FlashAttention-3 (2024)

IO-aware exact attention algorithm:
- FA1: Tiling + recomputation, 10-20x memory savings
- FA2: 2-4x speedup over FA1, supports MQA/GQA natively
- FA3 (2024): Hopper GPU optimized, 1.5-2x over FA2, FP8 support up to 1.2 PFLOPS

**Relevance**: Not relevant for architecture design (it computes exact attention), but critical for training efficiency. Use FA2/FA3 in training loop for speed.

Source: [FlashAttention-3 blog](https://tridao.me/blog/2024/flash3/), [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

---

## 3. Sparsity and Matrix Decomposition

### 3.1 Low-Rank Decomposition of Weight Matrices

#### 3.1.1 LoRA and Variants

**Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., arXiv:2106.09685, ICLR 2022)

Core idea: Weight update as low-rank factorization:
```
W = W_0 + BA    where B in R^{d x r}, A in R^{r x k}, r << min(d,k)
```

**Key variants (2024)**:
- **DoRA** (Weight-Decomposed LoRA): Decomposes weight into magnitude and direction components
- **ALoRA**: Dynamically adjusts rank per module during adaptation
- **EDoRA, LoRA-Mini, Q3R**: Sublinear memory footprint

**Relevance**: For a minimal model, the entire weight matrix could be parameterized as a low-rank factorization from the start (not as an adaptation). For a 16x16 matrix with rank 2: 16*2 + 2*16 = 64 parameters instead of 256.

Source: [arXiv:2106.09685](https://arxiv.org/pdf/2106.09685), [LoRA survey](https://arxiv.org/html/2501.00365v1)

#### 3.1.2 ResidualTransformer

**Paper:** "ResidualTransformer: Residual Low-Rank Learning with Weight-Sharing for Transformer Layers" (Wang et al., ICASSP 2024, arXiv:2310.02489)

Each weight matrix = shared full-rank component + unique low-rank residual:
```
W_i = W_shared + D_i * L_i * R_i
```
where D_i is a diagonal scaling matrix, and L_i, R_i are low-rank factors.

Results: **~3x encoder size reduction** with slight performance degradation on speech tasks.

**Relevance**: This is the most directly relevant approach for our goal. We can share a base weight matrix across layers and add tiny low-rank corrections per layer. For a 16x16 weight with rank-1 residual: 256 (shared) + 16+16+16 = 304 per layer instead of 256 per layer (but shared across N layers, so amortized cost is much lower).

Source: [arXiv:2310.02489](https://arxiv.org/abs/2310.02489)

#### 3.1.3 PELA: Learning Parameter-Efficient Models with Low-Rank Approximation

**Paper:** PELA (CVPR 2024)

Trains a model that is inherently low-rank, keeping only the low-rank factors without the original large matrix. Specifically designed for training from scratch with low-rank constraints.

Source: [PELA paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_PELA_Learning_Parameter-Efficient_Models_with_Low-Rank_Approximation_CVPR_2024_paper.pdf)

### 3.2 SVD-Based Compression

**Key approaches**:
- **Standard SVD**: Truncate to top-k singular values. W ≈ U_k * S_k * V_k^T
- **Fisher-Weighted SVD (FWSVD)**: Weight importance by Fisher information before decomposition
- **Layer-Collaborative SVD**: Consider cross-layer interactions when choosing ranks
- **Progressive decomposition**: Start full-rank, gradually reduce

**Compression results**: 9-30% parameter reduction with insignificant accuracy impact on already-compact models. Progressive low-rank can achieve 4x compression vs 1.5x for direct truncation.

**Relevance**: For training a minimal model from scratch, we can directly train with factorized weights (W = UV where U is d x r, V is r x d). SVD post-training is more relevant for compressing a trained model.

Source: [Language model compression with weighted low-rank factorization](https://arxiv.org/abs/2207.00112)

### 3.3 Structured Sparsity

#### 3.3.1 N:M Sparsity

Hardware-accelerated pattern (e.g., 2:4 on NVIDIA A100+): For every 4 consecutive weights, 2 must be zero.
- Achieves 50% sparsity with hardware support
- Recent ICLR 2024 work on dynamic sparse training for transformers

#### 3.3.2 Block Sparsity

- **Block diagonal**: Weight matrix as block-diagonal -> independent neuron groups
- **BLAST matrices** (2024): Block-Level Adaptive Structured matrices with shared bases and block-wise diagonal coupling. Achieves up to 70% computational reduction while recovering original accuracy.
- **Monarch matrices** (Dao et al., ICML 2022): Products of two block-diagonal matrices (up to permutation). 2x speedup over dense multiplication.

**Relevance**: For tiny models, structured sparsity can be baked into the architecture:
- Block diagonal attention projections (e.g., 2 independent 8x8 blocks instead of one 16x16)
- Monarch-style factorization of FFN layers

Source: [BLAST paper](https://arxiv.org/html/2410.21262v1), [Monarch paper](https://proceedings.mlr.press/v162/dao22a/dao22a.pdf), [Weight-sparse transformers have interpretable circuits](https://cdn.openai.com/pdf/41df8f28-d4ef-43e9-aed2-823f9393e470/circuit-sparsity-paper.pdf)

### 3.4 Kronecker Products for Parameter Efficiency

#### 3.4.1 AdaKron (LREC-COLING 2024)

Uses Kronecker product to combine outputs of two small networks:
- Only 0.55% of original parameters needed for fine-tuning
- Output dimension is product of individual output dimensions

#### 3.4.2 Krony-PT (December 2024)

Applies Kronecker product decomposition to compress GPT-2, representing large weight matrices as Kronecker products of smaller matrices.

**Mathematical basis**: For matrices A (m x n) and B (p x q):
```
A ⊗ B has dimensions (mp x nq) with only m*n + p*q parameters
```
Example: 16x16 matrix as Kronecker product of 4x4 and 4x4: 32 parameters instead of 256.

**Relevance**: Kronecker factorization is extremely powerful for our use case. A 16x16 weight matrix can be represented as the Kronecker product of two 4x4 matrices using only 32 parameters (8x reduction). This could be the key technique for staying under 500 parameters.

Source: [AdaKron](https://aclanthology.org/2024.lrec-main.32/), [Krony-PT](https://arxiv.org/html/2412.12351v1)

### 3.5 Matrix Factorization: NMF and Others

- **NMF (Non-negative Matrix Factorization)**: Constrains factors to be non-negative, yielding interpretable and sparse decompositions. Superior to SVD for feature interpretability.
- **Sparse Low-Rank Factorization**: Combines sparsity with low-rank constraint for better compression-performance tradeoff.
- **Combined pruning + factorization**: Recent work identifies that pruned models exhibit low-rank sparsity patterns that can be further factorized.

**Relevance**: NMF could be useful if we want interpretable weight decompositions. The non-negativity constraint might be too restrictive for attention weights, but could work well for FFN weights combined with ReLU activations.

Source: [NMF survey](https://oa.ee.tsinghua.edu.cn/~zhangyujin/Download-Paper/E224=TKDE-13.pdf), [LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.945.pdf)

### 3.6 Combining Base Matrix + Low-Rank Update

The pattern W = W_base + low_rank_correction appears in multiple works:

| Method | Base | Correction | Application |
|--------|------|-----------|-------------|
| LoRA | Frozen pretrained | BA (rank r) | Fine-tuning |
| ResidualTransformer | Shared across layers | D*L*R per layer | Training from scratch |
| Relaxed Recursive Transformer | Shared (looped) | LoRA per iteration | Training from scratch |
| MoL | Shared FFN | Mixture of LoRA experts | Training from scratch |

**For minimal addition**: The "shared base + per-layer residual" pattern is ideal. One shared attention layer (~256 params for 16-dim) + tiny per-iteration residuals (~32 params each for rank-1) allows effective depth with minimal parameter increase.

---

## 4. Layer Reuse and Sharing

### 4.1 Universal Transformers

**Paper:** "Universal Transformers" (Dehghani et al., ICLR 2019, arXiv:1807.03819)

Key design:
- **Weight tying across depth**: Same self-attention and FFN weights reused at every step
- **Parallel RNN interpretation**: Block of parallel RNNs evolving per-symbol hidden states
- **Adaptive Computation Time (ACT)**: Dynamic halting -- different tokens get different numbers of processing steps based on input difficulty

**Results**: Turing complete (unlike standard Transformers), strong on algorithmic tasks, competitive on NLP.

**Relevance**: Universal Transformers are perfect for addition. The addition algorithm applies the same operation (add digits, propagate carry) at each position. A shared layer that loops N times with ACT could learn this naturally. The carry chain requires information to flow from right to left across iterations.

Source: [ICLR 2019 paper](https://openreview.net/pdf?id=HyzdRiR9Y7), [Semantic Scholar](https://www.semanticscholar.org/paper/Universal-Transformers-Dehghani-Gouws/ac4dafdef1d2b685b7f28a11837414573d39ff4e)

### 4.2 ALBERT

**Paper:** "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (Lan et al., ICLR 2020)

Three key parameter reduction techniques:
1. **Embedding factorization**: Decompose V x H embedding into V x E and E x H (where E << H)
2. **Cross-layer parameter sharing**: All layers share the same parameters (both attention and FFN)
3. **Sentence-order prediction** (replacing NSP)

**Results**: 18x fewer parameters than BERT-large, 1.7x faster training, new GLUE/SQuAD/RACE SOTA.

**Relevance**: ALBERT's embedding factorization is critical for our tiny model. With a vocabulary of ~15 tokens (digits 0-9, +, =, EOS, PAD, etc.) and hidden dim 16, the embedding is only 15*16 = 240 parameters. Factorizing through dim 4: 15*4 + 4*16 = 124 parameters. Cross-layer sharing is exactly what we need.

Source: [ICLR 2020 paper](https://openreview.net/pdf?id=H1eA7AEtvS), [arXiv:1909.11942](https://www.arxiv.org/pdf/1909.11942v2)

### 4.3 Lessons on Parameter Sharing

**Paper:** "Lessons on Parameter Sharing across Layers in Transformers" (Takase & Kiyono, arXiv:2104.06022, 2021)

Three sharing strategies for M unique layers in an N-layer model:
1. **SEQUENCE**: Share in sequential blocks (layers 1-k share, k+1-2k share, etc.)
2. **CYCLE**: Repeat layers 1..M cyclically (1,2,..,M,1,2,..,M,...)
3. **CYCLE (REV)**: Forward then reverse cycling (1,2,..,M,..,2,1,2,..,M)

**Results**: CYCLE (REV) generally performs best. All strategies are efficient in both parameters and computation.

**Relevance**: For a model with 2 unique layers repeated 4 times, CYCLE(REV) would give pattern [1,2,2,1,1,2,2,1]. This palindromic pattern might help carry propagation flow in both directions.

Source: [arXiv:2104.06022](https://arxiv.org/abs/2104.06022), [GitHub implementation](https://github.com/jaketae/param-share-transformer)

### 4.4 Relaxed Recursive Transformers

**Paper:** "Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA" (Nouriborji et al., ICLR 2025, arXiv:2410.20672)

Architecture: A block of K unique layers is repeated T times, with **per-iteration LoRA modules** to relax strict weight tying:
```
W_iteration_t = W_shared + B_t * A_t    (LoRA for iteration t)
```

Key results:
- Recursive Gemma 1B outperforms TinyLlama 1.1B and Pythia 1B
- Recovers most performance of the full-size model
- **Continuous Depth-wise Batching**: 2-3x inference throughput gains

**Relevance**: This is arguably the most important paper for our approach. We can have a single shared attention+FFN block (~300 params) that loops T times, with per-iteration LoRA of rank 1 adding ~32 params per iteration. For T=4: 300 + 4*32 = 428 parameters, well under 500.

Source: [arXiv:2410.20672](https://arxiv.org/abs/2410.20672), [ICLR 2025](https://openreview.net/pdf?id=WwpYSOkkCt)

### 4.5 Mixture of LoRAs (MoL) for Recursive Transformers

**Paper:** "Improving Recursive Transformers with Mixture of LoRAs" (arXiv:2512.12880, December 2025)

Extends recursive transformers with token-conditional weight modulation:
- **MoL**: Low-rank experts inside shared FFN, enabling token-conditional modulation
- **ModernALBERT**: Integrates RoPE, GeGLU, FlashAttention with the recursive architecture
- **Expert merging**: At inference, MoL can be compressed to a single adapter while preserving accuracy

**Relevance**: Token-conditional modulation means different digit positions can activate different "experts" within the same shared layer. This is ideal for addition, where different positions require different processing (e.g., LSB vs MSB, carry positions vs non-carry).

Source: [arXiv:2512.12880](https://arxiv.org/abs/2512.12880)

### 4.6 Subformer

**Paper:** "Subformer: Exploring Weight Sharing for Parameter Efficiency in Generative Transformers" (Reid et al., EMNLP 2021 Findings, arXiv:2101.00234)

Two key techniques:
1. **Sandwich-style sharing**: First and last layers are unique; middle layers share parameters. This lets unique I/O layers focus on token prediction while shared middle layers learn general representations.
2. **SAFE (Self-Attentive Factorized Embeddings)**: Reduces embedding parameters using self-attention over factorized embeddings.

**Results**: 25-70% parameter reduction while outperforming standard Transformers.

**Relevance**: The sandwich pattern is smart -- having unique input/output layers with a shared computation core. For addition, the unique input layer handles digit parsing, the shared layers do computation, and the unique output layer handles digit generation.

Source: [arXiv:2101.00234](https://arxiv.org/abs/2101.00234), [GitHub](https://github.com/machelreid/subformer)

### 4.7 DenseFormer

**Paper:** "DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging" (arXiv:2402.02622, 2024)

Adds a **Depth-Weighted Average (DWA)** after each transformer block:
```
h_out = sum(alpha_i * h_i for i in range(current_layer+1))
```

DWA weights reveal strong reuse of activations from distant layers. DenseFormer is more data-efficient, matching deeper models' perplexity with fewer unique layers.

**Relevance**: Rather than strict weight sharing, we could add learnable mixing weights between layer outputs. This adds minimal parameters (one scalar per layer pair) but enables richer information flow. For 4 iterations of a shared layer: 4*5/2 = 10 additional scalar parameters.

Source: [arXiv:2402.02622](https://arxiv.org/abs/2402.02622)

### 4.8 Sparse Universal Transformer (SUT)

**Paper:** "Sparse Universal Transformer" (Tan et al., EMNLP 2023, arXiv:2310.07096)

Combines Universal Transformer with Sparse Mixture of Experts and stick-breaking dynamic halting:
- Same generalization benefits as UT
- Half the computation and parameters of strong baselines on WMT'14
- Strong on formal language tasks (logical inference, CFQ)

**Relevance**: Dynamic halting is appealing -- for "easy" additions (no carries), 1-2 iterations might suffice; for complex carry chains, more iterations are needed. The stick-breaking mechanism is more principled than ACT.

Source: [arXiv:2310.07096](https://arxiv.org/abs/2310.07096)

### 4.9 Weight Tying (Input/Output Embeddings)

**Key paper:** "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017)

Sharing input and output embedding matrices:
- **Full tying**: Same matrix for input embedding and output projection
- **Savings**: For vocab V and dim d: saves V*d parameters (one matrix instead of two)
- **Quality**: Often improves perplexity, not just parameter count
- **Universal adoption**: Used in T5, GPT-2/3/4, ALBERT, etc.

For our task with V=15, d=16: saves 240 parameters (significant at our scale).

Source: [Weight Tying explanation](https://mbrenndoerfer.com/writing/weight-tying-shared-embeddings-transformers), [MartinLwx blog](https://martinlwx.github.io/en/an-explanation-of-weight-tying/)

---

## 5. Causality and Masking

### 5.1 Causal (Autoregressive) vs. Non-causal (Bidirectional)

**Causal masking** (decoder-only): Each token can only attend to previous tokens. Standard for generation tasks.

**Bidirectional masking** (encoder-only): Each token attends to all tokens. Standard for understanding tasks.

**For addition**: The input (two numbers + operator) benefits from bidirectional attention, as each digit's significance depends on the full input context. The output (result) must be generated sequentially but benefits from knowing the full input.

### 5.2 Prefix Language Models (Semi-Causal)

**Approach**: Bidirectional attention on a prefix (the input), then causal attention for generation (the output).

For addition: "123+456=" is the prefix (bidirectional), and "579" is generated causally.

**Advantages**:
- Input digits can attend to each other bidirectionally (important for carry detection)
- Output still generated autoregressively
- No additional parameters vs. standard causal model -- just a different attention mask

**Intermittent Semi-working Mask (ISM)** (2024): Alternates bidirectional and unidirectional attention across turns. Achieves SOTA quality with low latency.

Source: [Prefix LM blog](https://haileyschoelkopf.github.io/blog/2024/prefix-lm/), [ISM paper](https://arxiv.org/html/2408.00539v1)

### 5.3 Masked Language Modeling for Arithmetic

**MEAP (Mask-Enhanced Autoregressive Prediction)** (2025): Randomly masks input tokens during next-token prediction training with decoder-only architecture. Improves information retrieval and reasoning without requiring bidirectional attention or encoder-decoder architecture.

**Diffusion LLMs vs. Autoregressive** (2025): Diffusion models exhibit "order robustness" -- unlike AR models that degrade up to 67% under answer-first prompting, diffusion models remain stable. This suggests that non-autoregressive approaches could be advantageous for arithmetic.

**Relevance**: For addition, consider training with masked tokens in the input to force the model to learn robust digit relationships. Also, non-autoregressive generation (predicting all output digits simultaneously) could eliminate carry-propagation bottlenecks.

Source: [MEAP paper](https://arxiv.org/abs/2502.07490), [Diffusion LMs](https://arxiv.org/html/2602.03769)

### 5.4 Encoder-Decoder vs. Decoder-Only for Arithmetic

**Recent comparison** (2024-2025): A systematic study revisiting encoder-decoder LLMs found:
- For tasks where input and output are structurally different, encoder-decoder has advantages
- Encoder-decoder benefits from bi-directionality even with fewer parameters
- The shift to decoder-only is driven by GPT's success, not encoder-decoder's incapability

**Ettin findings** (2025): Controlled comparison shows decoders remain dominant for generation; encoders for understanding. The optimal choice depends on the task.

**For addition**:
- **Encoder-decoder**: Natural fit -- encoder processes "123+456=" bidirectionally, decoder generates "579" autoregressively. Parameter cost: duplicated attention/FFN weights.
- **Decoder-only with prefix mask**: Simpler, single set of weights, prefix gets bidirectional attention.
- **Encoder-only (masked prediction)**: Predict each output position independently. No autoregressive dependency -- all digits predicted in parallel. Most parameter-efficient if it works.

**Recommendation**: Prefix-LM (decoder-only with bidirectional prefix) is the best balance of expressiveness and parameter efficiency for our task.

Source: [Encoder-Decoder revisited](https://arxiv.org/html/2510.26622v1), [Ettin](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder)

### 5.5 Bidirectional Attention Advantages for Arithmetic

Key findings from the literature:

1. **Bitune** (2024): Adding bidirectional attention to decoder-only models consistently improves arithmetic performance.

2. **Mechanistic interpretability** (Quirke & Barez, 2024): Addition circuits require information flow from right-to-left (for carry propagation). In a causal model, this must be accomplished through multiple sequential attention operations. In a bidirectional model, carry information can flow directly.

3. **Position Coupling** (NeurIPS 2024): Even in a causal 1-layer model, proper positional encoding (assigning same position ID to same-significance digits) enables addition of exponentially long numbers. This suggests that carry propagation can be partially encoded in position embeddings rather than requiring multiple attention layers.

4. **Abacus Embeddings** (McLeish et al., NeurIPS 2024): Position encoding relative to number start enables 99% accuracy on 100-digit addition when trained on only 20-digit numbers. The embedding encodes digit significance, solving a key limitation of standard positional encodings.

**Practical implication**: For a 1-layer minimal model, use either:
- Position coupling (assign same position to same-significance digits across input/output)
- Abacus embeddings (position relative to number boundaries)
- Or handcraft positional embeddings that encode digit significance directly

Source: [Bitune](https://arxiv.org/html/2405.14862), [Position Coupling](https://arxiv.org/abs/2405.20671), [Abacus Embeddings](https://arxiv.org/abs/2405.17399)

---

## 6. Synthesis: Implications for Minimal Addition Transformers

### 6.1 What the Literature Tells Us About Minimal Addition

From mechanistic interpretability studies (Quirke & Barez 2024, Barez et al. 2024), we know:

1. **The algorithm is consistent**: All trained addition models converge on the same core algorithm regardless of model size, random seeds, or optimizers.
2. **Parallel digit streams**: The model processes each digit position in parallel.
3. **Three key operations**: (a) digit pair lookup, (b) carry detection, (c) carry propagation.
4. **1 layer, 3 heads is sufficient**: For >99.999% accuracy on up to 15-digit addition.
5. **The hard part is carry chains**: Long carry chains (e.g., 999+1=1000) are the main failure mode.

### 6.2 Existing Records

| Entry | Parameters | Method | Accuracy |
|-------|-----------|--------|----------|
| Papailiopoulos challenge (trained) | 491 | Low-rank transformer, trained | >=99.97% |
| N8python (hand-coded) | 343 | Hand-crafted Qwen3 architecture | 100% |
| xangma (hand-coded) | 197 | Hand-crafted GPT weights | 100% |
| Cosmin Negruseri (hand-coded) | 130 | Factorized embeddings, rank-1 layers, parabolic decode, 4 ReLU carry neurons | 100% |

The 130-parameter hand-coded solution uses:
- Rank-1 factorized embeddings
- Rank-1 layer weights
- Tuned sinusoidal PE (period 11)
- Parabolic decode head
- 4 ReLU carry neurons
- LSB-first output format

### 6.3 Recommended Architecture for Trained Model Under 500 Parameters

Based on this literature review, the optimal architecture combines:

1. **Vocabulary**: 15 tokens (0-9, +, =, EOS, BOS, PAD) with **weight-tied embeddings**
2. **Positional encoding**: **Position coupling** or **Abacus-style** (same position ID for same-significance digits)
3. **Architecture**: **Recursive transformer** with 1 shared block looped T times
4. **Per-iteration differentiation**: **Rank-1 LoRA** per iteration (Relaxed Recursive Transformer pattern)
5. **Attention**: Single head (minimal), with **MLA-style low-rank KV projection** (rank 2-4)
6. **FFN**: **GeGLU** with small hidden dim, or Kronecker-factorized linear layers
7. **Normalization**: **RMSNorm** (no bias, no mean computation)
8. **No biases anywhere**: Every bias parameter eliminated
9. **Masking**: **Prefix-LM** (bidirectional on input, causal on output)
10. **Weight factorization**: All weight matrices as **low-rank or Kronecker products**

### 6.4 Parameter Budget Estimate

For d_model=8, vocab=15, 1 shared block looped 4 times with rank-1 LoRA:

| Component | Parameters |
|-----------|-----------|
| Token embedding (tied, factorized 15x2 + 2x8) | 46 |
| Positional embedding (23 positions x 8 or learned coupling) | ~40-184 |
| Attention Q,K,V (low-rank, rank 2: 3*(8*2+2*8)) | 96 |
| Attention output (rank 2: 8*2+2*8) | 32 |
| FFN (GeGLU, 8->4->8: 8*4*3+4*8) | 128 |
| RMSNorm (2 per block: 2*8) | 16 |
| Per-iteration LoRA (4 iterations, rank 1, ~16 each) | 64 |
| Output head (tied with embedding) | 0 |
| **Total (rough estimate)** | **~422-566** |

This can be further reduced by:
- Using Kronecker factorization on FFN (128 -> ~32)
- Reducing d_model to 6 or even 4
- Using fixed (non-learned) positional embeddings
- Sharing Q/K projections

### 6.5 Key Trade-offs

| Decision | Option A | Option B | Recommendation |
|----------|----------|----------|---------------|
| Attention type | Full MHA | Low-rank MLA | MLA (fewer params) |
| Layer strategy | Multiple unique | Shared + LoRA residuals | Shared + residuals |
| Masking | Causal only | Prefix-LM | Prefix-LM |
| Position encoding | Learned | Position coupling/Abacus | Abacus or coupling |
| FFN type | Standard MLP | GeGLU | GeGLU (better expressiveness/param) |
| Normalization | LayerNorm | RMSNorm | RMSNorm (no mean computation) |
| Weight matrices | Dense | Kronecker factorized | Kronecker for large matrices |
| Output format | MSB-first | LSB-first | LSB-first (natural for carry propagation) |

### 6.6 Open Questions for Experimentation

1. **How many iterations of the shared block are needed?** Literature suggests 2-4 for addition.
2. **Can we use non-autoregressive output?** Predicting all digits simultaneously eliminates sequential dependency but may struggle with carries.
3. **What is the minimum d_model?** The 130-param hand-coded solution suggests rank-1 representations suffice; d_model=4 might work.
4. **Can we combine position coupling with Abacus embeddings?** Both encode digit significance but in different ways.
5. **Is GeGLU worth the extra parameters over ReLU?** For tiny models, the gating mechanism might not help.
6. **Can we train a model that uses the same algorithmic structure as the 130-param hand-coded solution?** If we can learn the same structure (factorized embeddings, 4 carry neurons, parabolic decode), we might achieve similar parameter counts with a trained model.

---

## 7. References

### 7.1 Modern Architectures

1. DeepSeek-V2. "A Strong, Economical, and Efficient Mixture-of-Experts Language Model." [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) (2024)
2. DeepSeek-V3. "Technical Report." [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) (2024)
3. DeepSeek-R1. "Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." [arXiv:2501.12948](https://arxiv.org/pdf/2501.12948) (2025)
4. NVIDIA Nemotron 3. "Efficient and Open Intelligence." [arXiv:2512.20856](https://arxiv.org/pdf/2512.20856) (2025)
5. ModernBERT. "Smarter, Better, Faster, Longer." [arXiv:2412.13663](https://arxiv.org/abs/2412.13663) (2024)
6. Ettin. "Encoder vs Decoder Models." [GitHub: JHU-CLSP/ettin-encoder-vs-decoder](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder) (2025)
7. NanoGPT. [GitHub: karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (2022)
8. microGPT. [Blog post](http://karpathy.github.io/2026/02/12/microgpt/) (2026)
9. ByteDance MegaScale. [arXiv:2402.15627](https://arxiv.org/html/2402.15627v1) (2024)

### 7.2 Attention Mechanisms

10. Vaswani et al. "Attention Is All You Need." NeurIPS 2017.
11. Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models." [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) EMNLP 2023.
12. Shazeer. "Fast Transformer Decoding: One Write-Head is All You Need." (2019)
13. Choromanski et al. "Rethinking Attention with Performers." [arXiv:2009.14794](https://arxiv.org/abs/2009.14794) (2020)
14. Wang et al. "Linformer: Self-Attention with Linear Complexity." [arXiv:2006.04768](https://arxiv.org/abs/2006.04768) (2020)
15. Zaheer et al. "Big Bird: Transformers for Longer Sequences." [arXiv:2007.14062](https://arxiv.org/abs/2007.14062) NeurIPS 2020.
16. Yang et al. "Gated Linear Attention Transformers with Hardware-Efficient Training." [arXiv:2312.06635](https://arxiv.org/abs/2312.06635) ICML 2024.
17. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention." [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) (2022)
18. Fuchs et al. "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks." [arXiv:2006.10503](https://arxiv.org/abs/2006.10503) NeurIPS 2020.
19. Sebastian Raschka. "Multi-Head Latent Attention (MLA)." [Blog](https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/)

### 7.3 Arithmetic in Transformers

20. Quirke & Barez. "Understanding Addition in Transformers." [arXiv:2310.13121](https://arxiv.org/abs/2310.13121) ICLR 2024.
21. Barez et al. "Understanding Addition and Subtraction in Transformers." [arXiv:2402.02619](https://arxiv.org/abs/2402.02619) (2024)
22. McLeish et al. "Transformers Can Do Arithmetic with the Right Embeddings." [arXiv:2405.17399](https://arxiv.org/abs/2405.17399) NeurIPS 2024.
23. Cho et al. "Position Coupling: Improving Length Generalization of Arithmetic Transformers." [arXiv:2405.20671](https://arxiv.org/abs/2405.20671) NeurIPS 2024.
24. Lee et al. "Teaching Arithmetic to Small Transformers." [arXiv:2307.03381](https://arxiv.org/abs/2307.03381) (2023)
25. Patriota. "Arbitrary-Length Generalization for Addition in a Tiny Transformer." [arXiv:2406.00075](https://arxiv.org/abs/2406.00075) (2024)
26. Cho et al. "Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count." [arXiv:2410.15787](https://arxiv.org/abs/2410.15787) (2024)

### 7.4 Parameter Efficiency and Weight Sharing

27. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." [arXiv:2106.09685](https://arxiv.org/pdf/2106.09685) ICLR 2022.
28. Wang et al. "ResidualTransformer: Residual Low-Rank Learning with Weight-Sharing." [arXiv:2310.02489](https://arxiv.org/abs/2310.02489) ICASSP 2024.
29. Nouriborji et al. "Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA." [arXiv:2410.20672](https://arxiv.org/abs/2410.20672) ICLR 2025.
30. Nouriborji et al. "Improving Recursive Transformers with Mixture of LoRAs." [arXiv:2512.12880](https://arxiv.org/abs/2512.12880) (2025)
31. Lan et al. "ALBERT: A Lite BERT for Self-supervised Learning." ICLR 2020.
32. Dehghani et al. "Universal Transformers." [arXiv:1807.03819](https://openreview.net/pdf?id=HyzdRiR9Y7) ICLR 2019.
33. Reid et al. "Subformer: Exploring Weight Sharing for Parameter Efficiency." [arXiv:2101.00234](https://arxiv.org/abs/2101.00234) EMNLP 2021 Findings.
34. Takase & Kiyono. "Lessons on Parameter Sharing across Layers in Transformers." [arXiv:2104.06022](https://arxiv.org/abs/2104.06022) (2021)
35. "DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging." [arXiv:2402.02622](https://arxiv.org/abs/2402.02622) (2024)
36. Tan et al. "Sparse Universal Transformer." [arXiv:2310.07096](https://arxiv.org/abs/2310.07096) EMNLP 2023.
37. Press & Wolf. "Using the Output Embedding to Improve Language Models." (2017)

### 7.5 Sparsity and Matrix Decomposition

38. Guo et al. "PELA: Learning Parameter-Efficient Models with Low-Rank Approximation." CVPR 2024.
39. "Kronecker Factorization Improves Efficiency." [arXiv:2505.22255](https://arxiv.org/pdf/2505.22255) (2025)
40. "AdaKron: An Adapter-based Parameter Efficient Model Tuning with Kronecker Product." [LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.32/)
41. "Krony-PT: GPT2 compressed with Kronecker Products." [arXiv:2412.12351](https://arxiv.org/html/2412.12351v1) (2024)
42. Dao et al. "Monarch: Expressive Structured Matrices for Efficient and Accurate Training." [ICML 2022](https://proceedings.mlr.press/v162/dao22a/dao22a.pdf)
43. "BLAST: Block-Level Adaptive Structured Matrices." [arXiv:2410.21262](https://arxiv.org/html/2410.21262v1) (2024)
44. "SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot." [arXiv:2301.00774](https://arxiv.org/pdf/2301.00774) (2023)

### 7.6 Causality and Masking

45. "Intermittent Semi-working Mask: A New Masking Paradigm for LLMs." [arXiv:2408.00539](https://arxiv.org/html/2408.00539v1) (2024)
46. "Mask-Enhanced Autoregressive Prediction: Pay Less Attention to Learn More." [arXiv:2502.07490](https://arxiv.org/abs/2502.07490) (2025)
47. "Bitune: Leveraging Bidirectional Attention to Improve Decoder-Only LLMs." [arXiv:2405.14862](https://arxiv.org/html/2405.14862) (2024)
48. "Segment-Based Attention Masking for GPTs." [arXiv:2412.18487](https://arxiv.org/abs/2412.18487) (2024)

### 7.7 Additional Architecture References

49. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) (2023)
50. "Efficient Attention Mechanisms for Large Language Models: A Survey." [arXiv:2507.19595](https://arxiv.org/abs/2507.19595) (2025)
51. "Weight-sparse transformers have interpretable circuits." [OpenAI paper](https://cdn.openai.com/pdf/41df8f28-d4ef-43e9-aed2-823f9393e470/circuit-sparsity-paper.pdf) (2024)
52. "A Survey of Linear Attention: Algorithm, Theory, Application, and Infrastructure." [GitHub](https://github.com/btzyd/Awesome-Linear-Attention-Survey)
