# Positional Embeddings & Input Encoding Strategies for Arithmetic Transformers

**Research Report -- February 2025**

---

## Table of Contents

1. [Overview and Motivation](#1-overview-and-motivation)
2. [Positional Embedding Types](#2-positional-embedding-types)
   - 2.1 [Sinusoidal (Original Transformer)](#21-sinusoidal-original-transformer)
   - 2.2 [Learned Positional Embeddings](#22-learned-positional-embeddings)
   - 2.3 [RoPE (Rotary Position Embedding)](#23-rope-rotary-position-embedding)
   - 2.4 [ALiBi (Attention with Linear Biases)](#24-alibi-attention-with-linear-biases)
   - 2.5 [Relative Position Encodings (Shaw et al., T5)](#25-relative-position-encodings-shaw-et-al-t5)
   - 2.6 [FIRE (Functional Interpolation for Relative Positions)](#26-fire-functional-interpolation-for-relative-positions)
   - 2.7 [KERPLE (Kernelized Relative Positional Embedding)](#27-kerple-kernelized-relative-positional-embedding)
   - 2.8 [Fourier Features for Position Encoding](#28-fourier-features-for-position-encoding)
   - 2.9 [GRPE (Graph Relative Positional Encoding)](#29-grpe-graph-relative-positional-encoding)
   - 2.10 [Stochastic Positional Encoding (SPE)](#210-stochastic-positional-encoding-spe)
   - 2.11 [NoPE (No Position Encoding)](#211-nope-no-position-encoding)
   - 2.12 [Abacus Positional Embeddings](#212-abacus-positional-embeddings)
   - 2.13 [Position Coupling](#213-position-coupling)
3. [Custom PE for Addition](#3-custom-pe-for-addition)
   - 3.1 [Why Abacus Embeddings Work for Arithmetic](#31-why-abacus-embeddings-work-for-arithmetic)
   - 3.2 [Period-11 Sinusoidal PE (Cosmin's 130-param Solution)](#32-period-11-sinusoidal-pe-cosmins-130-param-solution)
   - 3.3 [Position Coupling for Column-Wise Alignment](#33-position-coupling-for-column-wise-alignment)
   - 3.4 [Index Hints / Positional Descriptions](#34-index-hints--positional-descriptions)
   - 3.5 [Design Principles for Arithmetic PE](#35-design-principles-for-arithmetic-pe)
4. [Input Encoding Strategies for Addition](#4-input-encoding-strategies-for-addition)
   - 4.1 [Full Number vs Digit-by-Digit Representation](#41-full-number-vs-digit-by-digit-representation)
   - 4.2 [LSB-First vs MSB-First](#42-lsb-first-vs-msb-first)
   - 4.3 [Special Tokens: +, =, Padding](#43-special-tokens---padding)
   - 4.4 [Reversed Output](#44-reversed-output)
   - 4.5 [Scratchpad / Chain-of-Thought](#45-scratchpad--chain-of-thought)
   - 4.6 [Binary vs Decimal Representation](#46-binary-vs-decimal-representation)
   - 4.7 [Interleaved Formats](#47-interleaved-formats)
   - 4.8 [Factorized (Rank-1) Embeddings](#48-factorized-rank-1-embeddings)
5. [Parameter Efficiency in Embeddings](#5-parameter-efficiency-in-embeddings)
   - 5.1 [Factorized Embedding Matrices (ALBERT)](#51-factorized-embedding-matrices-albert)
   - 5.2 [Shared Input/Output Embeddings (Weight Tying)](#52-shared-inputoutput-embeddings-weight-tying)
   - 5.3 [Hash-Based Embeddings](#53-hash-based-embeddings)
   - 5.4 [Minimal Vocabulary Considerations](#54-minimal-vocabulary-considerations)
   - 5.5 [Parameter Costs of Different PE Approaches](#55-parameter-costs-of-different-pe-approaches)
6. [Comparative Summary Table](#6-comparative-summary-table)
7. [Recommendations for Sub-500 Parameter Models](#7-recommendations-for-sub-500-parameter-models)
8. [References](#8-references)

---

## 1. Overview and Motivation

In a sub-500 parameter transformer for 10-digit addition, positional embeddings are arguably the most critical design choice. The vocabulary is tiny (digits 0-9 plus a few special tokens like `+` and `=`), so token embeddings consume very few parameters. However, position information is essential for arithmetic because:

- **Addition is column-aligned**: Digits of the same significance (ones, tens, hundreds, etc.) must be identified and matched across the two operands.
- **Carry propagation is sequential**: The carry from column `i` depends on the sum at column `i`, creating a right-to-left dependency chain.
- **Length generalization requires structural understanding**: The model must learn the *algorithm* of addition, not just memorize input-output pairs.

The choice of positional encoding directly determines whether the model can learn these structural properties, and at what parameter cost.

---

## 2. Positional Embedding Types

### 2.1 Sinusoidal (Original Transformer)

**Paper**: Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
**Link**: https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf

**Mechanism**: Fixed, non-learned sinusoidal functions encode absolute position:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each dimension corresponds to a sinusoid with wavelength forming a geometric progression from `2*pi` to `10000 * 2*pi`. This allows the model to attend to relative positions because `PE(pos+k)` can be expressed as a linear function of `PE(pos)`.

**Parameter cost**: **Zero** trainable parameters (fixed at initialization).

**Strengths**:
- No learnable parameters -- ideal for parameter-constrained models.
- Theoretically supports extrapolation to longer sequences than training.
- The linear-relationship property enables relative-position reasoning.

**Weaknesses**:
- Fixed, so cannot adapt to task-specific structure.
- The standard frequency schedule (geometric progression from 2*pi to 10000*2*pi) is designed for NLP, not arithmetic.
- Does not inherently encode digit significance.

**Relevance to our task**: The 130-param solution uses a *modified* sinusoidal PE with period 11 (not the standard Vaswani schedule). This shows that while the sinusoidal *form* is useful, the *frequency selection* is critical for arithmetic.

---

### 2.2 Learned Positional Embeddings

**Paper**: Devlin et al., "BERT" (2018); Radford et al., "GPT-2" (2019)

**Mechanism**: A learnable embedding matrix `E_pos` of shape `(max_seq_len, d_model)` maps each absolute position index to a dense vector. These are added to token embeddings.

**Parameter cost**: `max_seq_len * d_model`. For our setup (seq_len ~35, d_model ~4): **140 parameters** -- already nearly the entire budget.

**Strengths**:
- Fully flexible; can learn arbitrary position-dependent patterns.
- Used in most modern LLMs (GPT-2, BERT, etc.).

**Weaknesses**:
- Cannot extrapolate beyond trained sequence length.
- Parameter-hungry for parameter-constrained models.
- May overfit with limited training data.

**Relevance to our task**: Too expensive for sub-500 param models unless heavily factorized (rank-1). Abacus embeddings are a specialized form of learned PE that reuses embeddings across digit positions of the same significance.

---

### 2.3 RoPE (Rotary Position Embedding)

**Paper**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
**Link**: https://arxiv.org/abs/2104.09864

**Mechanism**: RoPE encodes position by *rotating* query and key vectors in 2D subspaces. For each pair of dimensions `(2i, 2i+1)`, vectors are rotated by angle `theta_i * pos`, where `theta_i = 1/10000^(2i/d)`. The key insight: the dot product between rotated Q and K vectors depends only on the *relative* position difference, not absolute positions.

```
f(q, pos_m) . f(k, pos_n) = g(q, k, m-n)
```

**Parameter cost**: **Zero** trainable parameters (rotation angles are fixed).

**Strengths**:
- Unifies absolute and relative position encoding.
- Zero parameters -- rotation matrices are computed on the fly.
- Widely adopted in modern LLMs (LLaMA, PaLM, etc.).
- Decaying inter-token dependency with increasing distance.

**Weaknesses**:
- Requires even embedding dimension (dimensions are paired).
- Length generalization is limited without extensions (e.g., NTK-aware scaling, YaRN).
- Not specifically designed for arithmetic structure.

**Relevance to our task**: RoPE is parameter-free, which is attractive. However, it encodes *sequence* position, not *digit significance* position. For very small d_model (e.g., 4), only 2 rotation pairs are available, limiting expressiveness.

---

### 2.4 ALiBi (Attention with Linear Biases)

**Paper**: Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (ICLR 2022)
**Link**: https://arxiv.org/abs/2108.12409

**Mechanism**: Instead of adding positional embeddings to token embeddings, ALiBi adds a static linear bias directly to attention scores:

```
softmax(q_i * K^T + m * [-(i-1), ..., -2, -1, 0])
```

where `m` is a head-specific slope (not learned -- set before training). Slopes are geometric: `m_h = 1/2^(8h/H)` for head `h` of `H` total heads.

**Parameter cost**: **Zero** trainable parameters.

**Strengths**:
- Zero parameters.
- Strong length extrapolation: models trained on 1024 tokens extrapolate to 2048.
- Introduces recency bias (recent tokens get higher attention).
- Trivially simple to implement.

**Weaknesses**:
- Linear bias is a strong inductive bias that may not suit all tasks.
- Recency bias may hurt tasks where distant positions are equally important.
- Not digit-significance-aware.

**Relevance to our task**: The recency bias is problematic for addition where the carry from the rightmost digit affects the leftmost. However, ALiBi's zero-parameter cost and length extrapolation are appealing. Could be combined with digit-significance information.

---

### 2.5 Relative Position Encodings (Shaw et al., T5)

**Shaw et al. (2018)**:
**Paper**: "Self-Attention with Relative Position Representations" (NAACL 2018)
**Link**: https://arxiv.org/abs/1803.02155

**Mechanism**: Modifies attention computation to incorporate learnable relative position embeddings. Instead of using absolute positions, the offset `(i - j)` between query position `i` and key position `j` indexes into a learned embedding table.

**Parameter cost**: `(2*clip_distance + 1) * d_model` or `(2*clip_distance + 1) * d_k` (for clipped relative distances).

**T5 Relative Position Bias (Raffel et al., 2020)**:
**Paper**: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (JMLR 2020)

**Mechanism**: Simplified version where each relative position maps to a scalar bias added to attention logits. Uses logarithmic bucketing: close distances get unique bins, far distances share bins.

```
num_buckets = 32 (typically)
Close distances: unique embedding each
Far distances: log-spaced buckets
```

**Parameter cost**: `num_buckets * n_heads` (relatively small).

**Strengths**:
- Explicitly models relative distance.
- T5 bucketing handles unseen distances gracefully.
- More expressive than ALiBi's fixed linear bias.

**Weaknesses**:
- Shaw et al. adds moderate parameters.
- T5 bucketing still doesn't encode digit significance.

**Relevance to our task**: Relative position is useful for addition (e.g., "the digit 3 positions to the left" is meaningful), but the standard relative PE does not distinguish between "3 positions left in operand A" vs "3 positions left in operand B."

---

### 2.6 FIRE (Functional Interpolation for Relative Positions)

**Paper**: Li et al., "Functional Interpolation for Relative Positions Improves Long Context Transformers" (2023)
**Link**: https://arxiv.org/abs/2310.04418

**Mechanism**: Uses a learned function (small MLP) to map relative distances to bias values, with progressive interpolation that normalizes distances by query position. This allows generalization to positions not seen during training.

FIRE can theoretically represent T5's RPE, ALiBi, and KERPLE as special cases.

**Parameter cost**: Small MLP parameters (few hundred to few thousand, depending on architecture).

**Strengths**:
- Unifies multiple RPE approaches.
- Strong length generalization.
- Progressive interpolation prevents OOD position values.

**Weaknesses**:
- Requires a small MLP, which adds parameters.
- More complex to implement than ALiBi or RoPE.

**Relevance to our task**: FIRE was used as a comparison in the Abacus embeddings paper. Abacus + FIRE showed synergy, outperforming either alone. However, the MLP overhead may be too much for sub-500 param models.

---

### 2.7 KERPLE (Kernelized Relative Positional Embedding)

**Paper**: Chi et al., "KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation" (NeurIPS 2022)
**Link**: https://papers.neurips.cc/paper_files/paper/2022/file/37a413841a614b5414b333585e7613b8-Paper-Conference.pdf

**Mechanism**: Generalizes relative position encoding using conditionally positive definite (CPD) kernels. The key variant uses a logarithmic kernel:

```
bias(i,j) = -r1 * log(1 + r2 * |i-j|)
```

where `r1, r2` are learned per head.

**Parameter cost**: `2 * n_heads` (very small -- just two scalars per head).

**Strengths**:
- Principled framework generalizing many RPE methods.
- Logarithmic variant achieves excellent length extrapolation.
- Tiny parameter cost.

**Weaknesses**:
- Still based on absolute-position distance, not digit significance.

**Relevance to our task**: The ultra-low parameter cost (e.g., 4 parameters for 2 heads) is very attractive. The logarithmic decay roughly encodes "nearby tokens matter more," which partially captures carry locality.

---

### 2.8 Fourier Features for Position Encoding

**Paper**: Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" (NeurIPS 2020)
**Link**: https://arxiv.org/abs/2006.10739

**Mechanism**: Maps low-dimensional inputs (like position) through a set of sinusoidal functions at various frequencies:

```
gamma(p) = [sin(2*pi*b_1*p), cos(2*pi*b_1*p), ..., sin(2*pi*b_k*p), cos(2*pi*b_k*p)]
```

Frequencies `b_i` can be fixed, sampled from a Gaussian, or learned. This lifts the input into a higher-dimensional space where the neural tangent kernel becomes stationary with tunable bandwidth.

**Parameter cost**: Zero (fixed) or `k` (learned frequencies).

**Strengths**:
- Overcomes spectral bias of MLPs (enables learning high-frequency functions).
- Frequencies can be tuned to match task periodicity.
- Direct connection to the Vaswani sinusoidal PE (which is a special case).

**Weaknesses**:
- Choosing the right frequency distribution requires domain knowledge.

**Relevance to our task**: Extremely relevant. The 130-param solution uses exactly this idea: a period-11 sinusoidal encoding that matches the task structure. Fourier features provide the theoretical justification for why hand-tuned frequencies work.

---

### 2.9 GRPE (Graph Relative Positional Encoding)

**Paper**: Park et al., "GRPE: Relative Positional Encoding for Graph Transformer" (ICLR 2022 Workshop)
**Link**: https://arxiv.org/abs/2201.12787

**Mechanism**: Designed for graph transformers, not sequence transformers. Encodes (a) topological relationships (shortest path distances) and (b) edge features between graph nodes.

**Relevance to our task**: Minimal direct relevance. GRPE addresses graph structure, not sequential digit processing. However, the concept of encoding task-specific relational structure (rather than sequential position) aligns with the philosophy of Abacus embeddings and position coupling.

---

### 2.10 Stochastic Positional Encoding (SPE)

**Paper**: Liutkus et al., "Relative Positional Encoding for Transformers with Linear Complexity" (ICML 2021)
**Link**: https://arxiv.org/abs/2105.08399

**Mechanism**: Approximates relative position interactions as cross-covariance structures of correlated Gaussian processes. This enables relative position encoding in linear-complexity attention mechanisms (e.g., Performer) that do not compute the full attention matrix.

**Parameter cost**: Depends on kernel parameterization.

**Strengths**:
- Enables RPE in linear attention.
- Good extrapolation behavior.

**Weaknesses**:
- Complex theoretical framework.
- Designed for linear attention variants, not standard softmax attention.

**Relevance to our task**: Limited. Our model uses standard softmax attention and the complexity savings of linear attention are irrelevant at our tiny scale.

---

### 2.11 NoPE (No Position Encoding)

**Paper**: Kazemnejad et al., "The Impact of Positional Encoding on Length Generalization in Transformers" (NeurIPS 2023)
**Link**: https://arxiv.org/abs/2305.19466

Also: Haviv et al., "Transformer Language Models without Positional Encodings Still Learn Positional Information" (2022)
**Link**: https://arxiv.org/abs/2203.16634

**Mechanism**: Simply omit all positional encodings. Causal attention masks implicitly provide some position information because each token can attend to a different number of predecessors.

**Key findings**:
- Kazemnejad et al. showed NoPE **outperforms** ALiBi, RoPE, and APE on length generalization benchmarks for reasoning tasks.
- Haviv et al. proved that NoPE transformers acquire implicit absolute position information through the causal mask.
- Theoretically, NoPE can represent both absolute PE (from layer 1) and relative PE (from layer 2).

**Parameter cost**: **Zero** parameters.

**Strengths**:
- Absolute minimum parameter cost (zero).
- Surprisingly strong length generalization.
- No hyperparameters to tune.

**Weaknesses**:
- Implicit position information may be too weak for fine-grained digit alignment.
- Requires multiple layers to build position representations.
- Not suitable for tasks requiring precise position-digit correspondence.

**Relevance to our task**: Intriguing for parameter efficiency but likely insufficient for arithmetic. Addition requires precise digit alignment, and NoPE's implicit position signal may be too noisy. However, the finding that explicit PE can *hurt* length generalization is important -- it suggests our PE should encode *task structure*, not just absolute position.

---

### 2.12 Abacus Positional Embeddings

**Paper**: McLeish et al., "Transformers Can Do Arithmetic with the Right Embeddings" (NeurIPS 2024)
**Link**: https://arxiv.org/abs/2405.17399
**Code**: https://github.com/mcleish7/arithmetic

**THIS IS THE MOST RELEVANT APPROACH FOR OUR TASK.**

**Mechanism**: Assigns positional embeddings based on **digit significance within each number**, not absolute sequence position. All digits of the same significance across different numbers in the input receive the same positional index.

For input "98282+3859172=..." (reversed format):
- Each number's digits are indexed 1, 2, 3, ... from the start of that number
- Digit in ones place -> index 1, tens -> index 2, hundreds -> index 3, etc.
- Non-digit tokens ('+', '=') receive index 0

**Training-time randomization**: During training, a random offset `beta ~ Uniform[1, k]` (k=100 by default) is added to all digit indices. This exposes the model to embedding positions beyond training lengths.

**Test-time**: `beta = 1` (fixed).

**Implementation** (from the GitHub repository):

```python
class Abacus(torch.nn.Module):
    def __init__(self, digit_tokens, embedding_dim, max_seq_length=1024, max_k=99):
        super().__init__()
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer("digits", torch.tensor(digit_tokens))
        self.max_k = max_k

    def helper(self, mask, device):
        """Converts a binary mask of digit locations into consecutive spans"""
        # Identifies consecutive digit sequences
        # Assigns each position within a sequence an index starting from 1
        shifted_mask = torch.cat([torch.zeros(...), mask[:, :-1]], dim=1)
        starts = (shifted_mask != mask) & mask
        segment_ids = torch.cumsum(starts, dim=1)
        # ... (reset index computation)
        positions = index - reset_index.gather(1, segment_ids) + 1
        return positions * mask

    def forward(self, input_ids):
        mask = torch.isin(input_ids, self.digits)
        output = self.helper(mask, input_ids.device)
        k = random.randint(0, self.max_k) if self.training else 0
        output[output > 0] += k
        return self.embedding(output)
```

**Parameter cost**: `max_seq_length * d_model` for the embedding table. But since the effective number of positions used is small (just the max number of digits per number + k), this can be reduced.

**Results**:
- Trained on max 20-digit numbers, achieves 99% accuracy on **100-digit** addition (5x generalization).
- With input injection + recurrent layers: 99.1% on 100-digit (87% error reduction over base).
- Achieves 6x generalization factor (120-digit from 20-digit training).
- Also improves multiplication and sorting tasks.

**Combined with other techniques**:
- Abacus + FIRE: synergistic, outperforms either alone.
- Abacus + Input Injection: ~50% error reduction.
- Abacus + Recurrent Layers (8x2 looped transformer): further ~50% error reduction.

---

### 2.13 Position Coupling

**Paper**: Cho et al., "Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure" (NeurIPS 2024)
**Link**: https://arxiv.org/abs/2405.20671

**Mechanism**: Assigns the **same position ID** to digits of the same significance across operands and the result. Unlike Abacus (which assigns positions within each number independently), position coupling explicitly ties positions across the entire input-output sequence.

For "653+049=2070":
- Ones digits: 3, 9, 0 all get the same position ID
- Tens digits: 5, 4, 7 all get the same position ID
- Hundreds digits: 6, 0, 0 all get the same position ID
- Thousands digit: 2 gets its own ID
- Special tokens (+, =, BOS, EOS) get position 0

Uses standard learned positional embeddings with these coupled IDs.

**Training-time**: Random starting position ID (like Abacus).

**Results**:
- Trained on 1-30 digit addition: generalizes to **200-digit** (6.67x).
- Theoretical proof that a 1-layer 2-head transformer with coupled positions can solve addition for exponentially many digits.
- Proof that a 1-layer transformer *without* positional encoding **cannot** solve addition.

**Parameter cost**: Standard learned PE table, but since many positions share the same ID, the effective table size is small.

**Comparison with Abacus**:
- Position coupling ties positions across the full input, Abacus ties within each number separately.
- Position coupling achieves slightly better generalization (6.67x vs 6x).
- Position coupling does not expand sequence length (unlike index hinting which doubles it).

---

## 3. Custom PE for Addition

### 3.1 Why Abacus Embeddings Work for Arithmetic

The fundamental challenge in arithmetic: **transformers lose track of exact digit positions** within long sequences. Standard positional encodings tell the model "this token is at position 17 in the sequence," but what the model needs to know is "this is the 5th digit of the second operand" (i.e., the ten-thousands place).

Abacus embeddings work because they:

1. **Provide an explicit significance signal**: All ones-place digits get embedding index 1, all tens-place digits get index 2, etc. This is analogous to how humans align columns on an abacus or paper.

2. **Enable cross-operand alignment**: When the model sees that digit `A[i]` and digit `B[j]` have the same Abacus index, it knows they should be added together.

3. **Decouple position from sequence order**: The model does not need to learn "position 5 in the sequence = ones place of the second operand." The Abacus embedding directly encodes "this is a ones-place digit."

4. **Training-time randomization enables extrapolation**: By randomly shifting indices during training (beta ~ U[1,100]), the model sees embedding vectors for positions 1-120 even when training on 20-digit numbers. This is why 6x length generalization is possible.

---

### 3.2 Period-11 Sinusoidal PE (Cosmin's 130-param Solution)

**Source**: https://gist.github.com/cosminscn/89c110dbae76ea0c873d67607e466f5b

This hand-crafted 130-parameter transformer solves 10-digit addition using a brilliant positional encoding strategy.

**The input format**: `AAAAAAAAAA+BBBBBBBBBB=` (10 digits + plus + 10 digits + equals = 22 input tokens), followed by 11 output digits generated autoregressively (with reversed output), for a total block size of 35.

**The PE implementation**:

```python
def generate_pe(self, seq_len, device):
    pe = torch.zeros(seq_len, self.config.n_embd, device=device)  # n_embd = 4
    positions = torch.arange(seq_len, dtype=torch.float32)
    th = 2 * math.pi / 11  # Period = 11
    amp = torch.where(positions <= 21, 100.0, 1.0)
    pe[:, 1] = amp * torch.sin(positions * th)
    pe[:, 2] = amp * torch.cos(positions * th)
    return pe
```

**Why period 11?**

The period of 11 is chosen because there are **11 distinct digit positions** in a 10-digit number (indices 0-9 for the 10 digits, plus one extra for carry). The sinusoidal PE with period 11 creates a unique (sin, cos) pair for each digit significance level.

Key observations:
- **Dimension 0**: Reserved for digit *value* (the factorized embedding puts digit values here).
- **Dimensions 1-2**: Encode position via sin/cos with period 11.
- **Dimension 3**: Available for other purposes (or unused).
- **Amplitude of 100 for input tokens (positions 0-21)**: Strongly encodes position in the input.
- **Amplitude of 1 for output tokens (positions 22+)**: Weakly encodes position during generation, letting the model rely more on causal context.

**Why this works for addition**:

The period-11 PE ensures that digits of the same significance in operand A, operand B, and the output are related by a fixed phase shift. Specifically:
- Position 0 (A's first digit) and position 11 (B's first digit) differ by exactly one full period (11), so they get the **same** sin/cos values.
- This is identical in effect to Abacus-style alignment, but achieved with zero learned parameters using hand-picked frequencies.

The matching is:
```
Operand A:  pos 0,  1,  2,  3,  4,  5,  6,  7,  8,  9   (10 digits)
Operator +:     pos 10
Operand B:  pos 11, 12, 13, 14, 15, 16, 17, 18, 19, 20  (10 digits)
Operator =:     pos 21

Position 0  mod 11 = 0   ->   sin(0),     cos(0)
Position 11 mod 11 = 0   ->   sin(0),     cos(0)   [SAME as pos 0!]
Position 1  mod 11 = 1   ->   sin(th),    cos(th)
Position 12 mod 11 = 1   ->   sin(th),    cos(th)  [SAME as pos 1!]
...
Position 9  mod 11 = 9   ->   sin(9*th),  cos(9*th)
Position 20 mod 11 = 9   ->   sin(9*th),  cos(9*th) [SAME as pos 9!]
```

**The `+` sign at position 10**: `10 mod 11 = 10`, giving it a unique encoding that distinguishes it from any digit position.

**The `=` sign at position 21**: `21 mod 11 = 10`, which gives it the same encoding as `+`. This is acceptable because the model also has the token identity to distinguish `+` from `=`.

This is a **masterful zero-parameter encoding** that exploits the fixed structure of 10-digit addition. The period exactly matches the spacing between corresponding digits of the two operands, creating automatic column alignment.

**Limitation**: This only works for fixed-length (10-digit) inputs. For variable-length inputs, Abacus embeddings or position coupling would be needed.

---

### 3.3 Position Coupling for Column-Wise Alignment

Position coupling (Cho et al., NeurIPS 2024) takes the Abacus idea further by explicitly assigning the same position ID to corresponding digits across operands *and* the result.

For our 10-digit addition task, this would assign:
- All ones-place digits (A[9], B[9], Result[0]) -> position ID 1
- All tens-place digits (A[8], B[8], Result[1]) -> position ID 2
- ... etc.

**Theoretical guarantee**: A single-layer, 2-head transformer with coupled positions can solve addition for arbitrarily long numbers (exponential in embedding dimension).

---

### 3.4 Index Hints / Positional Descriptions

**Paper**: Shen et al., "Positional Description Matters for Transformers Arithmetic" (2023)
**Link**: https://arxiv.org/abs/2311.14737

An alternative approach: prepend explicit position markers to each token in the input. E.g., "1:A5 2:A3 3:A7 + 1:B2 2:B8 3:B1 = ...". This doubles the sequence length but provides explicit position information as tokens.

**Disadvantage**: Doubles sequence length and parameter requirements.

---

### 3.5 Design Principles for Arithmetic PE

Based on the literature, successful positional encodings for arithmetic share these properties:

1. **Digit-significance alignment**: Digits of the same column/significance should share positional information.
2. **Operand independence**: The PE should reset or be independent between operands (Abacus) or explicitly couple corresponding digits (Position Coupling).
3. **Low parameter cost**: Fixed/formula-based PEs (sinusoidal, RoPE) or very small learned PEs.
4. **Training-time randomization**: Random offsets during training enable length generalization (Abacus, Position Coupling).
5. **Compatibility with reversed format**: Since LSB-first output is crucial for arithmetic, the PE should work well with reversed number representations.

---

## 4. Input Encoding Strategies for Addition

### 4.1 Full Number vs Digit-by-Digit Representation

**Digit-by-digit** is universal in the literature for transformer arithmetic. Full number representation (e.g., treating "1234" as a single token) would require an impractically large vocabulary (10^n tokens for n-digit numbers).

For our task: **digit-by-digit is the only viable option** with a vocab of 10-14 tokens.

---

### 4.2 LSB-First vs MSB-First

**LSB-first (Least Significant Bit/Digit first)**: Numbers are written with the ones digit first. E.g., "1234" becomes "4321".

**MSB-first**: Standard human-readable order.

**Key theoretical result** (Lee et al., ICLR 2024): Outputting MSB-first requires a **global** algorithm (must scan all digits to determine the first output digit, because of carry propagation). LSB-first requires only a **local** algorithm (each output digit depends on two input digits and one carry bit).

**Lemma (Lee et al.)**: There exists an algorithm computing C=A+B from LSB that, at each position i, only requires the i-th digits of A and B plus the carry from position i-1.

**Practical impact**: LSB-first achieves 100% accuracy with ~2,500 training samples for 3-digit addition, while MSB-first plateaus at ~85% even with 10,000+ samples.

**In the 130-param solution**: Both operands and the result are in standard MSB order in the input, but the *output is generated in reverse* (LSB-first), then flipped. This combines readability with computational efficiency.

---

### 4.3 Special Tokens: +, =, Padding

**Standard practice**: Use special tokens `+` and `=` as delimiters. Some approaches also use:
- `$` as BOS/EOS tokens (Position Coupling paper)
- Zero-padding to equalize operand lengths

**In the 130-param solution**: The vocabulary has only 10 tokens (digits 0-9). The `+` and `=` are encoded using the same digit embedding but distinguished by their position in the sequence (the PE handles this). This saves 2 embedding entries.

**Zero-padding**: Padding shorter operands to equal length simplifies the PE design since corresponding digits always appear at fixed relative offsets.

---

### 4.4 Reversed Output

**Reversed output** means generating the sum starting from the least significant digit.

**Why it is critical** (Lee et al., ICLR 2024):

| Format | Samples for 100% accuracy (3-digit) | Scales to longer digits? |
|--------|--------------------------------------|--------------------------|
| Plain (MSB-first output) | 10,000+ (caps at ~85%) | Poorly |
| Reversed (LSB-first output) | ~2,500 | Yes |
| Simplified Scratchpad | ~2,000 | Yes |
| Detailed Scratchpad | ~1,000 | Yes |

Reversed output is robust to noise: even with random perturbations in preceding output tokens, the reversed format "consistently outputs a result that deviates by no more than 1 from the true answer."

**For our 130-param solution**: Output is generated reversed (LSB first), then programmatically flipped.

---

### 4.5 Scratchpad / Chain-of-Thought

**Simplified Scratchpad**: Includes carry and digit-sum for each step.
Example: `128+367=` becomes `128+367=5c1 9c0 4c0` (digit sums with carry flags).

**Detailed Scratchpad**: Full natural-language intermediate steps.
Example: `8+7=15, write 5, carry 1. 2+6+1=9, write 9. 1+3=4, write 4. Answer: 495.`

**Advantages**: Achieves 100% accuracy with the fewest training samples (1,000 for detailed).

**Disadvantages for our task**: Scratchpad dramatically increases sequence length and requires predicting intermediate tokens, which increases model capacity requirements. **Not viable for sub-500 parameter models.**

---

### 4.6 Binary vs Decimal Representation

Binary representation halves the number of distinct digit tokens (2 vs 10) but increases sequence length by ~3.3x (since log_2(10) ~ 3.32).

For 10-digit decimal numbers (up to 10^10):
- Decimal: 10 tokens per number, vocab size 10
- Binary: ~34 tokens per number, vocab size 2

**Trade-off for our task**: Binary saves token embedding parameters (2 vs 10 entries) but drastically increases sequence length, requiring more positional embedding capacity and longer attention spans. **Decimal is clearly better for our constrained setting.**

---

### 4.7 Interleaved Formats

**Concept**: Instead of presenting `A0A1A2...+B0B1B2...=`, interleave digits: `A0B0A1B1A2B2...=`.

This places corresponding digits adjacent in the sequence, potentially making column alignment easier for the attention mechanism.

**Status in literature**: Not widely studied. The Abacus and Position Coupling approaches solve the alignment problem more elegantly without rearranging the input.

**Trade-off**: Simplifies attention patterns but may break the sequential generation of the sum (since the model needs both digits at each position before computing the local sum and carry).

---

### 4.8 Factorized (Rank-1) Embeddings

**From the 130-param solution**:

```python
class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, rank=1):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(vocab_size, rank))  # 10x1
        self.B = nn.Parameter(torch.zeros(rank, emb_dim))      # 1x4
    
    def forward(self, x):
        return self.A[x] @ self.B
```

Instead of a full `vocab_size x d_model` embedding matrix (10 x 4 = 40 params), this uses rank-1 factorization: `vocab_size x 1` + `1 x d_model` = 10 + 4 = **14 parameters**.

**Why it works for digits**: Each digit (0-9) is mapped to a single scalar `A[digit]`, then projected to all embedding dimensions by the shared vector `B`. This means all digits live on a 1D manifold in embedding space -- they differ only in magnitude, not direction.

For addition, this is sufficient because digit identity primarily needs to encode *value* (0-9), and the positional embedding encodes *position*. A rank-1 embedding cleanly separates these two concerns.

---

## 5. Parameter Efficiency in Embeddings

### 5.1 Factorized Embedding Matrices (ALBERT)

**Paper**: Lan et al., "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (ICLR 2020)
**Link**: https://openreview.net/pdf?id=H1eA7AEtvS

**Mechanism**: Decomposes the embedding matrix `V x H` into `V x E` and `E x H`, where `E << H`.

Parameter reduction: from `O(V * H)` to `O(V * E + E * H)`.

For BERT: from 23M parameters to ~3M (with E=128, V=30K, H=768).

**For our task**: With V=10, H=4, a full embedding is only 40 parameters. Rank-1 factorization reduces this to 14 parameters. The savings are modest in absolute terms but significant as a fraction of a 130-500 parameter budget.

---

### 5.2 Shared Input/Output Embeddings (Weight Tying)

**Paper**: Press & Wolf, "Using the Output Embedding to Improve Language Models" (EACL 2017)
**Link**: https://arxiv.org/abs/1608.05859

**Mechanism**: The output projection matrix (which maps hidden states to vocab logits) shares weights with the input embedding matrix.

**Parameter savings**: Eliminates one copy of the `vocab_size x d_model` matrix.

**For our task**: If the input embedding is `V x d_model = 10 x 4 = 40`, weight tying saves 40 parameters. With rank-1 factorization in the input and a separate rank-1 output head, the savings structure changes.

In the 130-param solution, the LM head (output projection) uses a separate Rank1Linear:
```python
# LM head: Rank1Linear(4, 10, bias=True) = 4 + 10 + 10 = 24 params
```
Full weight tying would replace this with the transposed embedding (14 params), saving 10 params.

---

### 5.3 Hash-Based Embeddings

**Paper**: Svenstrup et al., "Hash Embeddings for Efficient Word Representations" (NeurIPS 2017)
**Link**: https://arxiv.org/abs/1709.03933

**Mechanism**: Each token is represented by k d-dimensional vectors selected from a shared pool of B vectors via hashing, combined with a learned k-dimensional importance weight. This allows an effectively infinite vocabulary to be represented with a fixed-size embedding table.

**For our task**: Irrelevant. Hash embeddings solve the problem of *large vocabularies* (millions of tokens). Our vocabulary is 10-14 tokens. Standard embeddings are already tiny.

---

### 5.4 Minimal Vocabulary Considerations

For 10-digit addition, the minimal vocabulary is:

| Strategy | Tokens | Vocab Size |
|----------|--------|------------|
| Digits + operators | 0-9, +, = | 12 |
| Digits + operators + padding | 0-9, +, =, PAD | 13 |
| Digits only (operators implicit by position) | 0-9 | 10 |
| Binary | 0, 1, +, = | 4 |
| With BOS/EOS | 0-9, +, =, BOS, EOS | 14 |

The **130-param solution uses vocab_size=10** (digits only), encoding operator positions purely through the positional embedding. This is the minimum viable vocabulary.

**Parameter impact**: Full embedding: `vocab_size * d_model`. At d_model=4:
- Vocab 10: 40 params (or 14 with rank-1)
- Vocab 14: 56 params (or 18 with rank-1)
- The difference is small but matters at the 130-param scale.

---

### 5.5 Parameter Costs of Different PE Approaches

| PE Method | Parameters | Formula | At d=4, seq=35, heads=2 |
|-----------|-----------|---------|------------------------|
| Sinusoidal | 0 | Fixed | 0 |
| Period-11 sinusoidal | 0 | Fixed | 0 |
| Learned absolute | seq * d | 35 * 4 | 140 |
| Learned (rank-1) | seq + d | 35 + 4 | 39 |
| RoPE | 0 | Fixed | 0 |
| ALiBi | 0 | Fixed | 0 |
| KERPLE (log) | 2 * heads | 2 * 2 | 4 |
| T5 relative bias | buckets * heads | 32 * 2 | 64 |
| Shaw relative | clip * d | 10 * 4 | 40 |
| FIRE | MLP params | ~hidden^2 | 50-200 |
| Abacus | max_pos * d | 100 * 4 | 400 |
| Abacus (reduced k) | k * d | 15 * 4 | 60 |
| Position Coupling | max_pos * d | 10 * 4 | 40 |
| NoPE | 0 | None | 0 |

**For sub-500 parameter models**: Zero-parameter methods (sinusoidal, RoPE, ALiBi, NoPE) or ultra-low-parameter methods (KERPLE, small Abacus) are the only viable options.

---

## 6. Comparative Summary Table

| Method | Params | Digit Alignment? | Length Gen? | Arithmetic-Tested? | Complexity |
|--------|--------|------------------|------------|---------------------|------------|
| Sinusoidal (standard) | 0 | No | Moderate | No | Low |
| **Period-11 sinusoidal** | **0** | **Yes (fixed-length)** | **No** | **Yes (130p solution)** | **Low** |
| Learned | High | No | Poor | No | Low |
| RoPE | 0 | No | Moderate | Limited | Low |
| ALiBi | 0 | No | Good | No | Low |
| T5 relative | Low | No | Good | No | Medium |
| KERPLE | Very low | No | Good | No | Low |
| FIRE | Medium | No | Very good | Yes (with Abacus) | Medium |
| **Abacus** | **Medium** | **Yes** | **Excellent (6x)** | **Yes (SOTA)** | **Medium** |
| **Position Coupling** | **Low** | **Yes** | **Excellent (6.67x)** | **Yes (SOTA)** | **Low** |
| NoPE | 0 | No | Surprisingly good | Limited | None |
| Index Hints | 0 (extra tokens) | Yes | Good | Yes | Low |

---

## 7. Recommendations for Sub-500 Parameter Models

### Best Option: Hand-Crafted Period-Matched Sinusoidal PE

For a fixed-length 10-digit addition task, the **period-11 sinusoidal PE** from Cosmin's 130-param solution is optimal:
- **Zero learnable parameters**
- Automatically aligns corresponding digits across operands
- Works perfectly with the specific input format (`AAAAAAAAAA+BBBBBBBBBB=`)
- Proven to work at 130 parameters

### For Variable-Length Generalization: Minimal Abacus

If length generalization is needed, a **minimal Abacus embedding** with small `max_k`:
- Use `max_k = 15` (supports up to ~15-digit numbers)
- With d_model=4: only 60 parameters
- Provides digit-significance alignment
- Can be combined with input injection for better results

### Key Design Choices for Minimal Models

1. **Use LSB-first (reversed) output**: This is non-negotiable for good arithmetic performance.
2. **Use rank-1 factorized token embeddings**: Saves ~26 parameters over full embeddings.
3. **Use zero-parameter PE when possible**: Sinusoidal with task-matched frequency.
4. **Vocab size = 10** (digits only): Encode operators purely through position.
5. **Consider weight tying**: Saves 10-40 parameters depending on output head design.
6. **Combine PE with strong amplitude in input region**: The 130-param solution uses amplitude 100 for input positions and 1 for output positions, letting the PE strongly guide input processing while allowing flexible output generation.

### Novel Directions to Explore

1. **Hybrid fixed + learned PE**: Use period-11 sinusoidal as a fixed base, add a tiny (rank-1) learned residual.
2. **Multi-period sinusoidal**: Use period-11 for digit alignment plus period-2 for odd/even position discrimination.
3. **RoPE with custom frequencies**: Instead of the standard geometric progression, use period-11 frequencies in the RoPE rotation -- getting both relative-position benefits and digit alignment.
4. **KERPLE for arithmetic**: The logarithmic kernel (4 parameters for 2 heads) provides a learnable distance-decay that could complement fixed digit-alignment PE.

---

## 8. References

### Core Papers for Arithmetic in Transformers

1. McLeish et al., "Transformers Can Do Arithmetic with the Right Embeddings" (NeurIPS 2024). https://arxiv.org/abs/2405.17399

2. Cho et al., "Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure" (NeurIPS 2024). https://arxiv.org/abs/2405.20671

3. Quirke & Barez, "Understanding Addition in Transformers" (ICLR 2024). https://arxiv.org/abs/2310.13121

4. Lee et al., "Teaching Arithmetic to Small Transformers" (ICLR 2024). https://arxiv.org/abs/2307.03381

5. Shen et al., "Positional Description Matters for Transformers Arithmetic" (2023). https://arxiv.org/abs/2311.14737

6. Patriota, "Arbitrary-Length Generalization for Addition in a Tiny Transformer" (2024). https://arxiv.org/abs/2406.00075

7. Cosmin's 130-param solution (gist). https://gist.github.com/cosminscn/89c110dbae76ea0c873d67607e466f5b

### Positional Encoding Papers

8. Vaswani et al., "Attention Is All You Need" (NeurIPS 2017). https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf

9. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021). https://arxiv.org/abs/2104.09864

10. Press et al., "Train Short, Test Long: Attention with Linear Biases" (ICLR 2022). https://arxiv.org/abs/2108.12409

11. Shaw et al., "Self-Attention with Relative Position Representations" (NAACL 2018). https://arxiv.org/abs/1803.02155

12. Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (JMLR 2020). -- T5 relative position bias.

13. Chi et al., "KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation" (NeurIPS 2022). https://papers.neurips.cc/paper_files/paper/2022/file/37a413841a614b5414b333585e7613b8-Paper-Conference.pdf

14. Li et al., "Functional Interpolation for Relative Positions Improves Long Context Transformers" (2023). https://arxiv.org/abs/2310.04418

15. Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" (NeurIPS 2020). https://arxiv.org/abs/2006.10739

16. Kazemnejad et al., "The Impact of Positional Encoding on Length Generalization in Transformers" (NeurIPS 2023). https://arxiv.org/abs/2305.19466

17. Haviv et al., "Transformer Language Models without Positional Encodings Still Learn Positional Information" (2022). https://arxiv.org/abs/2203.16634

18. Liutkus et al., "Relative Positional Encoding for Transformers with Linear Complexity" (ICML 2021). https://arxiv.org/abs/2105.08399

19. Park et al., "GRPE: Relative Positional Encoding for Graph Transformer" (ICLR 2022 Workshop). https://arxiv.org/abs/2201.12787

### Parameter Efficiency Papers

20. Lan et al., "ALBERT: A Lite BERT for Self-supervised Learning" (ICLR 2020). https://openreview.net/pdf?id=H1eA7AEtvS

21. Press & Wolf, "Using the Output Embedding to Improve Language Models" (EACL 2017). https://arxiv.org/abs/1608.05859

22. Svenstrup et al., "Hash Embeddings for Efficient Word Representations" (NeurIPS 2017). https://arxiv.org/abs/1709.03933
