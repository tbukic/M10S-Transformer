# Report 06: Activation Functions, Normalization, Skip Connections, and FFN Design for Minimal Transformers

## Executive Summary

This report examines the design choices for activation functions, normalization layers, skip/residual connections, and feed-forward network (FFN) architectures in the context of building a sub-500 parameter transformer for 10-digit addition. At this extreme scale, every parameter must be justified. The key finding is that many "standard" transformer components can be eliminated or drastically simplified. The 130-parameter hand-coded solution by Cosmin demonstrates that **ReLU activation, no normalization, standard residual connections, and a 4-neuron MLP** suffice to solve the task perfectly. The challenge for a *learned* model is achieving comparable efficiency through gradient-based training.

---

## 1. Activation Functions

### 1.1 Standard Choices

#### ReLU (Rectified Linear Unit)
- **Formula**: `f(x) = max(0, x)`
- **Parameters**: 0 (parameter-free)
- **Pros**: Simplest computation, promotes sparsity, no vanishing gradient for positive inputs, enables piecewise-linear function approximation
- **Cons**: "Dying neuron" problem (neurons that output 0 for all inputs), not smooth
- **Relevance to digit addition**: ReLU is the activation used in Cosmin's 130-parameter solution. Its piecewise-linear nature is ideal for threshold-based carry detection (e.g., "is digit_sum >= 10?"). Each ReLU neuron naturally implements a half-plane detector, which maps directly to carry logic.

**Source**: [Squared ReLU Explained - Papers With Code](https://paperswithcode.com/method/squared-relu)

#### GELU (Gaussian Error Linear Unit)
- **Formula**: `f(x) = x * Phi(x)` where Phi is the standard Gaussian CDF
- **Parameters**: 0
- **Pros**: Smooth, good default for encoder models (BERT, etc.), stochastic regularization effect
- **Cons**: More expensive to compute than ReLU, less sparse activations
- **Relevance**: GELU is the default in many transformer implementations including nanoGPT. For a sub-500 parameter model, the smoothness offers no clear advantage over ReLU, and the reduced sparsity may actually hurt when we need sharp threshold behavior for carry detection.

#### SiLU / Swish
- **Formula**: `f(x) = x * sigmoid(x)`
- **Parameters**: 0 (or 1 if beta is learnable: `x * sigmoid(beta*x)`)
- **Pros**: Smooth, non-monotonic, unbounded above; standard in LLaMA, Mistral, GPT-NeoX
- **Cons**: Like GELU, smoother but less sparse than ReLU
- **Relevance**: Used in modern large-scale models but offers no special advantage for tiny arithmetic models. The non-monotonicity could theoretically help with modular arithmetic but adds unnecessary complexity at this scale.

**Source**: [FFN Activation Functions: ReLU, GELU, and SiLU for Transformer Models](https://mbrenndoerfer.com/writing/ffn-activation-functions)

### 1.2 Scalable Softmax (SSMax)

Scalable Softmax is a recent innovation (January 2025) designed to replace standard softmax in the **attention mechanism** (not the FFN activation).

- **Formula**: `SSMax(z_i) = n^(s*z_i) / sum(n^(s*z_j))` equivalently `= exp((s*log(n))*z_i) / sum(exp((s*log(n))*z_j))`
- **Parameters**: 1 learnable scalar `s` per attention head per layer
- **Purpose**: Prevents "attention fading" -- the phenomenon where softmax outputs flatten as sequence length grows, making it hard for the model to focus on key positions
- **Results**: Faster loss reduction during pretraining, significantly improved long-context performance, better key information retrieval
- **Extra parameter cost**: Minimal -- 144 parameters for a 768-dim, 12-layer, 12-head model (1 per head per layer)

**Relevance to our task**: With a fixed sequence length of 33 tokens (10+1+10+1+11), attention fading is not a major concern. However, SSMax could help the model learn sharper attention patterns that isolate the correct digit pairs for addition. At our scale (1 layer, 2 heads), SSMax would add only 2 parameters -- potentially worth exploring.

**Source**: [Scalable-Softmax Is Superior for Attention (arXiv 2501.19399)](https://arxiv.org/abs/2501.19399)

### 1.3 GLU Variants (Gated Linear Units)

GLU variants replace the standard FFN activation with a gating mechanism where one linear projection gates another.

#### Original GLU
- **Formula**: `GLU(x) = (xW) * sigmoid(xV)`
- **Parameters**: 3 weight matrices instead of 2 (W, V, W2 vs W1, W2)
- **Key insight**: To maintain parameter parity, Shazeer (2020) reduces hidden dimension by factor of 2/3

#### SwiGLU
- **Formula**: `SwiGLU(x) = (Swish(xW)) * (xV)`
- **Used in**: LLaMA, LLaMA 2, Mistral, PaLM
- **Performance**: Best perplexity among GLU variants

#### GeGLU
- **Formula**: `GeGLU(x) = (GELU(xW)) * (xV)`
- **Performance**: Second-best among GLU variants, close to SwiGLU

#### ReGLU
- **Formula**: `ReGLU(x) = (ReLU(xW)) * (xV)`
- **Performance**: Competitive with SwiGLU, simpler

**Relevance to our task**: GLU variants require **3 matrices** instead of 2 in the FFN. For a sub-500 parameter model, this is a significant cost. With n_embd=4 and mlp_hidden=4, a standard FFN uses 4x4 + 4x4 = 32 parameters (plus biases). A GLU variant would need 4x4 + 4x4 + 4x4 = 48 parameters for the same hidden dimension, or must reduce hidden dim to ~3, potentially losing a carry neuron. **Not recommended at this scale** unless the gating mechanism is critical for learning carry logic.

**Source**: [GLU Variants Improve Transformer (Shazeer 2020, arXiv 2002.05202)](https://arxiv.org/abs/2002.05202)

### 1.4 ReLU-Squared (Squared ReLU)

- **Formula**: `f(x) = (max(0, x))^2 = ReLU(x)^2`
- **Parameters**: 0
- **Origin**: Discovered by the Primer architecture search (Google, 2021) as one of two key modifications that improve transformer efficiency
- **Pros**: Greater nonlinearity, sparser activations, enables better function approximation with fewer layers, can replace deeper ReLU networks
- **Cons**: Gradient magnitude grows with activation value (can cause instability without normalization)

**Key research findings**:
- **Primer** (2021): ReLU-squared + depthwise convolution after Q/K/V projections reduces training cost by up to 4x at 500M parameter scale
- **ReLU Strikes Back** (ICLR 2024): ReLU (and by extension ReLU^2) provides activation sparsity that enables 32% computation savings at inference. The choice between GELU, SiLU, and ReLU does not significantly impact accuracy, but ReLU provides sparsity benefits
- **Neural network depth**: ReLU^2 networks with 2 hidden layers consistently outperform ReLU networks with the same layers for smooth function approximation

**Relevance to our task**: ReLU^2 could be valuable. The quadratic nonlinearity provides more expressive power per neuron. Where a ReLU neuron can only implement a half-plane threshold, a ReLU^2 neuron creates a quadratic response that could potentially encode both threshold detection AND magnitude information in a single neuron. This is relevant for carry detection where we need to know both *whether* a carry occurs and *how much* to adjust the digit. However, the gradient behavior may require careful initialization.

**Sources**:
- [Primer: Searching for Efficient Transformers (arXiv 2109.08668)](https://arxiv.org/abs/2109.08668)
- [ReLU Strikes Back (ICLR 2024)](https://openreview.net/forum?id=osoWxY8q2E)
- [Squared ReLU Explained](https://paperswithcode.com/method/squared-relu)

### 1.5 Sigmoid and Tanh

#### Sigmoid
- **Formula**: `f(x) = 1 / (1 + exp(-x))`
- **Range**: (0, 1)
- **Pros**: Bounded output, probabilistic interpretation
- **Cons**: Vanishing gradients, non-zero-centered, expensive to compute

#### Tanh
- **Formula**: `f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
- **Range**: (-1, 1)
- **Pros**: Zero-centered output (reduces bias accumulation), stronger gradients than sigmoid near zero, works well in shallow networks where vanishing gradient is less severe
- **Cons**: Still suffers vanishing gradient, computationally heavier than ReLU

**Relevance**: For very shallow models (1-2 layers), the vanishing gradient problem is minimal, making tanh viable. Tanh's bounded nature could also serve as a natural normalization mechanism, replacing explicit LayerNorm. However, for carry detection, the smooth S-shape of tanh/sigmoid does not create the sharp thresholds that ReLU provides naturally.

**Sources**: [Tanh vs Sigmoid vs ReLU - GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/tanh-vs-sigmoid-vs-relu/)

### 1.6 SoLU (Softmax Linear Units)

- **Formula**: `SoLU(x) = x * softmax(x)` (applied element-wise across the hidden dimension)
- **Origin**: Anthropic's Transformer Circuits research (2022)
- **Purpose**: Increases interpretability by encouraging basis-aligned (monosemantic) features. Large values suppress smaller ones, promoting sparse, interpretable neuron activations
- **Results**: Increases human-interpretable MLP neurons from 35% to 60% without performance loss
- **Trade-off**: Requires LayerNorm after SoLU to recover suppressed features

**Relevance**: SoLU is fascinating for interpretability but requires a subsequent LayerNorm layer (consuming parameters). At sub-500 parameter scale, the interpretability benefit does not outweigh the parameter cost. However, the competitive suppression mechanism could theoretically help with one-hot carry decisions.

**Source**: [Softmax Linear Units - Anthropic Transformer Circuits](https://transformer-circuits.pub/2022/solu/index.html)

### 1.7 Custom Activations for Digit Addition

Given the task structure, we can reason about what activation properties are most useful:

1. **Threshold behavior**: Carry detection requires testing if digit_sum >= 10. This maps directly to ReLU with appropriate bias: `ReLU(digit_sum - 10)` fires when carry occurs.

2. **Magnitude preservation**: The output digit is `(digit_sum + carry_in) mod 10`. We need the activation to preserve the actual sum value, not just indicate presence/absence.

3. **Chain propagation**: Carries chain across digits (e.g., 9999+1=10000). The activation must support cascading computations.

**Ideal properties for our activation**:
- Sharp thresholds (favors ReLU, ReLU^2 over GELU, SiLU)
- Unbounded positive range (carry signals may need amplification for propagation)
- Sparse activation (not all neurons fire for all inputs)
- Zero parameters (cannot afford learnable activation parameters)

**Recommendation**: ReLU or ReLU^2. ReLU for simplicity and proven effectiveness (Cosmin's solution uses it). ReLU^2 as an alternative worth exploring for its greater per-neuron expressiveness.

### 1.8 Parameter Count Summary for Activations

| Activation | Extra Parameters | Computation Cost | Sparsity |
|-----------|-----------------|-----------------|----------|
| ReLU | 0 | Lowest | High |
| ReLU^2 | 0 | Very Low | Very High |
| GELU | 0 | Medium | Low |
| SiLU/Swish | 0 (or 1) | Medium | Low |
| Sigmoid | 0 | Medium | None |
| Tanh | 0 | Medium | None |
| SoLU | 0 (needs LayerNorm after) | Medium | High |
| GLU variants | +1 weight matrix | High | Varies |
| SSMax (attention) | 1 per head per layer | Low | N/A |

---

## 2. Normalization

### 2.1 LayerNorm

- **Formula**: `LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta`
- **Parameters per layer**: 2 * d_model (gamma and beta vectors). For d_model=4: **8 parameters per LayerNorm instance**
- **Standard transformer usage**: 2 LayerNorm layers per block (pre-attention and pre-FFN), plus one final LayerNorm. For 1 block: 3 * 8 = **24 parameters** for normalization alone
- **Function**: Centers activations (mean=0) and normalizes variance (var=1), then applies learned affine transformation

**Critical concern for our task**: At d_model=4, each LayerNorm costs 8 parameters. With a 130-parameter budget, spending 24 parameters on normalization (18.5% of budget) seems wasteful -- especially since Cosmin's solution proves it is unnecessary.

**Source**: [LayerNorm and RMS Norm in Transformer Models](https://machinelearningmastery.com/layernorm-and-rms-norm-in-transformer-models/)

### 2.2 RMSNorm

- **Formula**: `RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)`
- **Parameters per layer**: 1 * d_model (gamma only, no beta). For d_model=4: **4 parameters per instance**
- **Savings over LayerNorm**: Eliminates the mean-centering step and the beta (shift) parameter
- **Performance**: Comparable to LayerNorm in most settings, with 7-64% running time reduction depending on model
- **Used in**: LLaMA, LLaMA 2, Mistral, Gemma

**For our task**: 3 RMSNorm instances would cost 12 parameters -- still significant but half of LayerNorm. At small scales, the relative importance of normalization computation is higher (it does not scale with d_model^2 like attention/FFN weights).

**Source**: [Root Mean Square Layer Normalization (arXiv 1910.07467)](https://arxiv.org/abs/1910.07467)

### 2.3 BatchNorm

- **Not suitable for transformers** due to:
  - Dependence on batch size (fails with small batches)
  - Inconsistency between training and inference modes
  - Poor handling of variable-length sequences
  - Requires running statistics that complicate deployment

**For our task**: Not recommended. The variable-length concern is minor (fixed 33-token sequences), but the batch-size dependency and train/inference mismatch add unnecessary complexity.

**Source**: [Understanding the Failure of Batch Normalization for Transformers in NLP](https://openreview.net/forum?id=X8mmH03wFlD)

### 2.4 Pre-Norm vs Post-Norm

| Aspect | Pre-Norm | Post-Norm |
|--------|----------|-----------|
| **Gradient flow** | Better (direct residual path) | Can cause vanishing gradients |
| **Training stability** | No warmup needed | Requires LR warmup |
| **Performance** | Slightly lower final quality | Better for shallow models (<=6 layers) |
| **Adoption** | GPT-2, GPT-3, LLaMA, Mistral | Original transformer, some BERT variants |

**Key finding**: Post-LN has consistently achieved better performance than Pre-LN in relatively shallow Transformers (6 or fewer layers). For a 1-layer model, if normalization is used at all, post-norm may be slightly preferable.

**However**: For a 1-layer model, the distinction is minimal since there is only one block. The more important question is whether normalization is needed at all.

**Sources**:
- [On Layer Normalization in the Transformer Architecture (arXiv 2002.04745)](https://arxiv.org/pdf/2002.04745)
- [Why Pre-Norm Became the Default in Transformers](https://medium.com/@ashutoshs81127/why-pre-norm-became-the-default-in-transformers-4229047e2620)

### 2.5 QK-Norm (Query-Key Normalization)

- **Mechanism**: Applies L2 normalization to query and key vectors before computing attention scores, replacing the standard `1/sqrt(d_k)` scaling with a learnable parameter
- **Effect**: All dot products become cosine similarities, preventing "winner-take-all" collapse and supporting more uniform attention distributions
- **Parameters**: 1 learnable scaling parameter per head (or uses existing LayerNorm parameters)
- **Benefits**: Allows higher learning rates, reduces perplexity, stabilizes attention

**For our task**: With d_head=2 (n_embd=4, n_head=2), attention scores are already very low-dimensional. QK-Norm adds minimal cost (1-2 parameters) and could help stabilize training. Worth considering if attention training proves unstable.

**Source**: [Query-Key Normalization for Transformers (arXiv 2010.04245)](https://arxiv.org/abs/2010.04245)

### 2.6 Normalization-Free Architectures

Recent research has shown that normalization layers can be completely replaced:

#### Dynamic Tanh (DyT) - CVPR 2025
- **Formula**: `DyT(x) = gamma * tanh(alpha * x) + beta`
- **Parameters**: Same as LayerNorm (gamma, beta vectors + 1 scalar alpha)
- **Key insight**: LayerNorm in transformers often produces tanh-like S-shaped input-output mappings. DyT captures this directly without computing statistics
- **Performance**: Matches or exceeds normalized counterparts across vision, speech, and language tasks (including LLaMA 7B-70B)
- **Speed**: 52.4% faster inference, 42.2% faster training at the layer level

#### Derf (Stronger Normalization-Free Transformers)
- **Formula**: `Derf(x) = erf(alpha*x + s)` where erf is the rescaled Gaussian CDF
- **Performance**: Outperforms LayerNorm, RMSNorm, and DyT across vision, speech, and DNA modeling

**For our task**: DyT is intriguing because it replaces normalization with an element-wise nonlinearity, effectively merging normalization and activation. However, it has the same parameter count as LayerNorm. The real question is whether we need *any* form of normalization.

**Sources**:
- [Transformers without Normalization (arXiv 2503.10622)](https://arxiv.org/abs/2503.10622)
- [Stronger Normalization-Free Transformers (arXiv 2512.10938)](https://arxiv.org/abs/2512.10938)

### 2.7 Is Normalization Needed at Sub-500 Parameter Scale?

**Evidence that it is NOT needed**:

1. **Cosmin's 130-parameter solution** uses `nn.Identity()` for both ln_1 and ln_2, completely eliminating normalization. The model achieves 100% accuracy on 10-digit addition.

2. **Shallow depth**: With only 1 layer, there is no deep gradient flow problem to solve. Normalization's primary benefit -- stabilizing training in deep networks -- is irrelevant.

3. **Small embedding dimension**: With d_model=4, activations have very few dimensions to normalize. The statistical estimates (mean, variance) are noisy with only 4 values, potentially doing more harm than good.

4. **Parameter budget**: 8-24 parameters for normalization represents 1.6-4.8% of a 500-parameter budget or 6-18% of a 130-parameter budget. These parameters are better spent on attention or FFN weights.

5. **Recent LayerNorm removal research**: Studies show LayerNorm can be removed from pre-trained models (GPT-2 scale) at inference time with minimal impact, and "Transformers without Normalization" (CVPR 2025) provides principled alternatives.

**Evidence that it MIGHT help training**:
1. Normalization can stabilize training dynamics, helping gradient-based optimization converge
2. Without normalization, careful weight initialization becomes more critical
3. Pre-LN transformers rely on LayerNorm parameters for stable learning -- removing them can cause overfitting

**Recommendation**: Start without normalization (following Cosmin's approach). If training proves unstable, try RMSNorm (4 params per instance) as a lightweight stabilizer, or DyT as a normalization-activation hybrid. QK-Norm (1-2 params) is the cheapest option if attention stability is the concern.

### 2.8 Parameter Cost Summary for Normalization

| Method | Params (d=4) | Instances in 1-layer transformer | Total Params |
|--------|-------------|--------------------------------|-------------|
| LayerNorm | 8 per instance | 3 (pre-attn, pre-FFN, final) | 24 |
| RMSNorm | 4 per instance | 3 | 12 |
| DyT | 9 per instance | 3 | 27 |
| QK-Norm | 1-2 per head | 2 heads | 2-4 |
| None (Identity) | 0 | -- | **0** |

---

## 3. Skip / Residual Connections

### 3.1 Standard Residual Connections

- **Formula**: `output = x + sublayer(x)` (where sublayer is attention or FFN)
- **Parameters**: 0 (parameter-free)
- **Function**: Provides shortcut for gradient flow, enables identity mapping, prevents information loss through layers
- **Used in**: Virtually all modern transformers

**Cosmin's approach**: Uses standard residual connections: `x = x + self.attn(self.ln_1(x))` and `return x + self.mlp(self.ln_2(x))`. Both residual paths are present even in the 130-parameter solution.

### 3.2 Are Residual Connections Needed for 1-2 Layer Models?

**Arguments FOR keeping them** (even with 1 layer):

1. **Information preservation**: The residual connection preserves the original token embedding (including positional information) through the attention sublayer. Without it, attention output completely replaces the input, potentially losing positional encoding information.

2. **Dual information streams**: With residual connections, the FFN receives both the original embedding AND the attention output. This allows the FFN to compute functions of both.

3. **Training dynamics**: Even for 1 layer, residual connections provide a "highway" that ensures gradients flow directly from the loss to the embedding layer. Without them, gradients must pass through both the FFN and attention backward passes.

4. **Zero-cost**: Residual connections add 0 parameters. There is no reason to remove them from a parameter-budget perspective.

**Arguments for REMOVING them**:
1. If the model is truly 1 layer, the gradient path is already short enough
2. Residual connections can constrain the model's output (it must be close to its input plus a correction)
3. More architectural flexibility without them

**Recommendation**: Keep residual connections. They are free in parameter cost and provide significant training benefits. Cosmin's solution retains them even at 130 parameters.

### 3.3 Dense Connections (DenseNet-style)

- **Concept**: Each layer receives concatenated outputs from all previous layers
- **Parameters**: Increases input dimensionality of each subsequent layer
- **Not applicable**: With only 1 layer, there are no previous layers to connect to. Even with 2 layers, the added complexity may not justify the parameter increase from wider input projections.

### 3.4 Highway Networks

- **Formula**: `output = T(x) * H(x) + (1 - T(x)) * x` where T is a learned gate and H is the transform
- **Parameters**: Additional gating parameters (T requires its own weight matrix)
- **Benefit**: Learned blending between identity and transformation

**For our task**: Highway networks add gating parameters that are hard to justify at sub-500 scale. The standard residual connection (effectively T=0.5 fixed) is sufficient and parameter-free.

### 3.5 DeepCrossAttention (DCA)

- **Concept**: Uses input-dependent weights to dynamically combine outputs from all previous layers via cross-attention
- **Parameters**: Negligible (0.2% of model) for large models
- **Performance**: Up to 3x faster training to reach same quality

**For our task**: Not applicable to a 1-layer model, but the principle of dynamic layer weighting could be relevant if scaling to 2+ layers.

**Source**: [DeepCrossAttention: Supercharging Transformer Residual Connections (arXiv 2502.06785)](https://arxiv.org/abs/2502.06785)

### 3.6 MUDDFormer (Multiway Dynamic Dense Connections)

- **Concept**: Generates connection weights dynamically based on hidden states, separately for each stream (Q, K, V, residual)
- **Performance**: MUDDPythia-2.8B matches Pythia-6.9B, adding only 0.23% parameters and 0.4% computation
- **Innovation**: Unlike static dense connections, MUDD adapts weights per-token and per-stream

**For our task**: Overkill for 1-layer models. However, the per-stream dynamic weighting concept is theoretically interesting -- if we had 2 layers, different tokens (carry positions vs non-carry positions) might benefit from different cross-layer connections.

**Source**: [MUDDFormer (arXiv 2502.12170)](https://arxiv.org/abs/2502.12170)

### 3.7 Skip-Layer Attention

- **Concept**: Queries in layer L interact with keys/values from layer L-1
- **Benefit**: Richer multi-head attention diversity without added computation
- **Not applicable**: Requires 2+ layers

**Source**: [Skip-Layer Attention (arXiv 2406.11274)](https://arxiv.org/html/2406.11274v1)

### 3.8 Summary for Residual Connections

| Method | Extra Parameters | Applicable (1 layer) | Recommendation |
|--------|-----------------|---------------------|----------------|
| Standard residual | 0 | Yes | **Use** |
| Dense connections | Increases with depth | No | Skip |
| Highway networks | Gating weights | Yes (but costly) | Skip |
| DCA | ~0.2% of model | No (needs depth) | Skip |
| MUDDFormer | ~0.23% of model | No (needs depth) | Skip |
| Skip-Layer Attention | 0 | No (needs 2+ layers) | Skip |

---

## 4. Interactions and Combined Effects

### 4.1 How Components Interact

The interaction between activation, normalization, and residual connections forms a coupled system:

1. **Residual + Normalization**: Pre-norm (normalize then add residual) vs post-norm (add residual then normalize). Pre-norm provides better gradient flow but may yield slightly worse final quality in shallow models.

2. **Activation + Normalization**: Some activations (like SoLU) require normalization to recover suppressed features. Others (like ReLU) work fine without normalization. ReLU^2 may need normalization to control growing gradient magnitudes.

3. **Residual + Activation**: The residual connection means the FFN output is *added* to the input. The activation function determines what kind of correction the FFN can produce. ReLU can only produce non-negative corrections (in each dimension); tanh bounds corrections to [-1, 1].

4. **EvoNorms**: Google/DeepMind's research on evolving combined normalization-activation layers found novel structures that go beyond human design patterns. This suggests the optimal combination may not be any standard pairing.

**Source**: [Evolving Normalization-Activation Layers (arXiv 2004.02967)](https://arxiv.org/abs/2004.02967)

### 4.2 Ablation Studies on Small Transformers

Research on transformer component ablation reveals:

- **Removing LayerNorm from Pre-LN**: Results in severe overfitting and collapsed test accuracy, especially in early layers. But this applies to Pre-LN specifically; models designed without normalization from the start can work fine.

- **Removing FFN**: Models with zero FFN layers can still function but require higher embedding dimension to compensate. There is an essential balance between model dimensionality and depth.

- **Removing residual connections**: Destabilizes training by blocking gradient flow. Even for 1-layer models, residual connections preserve important input information.

- **The interaction effect**: Removing multiple components simultaneously has compounding negative effects. However, if the architecture is designed from scratch without these components (like Cosmin's solution), the remaining components can be engineered to compensate.

**Source**: [One Wide Feedforward is All You Need (arXiv 2309.01826)](https://arxiv.org/abs/2309.01826)

### 4.3 What is Truly Essential for a Sub-500 Parameter Model?

Based on all evidence:

| Component | Essential? | Justification |
|-----------|-----------|---------------|
| Token embedding | Yes | Must convert digits to vectors |
| Positional encoding | Yes | Must identify digit positions (which digit pair to add) |
| Self-attention | Yes | Must align corresponding digit pairs (digit i of A with digit i of B) |
| Residual connections | Strongly recommended | Free (0 params), helps training, preserves position info |
| FFN with nonlinear activation | Yes | Must compute carry logic (threshold detection) |
| Normalization | No | Proven unnecessary by 130-param solution |
| Output projection | Yes | Must convert hidden state to digit prediction |

### 4.4 The Parabolic Decode Head (Cosmin's 130-param Solution)

This is one of the most ingenious components of the hand-coded solution. Here is a detailed analysis:

#### Architecture
The LM head is a Rank-1 linear layer: `W = u @ v^T` where u is (10,1) and v is (1,4).

#### Weight assignment
```python
for v in range(10):
    lm_head.u[v, 0] = 2.0 * v      # u = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    lm_head.bias[v] = -float(v * v)  # bias = [0, -1, -4, -9, -16, -25, -36, -49, -64, -81]
```
And `v[0, 3] = 1.0`, meaning only dimension 3 of the hidden state is used.

#### How it works
For a hidden state `h` with value `d` in dimension 3, the logit for digit `k` is:
```
logit(k) = 2k * d - k^2 = -(k - d)^2 + d^2
```

This is a **downward parabola centered at k = d**. The maximum logit occurs at `k = d`, meaning if the hidden state encodes the correct output digit value in dimension 3, the argmax of the logits will select that digit.

#### Why "parabolic"
The term `-k^2` in the bias creates the parabolic shape. Combined with the linear term `2k*d`, this implements a nearest-neighbor decoder: the predicted digit is whichever digit `k` is closest to the value `d`. This is far more parameter-efficient than a full 4x10 linear projection (40 params) -- the parabolic decode uses only 10 (u) + 10 (bias) + 4 (v) = **24 parameters** for the same function.

#### Implications for learned models
A learned model could potentially discover this parabolic structure, but it would require the training process to:
1. Learn to encode the output digit as a scalar in one hidden dimension
2. Learn the parabolic relationship between that scalar and the logits
3. This is a highly structured solution that gradient descent may not find easily

**Alternative for learned models**: A standard linear head (4x10 = 40 params + 10 bias = 50 params) is more general and easier to learn, though more expensive. A Rank-1 head (10 + 4 = 14 params + 10 bias = 24 params) saves parameters but constrains the output to a rank-1 mapping.

### 4.5 Full Analysis of Cosmin's 130-Parameter Solution

The complete architecture:

```
GPTConfig:
  block_size = 35
  vocab_size = 10
  n_layer = 1
  n_head = 2
  n_embd = 4
  mlp_hidden = 4
  bias = False (except where explicitly added)
```

**Parameter breakdown**:
| Component | Parameters | Description |
|-----------|-----------|-------------|
| Factorized Embedding (A) | 10 | 10x1 digit-to-scalar mapping |
| Factorized Embedding (B) | 4 | 1x4 scalar-to-embedding routing |
| Attention QKV (c_attn) | 48 | 4x12 (3*n_embd x n_embd) |
| Attention Projection (c_proj) | 16 | 4x4 |
| MLP input (c_fc) weights | 16 | 4x4 |
| MLP input (c_fc) bias | 4 | 4 carry neuron thresholds |
| MLP projection (Rank-1 u) | 4 | 4x1 |
| MLP projection (Rank-1 v) | 4 | 1x4 |
| LM Head (Rank-1 u) | 10 | 10x1 parabolic decode |
| LM Head (Rank-1 v) | 4 | 1x4 dimension selection |
| LM Head bias | 10 | 10 parabolic bias (-k^2) |
| **Total** | **130** | |

**Key design decisions**:
1. **No normalization** (`nn.Identity()`)
2. **ReLU activation** (not GELU)
3. **Factorized embeddings** (rank-1: 14 params instead of 40)
4. **Rank-1 projections** where possible (8 params instead of 16)
5. **Dynamic sinusoidal PE** (no learned position embeddings)
6. **Only 4 MLP neurons** for carry detection
7. **Parabolic decode head** (24 params instead of 50)

**The 4 carry neurons**:
- Neuron 0: `100*dim1 - 100*dim0 - 50` -- fires when digit_sum >= ~5 (detects potential carry contribution)
- Neuron 1: `100*dim1 - 100*dim0 - 150` -- fires when digit_sum >= ~15 (detects definite double-carry)
- Neuron 2: `1000*dim3 + 10*dim1 - 10*dim0 - 9005` -- carry chain amplifier (high threshold)
- Neuron 3: `1000*dim3 + 10*dim1 - 10*dim0 - 9015` -- carry chain amplifier (higher threshold)

The key insight: neurons 2-3 have a 1000x weight on dim3, which contains carry information from previous positions (fed back through the residual connection and attention mechanism). This creates an amplification cascade that propagates carries across digit positions.

**Source**: [Cosmin's 130-param nanoGPT gist](https://gist.github.com/cosminscn/89c110dbae76ea0c873d67607e466f5b)

---

## 5. Feed-Forward Network (FFN) Design

### 5.1 Standard 2-Layer FFN

- **Formula**: `FFN(x) = W2 * activation(W1 * x + b1) + b2`
- **Parameters**: `d_model * d_ff + d_ff + d_ff * d_model + d_model` = `2 * d_model * d_ff + d_ff + d_model`
- **For d_model=4, d_ff=4**: 4*4 + 4 + 4*4 + 4 = 40 parameters (with biases)
- **Standard ratio**: d_ff = 4 * d_model (in standard transformers), but this is arbitrary

### 5.2 GLU-Style FFN

- **Formula**: `FFN(x) = W2 * (activation(W1*x) * (V*x))`
- **Parameters**: `d_model * d_ff + d_model * d_ff + d_ff * d_model` = `3 * d_model * d_ff` (plus biases)
- **For d_model=4, d_ff=3** (reduced by 2/3): 4*3 + 4*3 + 3*4 = 36 parameters
- **Trade-off**: More parameters for same hidden size, or fewer hidden neurons for same parameter count

### 5.3 Can We Eliminate the FFN Entirely?

Research ("One Wide Feedforward is All You Need" by Apple, 2023) shows that FFN layers can be shared or removed from decoder layers with modest accuracy drops. However, for arithmetic:

**The FFN is essential for digit addition because**:
1. Attention can only perform linear operations (weighted averages) on values
2. Carry detection requires nonlinear thresholding (`is sum >= 10?`)
3. The FFN provides the only nonlinear computation in the transformer block
4. Without the FFN, the entire model is a linear function of the input (attention is softmax-weighted averaging, and without FFN nonlinearity, the composition is still approximately linear for the learned task)

**Confirmed by Cosmin's solution**: The 4-neuron MLP is where all carry logic resides. The attention mechanism only routes information (pairs digits and propagates carry signals); the MLP does the actual arithmetic.

**Source**: [One Wide Feedforward is All You Need (arXiv 2309.01826)](https://arxiv.org/abs/2309.01826)

### 5.4 Minimal FFN: How Many Neurons for Carry Logic?

#### Theoretical analysis

For 10-digit addition with carries, the FFN must compute:
1. **Digit sum**: `a_i + b_i + carry_in` (this can be done by attention routing + linear combination)
2. **Carry detection**: `is digit_sum >= 10?` (requires nonlinear threshold -- 1 ReLU neuron minimum)
3. **Carry propagation**: Handle chains like 99999+1=100000 where carry propagates through all digits

**Cosmin's solution uses 4 ReLU neurons**:
- 2 neurons for basic carry detection (thresholds at sum=5 and sum=15)
- 2 neurons for carry chain amplification (using 1000x scaling)

**Research on transformer addition** (arXiv 2402.02619): Transformers learn a "TriCase" operation that classifies digit pairs into three states: definite carry (sum >= 10), possible carry (sum = 9), no carry (sum <= 8). This requires at minimum 2 threshold neurons per digit position (to distinguish 3 states).

**Minimum neurons for N-digit addition**:
- **1 neuron**: Can detect single carries but cannot propagate chains
- **2 neurons**: Can implement TriCase (3-state classification) for carry detection
- **4 neurons**: Sufficient for full carry chain propagation (as demonstrated by Cosmin)
- **The theoretical minimum is likely 2-3 neurons**, but 4 provides robustness and clean separation of concerns

**Source**: [Arithmetic in Transformers Explained (arXiv 2402.02619)](https://arxiv.org/html/2402.02619v9)

### 5.5 How Many Neurons for 10-Digit Carry Propagation?

The key insight from both Cosmin's solution and the "Arithmetic in Transformers" paper is that carry propagation is handled through the **attention mechanism**, not just the FFN. Here is how:

1. **Attention** identifies "possible carry" positions (where digit sum = 9)
2. **FFN** applies the threshold to detect carries
3. **Autoregressive generation** propagates carries left-to-right (digit by digit), so the carry from position i is available as input for position i+1

This means the FFN does NOT need to handle the entire carry chain in one forward pass. It only needs to:
- Detect whether the current digit pair generates a carry
- Incorporate the incoming carry from the previous digit (which is already part of the input context through autoregressive generation)

**This reduces the neuron requirement**: 2-4 neurons suffice because each digit position is processed sequentially, with carry information flowing through the autoregressive generation loop rather than through a single forward pass.

### 5.6 FFN Alternatives for Minimal Models

| FFN Design | Params (d=4, h=4) | Carry Capable? | Notes |
|-----------|-------------------|---------------|-------|
| Standard 2-layer (with bias) | 40 | Yes | Baseline |
| Standard 2-layer (no bias) | 32 | Partially | Harder to set thresholds without bias |
| Rank-1 projection | 24 | Yes | Cosmin's approach (full W1, rank-1 W2) |
| GLU-style (h=3) | 36 | Yes | Extra gating, fewer hidden neurons |
| Single linear + activation | 20 | Limited | No output projection |
| No FFN | 0 | **No** | Cannot detect carries |

### 5.7 Importance of Bias in the FFN

The bias term in the first FFN layer is **critical** for carry detection. Without it, the threshold is fixed at 0 (ReLU fires for any positive input). With bias, we can set arbitrary thresholds:

```python
# Cosmin's carry neuron thresholds:
fb[0] = -50   # fires when weighted sum > 50
fb[1] = -150  # fires when weighted sum > 150
fb[2] = -9005 # carry chain amplifier threshold
fb[3] = -9015 # carry chain amplifier threshold
```

**Recommendation**: Always include bias in the first FFN layer (costs only d_ff = 4 parameters). The output projection bias is less critical and can be omitted for parameter savings.

---

## 6. Comprehensive Recommendations for Sub-500 Parameter Transformers

### 6.1 Recommended Architecture

Based on all research, the recommended starting configuration for a **learned** sub-500 parameter 10-digit addition model:

| Component | Choice | Params | Rationale |
|-----------|--------|--------|-----------|
| Embedding | Factorized (rank-1) | 14 | Following Cosmin; full embedding costs 40 |
| Position encoding | Sinusoidal (fixed) | 0 | Dynamic PE eliminates learned position params |
| Normalization | None (Identity) | 0 | Proven unnecessary; saves 12-24 params |
| Attention (QKV) | Full linear | 48 | Core routing mechanism; hard to factorize |
| Attention projection | Full or Rank-1 | 16 or 8 | Rank-1 saves 8 params |
| FFN activation | ReLU | 0 | Sharp thresholds for carry detection |
| FFN hidden dim | 4-8 | 16-32 + 4-8 bias | 4 minimum for carries; 8 provides headroom |
| FFN projection | Rank-1 | 8 | Sufficient to route carry info |
| Output head | Rank-1 with bias | 24 | Parabolic decode or learned |
| Residual connections | Standard | 0 | Free, beneficial |
| **Total** | | ~130-160 | |

### 6.2 Variants Worth Exploring

1. **ReLU-squared activation**: Replace ReLU with ReLU^2 in FFN. Zero extra parameters, potentially more expressive per neuron, may reduce the number of neurons needed.

2. **SSMax in attention**: Replace softmax with SSMax. 2 extra parameters for sharper attention patterns.

3. **Wider FFN (8 neurons)**: Increase mlp_hidden to 8, adding ~36 parameters but providing more capacity for carry chain logic. May help learnability even if 4 neurons suffice for hand-coding.

4. **QK-Norm**: Add lightweight normalization to attention if training proves unstable. 2-4 extra parameters.

5. **Full (non-factorized) embedding**: Trade 26 extra parameters for easier learning of digit representations.

6. **2 layers with shared weights**: Double the depth without doubling parameters. The second pass through the same weights could help with carry chain propagation.

### 6.3 What NOT to Use

- **GELU/SiLU**: Smoother activations lose the sharp threshold behavior needed for carry detection
- **Full LayerNorm**: Too expensive (24 params) for no proven benefit at this scale
- **GLU variants**: 3 weight matrices are too expensive for the minimal FFN
- **SoLU**: Requires subsequent LayerNorm, compounding parameter cost
- **Highway connections**: Gating parameters not justified at this scale
- **Dense connections**: Not applicable to 1-layer models
- **BatchNorm**: Fundamentally unsuited for sequence models

---

## 7. Open Questions and Areas for Further Investigation

1. **Can gradient descent discover the parabolic decode head?** The hand-coded solution uses a very specific weight structure. It is unclear whether SGD/Adam can find this structure from random initialization.

2. **ReLU vs ReLU^2 for learnability**: While ReLU is proven sufficient (Cosmin's solution), ReLU^2 might be easier to *learn* because its quadratic response could encode threshold + magnitude in one neuron.

3. **Optimal hidden dimension for learned models**: Cosmin uses d_ff=4 (same as d_model), but a learned model may need d_ff=8 or d_ff=16 for gradient descent to find the carry logic.

4. **Sinusoidal vs learned positional encoding**: Cosmin uses carefully designed sinusoidal PE with amplitude modulation (100 for operand positions, 1 for result positions). Can a learned model discover similarly effective positional encoding?

5. **Attention pattern learnability**: The hand-coded attention uses sinusoidal resonance to pair digits. Can gradient descent discover these attention patterns with only 48 parameters in c_attn?

---

## Sources

- [Scalable-Softmax Is Superior for Attention (arXiv 2501.19399)](https://arxiv.org/abs/2501.19399)
- [GLU Variants Improve Transformer (Shazeer 2020, arXiv 2002.05202)](https://arxiv.org/abs/2002.05202)
- [Squared ReLU Explained - Papers With Code](https://paperswithcode.com/method/squared-relu)
- [Primer: Searching for Efficient Transformers (arXiv 2109.08668)](https://arxiv.org/abs/2109.08668)
- [ReLU Strikes Back (ICLR 2024)](https://openreview.net/forum?id=osoWxY8q2E)
- [Root Mean Square Layer Normalization (arXiv 1910.07467)](https://arxiv.org/abs/1910.07467)
- [On Layer Normalization in the Transformer Architecture (arXiv 2002.04745)](https://arxiv.org/pdf/2002.04745)
- [Query-Key Normalization for Transformers (arXiv 2010.04245)](https://arxiv.org/abs/2010.04245)
- [Transformers without Normalization (arXiv 2503.10622)](https://arxiv.org/abs/2503.10622)
- [Stronger Normalization-Free Transformers (arXiv 2512.10938)](https://arxiv.org/abs/2512.10938)
- [Understanding the Failure of Batch Normalization for Transformers in NLP](https://openreview.net/forum?id=X8mmH03wFlD)
- [DeepCrossAttention (arXiv 2502.06785)](https://arxiv.org/abs/2502.06785)
- [MUDDFormer (arXiv 2502.12170)](https://arxiv.org/abs/2502.12170)
- [Skip-Layer Attention (arXiv 2406.11274)](https://arxiv.org/html/2406.11274v1)
- [One Wide Feedforward is All You Need (arXiv 2309.01826)](https://arxiv.org/abs/2309.01826)
- [Teaching Arithmetic to Small Transformers (arXiv 2307.03381)](https://arxiv.org/abs/2307.03381)
- [Arithmetic in Transformers Explained (arXiv 2402.02619)](https://arxiv.org/html/2402.02619v9)
- [Softmax Linear Units - Anthropic Transformer Circuits](https://transformer-circuits.pub/2022/solu/index.html)
- [Evolving Normalization-Activation Layers (arXiv 2004.02967)](https://arxiv.org/abs/2004.02967)
- [Cosmin's 130-param nanoGPT gist](https://gist.github.com/cosminscn/89c110dbae76ea0c873d67607e466f5b)
- [Reverse Engineering a Neural Network for Binary Addition](https://cprimozic.net/blog/reverse-engineering-a-small-neural-network/)
- [LayerNorm and RMS Norm in Transformer Models](https://machinelearningmastery.com/layernorm-and-rms-norm-in-transformer-models/)
- [FFN Activation Functions for Transformers](https://mbrenndoerfer.com/writing/ffn-activation-functions)
- [On Layer Normalizations and Residual Connections in Transformers (arXiv 2206.00330)](https://arxiv.org/abs/2206.00330)
