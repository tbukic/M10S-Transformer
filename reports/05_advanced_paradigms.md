# Advanced Training Paradigms for Minimal Transformers

*Research Report -- February 2026*

This report surveys advanced training paradigms that may help achieve sub-500 parameter transformers for 10-digit addition at >=99% accuracy. We cover distillation, recursive/looped transformers, network growing, diffusion language models, Sakana AI's evolutionary approaches, algorithm learning via transformers, and combinatorial optimization for architecture design.

---

## Table of Contents

1. [Distillation for Tiny Models](#1-distillation-for-tiny-models)
2. [Recursive / Iterative Transformers](#2-recursive--iterative-transformers)
3. [Network Growing](#3-network-growing)
4. [Diffusion-Based Language Models](#4-diffusion-based-language-models)
5. [Sakana AI Work](#5-sakana-ai-work)
6. [Learning Algorithms via Transformers](#6-learning-algorithms-via-transformers)
7. [Combinatorial Optimization Approaches](#7-combinatorial-optimization-approaches)
8. [Synthesis: Implications for Sub-500 Parameter Addition](#8-synthesis-implications-for-sub-500-parameter-addition)

---

## 1. Distillation for Tiny Models

### 1.1 Knowledge Distillation Basics

Knowledge distillation (KD) transfers learned representations from a large "teacher" model to a smaller "student" model. The student learns to match the teacher's output distribution (soft labels) rather than only the hard ground-truth labels, capturing inter-class relationships and learned structure that pure supervised training misses.

**Core taxonomy:**

| Approach | What is Transferred | Pros | Cons |
|----------|-------------------|------|------|
| **Logit-based** | Output probability distribution | Simple, no architectural coupling | Misses internal representations |
| **Feature-based** | Intermediate layer activations | Richer signal, captures internal structure | Requires architectural alignment, high memory |
| **Relation-based** | Relationships between samples/layers | Captures structural knowledge | Complex implementation |

**Source:** [Knowledge Distillation survey (Springer, 2025)](https://link.springer.com/article/10.1007/s10462-025-11423-3)

### 1.2 Logit-Based vs Feature-Based Distillation

Recent evidence strongly favors logit-based distillation for practical efficiency:

- **Logit distillation** requires no storage or transmission of high-dimensional intermediate features, making it simpler and more memory-efficient.
- **Full-vocabulary logit distillation** consistently outperforms sampled-token objectives, though it requires storing vocabulary-sized logits for many positions.
- The credit assignment hierarchy follows: logit-level > token-level > sequence-level, with denser signals yielding better learning.

For tiny models (sub-1000 parameters), logit-based distillation is the pragmatic choice because the student has too few layers to meaningfully align intermediate features with the teacher.

**Source:** [Logit Distillation Overview (Emergent Mind)](https://www.emergentmind.com/topics/logit-distillation); [Comprehensive KD Survey (arXiv, 2025)](https://arxiv.org/pdf/2503.12067)

### 1.3 On-Policy Distillation

Traditional KD is "off-policy" -- the student learns from the teacher's generated outputs. **On-policy distillation** has the student generate its own outputs, then uses teacher feedback on those outputs.

**MiniLLM** (Gu et al., 2023) is the landmark on-policy distillation method for language models:
- Replaces forward KL divergence with **reverse KLD**, which prevents the student from overestimating low-probability regions of the teacher distribution
- Uses **policy optimization** (student generates, teacher evaluates) to reduce exposure bias
- Scales from 120M to 13B parameters with consistent improvements

**Relevance to our task:** On-policy distillation is particularly promising for tiny arithmetic models because the student can explore its own failure modes (carry errors, digit alignment mistakes) and receive targeted teacher feedback on exactly those cases.

**Source:** [MiniLLM (arXiv, 2023)](https://arxiv.org/abs/2306.08543)

### 1.4 Self-Distillation

**Born-Again Networks (BANs)** demonstrated that training a student with the *same architecture* as the teacher yields surprising improvements:
- A model trained to match itself (via soft labels from a previous training run) consistently outperforms standard training
- Repeated self-distillation provides additional gains at each iteration
- On CIFAR-10/100, BANs achieved state-of-the-art results

**Understanding Gains from Repeated Self-Distillation** (NeurIPS 2024) provides theoretical grounding: self-distillation acts as implicit regularization, smoothing the learned function and improving generalization.

**Relevance:** For a sub-500 parameter model, self-distillation could be a free accuracy boost -- train the model, then train it again using its own soft labels as targets. No additional parameters needed.

**Source:** [Born Again Neural Networks (arXiv, 2018)](https://arxiv.org/abs/1805.04770); [Understanding Self-Distillation (NeurIPS 2024)](https://papers.nips.cc/paper_files/paper/2024/file/0eb1ac7551ddbae575415aa5183a88be-Paper-Conference.pdf)

### 1.5 Distilling into Sub-1000 Parameter Models

No published work specifically targets distilling into models with fewer than 1000 parameters. The literature focuses on compressing large LLMs (billions of parameters) down to millions. However, the principles transfer:

**Strategy for extreme distillation:**
1. **Train a large teacher** (e.g., 100K+ parameter transformer) on 10-digit addition to high accuracy
2. **Distill using logit matching** with temperature scaling (T=2-10) to soften the distribution
3. **Apply on-policy distillation** so the tiny student explores its own mistakes
4. **Iterate with self-distillation** on the resulting student for additional gains

**Key concern:** Below a critical parameter count, the student may lack sufficient representational capacity to capture the teacher's knowledge regardless of distillation quality. For addition, the question is whether ~500 parameters suffice to represent the carry-propagation algorithm -- distillation cannot overcome fundamental capacity limits.

### 1.6 Teacher Forcing Strategies

Teacher forcing exposes the student to the teacher's intermediate outputs during training (e.g., providing correct partial sums as context). For autoregressive generation:
- **Full teacher forcing:** Student always sees teacher's previous tokens -- fast training but exposure bias at inference
- **Scheduled sampling:** Gradually shift from teacher tokens to student tokens during training
- **On-policy (MiniLLM-style):** Student always generates its own tokens, teacher provides distribution feedback

For arithmetic, teacher forcing with a scratchpad (showing carry digits) could be particularly powerful, as it decomposes the hard problem into easier per-digit steps.

---

## 2. Recursive / Iterative Transformers

### 2.1 Universal Transformers

The **Universal Transformer** (Dehghani et al., 2018, ICLR 2019) was the foundational work connecting transformers to iterative computation:

- Instead of stacking distinct layers, it **recurrently applies the same transformation** across depth
- Incorporates **Adaptive Computation Time (ACT)** from Graves (2016) to dynamically choose the number of iterations per position
- Proven to be **Turing-complete** under certain conditions (unlike standard transformers)
- Crucially: a single set of shared weights replaces many distinct layer parameters

**Key insight for our task:** Parameter sharing across depth is the most direct way to reduce parameter count while maintaining expressive depth. A 1-layer transformer applied 10 times has the same parameter count as a 1-layer transformer but potentially the computational capacity of a 10-layer one.

**Source:** [Universal Transformers (arXiv, 2018)](https://arxiv.org/abs/1807.03819); [ICLR 2019 paper](https://openreview.net/pdf/6ee41939003eaa38439a2607d081864b4ba5fea4.pdf)

### 2.2 PonderNet: Learned Halting

**PonderNet** (Banino et al., DeepMind, 2021) improved upon ACT with a probabilistic halting mechanism:
- Reformulates halting as a **geometric random variable** trained via variational inference
- Provides **unbiased, low-variance gradient estimates** (unlike ACT)
- On bAbI QA tasks: matched Universal Transformer accuracy with **6x fewer computation steps** (1,658 vs 10,161)
- Applicable to any architecture (MLP, LSTM, Transformer)

**Source:** [PonderNet (arXiv, 2021)](https://arxiv.org/pdf/2107.05407); [Ponder Transformer implementation](https://github.com/lucidrains/ponder-transformer)

### 2.3 Looped Transformers as Programmable Computers

**Giannou et al. (ICML 2023)** proved that looped transformers can simulate arbitrary computation:
- A **constant-depth (13-layer) looped transformer** can emulate a basic computer, including conditional branches, function calls, and program counters
- The input sequence acts as a "punchcard" with both instructions and memory
- Can implement **basic calculator operations, linear algebra, and even backpropagation**
- Number of iterations (loop count) handles algorithmic complexity, not model depth

This is the theoretical foundation for using tiny looped models for arithmetic: addition is an iterative algorithm (process each digit position with carries), and a constant-size transformer in a loop should suffice.

**Source:** [Looped Transformers as Programmable Computers (ICML 2023)](https://proceedings.mlr.press/v202/giannou23a.html)

### 2.4 Looped Transformers for Learning Algorithms (ICLR 2024)

Yang et al. demonstrated that **looped transformers learn better algorithmic representations**:
- On linear regression, sparse functions, decision trees, and neural network tasks
- Looped transformers match standard transformer performance with **less than 10% of parameters** (0.79M vs 9.48M)
- The advantage comes from **task decomposition**: instead of learning complex operations directly, the looped model learns simpler per-step operations (analogous to gradient descent steps)
- Limitation: looped models may learn distribution-specific solutions rather than truly general algorithms

**Source:** [Looped Transformers are Better at Learning Learning Algorithms (ICLR 2024)](https://arxiv.org/html/2311.12424v3)

### 2.5 Looped Transformers for Length Generalization (ICLR 2025)

This is perhaps the most directly relevant work:
- **Adaptive loop count** significantly improves length generalization on arithmetic tasks
- A small looped model (k blocks applied multiple times) achieves **near-perfect addition accuracy** even at lengths exceeding training distribution
- For addition, looped models with k*12/k configuration nearly match the non-looped iso-FLOP baseline
- On parity: generalizes to **50+ digits** when trained on only 20 digits
- Theoretical result: any non-looped model with few distinct layers can be simulated by a looped model with small overhead

**Critical finding for our task:** If a transformer can do 10-digit addition with N layers, a looped transformer might need only N/k parameters (for k loops), potentially enabling sub-500 parameter addition.

**Source:** [Looped Transformers for Length Generalization (ICLR 2025)](https://openreview.net/forum?id=2edigk8yoU); [GitHub](https://github.com/UW-Madison-Lee-Lab/looped-tf)

### 2.6 LoopFormer: Elastic-Depth Looped Transformers (2025)

**LoopFormer** extends looped transformers with elastic depth:
- **Budget-conditioned inference**: specify compute budget M <= L at inference, quality scales smoothly
- **Shortcut modulation**: AdaLN-style conditioning on normalized time t and step size Delta_t
- **Shortcut-consistency training**: aligns trajectories of different lengths via stop-gradient self-distillation
- Maintains non-degenerate representations across variable budgets (unlike early-exit methods which collapse)
- Tested at ~1B parameter scale with configurations: k in {1,2,3} blocks, L in {8,12,24} loops

**Source:** [LoopFormer (arXiv, 2025)](https://arxiv.org/abs/2602.11451)

### 2.7 Mixture-of-Recursions (NeurIPS 2025)

**MoR** combines parameter sharing with adaptive per-token recursion depth:
- **Lightweight routers** dynamically assign recursion depths to individual tokens
- Tokens that need more computation get more loops; simple tokens get fewer
- Achieves **2x inference throughput** vs standard transformers at similar accuracy
- Tested from 135M to 1.7B parameters
- **KV sharing variant** reuses KV pairs from first recursion to reduce memory

**Relevance:** For addition, different digit positions may need different computation (positions with carries need more). MoR's per-token adaptive depth could concentrate compute where it matters.

**Source:** [Mixture-of-Recursions (NeurIPS 2025)](https://arxiv.org/abs/2507.10524); [GitHub](https://github.com/raymin0223/mixture_of_recursions)

### 2.8 How Many Iterations for 10-Digit Addition?

Based on the literature:
- Addition is fundamentally sequential in the carry chain: worst case requires propagating carries across all 10 digits
- A **10-iteration loop** should theoretically suffice (one per digit position)
- In practice, looped transformers may need **2-3x more iterations** than the theoretical minimum due to imperfect per-step computation
- Estimate: **15-30 loop iterations** with a single-layer or 2-layer transformer block

### 2.9 Minimum Network Size with Recursion

The key question: what is the smallest transformer block that, when looped sufficiently many times, can perform 10-digit addition?

**Theoretical lower bound analysis:**
- Each step must track: current digit position, carry state (0/1), and two input digits
- Minimum hidden dimension: likely d=8-16 to represent this state
- A single attention head may suffice if the looping handles sequential processing
- Estimated minimum: a 1-layer, 1-head, d=16 transformer block = ~300-600 parameters when looped 20-30 times

This is the most promising path to sub-500 parameter addition.

---

## 3. Network Growing

### 3.1 Net2Net (Chen et al., 2015)

The foundational work on function-preserving network transformations:
- **Net2WiderNet**: Adds neurons to a layer while preserving the learned function (via weight copying and normalization)
- **Net2DeeperNet**: Adds layers initialized as identity functions
- Enables instantaneous acceleration of learning by starting from a pretrained smaller model
- Limitation: only works for piecewise-linear activations; restricted to width and depth morphisms

**Source:** [Accelerate Training Using Net2Net (Analytics Vidhya, 2024)](https://www.analyticsvidhya.com/blog/2024/02/accelerate-neural-network-training-using-the-net2net-method/)

### 3.2 Network Morphism (Wei et al., 2016)

Extends Net2Net to a broader framework:
- Handles **arbitrary non-linear activation functions** (not just piecewise linear)
- Supports morphisms for **depth, width, kernel size, and subnet changes**
- Defines the mathematical conditions for function-preserving transformations
- More flexible than Net2Net but requires solving embedding equations for each transformation type

**Source:** [Network Morphism (ICML 2016)](https://arxiv.org/pdf/1603.01670)

### 3.3 MixtureGrowth (WACV 2024)

A recent approach to growing networks:
- Constructs model weights as **linear combinations of learned templates**
- Trains two models with different templates, then **merges along the diagonal** to form a larger model
- Preserves learned knowledge during growth while doubling capacity
- Demonstrated on vision tasks

**Source:** [MixtureGrowth (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/papers/Pham_MixtureGrowth_Growing_Neural_Networks_by_Recombining_Learned_Parameters_WACV_2024_paper.pdf)

### 3.4 GrowNN: Growing with Experience (2025)

Applied progressive network growth to deep reinforcement learning:
- **Start with a small network**, learn initial policy
- **Add layers** without changing the encoded function (function-preserving growth)
- New layers learn more expressive policies on top of prior knowledge
- Results: incrementally deeper networks outperform static counterparts by **48-72%** on MiniHack and Ant environments

**Relevance:** For our task, start with a 1-layer model on 1-digit addition, grow to 2 layers for 5-digit, continue growing for 10-digit. The curriculum matches the growing architecture.

**Source:** [Growing with Experience (arXiv, 2025)](https://arxiv.org/abs/2506.11706)

### 3.5 Lottery Ticket Hypothesis and Pruning

The **Lottery Ticket Hypothesis** (Frankle & Carlin, 2018) demonstrates that:
- Dense networks contain sparse subnetworks ("winning tickets") that, trained in isolation, match full network accuracy
- These subnetworks can be **90%+ smaller** than the original
- Found via **iterative magnitude pruning**: train, prune smallest weights, reset remaining weights to initialization, retrain

**Grow-then-prune strategy for our task:**
1. Train a larger model (e.g., 5000 parameters) on 10-digit addition
2. Apply iterative magnitude pruning to find the winning ticket
3. The resulting sparse network may be well under 500 parameters
4. Retrain the sparse architecture from the winning initialization

**Challenge:** The lottery ticket hypothesis has been validated primarily for large networks. Whether winning tickets exist in already-small networks (~500 params) is unclear. However, pruning from a moderately larger network (5K-10K params) down to <500 is plausible.

**Source:** [Lottery Ticket Hypothesis (ICLR 2019)](https://arxiv.org/abs/1803.03635)

### 3.6 Progressive Growing Strategy for 10-Digit Addition

A concrete curriculum-based growing strategy:

| Stage | Digit Length | Architecture | Parameters (est.) |
|-------|-------------|-------------|-------------------|
| 1 | 1-2 digits | 1 layer, d=8 | ~100 |
| 2 | 3-4 digits | 1 layer, d=12 | ~200 |
| 3 | 5-7 digits | 1 layer, d=16 | ~350 |
| 4 | 8-10 digits | 1 layer, d=20 | ~500 |

At each stage, apply Net2WiderNet to preserve learned representations while expanding capacity. If accuracy stalls, grow depth instead of width.

---

## 4. Diffusion-Based Language Models

### 4.1 Mercury (Inception Labs, 2025-2026)

Mercury is the first family of commercial-scale **diffusion large language models (dLLMs)**:

- Uses **discrete diffusion** to generate multiple tokens in parallel
- Mercury Coder Mini: **1,109 tokens/sec** on H100 (up to 10x faster than GPT-4o Mini)
- Mercury Coder Small: **737 tokens/sec**
- **Mercury 2** (Feb 2026): first reasoning-capable diffusion LLM, 5x faster than speed-optimized frontier models
- Architecture: Transformer-based, but trained with diffusion objectives instead of autoregressive next-token prediction

**Key advantage:** Parallel token generation means the model can predict all output digits simultaneously rather than sequentially. For addition, this could mean generating the 11 output digits (10 digits + potential carry) in one or few diffusion steps.

**Source:** [Mercury blog (Inception Labs)](https://www.inceptionlabs.ai/blog/introducing-mercury); [Mercury paper (arXiv, 2025)](https://arxiv.org/abs/2506.17298); [Mercury 2 announcement (Business Wire, Feb 2026)](https://www.businesswire.com/news/home/20260224034496/en/)

### 4.2 MDLM: Masked Diffusion Language Model (NeurIPS 2024)

**MDLM** simplifies discrete diffusion for language:
- Uses **absorbing-state (masking) diffusion** only -- simpler than general discrete noise processes
- Derives a **Rao-Blackwellized objective** that reduces to a mixture of masked language modeling losses
- On LM1B: **23.00 perplexity** (vs 20.86 for autoregressive Transformer, 32.79 for SEDD)
- Tested primarily at **110M parameters**
- Powers Bytedance's Seed Diffusion and Nvidia's Genmol

**Source:** [MDLM project page](https://s-sahoo.com/mdlm/); [MDLM GitHub](https://github.com/kuleshov-group/mdlm)

### 4.3 SEDD: Score Entropy Discrete Diffusion (ICML 2024 Best Paper)

**SEDD** models ratios between data distributions rather than absolute probabilities:
- Achieves **25-75% improvements in perplexity** over previous diffusion approaches
- Won **ICML 2024 Best Paper Award**
- Provides the theoretical foundation for much of the subsequent work on discrete diffusion

**Source:** [SEDD topic (Emergent Mind)](https://www.emergentmind.com/topics/discrete-diffusion-language-model-dlm)

### 4.4 Block Diffusion (ICLR 2025 Oral)

**BD3-LMs** interpolate between autoregressive and diffusion generation:
- Decompose sequences into **blocks**; within each block, apply discrete diffusion
- Between blocks, use autoregressive conditioning
- Supports **KV caching** and flexible-length generation
- Sets new state-of-the-art among diffusion models on language benchmarks

This is relevant because for addition, the "blocks" could correspond to groups of output digits, enabling parallel generation within each group.

**Source:** [Block Diffusion (ICLR 2025)](https://arxiv.org/abs/2503.09573); [GitHub](https://github.com/kuleshov-group/bd3lms)

### 4.5 Could Diffusion Help with Arithmetic?

**Potential advantages:**
- **Parallel generation**: All output digits predicted simultaneously, no sequential error propagation
- **Iterative refinement**: Multiple diffusion steps refine all positions, similar to how humans might check and correct their work
- **No left-to-right bias**: Diffusion models can attend to all positions equally, potentially learning carry propagation more naturally

**Potential disadvantages:**
- **Parameter overhead**: Diffusion models typically need extra parameters for noise schedule handling and denoising
- **Multiple forward passes**: Still requires multiple diffusion steps (typically 10-100), so total compute may be higher
- **Less studied at tiny scale**: All current diffusion LLMs are 100M+ parameters

**Assessment for sub-500 parameter models:** Diffusion is unlikely to help at this scale. The mechanism overhead (noise conditioning, step embeddings) would consume a significant fraction of the parameter budget. The parallel generation advantage is less relevant when the output is only 11 digits. However, the iterative refinement aspect overlaps with looped transformer benefits and could be worth exploring in a hybrid approach.

### 4.6 Parameter Efficiency Comparison

| Approach | Typical Scale | Smallest Reported | Key Overhead |
|----------|-------------|-------------------|--------------|
| Autoregressive | 100M-175B | <1K (this project) | None |
| MDLM | 110M-460M | ~100M | Masking/diffusion schedule |
| Mercury | 1B+ | ~500M | Diffusion denoising |
| Block Diffusion | 100M+ | ~100M | Block conditioning |

Diffusion LLMs have not been explored at sub-million parameter scales. This represents both a gap in the literature and a risky bet for our project.

---

## 5. Sakana AI Work

### 5.1 Evolutionary Model Merging

Sakana AI's core innovation uses **evolutionary algorithms** to combine existing models:

- **Parameter Space (PS) merging**: Evolve optimal weight combinations between models
- **Data Flow Space (DFS) merging**: Evolve optimal layer ordering/selection across models
- Published in **Nature Machine Intelligence** (2025)
- Implemented in **mergekit** and **Optuna Hub** open-source frameworks
- **M2N2 (Model Merging of Natural Niches)**: Can evolve new models from scratch using evolutionary search

**Relevance:** Could evolve optimal weight configurations for tiny addition models, exploring combinations that gradient descent might not find.

**Source:** [Sakana AI Evolutionary Model Merge](https://sakana.ai/evolutionary-model-merge/); [GitHub](https://github.com/SakanaAI/evolutionary-model-merge)

### 5.2 The AI Scientist

Sakana's system for **fully automated scientific discovery**:
- Automates the complete research cycle: idea generation, code writing, experimentation, paper writing
- Cost: **~$15 per paper**
- Automated reviewer achieves near-human accuracy
- Has conducted research on diffusion models, transformers, and grokking

**Independent evaluation (2025):**
- 42% of experiments failed due to coding errors
- Literature review was inadequate (simplistic keyword searches)
- Some "novel" ideas were well-established concepts
- Still impressive as a proof of concept for automated research

**Relevance to our project:** The AI Scientist could potentially be tasked with exploring architecture variations for tiny addition models, automating the experimental loop of design-train-evaluate.

**Source:** [AI Scientist blog](https://sakana.ai/ai-scientist/); [AI Scientist v2 paper](https://pub.sakana.ai/ai-scientist-v2/paper/paper.pdf); [Independent evaluation (ACM SIGIR Forum)](https://dl.acm.org/doi/10.1145/3769733.3769747)

### 5.3 Continuous Thought Machines (CTM)

Sakana's biologically-inspired architecture (2025):
- Each artificial neuron retains a **short history of its previous activity**
- **Neural synchronization** emerges organically -- groups of neurons decide when to fire together based on internal alignment
- Computation unfolds over time steps within each neuron, rather than through fixed parallel layers
- Enhanced interpretability: you can observe which neurons synchronize for different inputs

**Relevance:** The per-neuron temporal dynamics could be relevant for tiny models where each parameter needs to contribute maximally. However, the approach has been demonstrated only at larger scales and may introduce overhead inappropriate for sub-500 parameter models.

**Source:** [Continuous Thought Machines (Sakana AI, 2025)](https://pub.sakana.ai/ctm/)

---

## 6. Learning Algorithms via Transformers

### 6.1 Transformers as Algorithm Learners

A growing body of work shows transformers can **discover and implement algorithms**:

- **In-context learning (ICL)** allows transformers to solve new tasks from examples without weight updates (Brown et al., 2020)
- Transformers implement **gradient descent-like updates** internally (Akyurek et al., 2022; von Oswald et al., 2023)
- Looped transformers can implement the **Expectation-Maximization algorithm** for mixture of linear regressions
- However, transformers struggle with **parity** and other complex Boolean functions, performing as badly as random guessing

**Source:** [ICL Universal Approximation (arXiv, 2025)](https://arxiv.org/pdf/2506.05200); [Understanding ICL (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/f7e7fabd73b3df96c54a320862afcb78-Paper-Conference.pdf)

### 6.2 Can a Transformer Learn the Addition Algorithm Itself?

The evidence is nuanced:

**What works:**
- With proper **position embeddings** (Abacus or Position Coupling), transformers learn addition that generalizes across digit lengths
- Abacus Embeddings: **99% accuracy on 100-digit addition** after training on up to 20 digits (McLeish et al., 2024)
- Position Coupling: generalizes up to **200-digit addition** (6.67x training length) (Cho et al., NeurIPS 2024)
- The key insight: standard positional encodings fail because they don't encode digit significance; specialized embeddings that mark "ones place," "tens place," etc. unlock generalization

**What fails:**
- Standard position encodings lead to complete failure on out-of-distribution lengths
- The source of failure is **digit alignment**: the model cannot match corresponding digit positions without explicit positional cues
- Scratchpads (intermediate computation traces) help but are not sufficient alone

**What this means for our task:** A sub-500 parameter model almost certainly needs specialized positional encodings (Abacus or Position Coupling style) to learn addition. This is not a parameter cost -- it is an architectural choice that enables the model to focus its limited capacity on the actual computation rather than on figuring out digit alignment.

**Source:** [Transformers Can Do Arithmetic with Right Embeddings (arXiv, 2024)](https://arxiv.org/abs/2405.17399); [Position Coupling (NeurIPS 2024)](https://arxiv.org/abs/2405.20671); [Length Generalization in Arithmetic Transformers (arXiv, 2023)](https://arxiv.org/abs/2306.15400)

### 6.3 Meta-Learning for Arithmetic

**Meta-learning transformers** can learn to adapt rapidly to new tasks:
- General-Purpose In-Context Learning reformulates meta-learning as sequence modeling
- Meta-Learning Transformers to Improve In-Context Generalization (2025) explores how to make transformers generalize better from few examples

For arithmetic, meta-learning could:
1. **Pre-train on diverse arithmetic tasks** (addition, subtraction, multiplication of various sizes)
2. **Fine-tune/adapt to 10-digit addition** with the meta-learned initialization
3. The meta-learned representations may be more compact and transferable

**Source:** [Meta-Learning Transformers (arXiv, 2025)](https://arxiv.org/pdf/2507.05019); [General-Purpose ICL (arXiv, 2022)](https://arxiv.org/abs/2212.04458)

### 6.4 The RASP Framework

**RASP (Restricted Access Sequence Processing)** provides a programming language for expressing transformer computations:
- Each RASP program maps to specific attention patterns and MLP operations
- The **RASP-Generalization Conjecture**: transformers tend to learn length-generalizing solutions when short RASP-L programs exist for the task
- Addition has a **short RASP description** (digit-wise addition with carry propagation), supporting the hypothesis that transformers should be able to learn it with generalization

This suggests that the addition algorithm is naturally expressible in the transformer computation model, and the challenge is primarily one of optimization (finding the right weights) rather than expressiveness.

**Source:** [What Algorithms Can Transformers Learn? (Apple ML Research)](https://machinelearning.apple.com/research/transformers-learn)

---

## 7. Combinatorial Optimization Approaches

### 7.1 Neural Architecture Search (NAS)

NAS automates the design of neural network architectures:

**Core framework:**
1. **Search space**: Define possible architectures (layer types, widths, connections)
2. **Search strategy**: Explore the space (RL, evolutionary, gradient-based)
3. **Performance estimation**: Evaluate candidate architectures (full training, weight sharing, predictors)

**Constraint-aware NAS:**
- MCUNet (MIT HAN Lab, NeurIPS 2020): co-designs architecture and inference engine for microcontrollers
- TinyNAS: two-stage approach -- first optimize search space for resource constraints, then search within
- MCUNet achieved **>70% ImageNet accuracy** on commercial microcontrollers with 3.5x less SRAM

**Source:** [MCUNet project page](https://hanlab.mit.edu/projects/mcunet); [MCUNet paper (NeurIPS 2020)](https://arxiv.org/abs/2007.10319)

### 7.2 LayerNAS: Polynomial Complexity NAS

Google's LayerNAS reformulates NAS as combinatorial optimization:
- Reduces to a **multi-objective optimization** problem
- **Order of magnitude reduction** in candidates that must be searched
- Makes NAS practical for very constrained settings

**Source:** [LayerNAS (Google Research)](https://research.google/blog/layernas-neural-architecture-search-in-polynomial-complexity/)

### 7.3 Mixed-Integer Programming for Architecture Design

Recent work formulates architecture optimization as **Mixed-Integer Linear Programs (MILPs)**:
- Decision variables: binary (include layer or not), integer (layer width), continuous (weights)
- Constraints: parameter budget, latency targets, memory limits
- Objective: minimize loss on validation set (or a proxy)

For our sub-500 parameter constraint, this could be formulated as:

```
Minimize:    validation_loss(architecture)
Subject to:  total_params(architecture) <= 500
             d_model in {4, 8, 12, 16, 20, 24}
             n_heads in {1, 2, 4}
             n_layers in {1, 2, 3}
             d_ff in {8, 16, 32, 64}
             ...other architectural choices
```

This is a small enough search space to enumerate exhaustively or solve with standard optimization tools.

**Source:** [Hybrid Learning-to-Optimize (arXiv, 2025)](https://arxiv.org/pdf/2511.19383); [Neural CO Tutorial (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/pii/S0305054825001303)

### 7.4 Constraint Programming for Architecture Design

**Constraint Programming (CP)** is well-suited for discrete architecture choices:
- Natural expression of parameter count constraints: `d_model * d_ff + d_ff * d_model + d_model * vocab_size + ... <= 500`
- Can encode architectural validity constraints (e.g., d_model divisible by n_heads)
- CP solvers (like OR-Tools, MiniZinc) can efficiently enumerate all valid architectures
- Combined with a learned surrogate model for accuracy prediction, this becomes practical

### 7.5 Foundation Models for Combinatorial Optimization

Emerging work applies foundation models to CO problems:
- Train transformers on large collections of optimization instances
- At inference, the model predicts good solutions to new instances
- Could potentially predict good architectures given a task description and parameter budget
- Active research area with dedicated NeurIPS 2025 workshop (DiffCoALG)

**Source:** [Foundation Models for CO (GitHub awesome list)](https://github.com/ai4co/awesome-fm4co); [DiffCoALG NeurIPS 2025](https://sites.google.com/view/diffcoalg2025)

---

## 8. Synthesis: Implications for Sub-500 Parameter Addition

### 8.1 Most Promising Approaches (Ranked)

| Rank | Approach | Expected Impact | Feasibility | Risk |
|------|----------|----------------|-------------|------|
| 1 | **Looped/Recursive Transformers** | Very High | High | Low |
| 2 | **Specialized Position Embeddings** | High | High | Low |
| 3 | **Knowledge Distillation (logit + on-policy)** | Medium-High | High | Low |
| 4 | **Network Growing + Pruning** | Medium | Medium | Medium |
| 5 | **Self-Distillation** | Medium | High | Low |
| 6 | **Constraint-based Architecture Search** | Medium | Medium | Low |
| 7 | **Evolutionary Model Merging** | Medium | Medium | Medium |
| 8 | **Diffusion-based Generation** | Low | Low | High |

### 8.2 Recommended Strategy

The strongest approach combines several techniques:

1. **Architecture**: Use a **looped transformer** with 1-2 layers and d=12-16, applying 15-30 iterations. This puts the "parameter budget" into a single high-quality block while getting depth from iteration.

2. **Position Encoding**: Adopt **Abacus Embeddings** or **Position Coupling** to encode digit significance. This is critical for the model to align digits properly and is essentially free in parameter cost.

3. **Training Pipeline**:
   - a. Train a **large teacher model** (10K+ params, looped or standard) to 99%+ accuracy on 10-digit addition
   - b. **Distill** the teacher into the tiny looped model using on-policy logit distillation
   - c. Apply **self-distillation** for 1-2 rounds on the student
   - d. Optionally: **grow** from a 1-digit model using curriculum + Net2WiderNet

4. **Architecture Search**: Use **constraint programming** to enumerate all valid architectures within the 500-parameter budget, combined with rapid training to find the optimal configuration.

5. **Post-training**: Apply **iterative magnitude pruning** (lottery ticket approach) starting from a slightly larger model (e.g., 1000 params) to find a sparse winning ticket under 500 parameters.

### 8.3 Estimated Feasibility

**Can sub-500 parameters achieve 99%+ on 10-digit addition?**

The looped transformer results are encouraging:
- Standard transformers need ~10K+ parameters for reliable 10-digit addition
- Looped transformers achieve ~10x parameter reduction with equivalent accuracy
- That puts us in the ~1000 parameter range with looping alone
- Adding distillation, specialized embeddings, and pruning could push this further

**Honest assessment:** Achieving **99%+ accuracy with exactly 500 trainable parameters** is at the boundary of what current techniques suggest is possible. The looped transformer approach is the most likely path, but may require **500-1000 parameters** rather than strictly sub-500. The carry propagation in 10-digit addition requires tracking state across 10 sequential steps, and each step requires a minimum representation capacity.

**Lower bound estimate:** Based on the information required per digit step (2 input digits, 1 carry bit, 1 output digit), a theoretical minimum is approximately **200-300 parameters** with perfect weight utilization and sufficient loop iterations (20+).

### 8.4 Key Open Questions

1. **What is the absolute minimum parameter count for a looped transformer to learn 10-digit addition?** (Needs systematic experimentation)
2. **Can distillation bridge the gap from ~1000 to ~500 parameters?** (Unknown at this scale)
3. **Does the lottery ticket hypothesis hold for networks this small?** (Uncharted territory)
4. **Can MoR-style per-token adaptive depth help for arithmetic?** (Promising but untested)
5. **Would a hybrid looped-diffusion approach offer any benefit at tiny scale?** (Speculative)

---

## References

### Distillation
- [Knowledge Distillation Survey (Springer, 2025)](https://link.springer.com/article/10.1007/s10462-025-11423-3)
- [MiniLLM: On-Policy Distillation (arXiv, 2023)](https://arxiv.org/abs/2306.08543)
- [Born-Again Neural Networks (arXiv, 2018)](https://arxiv.org/abs/1805.04770)
- [Understanding Gains from Repeated Self-Distillation (NeurIPS 2024)](https://papers.nips.cc/paper_files/paper/2024/file/0eb1ac7551ddbae575415aa5183a88be-Paper-Conference.pdf)
- [Comprehensive KD Survey (arXiv, 2025)](https://arxiv.org/pdf/2503.12067)
- [Logit Distillation Overview (Emergent Mind)](https://www.emergentmind.com/topics/logit-distillation)
- [On-Policy Distillation (Thinking Machines Lab)](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [KD: Principles, Algorithms, Applications (Neptune.ai)](https://neptune.ai/blog/knowledge-distillation)
- [HuggingFace KD Guide](https://huggingface.co/blog/Kseniase/kd)

### Recursive / Iterative Transformers
- [Universal Transformers (ICLR 2019)](https://arxiv.org/abs/1807.03819)
- [PonderNet: Learning to Ponder (arXiv, 2021)](https://arxiv.org/pdf/2107.05407)
- [Looped Transformers as Programmable Computers (ICML 2023)](https://proceedings.mlr.press/v202/giannou23a.html)
- [Looped Transformers are Better at Learning Learning Algorithms (ICLR 2024)](https://arxiv.org/html/2311.12424v3)
- [Looped Transformers for Length Generalization (ICLR 2025)](https://openreview.net/forum?id=2edigk8yoU)
- [LoopFormer: Elastic-Depth Looped Transformers (arXiv, 2025)](https://arxiv.org/abs/2602.11451)
- [Mixture-of-Recursions (NeurIPS 2025)](https://arxiv.org/abs/2507.10524)
- [Adaptive Transformer Programs (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf)
- [Ponder Transformer implementation (GitHub)](https://github.com/lucidrains/ponder-transformer)
- [Looped TF for Length Gen (GitHub)](https://github.com/UW-Madison-Lee-Lab/looped-tf)

### Network Growing
- [Net2Net (Analytics Vidhya, 2024)](https://www.analyticsvidhya.com/blog/2024/02/accelerate-neural-network-training-using-the-net2net-method/)
- [Network Morphism (ICML 2016)](https://arxiv.org/pdf/1603.01670)
- [MixtureGrowth (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/papers/Pham_MixtureGrowth_Growing_Neural_Networks_by_Recombining_Learned_Parameters_WACV_2024_paper.pdf)
- [Growing with Experience: GrowNN (arXiv, 2025)](https://arxiv.org/abs/2506.11706)
- [Lottery Ticket Hypothesis (ICLR 2019)](https://arxiv.org/abs/1803.03635)
- [Lottery Ticket Hypothesis Survey (arXiv, 2024)](https://arxiv.org/pdf/2403.04861)

### Diffusion Language Models
- [Mercury (Inception Labs)](https://www.inceptionlabs.ai/blog/introducing-mercury)
- [Mercury 2 (Business Wire, Feb 2026)](https://www.businesswire.com/news/home/20260224034496/en/)
- [Mercury Paper (arXiv, 2025)](https://arxiv.org/abs/2506.17298)
- [MDLM (NeurIPS 2024)](https://s-sahoo.com/mdlm/)
- [Block Diffusion BD3-LM (ICLR 2025)](https://arxiv.org/abs/2503.09573)
- [Diffusion LLMs Survey (arXiv, 2025)](https://arxiv.org/pdf/2506.13759)
- [Diffusion LLM Efficiency Analysis (arXiv, 2025)](https://arxiv.org/html/2510.18480)

### Sakana AI
- [Evolutionary Model Merge (Nature Machine Intelligence, 2025)](https://sakana.ai/evolutionary-model-merge/)
- [AI Scientist (Sakana AI, 2024)](https://sakana.ai/ai-scientist/)
- [AI Scientist v2](https://pub.sakana.ai/ai-scientist-v2/paper/paper.pdf)
- [Independent Evaluation of AI Scientist (ACM SIGIR Forum)](https://dl.acm.org/doi/10.1145/3769733.3769747)
- [Continuous Thought Machines (Sakana AI, 2025)](https://pub.sakana.ai/ctm/)
- [Evolutionary Model Merge GitHub](https://github.com/SakanaAI/evolutionary-model-merge)

### Learning Algorithms via Transformers
- [Transformers Can Do Arithmetic with Right Embeddings (arXiv, 2024)](https://arxiv.org/abs/2405.17399)
- [Position Coupling (NeurIPS 2024)](https://arxiv.org/abs/2405.20671)
- [Length Generalization in Arithmetic Transformers (arXiv, 2023)](https://arxiv.org/abs/2306.15400)
- [What Algorithms Can Transformers Learn? (Apple ML Research)](https://machinelearning.apple.com/research/transformers-learn)
- [Understanding Addition and Subtraction in Transformers (arXiv, 2024)](https://arxiv.org/html/2402.02619v10)
- [Arithmetic Length Generalization Thesis (U Hamburg, 2025)](https://edoc.sub.uni-hamburg.de/informatik/volltexte/2025/301/pdf/thesis.pdf)

### Combinatorial Optimization
- [MCUNet: Tiny Deep Learning on IoT (NeurIPS 2020)](https://arxiv.org/abs/2007.10319)
- [LayerNAS (Google Research)](https://research.google/blog/layernas-neural-architecture-search-in-polynomial-complexity/)
- [Neural CO Tutorial (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/pii/S0305054825001303)
- [Foundation Models for CO (GitHub)](https://github.com/ai4co/awesome-fm4co)
- [NAS Overview (MIT HAN Lab)](https://hanlab.mit.edu/techniques/nas)
