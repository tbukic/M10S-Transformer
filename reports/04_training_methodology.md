# Training Methodology Literature Review

## Overview

This report is a comprehensive literature review of training methodology, optimization, and training dynamics relevant to training minimal transformer models (sub-500 parameters) for arithmetic tasks such as 10-digit addition. It covers optimizers, learning rate scheduling, curriculum learning, grokking, training dynamics, WeightWatcher diagnostics, and architecture search.

---

## 1. Optimizers

### 1.1 AdamW: Deep Dive

**Paper:** Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017/2019)
- **URL:** https://arxiv.org/abs/1711.05101

**What it is:** AdamW fixes the problem of L2 regularization in Adam by decoupling weight decay from the gradient-based update. Instead of adding weight decay to the loss function (L2 regularization), it applies weight decay directly during the parameter update step.

**Why it matters for Adam:** Adam's adaptive learning rates break the equivalence between L2 regularization and weight decay that holds for SGD. This causes L2 regularization to behave in unexpected ways within Adam. AdamW restores proper regularization behavior.

**Best practices:**
- **Default learning rate:** 1e-3, with common starting points for weight decay around 0.01 or 0.1 (dataset/model dependent).
- **Parameter groups:** Decouple weight decay for parameters that should not be regularized (biases, layer-norm gain/shift) by using parameter groups with zero decay.
- **Coupling with schedulers:** AdamW works best when paired with a learning rate scheduler (cosine decay, linear warmup, etc.).
- **Hyperparameter sensitivity:** When doubling the learning rate, halve the weight decay. The advantage of decoupled weight decay is that tuning one does not require re-tuning the other, reducing a 2D grid search to two 1D line searches.

**For tiny models:** AdamW is the most commonly used baseline optimizer. For sub-500 parameter models, the adaptive learning rates may be less critical since there are fewer parameters with heterogeneous gradient magnitudes. However, the decoupled weight decay remains valuable for regularization.

**Source:** https://fabian-sp.github.io/posts/2024/02/decoupling/

---

### 1.2 Muon Optimizer

**Papers:**
- Original: Keller Jordan et al. (2024) — https://kellerjordan.github.io/posts/muon/
- Scaling paper: "Muon is Scalable for LLM Training" (2025) — https://arxiv.org/abs/2502.16982
- Theory: "Deriving Muon" by Jeremy Bernstein — https://jeremybernste.in/writing/deriving-muon
- Convergence: "On the Convergence Analysis of Muon" (2025) — https://arxiv.org/abs/2505.23737

**What it is:** Muon is an optimizer for hidden layers of neural networks that updates matrix parameters with orthogonalized gradient momentum using Newton-Schulz iteration. It arises naturally from the spectral norm duality map.

**How it works:**
1. Compute gradient momentum (Nesterov-style works slightly better than standard SGD momentum).
2. Orthogonalize the momentum using Newton-Schulz iterations (5 iterations suffice).
3. The orthogonalization ensures every parameter update has the same spectral scale.
4. Muon only applies to 2D parameters; scalar/vector parameters and embedding/classifier head layers must use AdamW.

**Key results:**
- Lowered the CIFAR-10 94% training record from 3.3 to 2.6 A100-seconds.
- Trained a transformer to GPT-2 (XL) performance in $175 of compute.
- ~2x computational efficiency vs. AdamW with compute-optimal training at scale (Kimi/Moonlight).
- FLOP overhead below 1%; lighter memory footprint than AdamW (only first moment).

**Scaling insights (Moonlight, 2025):**
- Two crucial techniques for scaling: (1) adding weight decay and (2) carefully adjusting per-parameter update scale.
- With these, Muon works out-of-the-box on large-scale training without hyperparameter tuning.

**Convergence theory (2025):**
- Muon benefits from the low-rank and approximate blockwise diagonal structure of Hessian matrices, phenomena widely observed in neural network training.

**For tiny models:** Muon is designed for 2D weight matrices in hidden layers. In a sub-500 parameter transformer, the weight matrices may be very small (e.g., 16x16). The orthogonalization step may have outsized effects on such small matrices. This is worth experimenting with, but the original validation is on much larger models.

**Source:** https://github.com/KellerJordan/Muon

---

### 1.3 LION Optimizer

**Paper:** Chen et al., "Symbolic Discovery of Optimization Algorithms" (2023)
- **URL:** https://arxiv.org/abs/2302.06675

**What it is:** EvoLved sIgn mOmeNtum (Lion) was discovered via AutoML evolutionary search over program spaces. It uses only the sign of the momentum for updates, making it simpler and more memory-efficient than Adam.

**How it works:**
1. Track momentum (no second-order statistics).
2. Use the sign of the momentum to compute updates.
3. Every parameter update has the same magnitude (uniform update norm).
4. This uniform norm acts as an implicit regularizer.

**Key properties:**
- Simpler than Adam; fewer hyperparameters.
- Requires learning rates 3-10x smaller than Adam, and weight decay 3-10x larger.
- More memory efficient (no second moment).
- 88.3% zero-shot / 91.1% fine-tuning on ImageNet; 2.3x training efficiency on diffusion models.
- Deployed in production at Google (search ads CTR model).

**For tiny models:** Lion's implicit regularization from uniform update norms could be beneficial for tiny models prone to overfitting. Its sign-based updates may help when gradients have high variance relative to their magnitude, which is common in small models.

**Source:** https://github.com/lucidrains/lion-pytorch

---

### 1.4 Sophia Optimizer

**Paper:** Liu et al., "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training" (2023, ICLR 2024)
- **URL:** https://arxiv.org/abs/2305.14342

**What it is:** Sophia uses a lightweight estimate of the diagonal Hessian as a pre-conditioner, with element-wise clipping to tame non-convexity.

**How it works:**
1. Moving average of gradients divided by moving average of estimated Hessian.
2. Element-wise clipping controls worst-case update size.
3. Hessian estimated only every few iterations (negligible overhead).

**Key results:**
- 2x speedup over Adam for GPT models (125M-1.5B parameters).
- Achieves same perplexity with 50% fewer steps.
- Adapts to heterogeneous curvatures across parameter dimensions.

**For tiny models:** The Hessian estimation may be noisy for very small models with few parameters. However, the curvature-aware scaling could help navigate the complex loss landscape that even tiny transformers exhibit.

**Source:** https://openreview.net/forum?id=3xHDeA8Noi

---

### 1.5 Sharpness-Aware Minimization (SAM)

**Paper:** Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR 2021)
- **URL:** https://openreview.net/forum?id=6Tm1mposlrM

**What it is:** SAM seeks parameters that lie in neighborhoods with uniformly low loss, resulting in a min-max optimization problem.

**How it works:**
1. Compute a perturbation in the direction of sharpest ascent.
2. Compute gradient at the perturbed point.
3. Use this "sharpness-aware" gradient for the update.
4. Requires two forward-backward passes per step (2x computational cost).

**Key results:**
- Improves generalization across CIFAR-10/100, ImageNet, and fine-tuning tasks.
- Provides robustness to label noise.
- Finds flatter minima that generalize better.

**Variants:** ASAM (adaptive), Efficient SAM, Friendly SAM.

**For tiny models:** SAM's flat-minima-seeking behavior could be especially valuable for tiny models where the loss landscape may be dominated by sharp minima. The 2x computational cost is negligible for sub-500 parameter models. SAM may help with grokking-like phenomena by biasing toward generalizable solutions.

**Source:** https://github.com/davda54/sam

---

### 1.6 Schedule-Free Optimizers

**Paper:** Defazio et al., "The Road Less Scheduled" (NeurIPS 2024)
- **URL:** https://arxiv.org/abs/2405.15682

**What it is:** Schedule-Free learning eliminates the need for a learning rate schedule by replacing momentum with a combination of interpolation and averaging.

**How it works:**
1. Uses a constant learning rate (no schedule).
2. Replaces standard momentum with interpolation and weight averaging.
3. Introduces no additional hyperparameters beyond standard optimizers with momentum.

**Key results:**
- Matches or outperforms SOTA schedules (cosine-decay, linear decay).
- Won the MLCommons 2024 AlgoPerf Self-Tuning track.
- Does not require specifying total training steps T in advance.

**For tiny models:** Extremely attractive because it removes one major hyperparameter (the schedule). For experiments where training duration varies widely (exploring 1K to 100K+ epochs), not needing to specify T in advance is a significant practical advantage.

**Source:** https://github.com/facebookresearch/schedule_free

---

### 1.7 Population-Based Training (PBT)

**Paper:** Jaderberg et al., "Population Based Training of Neural Networks" (2017, DeepMind)
- **URL:** https://deepmind.google/blog/population-based-training-of-neural-networks/

**What it is:** PBT jointly optimizes neural network weights and hyperparameters by training a population of models in parallel, periodically copying weights from better performers and mutating hyperparameters.

**How it works:**
1. Train many networks in parallel with random hyperparameters.
2. Periodically evaluate all members.
3. Poorly performing members copy weights from better performers ("exploit").
4. Mutate hyperparameters of copied models ("explore").
5. Continue training with warm-started weights and new hyperparameters.

**Key advantages:**
- Hyperparameters evolve during training (no fixed schedule needed).
- Warm-starting from good weights avoids training from scratch.
- Naturally discovers optimizer schedules.

**For tiny models:** With sub-500 parameter models, individual training runs are very cheap. PBT becomes extremely practical — a population of 50-100 models can be trained simultaneously on a single GPU. This is one of the most promising approaches for our use case.

**Source:** https://arxiv.org/pdf/1902.01894

---

### 1.8 Optimizer Recommendations for Tiny Models

For models with <1000 parameters, the following considerations apply:

| Factor | Recommendation |
|--------|---------------|
| **Baseline** | AdamW with careful weight decay tuning |
| **Regularization-focused** | SAM or Lion (implicit regularization) |
| **Ease of tuning** | Schedule-Free AdamW (fewer hyperparameters) |
| **Experimental** | Muon for 2D weight matrices, with AdamW for embeddings |
| **Meta-optimization** | PBT to jointly evolve optimizer hyperparameters during training |
| **Speed** | Lion or Muon (lower memory, potentially faster convergence) |

**Key insight:** For tiny models, the computational cost of the optimizer is negligible. The primary concern is which optimizer leads to the best generalization, not efficiency. SAM's 2x cost is irrelevant when training takes seconds per epoch.

---

## 2. Learning Rate Scheduling

### 2.1 Cosine Annealing with Warm Restarts (SGDR)

**Paper:** Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (ICLR 2017)
- **URL:** https://arxiv.org/abs/1608.03983

**How it works:**
1. Learning rate decays following a cosine curve from max to min.
2. At restart points, learning rate jumps back to maximum.
3. Restart intervals can increase by a multiplicative factor T_mult.
4. Momentum/information from previous cycles is preserved.

**PyTorch:** `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)`

**For tiny models:** Warm restarts can help escape local minima and explore different regions of the loss landscape. With tiny models, the landscape may have many equally deep but different minima; restarts help explore them.

---

### 2.2 Linear Warmup + Decay

**References:**
- Popel & Bojar, "Training Tips for the Transformer Model" (2018) — https://arxiv.org/abs/1804.00247
- "Why Warmup the Learning Rate? Underlying Mechanisms and Improvements" (2024) — https://arxiv.org/abs/2406.09405

**Why warmup matters:**
- Warmup gradually reduces the sharpness (top eigenvalue of Hessian), moving the model from poorly conditioned areas toward flatter regions.
- Without warmup, transformer training is prone to divergence, especially due to attention entropy collapse.
- Warmup gives the optimization process time to "settle down" before taking larger steps.
- Duration: 100 steps for simple models, up to 40K+ for large transformers.

**For tiny models:** Even tiny transformers benefit from a brief warmup (10-100 steps). The attention mechanism can be unstable with random initialization. After warmup, linear or cosine decay to a small minimum learning rate works well.

---

### 2.3 Cyclic Learning Rates

**Paper:** Smith, "Cyclical Learning Rates for Training Neural Networks" (WACV 2017)
- **URL:** https://arxiv.org/abs/1506.01186

**How it works:**
1. Learning rate oscillates between a minimum and maximum bound.
2. Triangular policy: linear increase then linear decrease per cycle.
3. Triangular2: same but halves the amplitude each cycle.
4. LR Range Test: sweep learning rates to find optimal min/max bounds.

**Key benefit:** Improved classification accuracy without explicit LR tuning, often in fewer iterations.

---

### 2.4 One-Cycle Policy (Super-Convergence)

**Papers:**
- Smith & Topin, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates" (2019) — https://arxiv.org/abs/1708.07120
- Smith, "A Disciplined Approach to Neural Network Hyper-parameters" (2018) — https://arxiv.org/abs/1803.09820

**How it works:**
1. Single cycle: LR increases from small to large, then decreases back.
2. Final phase: LR drops several orders of magnitude below initial.
3. Large max LR acts as a regularizer, requiring reduction of other regularization.
4. Achieves 93% accuracy on CIFAR-10 in 70 epochs (vs. hundreds normally).

**For tiny models:** Super-convergence may be especially powerful for tiny models because the large learning rate phase provides strong implicit regularization. For sub-500 parameter models, the regularization budget is tight, so having the learning rate schedule contribute regularization is valuable.

---

### 2.5 Learning Rate as a Function of Model Size

**References:**
- "Predictable Scale: Part I — Optimal Hyperparameter Scaling Law in Large Language Model Pretraining" (2025) — https://arxiv.org/abs/2503.04715
- "How to Scale" — https://howtoscalenn.github.io/

**Key findings:**
- Larger models require smaller learning rates for stability.
- Optimal LR follows a power-law relationship with model parameters and data size.
- When training data is small, higher learning rates tend to be better.
- The optimal LR for Adam does not monotonically increase with batch size — it first rises then falls ("surge phenomenon").

**For models <1000 parameters:**
- Learning rates can be significantly higher than for large models (e.g., 1e-2 to 1e-1 vs. 1e-4 to 1e-3).
- The sweet spot needs empirical search but start with 1e-3 to 3e-2.
- Weight decay should be proportionally larger for small models.

---

## 3. Curriculum Learning

### 3.1 Overview

**Seminal paper:** Bengio et al., "Curriculum Learning" (ICML 2009)
- **URL:** https://doi.org/10.1145/1553374.1553380

**Core idea:** Train on examples of increasing difficulty, analogous to how humans learn. Start with easy examples and progressively introduce harder ones.

**Variants:**
| Variant | Description |
|---------|-------------|
| Vanilla Curriculum | Pre-defined difficulty ordering, fixed schedule |
| Self-Paced Learning | Model selects examples based on current loss/confidence |
| Teacher-Student CL | Separate network determines difficulty |
| Progressive CL | Gradually increases task complexity |
| Balanced CL | Maintains class/difficulty balance during progression |

**Source:** https://en.wikipedia.org/wiki/Curriculum_learning

---

### 3.2 Teaching Arithmetic to Small Transformers

**Paper:** Lee et al., "Teaching Arithmetic to Small Transformers" (2023)
- **URL:** https://arxiv.org/abs/2307.03381

**Key findings for arithmetic:**
1. **Data format matters enormously.** Four formats tested:
   - Plain: `123+456=579` — never reaches 100% accuracy
   - Reverse: output digits in reverse order — much better
   - Simplified Scratchpad: intermediate carries shown
   - Detailed Scratchpad: full chain-of-thought — best accuracy and sample efficiency

2. **Curriculum helps:** Training on 1-digit addition first, then 2-digit, etc. improves convergence.

3. **Chain-of-thought training:** Including intermediate computation steps simultaneously improves accuracy, sample complexity, and convergence speed.

**For our project:** This is one of the most directly relevant papers. The data format and curriculum strategy will likely have more impact on performance than any optimizer choice.

**Source:** https://github.com/lee-ny/teaching_arithmetic

---

### 3.3 Transformers Can Do Arithmetic with the Right Embeddings

**Paper:** McLeish et al., "Transformers Can Do Arithmetic with the Right Embeddings" (NeurIPS 2024)
- **URL:** https://arxiv.org/abs/2405.17399

**Key innovation: Abacus Embeddings**
- Standard positional embeddings fail because they cannot represent a digit's position within a number.
- Abacus Embeddings add an embedding to each digit encoding its position relative to the start of its number.
- Training on 20-digit numbers achieves 99% accuracy on 100-digit addition.

**Additional architecture improvements:**
- Input injection (re-injecting input at each layer).
- Recurrent layers improve performance further.

**For our project:** Abacus embeddings could be crucial for length generalization. Even in a tiny model, proper positional encoding of digit significance (ones, tens, hundreds, etc.) may be the difference between memorization and true arithmetic learning.

**Source:** https://github.com/mcleish7/arithmetic

---

### 3.4 Progressive Difficulty and Growing Networks

**Net2Net Paper:** Chen et al., "Net2Net: Accelerating Learning via Knowledge Transfer" (ICLR 2016)
- **URL:** https://arxiv.org/abs/1511.05641

**Growing networks approach:**
1. Train a small network (e.g., 1 layer, narrow) on easy examples.
2. Use function-preserving transformations to grow the network (add layers/width).
3. New larger network starts with the same function as the smaller one.
4. Continue training on harder examples with the larger model.

**Net2WiderNet:** Increases layer width while preserving function.
**Net2DeeperNet:** Adds layers while preserving function (initialize new layers as identity).

**For our project:** Could start with a minimal 1-layer model learning 1-digit addition (~100 params), then grow to handle progressively more digits. This naturally implements curriculum learning at the architecture level.

**Source:** https://arxiv.org/pdf/1511.05641

---

### 3.5 Self-Paced Learning

**References:**
- Kumar et al., "Self-Paced Learning with Diversity" (NeurIPS 2010)
- "On The Power of Curriculum Learning in Training Deep Networks" (2019) — https://arxiv.org/abs/1904.03626

**How it works:**
1. Model starts by training on examples with lowest loss (easiest).
2. Gradually includes harder examples as training progresses.
3. The pace is determined by the model's own performance.
4. No external difficulty measure needed.

**For our project:** Self-paced learning for addition could naturally discover that single-digit additions are easy (low loss first), then progressively tackle longer additions. The difficulty ordering emerges from the model's own learning dynamics.

---

## 4. Grokking

### 4.1 The Original Grokking Paper

**Paper:** Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" (2022)
- **URL:** https://arxiv.org/abs/2201.02177

**Key discovery:** Neural networks trained on small algorithmic datasets (modular arithmetic) first memorize the training data (reaching ~100% training accuracy with ~0% test accuracy), then — after vastly more training — suddenly achieve perfect generalization.

**Experimental setup:**
- Small transformers on operations like modular addition: (a + b) mod p for prime p.
- Train/test split of the full operation table.
- Training continued far beyond overfitting.

**Key observations:**
- Smaller datasets require more optimization steps for generalization.
- The transition from memorization to generalization is sudden, not gradual.
- Weight decay appears essential for triggering grokking.

---

### 4.2 Mechanisms of Grokking

**Key papers:**
- "Towards Understanding Grokking: An Effective Theory of Representation Learning" (NeurIPS 2022) — https://papers.neurips.cc/paper_files/paper/2022/file/dfc310e81992d2e4cedc09ac47eff13e-Paper-Conference.pdf
- "Grokking as a First Order Phase Transition" (ICLR 2024) — https://proceedings.iclr.cc/paper_files/paper/2024/file/682f87a8c306098ec8be29019bd76aa4-Paper-Conference.pdf
- "Grokking at the Edge of Numerical Stability" (ICLR 2025) — https://arxiv.org/abs/2501.04697

**Three-phase model:**
1. **Memorization:** Network learns training data via brute-force lookup-table-like circuits.
2. **Circuit formation:** Network develops a generalizable algorithm (e.g., Fourier-based computation for modular arithmetic).
3. **Cleanup:** Weight decay removes the memorization components, and the generalizing circuit dominates. This is when test accuracy suddenly jumps.

**Weight decay's role:**
- Weight decay creates a "pressure" toward simpler solutions.
- Larger weight decay generally increases the parameter region where grokking occurs.
- The Goldilocks Zone: weight norms need to be in a narrow range for generalization.
- Some evidence that weight decay may not be strictly necessary — initial weight norm and learning rate can also control grokking by affecting whether the model is in the "lazy" vs. "rich" training regime.

**Phase transition perspective:**
- Grokking can be viewed as a first-order phase transition.
- Information-theoretic progress measures reveal it as an emergent phase transition.
- Weight decay, weight initialization, and data distribution all affect the transition timing.

---

### 4.3 Omnigrok: Grokking Beyond Algorithmic Data

**Paper:** Liu et al., "Omnigrok: Grokking Beyond Algorithmic Data" (ICLR 2023)
- **URL:** https://arxiv.org/abs/2210.01117

**Key finding:** Grokking is not limited to algorithmic datasets. It occurs in image classification, sentiment analysis, and molecular property prediction.

**The "LU Mechanism":**
- Training loss vs. weight norm resembles an "L" shape (low loss at any norm).
- Test loss vs. weight norm resembles a "U" shape (low loss only at specific norm range).
- Grokking occurs when optimization traverses from the "L" region to the "U" minimum.
- The severity depends on how much the task relies on learned representations.

---

### 4.4 Mechanistic Interpretability of Grokking

**Reference:** Nanda et al., "A Mechanistic Interpretability Analysis of Grokking" — https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB

**How models actually learn modular arithmetic:**
- The network learns to perform modular addition using discrete Fourier transforms.
- Specific trigonometric identities are discovered by the network.
- This provides strong evidence that grokking leads to genuine algorithm discovery, not just better pattern matching.

---

### 4.5 Does Grokking Happen at Sub-1000 Parameter Scale?

**Assessment based on literature:**

Grokking was originally observed in small transformers (hundreds of thousands of parameters). Key considerations for sub-1000 parameters:

1. **Overparameterization is key:** Grokking requires the model to first memorize, then generalize. With <500 parameters, the model may not have enough capacity to fully memorize large operation tables, which could prevent the memorization phase.

2. **Modular arithmetic studies** typically use models with 10K-100K+ parameters on datasets with ~10K examples. Sub-500 parameters would need proportionally smaller datasets.

3. **Evidence suggests:** For very small models on very small tasks (e.g., mod 7 addition with ~50 examples), grokking-like phenomena may occur, but the dynamics could differ significantly. The model may skip the extended memorization phase due to capacity constraints.

4. **Practical implication:** Rather than waiting for classical grokking, tiny models may benefit from the related insight that **very long training with weight decay can improve generalization even when training loss has plateaued**. This is a weaker form of the same phenomenon.

---

## 5. Training Dynamics

### 5.1 Loss Landscape Visualization

**Paper:** Li et al., "Visualizing the Loss Landscape of Neural Nets" (NeurIPS 2018)
- **URL:** https://arxiv.org/abs/1712.09913

**Key findings:**
- Shallow networks have smooth landscapes with wide, convex regions.
- Deeper networks have increasingly chaotic, non-convex landscapes.
- Skip connections dramatically convexify the loss landscape.
- "Filter normalization" enables meaningful comparisons between loss landscapes.

**Mode Connectivity:**
- **Paper:** Garipov et al., "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs" (NeurIPS 2018) — https://arxiv.org/abs/1802.10026
- Different optima are connected by simple curves of nearly constant loss.
- This property enables weight averaging and ensemble techniques.

**For tiny models:** With few parameters, loss landscapes can be visualized more completely. A 2D projection of a 500-parameter space captures more structure than the same projection of a million-parameter space. This makes loss landscape analysis particularly informative for our use case.

---

### 5.2 Stochastic Weight Averaging (SWA)

**Paper:** Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization" (2018)
- **URL:** https://arxiv.org/abs/1803.05407

**How it works:**
1. Train normally for initial period (e.g., 75% of training).
2. Switch to cyclical or constant learning rate.
3. Average the weights of checkpoints collected during the final 25%.
4. The averaged model tends to lie in wider minima.

**Key benefits:**
- Finds much flatter solutions than SGD alone.
- Essentially no computational overhead (just maintain a running average).
- Drop-in replacement for standard training.
- Built into PyTorch: `torch.optim.swa_utils`.

**Related: Latest Weight Averaging (LAWA)**
- "When, Where and Why to Average Weights?" (2025) — https://arxiv.org/abs/2502.06761
- Online algorithm that averages the latest k checkpoints in a rolling window.
- Significant speedups on vision and language tasks.

**For tiny models:** SWA is nearly free and consistently improves generalization. For tiny models trained for many epochs, checkpoint averaging over the last N epochs could significantly improve test performance, especially if the model oscillates near multiple solutions.

---

### 5.3 Loss Functions Beyond Cross-Entropy

#### Label Smoothing
- Instead of hard 0/1 targets, use soft targets (e.g., 0.1/0.9).
- Prevents overconfidence, improves calibration and generalization.
- Lower conditioning number leads to faster convergence.
- **Source:** https://arxiv.org/abs/2402.03979

#### Focal Loss
- **Paper:** Lin et al., "Focal Loss for Dense Object Detection" (2017)
- Down-weights loss for well-classified examples, focusing on hard examples.
- Implicitly regularizes via an entropy term.
- Better calibrated predictions than cross-entropy.
- **Source:** https://arxiv.org/abs/1708.02002

**For tiny models learning arithmetic:**
- **Label smoothing** may help prevent the model from being too confident about incorrect carry predictions.
- **Focal loss** could help the model focus on the difficult cases (e.g., additions with long carry chains) without explicitly implementing curriculum learning.
- Standard cross-entropy remains a strong baseline.

---

### 5.4 Batch Size Effects on Generalization

**Key references:**
- "Revisiting Small Batch Training for Deep Neural Networks" (2018) — https://arxiv.org/abs/1804.07612
- Google Tuning Playbook — https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/faq

**Key findings:**
- **Smaller batch sizes generally lead to better generalization.** Best results consistently at batch size 32 or smaller.
- Small batches introduce gradient noise that acts as implicit regularization.
- Small batches converge to flat minima; large batches converge to sharp minima.
- Flat minima generalize better because they are robust to distribution shift.
- Trade-off: small batches are noisy (higher variance in training metrics) but find better solutions.

**For tiny models:**
- With a sub-500 parameter model, very small batch sizes (2-8) may be optimal.
- The gradient noise from small batches provides regularization that tiny models desperately need.
- Training is so fast that the reduced throughput from small batches is irrelevant.

---

### 5.5 Gradient Accumulation for Tiny Models

**Reference:** "Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful" (2025) — https://arxiv.org/abs/2507.07101

**Key insight:** Gradient accumulation may be unnecessary and wasteful for small batch training when hyperparameters (especially Adam's beta2) are properly configured.

**For tiny models:** Gradient accumulation is almost certainly unnecessary. With <500 parameters, even batch size 1 fits trivially in memory. The gradient noise from small batches is beneficial. Use small batch sizes directly rather than accumulating.

---

### 5.6 Training for 100K+ Epochs on Small Models

**Context from grokking literature:**

The conventional wisdom is that training beyond convergence leads to overfitting. However, grokking research shows that:

1. **With proper regularization (weight decay), extremely long training can lead to sudden generalization improvements.** This is the grokking phenomenon.
2. **The key is weight decay:** Without it, long training past convergence degrades performance. With it, the model continues to simplify its learned solution.
3. **Monitoring:** Track both training and test accuracy. If training accuracy is high but test accuracy is low, continue training (with weight decay) — grokking may occur.
4. **SWA over long training:** Combine long training with checkpoint averaging over the last 20-50% of training to capture the best generalization window.

**For tiny models:**
- 100K epochs on a 500-parameter model is computationally trivial (minutes to hours).
- Use weight decay = 0.1 to 1.0 (much higher than typical large-model settings).
- Monitor for delayed generalization. Be prepared to train 10-100x longer than when training loss converges.
- Consider periodic evaluation every 1000 epochs to catch grokking transitions.

---

### 5.7 Learning Rate Warmup

**Key reference:** "Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training" (2024) — https://arxiv.org/abs/2410.23922

**Why warmup is critical:**
1. Prevents attention entropy collapse in transformers.
2. Gradually reduces sharpness of the loss landscape.
3. Allows Adam's moment estimates to stabilize before taking large steps.
4. Models move from poorly conditioned to well-conditioned regions.

**For tiny transformers:**
- Even a short warmup (10-50 steps) stabilizes training.
- Without warmup, small transformers can diverge immediately.
- "Taming Transformer without using learning rate warmup" (2025, https://arxiv.org/abs/2505.21910) proposes alternatives, but warmup remains the simplest approach for tiny models.

---

## 6. Physics of LLMs / WeightWatcher

### 6.1 WeightWatcher Tool

**Tool:** WeightWatcher — https://weightwatcher.ai/
**GitHub:** https://github.com/CalculatedContent/WeightWatcher
**Key Paper:** Martin et al., "Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data" (Nature Communications, 2021) — https://www.nature.com/articles/s41467-021-24025-8

**What it does:**
WeightWatcher analyzes the weight matrices of trained neural networks to assess quality without needing training or test data. It uses Random Matrix Theory (RMT) and the theory of Heavy-Tailed Self-Regularization (HT-SR).

**How it works:**
1. Compute the empirical spectral density (ESD) of each weight matrix.
2. Fit a power law to the tail of the ESD.
3. The power-law exponent alpha (α) is the key quality metric.

**Interpreting alpha:**
| Alpha Range | Interpretation |
|-------------|---------------|
| α ≈ 2.0 | Excellent layer quality (heavy-tailed, well-trained) |
| 2.0 < α < 6.0 | Acceptable quality |
| α > 6.0 | Poorly trained or over-parameterized layer |
| Random-looking ESD | Layer may not be contributing useful computation |

**Key capabilities:**
- Predict test accuracy without test data.
- Detect over-training or over-parameterization.
- Monitor training progress.
- Compare model quality across architectures.

---

### 6.2 Heavy-Tailed Self-Regularization Theory

**Paper:** Martin & Mahoney, "Traditional and Heavy-Tailed Self Regularization in Neural Network Models" (2019) — https://arxiv.org/abs/1901.08276

**Core theory:**
- Well-trained networks develop heavy-tailed distributions in their weight matrix spectra.
- This is analogous to self-organization in statistical physics of disordered systems.
- The heavy tails indicate that the network has found efficient representations.
- State-of-the-art DNNs exhibit a novel form of self-regularization visible in the spectral properties.

**Phases of training quality:**
1. **Random-like:** Marchenko-Pastur distribution (untrained/poorly trained).
2. **Bulk + spikes:** Some learned structure emerging.
3. **Heavy-tailed:** Well-trained, efficient representations.
4. **Very heavy-tailed (α → 2):** Optimal training.

---

### 6.3 Implications for Tiny Model Training

**Key considerations:**

1. **Small matrices, limited spectral analysis:** With weight matrices of size 16x16 or 32x8, the empirical spectral density has very few points. Power-law fitting may be unreliable.

2. **Qualitative guidance still valuable:** Even if quantitative alpha values are noisy, the direction of change during training (increasing vs. decreasing alpha) can indicate whether training is improving representation quality.

3. **Alternative metrics for tiny models:**
   - Effective rank of weight matrices.
   - Singular value spread (ratio of largest to smallest non-zero singular value).
   - Weight matrix norm trajectories during training.

4. **Practical use:** Run WeightWatcher on checkpoints during training to monitor whether the model is developing structured (heavy-tailed) or random weight distributions. A transition from random to structured correlates with developing useful representations.

**Installation:** `pip install weightwatcher`

---

## 7. Architecture Search

### 7.1 Neural Architecture Search (NAS) Overview

**Key references:**
- Elsken et al., "Neural Architecture Search: A Survey" (2019) — https://arxiv.org/abs/1808.05377
- Lilian Weng, "Neural Architecture Search" (2020) — https://lilianweng.github.io/posts/2020-08-06-nas/

**Three components of NAS:**
1. **Search space:** What architectures are possible.
2. **Search strategy:** How to explore the space.
3. **Performance estimation:** How to evaluate candidates.

**For tiny models, the search space is unusually constrained:**
- Total parameters < 500
- Must include embedding, attention, and output layers
- Architectural choices: number of layers, heads, hidden dimension, FFN ratio

---

### 7.2 Hyperparameter Optimization: Optuna

**Tool:** Optuna — https://optuna.org/
**Documentation:** https://optuna.readthedocs.io/en/stable/

**Why Optuna:**
- Uses TPE (Tree-structured Parzen Estimator) by default — a Bayesian optimization algorithm.
- Finds optimal hyperparameters in ~67 iterations vs. hundreds for random search.
- Supports pruning of unpromising trials (early stopping).
- Define-by-run API: search space defined programmatically.

**Bayesian vs. Random Search:**
- Random search explores uniformly; Bayesian builds a model of the objective.
- Bayesian finds better solutions in fewer trials.
- For expensive evaluations, Bayesian is clearly better.
- For cheap evaluations (tiny models), random search with more trials can be competitive.

**For tiny models:** Since each trial takes only seconds to minutes, you can run thousands of trials. Random search with 1000 trials may outperform Bayesian optimization with 100 trials. However, Optuna's TPE with 1000 trials is strictly better.

**Source:** https://neptune.ai/blog/optuna-vs-hyperopt

---

### 7.3 Ray Tune

**Tool:** Ray Tune — https://docs.ray.io/en/latest/tune/index.html

**Key features:**
- Distributed parallel hyperparameter search.
- ASHA (early stopping of unpromising trials), BOHB, PBT built in.
- Scales from local machine to cluster without code changes.
- Integrates with Optuna's TPE.

**For tiny models:** Ray Tune can parallelize thousands of tiny model trainings across available GPUs/CPUs. With sub-500 parameter models, hundreds of trials can run concurrently even on a single GPU.

---

### 7.4 Evolutionary Architecture Search (NEAT)

**Paper:** Stanley & Miikkulainen, "Evolving Neural Networks through Augmenting Topologies" (2002)
- **URL:** https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

**How NEAT works:**
1. Start with minimal topology (no hidden layers).
2. Evolve by adding nodes and connections via mutation.
3. Protect innovations via speciation.
4. Historical markings track gene lineage for crossover.

**For tiny models:** NEAT is uniquely suited for sub-500 parameter models because:
- It starts from minimal topologies naturally.
- It adds complexity only when beneficial.
- Speciation protects novel structures from being eliminated too early.
- The entire population stays small, making it computationally trivial.

**Modern variants:**
- HyperNEAT: Evolves connectivity patterns.
- ES-HyperNEAT: Adaptive resolution.
- CoDeepNEAT: Combines blueprint and module evolution.

---

### 7.5 Efficient Search for Sub-500 Parameter Models

**Practical strategy:**

Given the extremely constrained parameter budget, the search space is actually small:

| Hyperparameter | Range | Notes |
|---------------|-------|-------|
| d_model | 8, 12, 16, 20, 24 | Embedding dimension |
| n_layers | 1, 2, 3 | Transformer layers |
| n_heads | 1, 2, 4 | Attention heads (must divide d_model) |
| d_ff | 16, 24, 32, 48, 64 | FFN hidden dimension |
| Vocabulary | 12-14 | Digits 0-9 + special tokens |
| Max sequence length | 25-35 | Input + output length |

**Parameter counting constraint:** Each configuration must be validated against the <500 parameter budget. For example:
- Embedding: vocab_size * d_model (e.g., 13 * 16 = 208)
- Attention: 4 * d_model^2 per layer (e.g., 4 * 256 = 1024 for d_model=16) — this already exceeds 500!
- **This means:** Extreme parameter sharing, weight tying, or non-standard architectures are likely needed.

**Search approach recommendation:**
1. **Phase 1:** Random search over 500-1000 configurations with Optuna to map the feasible space.
2. **Phase 2:** Bayesian optimization focused on the promising region.
3. **Phase 3:** PBT to jointly evolve architecture and training hyperparameters.

---

## 8. Synthesis: Recommended Training Strategy for Sub-500 Parameter Arithmetic Transformer

Based on the literature review, here is a recommended integrated training strategy:

### Phase 1: Architecture Search (1-2 days)
1. Use Optuna to search the constrained architecture space.
2. Validate each config fits within the parameter budget.
3. Evaluate using short training runs (1000 epochs) with AdamW.
4. Identify top 5-10 architectures.

### Phase 2: Training Recipe Optimization (1-2 days)
1. Use PBT with a population of 50-100 models.
2. Evolve: learning rate, weight decay, batch size, data curriculum.
3. Train for 10K+ epochs; evaluate generalization periodically.
4. Consider Abacus-style embeddings or reverse output format.

### Phase 3: Extended Training (ongoing)
1. Take the best recipe from Phase 2.
2. Train for 100K+ epochs with weight decay = 0.1-1.0.
3. Apply SWA over the last 20% of training.
4. Monitor for grokking (sudden jumps in test accuracy).
5. Use WeightWatcher to assess layer quality during training.

### Key Optimizer Choices
- **Primary:** AdamW or Schedule-Free AdamW
- **Experimental:** SAM (for flat minima), Lion (for implicit regularization), Muon (for weight matrices)
- **Meta:** PBT to let the optimizer choice evolve

### Key Data Choices
- **Format:** Reverse output or chain-of-thought (not plain left-to-right)
- **Curriculum:** Start with 1-digit, progress to 10-digit as accuracy improves
- **Sampling:** Over-sample hard examples (long carry chains)
- **Embeddings:** Consider Abacus-style positional encoding for digit significance

### Key Regularization
- **Weight decay:** 0.1-1.0 (much higher than typical)
- **Small batch size:** 4-16
- **Label smoothing:** 0.05-0.1
- **SWA:** Apply in final training phase

---

## 9. Key Papers Reference Table

| Paper | Year | Topic | URL |
|-------|------|-------|-----|
| Power et al., "Grokking" | 2022 | Delayed generalization | https://arxiv.org/abs/2201.02177 |
| Liu et al., "Omnigrok" | 2023 | Grokking beyond algorithmic data | https://arxiv.org/abs/2210.01117 |
| Lee et al., "Teaching Arithmetic to Small Transformers" | 2023 | Data format, curriculum | https://arxiv.org/abs/2307.03381 |
| McLeish et al., "Arithmetic with Right Embeddings" | 2024 | Abacus embeddings | https://arxiv.org/abs/2405.17399 |
| Jordan et al., "Muon" | 2024 | Optimizer for hidden layers | https://kellerjordan.github.io/posts/muon/ |
| Liu et al., "Muon is Scalable" | 2025 | Scaling Muon | https://arxiv.org/abs/2502.16982 |
| Chen et al., "Lion" | 2023 | Evolved sign momentum | https://arxiv.org/abs/2302.06675 |
| Liu et al., "Sophia" | 2023 | Second-order optimizer | https://arxiv.org/abs/2305.14342 |
| Foret et al., "SAM" | 2021 | Sharpness-aware minimization | https://openreview.net/forum?id=6Tm1mposlrM |
| Defazio et al., "Schedule-Free" | 2024 | No LR schedule needed | https://arxiv.org/abs/2405.15682 |
| Loshchilov & Hutter, "AdamW" | 2019 | Decoupled weight decay | https://arxiv.org/abs/1711.05101 |
| Loshchilov & Hutter, "SGDR" | 2017 | Cosine annealing warm restarts | https://arxiv.org/abs/1608.03983 |
| Smith, "Super-Convergence" | 2019 | One-cycle policy | https://arxiv.org/abs/1708.07120 |
| Smith, "Cyclical LR" | 2017 | Cyclic learning rates | https://arxiv.org/abs/1506.01186 |
| Izmailov et al., "SWA" | 2018 | Stochastic weight averaging | https://arxiv.org/abs/1803.05407 |
| Martin et al., "WeightWatcher" | 2021 | Model quality diagnostics | https://www.nature.com/articles/s41467-021-24025-8 |
| Martin & Mahoney, "HT-SR" | 2019 | Heavy-tailed self-regularization | https://arxiv.org/abs/1901.08276 |
| Li et al., "Visualizing Loss Landscapes" | 2018 | Loss surface analysis | https://arxiv.org/abs/1712.09913 |
| Jaderberg et al., "PBT" | 2017 | Population-based training | https://arxiv.org/abs/1711.09846 |
| Stanley & Miikkulainen, "NEAT" | 2002 | Neuroevolution | https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf |
| Chen et al., "Net2Net" | 2016 | Function-preserving growth | https://arxiv.org/abs/1511.05641 |
| Nanda et al., "Mechanistic Grokking" | 2023 | Fourier circuits in grokking | https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB |

---

## 10. Open Questions and Areas for Experimentation

1. **Does Muon help for very small weight matrices (e.g., 16x8)?** The orthogonalization may behave differently than on large matrices.

2. **Can grokking be triggered with <500 parameters?** The model may lack capacity for the memorization phase. Needs empirical testing on mod-7 or mod-11 arithmetic.

3. **Optimal weight decay for tiny models?** Literature suggests 0.01-0.1 for large models. Tiny models may need 0.5-2.0 for proper regularization.

4. **Abacus embeddings vs. reverse output format?** Both address the digit alignment problem. In a tiny model with a severe parameter budget, which is more parameter-efficient?

5. **Is label smoothing redundant with weight decay?** Both regularize. For tiny models, the interaction may be different than for large models.

6. **Can NEAT-style evolution discover a better-than-transformer architecture for arithmetic within the parameter budget?** The standard transformer may not be the optimal architecture at this scale.

7. **How many training epochs to grok?** With strong weight decay and small data, can grokking happen within 10K epochs, or does it require 100K+?
