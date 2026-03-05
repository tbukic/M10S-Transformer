# Ideas to Try — Pushing Below 60 Parameters

## 1. Matrix Tying Strategies

Our current tying: `tieKV` (K=V), `tieQO` (O=Q^T). These save 12p each.
The key question: what other algebraically motivated relationships between matrices
can reduce parameters while preserving (or helping) trainability?

### 1A. Transpose Tying: M & M^T
- **Already used**: tieQO sets O = Q^T
- **Why it works**: transpose preserves singular values. If M maps from space A→B,
  M^T maps B→A with the same "strength" in each direction.
- **Where else to try**: tie gate = up^T in SwiGLU (gate and up have same shape d→ff)

### 1B. Negated Transpose: M & -M^T
- **Idea**: B = -A^T makes the combined [A; -A^T] an antisymmetric/skew-symmetric structure
- **Property**: skew-symmetric matrices have purely imaginary eigenvalues → rotational dynamics
- **Where to try**: Q = -K^T (attention becomes Q·K = Q·(-Q^T)^T = -Q·Q, antisymmetric)
- **Intuition**: might force attention to compute relative (signed) relationships between positions
- **Risk**: may kill positive attention scores needed for softmax

### 1C. Rotation Tying: B = R(φ)·A
- **Idea**: one matrix is a rotated version of another, with φ as a single learnable param
- **For 2D rotation**: R(φ) = [[cos φ, -sin φ], [sin φ, cos φ]], costs 1 extra param
- **For 3D**: use Rodrigues rotation (axis-angle), costs 3 extra params (axis) + 1 (angle) = 4
- **Simpler 3D**: fix rotation axis (e.g., always around [1,1,1]/√3), costs 1 param
- **Where to try**: gate = R(φ)·up in SwiGLU, or K = R(φ)·Q in attention
- **Property**: preserves norms, just reorients. The model learns "how much to rotate"
- **Especially natural for**: RoPE already uses rotations! tying via rotation extends this principle

### 1D. Scaled Identity Offset: B = αI - M^T
- **Idea**: B is the "complement" of M in some sense
- **Property**: if M has eigenvalues λ_i, then B has eigenvalues α - λ_i*
- **Intuition**: B "fills in" what M misses. Together they span the full space
- **Where to try**: down_proj = αI - up_proj^T (MLP learns residual around scaled identity)
- **Variant**: B = I - M (α=1, no extra param)
- **Savings**: full matrix → 1 param (just α) if we derive B from existing M

### 1E. Hadamard/Element-wise Tying: B = M ⊙ S
- **Idea**: B shares M's structure but with a learned element-wise scale/mask S
- **If S is rank-1**: S = u·v^T, then B = M ⊙ (u·v^T), costs len(u)+len(v) extra params
- **If S is diagonal**: costs d extra params
- **If S is scalar**: B = α·M, costs 1 param

### 1F. Scalar Multiple: B = α·M
- **Simplest possible tying**: same matrix, different scale
- **Where to try**: gate = α·up (saves ff*d - 1 params = 5p for d=3,ff=2)
- **Intuition**: gate controls "how much" while up controls "what direction"
- **This is close to what lokimorty does**: gate weight derived from up weight algebraically

### 1G. Rank-1 Perturbation: B = M + u·v^T
- **Idea**: B is "almost" M, with a rank-1 correction
- **Cost**: len(u) + len(v) extra params
- **Where to try**: K = Q + u·v^T (K is Q plus a small correction)
- **Property**: preserves most of M's structure while allowing targeted adjustment

### 1H. Shared Basis / Factored Tying: M = U·D1·V^T, N = U·D2·V^T
- **Idea**: two matrices share their singular vectors but have different singular values
- **Cost**: U (d×r) + V (d×r) shared + 2×r diagonal entries
- **Property**: M and N operate in the same subspace but with different magnitudes
- **Where to try**: Q and K share basis, differ only in scaling per dimension
- **This is what QK norms partially do**: they normalize Q and K to unit norm per head

### 1I. Exponential Map: B = exp(M) or B = exp(M^T)
- **Property**: exp of skew-symmetric = orthogonal matrix → B is always a valid rotation
- **Where to try**: if M is small (3×3), exp(M) is cheap to compute
- **Saves**: entire matrix (B is fully determined by M)
- **Risk**: gradient flow through matrix exponential can be tricky

### 1J. Cayley Transform: B = (I - M)(I + M)^{-1}
- **Property**: if M is skew-symmetric, B is orthogonal (a rotation/reflection)
- **Cost**: B is fully determined by M (0 extra params)
- **Where to try**: tie output projection to be the Cayley transform of value projection
- **Advantage over exp**: easier gradients, guaranteed orthogonality

### 1K. Triple/Multi-Duty Tying (evindor's approach)
- **evindor ties**: V_proj = head_proj^T = fc2 (one 10×d matrix serves 3 roles)
- **Savings**: massive — 3 matrices for the price of 1
- **Why it works**: the output space (10 logits) is the same for value, output, and FFN-to-logit
- **Where to try**: any matrices that share shape AND conceptual role
  - head_proj and embed (already done via tied embeddings)
  - V and O when they have compatible shapes
  - down_proj and embed^T if ff matches vocab or d

### 1L. Polynomial Tying: B = p(M) for low-degree polynomial p
- **Idea**: B = αM² + βM + γI (quadratic in M)
- **Cost**: 3 params (α, β, γ) instead of d×d
- **Property**: B shares eigenvectors with M; eigenvalues are polynomial transform
- **Where to try**: K = αQ² + βQ (quadratic relationship between Q and K)

## 2. Embedding Reduction Ideas

### 2A. Outer Product (Rank-1) Embedding: E = u ⊗ v^T
- Standard 10×3 embedding: 30 params
- Rank-1: u (10-dim) × v (3-dim) = 10+3 = 13 params (saves 17p!)
- **Problem**: rank-1 means all rows are scaled versions of v → poor digit separation
- **Fix**: rank-2 = u1⊗v1 + u2⊗v2 = 2×(10+3) = 26 params (only 4p savings)
- **Better fix**: parametric formula (circular arc = 3 params, quadratic = 2 params)

### 2B. Quadratic Embedding (lokimorty, 2 params)
- `e(d) = [c0 - c1*d², -d]` for d=2
- For d=3: `e(d) = [c0 - c1*d², -d, c2*d]` → 3 params
- **Advantage over circular arc**: no trig functions, simpler gradients
- **Our circular arc**: `e(d) = A*[cos(start + i*stride), sin(start + i*stride)]` → 3 params (A, start, stride)
- **Worth trying**: quadratic embedding for d=3

### 2C. Frozen Positional Encoding
- evindor uses 3D frozen sinusoidal positions (0 learned params)
- We already have 0 position params (RoPE is fixed)
- **No savings here** — we're already optimal on position params

### 2D. Learnable RoPE Theta
- Currently theta=3.0 is a hyperparameter
- Making theta learnable adds 1 param but lets the model find optimal period
- **Worth a sweep**: theta ∈ {1, 2, 3, 5, 10, 19, 100}
- Period-19 is what hand-coded models use (for their format)

### 2E. Random Fixed Embedding (0 params)
- Initialize 10×3 embedding randomly, freeze it
- The model must learn everything else around this fixed basis
- **Unlikely to work for training** but worth a sanity check
- If it works: would save ALL embedding params

## 3. Training Technique Ideas

### 3A. Grokfast-EMA (HIGHEST PRIORITY)
- 10 lines of code, >50x grokking speedup on modular arithmetic
- `alpha=0.98, lambda=2.0, WD=0.005`
- Amplifies slow-varying gradient components (the "generalizing" direction)
- Source: [arXiv:2405.20233](https://arxiv.org/abs/2405.20233)

### 3B. Carry-Mix Curriculum (REVISIT)
- 80% carry-heavy samples, LINEAR fade to 0% over steps 15K-45K
- MicroAdder: 3/3 convergence at 74p vs ~10% without
- **We tried this before and it FAILED** — but we may have used wrong fade schedule
- KEY: must be step-based linear fade, NOT metric-triggered

### 3C. Digit-Count Curriculum
- Steps 0-5K: 1-3 digit operands
- Steps 5K-15K: 1-6 digits
- Steps 15K+: full 10 digits
- Combine with carry-mix for both diversity and difficulty scheduling

### 3D. EMA Model Averaging
- `ema_decay=0.999` on model weights during training
- Already used for 62p, extend to all configs

### 3E. Weight Decay Sweep (with Grokfast)
- Test WD ∈ {0.005, 0.01, 0.02, 0.05} with Grokfast enabled
- Synergistic effect: Grokfast + WD accelerates more than either alone

### 3F. Egalitarian Gradient Descent (EGD)
- Normalize gradient singular values: G_hat = (G·G^T)^(-1/2) · G
- Simplified: column-normalize gradients (5 lines)
- Near-perfect accuracy in 5-10 epochs on modular arithmetic
- Source: [arXiv:2510.04930](https://arxiv.org/abs/2510.04930)

### 3G. Parameterless RMSNorm
- Remove learned scale from RMSNorm → divide by RMS only
- Saves 3p (d_model per norm layer), or 6p if applied to all norms
- Hand-coded models prove this is sufficient
- Keep final norm learnable (it acts as output channel scaling)

## 4. Architecture Ideas (from competition)

### 4A. d=5 Split Architecture (evindor)
- 2D for token (circular arc) + 3D for frozen sinusoidal position
- Orthogonal subspaces → attention can cleanly separate routing from content
- 57 params, 100% accuracy, 44K steps
- **Gap from our approach**: we mix token and position in d=3

### 4B. Triple-Duty Head Projection (evindor)
- One matrix serves as V projection, FFN fc2, AND output head
- Massive parameter savings
- Requires careful architectural design to make shapes compatible

### 4C. Phase-Rotation Q=K Tying (evindor)
- Single projection for both Q and K, differentiated by a 1-param phase rotation
- Saves an entire projection matrix minus 1 param
- We tie Q=O and K=V; they tie Q=K differently

### 4D. Period-19 RoPE (hand-coded models)
- Most hand-coded models use RoPE with period 19 for digit routing
- Our theta=3.0 may not be optimal
- Worth sweeping theta systematically

## 5. Priority Order for Experiments

1. **Grokfast-EMA** on 56p/68p/80p (fastest to implement, highest expected impact)
2. **Matrix tying exploration** on 68-80p configs (start with gate=α·up, then rotation tying)
3. **Carry-mix curriculum** with correct linear fade schedule
4. **RoPE theta sweep** on a fast-grokking config (80p or 68p)
5. **Quadratic embedding** vs circular arc comparison
6. **Parameterless norms** on 68p (saves 3-6p)
7. **Triple-duty head_proj** — requires architectural rethinking

## 6. What Definitely Doesn't Work

- **d=2 anything**: degenerate loss landscape, 0% across ALL configs/LR/batch
- **head_dim=2**: representationally sufficient but SGD can't find it (0% in 15 runs)
- **ReLU MLP at d=3**: all dead in wave2 experiments
- **Carry-mix with metric-based fade**: creates destructive oscillation
- **Shared layers rep≥3**: dead after 400K steps
- **Targeted FT on arc62p**: oscillates, doesn't converge (embedding too constrained)
- **ALiBi**: 0 param savings over RoPE, and RoPE matches hand-coded models better
