# AdderBoard Submission: 55 params

## Issue Title
[Submission] 55 params - Tom Bukic

## Fields

**Author:** Tom Bukic

**Unique Parameter Count:** 55

**Accuracy:** 100.00% on 10,010 tests (verify.py seed=2025: 10K random + 10 edge cases)

**Method:** Grokfast-EMA continuation + iterated targeted fine-tuning

**Architecture:** 1L decoder-only transformer decoder + circular arc embedding, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU, RMSNorm

**Key Tricks:**
- Circular arc embedding (3 params instead of 30): digit tokens mapped via A*cos(start + stride*i)
- K = alpha * Q (learned scalar replaces 12-param key projection matrix)
- gate = alpha * up (learned scalar replaces 6-param gate projection in SwiGLU MLP)
- Tied O=Q^T (output projection = transpose of query projection, saves 12 params)
- Shared block RMSNorms (ln1 and ln2 share weights, saves 3 params)
- Tied lm_head to dynamic embedding table
- RoPE positional encoding (zero trainable params, theta=3)
- QK norms (RMSNorm on Q and K, stabilizes training)
- SwiGLU MLP at ff=2
- Grokfast-EMA gradient filter (alpha=0.98, lambda=2.0) for breaking through plateaus
- Iterated targeted fine-tuning (6 iterations total, 547 cumulative error pairs)

**Link to Code:** https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_55p.py

**Verification Output:**
```
Model: M10S-55p
Author: Tom Bukic
Parameters (unique): 55
Architecture: 1L decoder-only transformer + circular arc embed, d=3, 1h/1kv, hd=4, ff=2, K=aQ, gate=a*up, tieQO, shbnorm
Tricks: Circular arc embedding (3 params instead of 30), K = alpha * Q (scalar replaces 12-param key projection), gate = alpha * up (scalar replaces 6-param gate projection), Tied Q=O projections (output = Q transposed), Shared block RMSNorms (-3 params), RoPE (zero params), QK norms, Grokfast-EMA (alpha=0.98, lambda=2.0), Iterated targeted fine-tuning (6 iterations total, 547 cumulative error pairs)

Results: 10010/10010 correct (100.00%)
Time: 43.7s (229 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

**Additional Notes:**
- Parameter budget (55 total):
  - 3 circular arc embedding (A, start, stride)
  - 3 ln1 RMSNorm (shared with ln2 via shbnorm)
  - 1 k_alpha scalar (K = alpha * Q)
  - 12 q_proj (shared with o_proj via tieQO)
  - 12 v_proj
  - 4+4 QK norms (q_norm + k_norm)
  - 1 gate_alpha scalar (gate = alpha * up)
  - 6 up_proj
  - 6 down_proj
  - 3 final_norm
- Training pipeline: 295K cosine LR (81.2%) → 325K Grokfast-EMA (100% on 500-sample) → 6 iterations targeted FT (547 cumulative pairs)
- Grokfast-EMA was the key breakthrough: amplifies slow-varying gradient components, pushed from 81% → 95%+
- 58p variant (without shared block norms) also achieves 100%
- Paper with full methodology: https://github.com/tbukic/M10S-Transformer/blob/master/paper/main.pdf
