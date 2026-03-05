# AdderBoard Submission: 58 params

## Issue Title
[Submission] 58 params - Tom Bukic

## Fields

**Author:** Tom Bukic

**Unique Parameter Count:** 58

**Accuracy:** 100.00% on 10,010 tests (verify.py seed=2025: 10K random + 10 edge cases)

**Method:** Grokfast-EMA continuation + iterated targeted fine-tuning

**Architecture:** 1L Qwen3 decoder + circular arc embedding, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU, RMSNorm

**Key Tricks:**
- Circular arc embedding (3 params instead of 30): digit tokens mapped via A*cos(start + stride*i)
- K = alpha * Q (learned scalar replaces 12-param key projection matrix)
- gate = alpha * up (learned scalar replaces 6-param gate projection in SwiGLU MLP)
- Tied O=Q^T (output projection = transpose of query projection, saves 12 params)
- Tied lm_head to dynamic embedding table
- RoPE positional encoding (zero trainable params, theta=3)
- QK norms (RMSNorm on Q and K, stabilizes training)
- SwiGLU MLP at ff=2
- Grokfast-EMA gradient filter (alpha=0.98, lambda=2.0) for breaking through plateaus
- Iterated targeted fine-tuning (1 iteration, 9 error pairs on 2K eval set)

**Link to Code:** https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_58p.py

**Verification Output:**
```
Model: M10S-58p
Author: Tom Bukic
Parameters (unique): 58
Architecture: 1L Qwen3 + circular arc embed, d=3, 1h/1kv, hd=4, ff=2, K=aQ, gate=a*up, tieQO
Tricks: Circular arc embedding (3 params instead of 30), K = alpha * Q (scalar replaces 12-param key projection), gate = alpha * up (scalar replaces 6-param gate projection), Tied Q=O projections (output = Q transposed), RoPE (zero params), QK norms, Grokfast-EMA (alpha=0.98, lambda=2.0), Iterated targeted fine-tuning (1 iteration, 9 error pairs)

Results: 10010/10010 correct (100.00%)
Time: 189.6s (53 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

**Additional Notes:**
- Parameter breakdown: base Qwen3 95p + 9*ff=18 - 27(arc embed) - 12(tieQO) - 11(K=aQ,+1scalar) - 5(gate=a*up,+1scalar) = 58
- Training pipeline: 290K cosine LR → 60K Grokfast-EMA (alpha=0.98, lambda=2.0) → targeted FT (1 iter, 9 error pairs)
- The Grokfast-EMA phase was the key breakthrough: pushed from 83.8% → 99.4% by amplifying slow-varying gradient components
- Verified 100% on independent 10K test set (seed=42)
- 55p variant with shared block norms also QUALIFIED at 99.99% (1 error on 10,010)
- Paper with full methodology: https://github.com/tbukic/M10S-Transformer/blob/master/paper/main.pdf
