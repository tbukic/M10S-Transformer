# AdderBoard Submission: 62 params

## Issue Title
[Submission] 62 params - Tom Bukic

## Fields

**Author:** Tom Bukic

**Unique Parameter Count:** 62

**Accuracy:** 100.00% on 10,010 tests (verify.py seed=2025: 10K random + 10 edge cases)

**Method:** 4-stage training with circular arc embedding + Adam (no weight decay)

**Architecture:** 1L decoder-only transformer decoder + circular arc embedding, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU, RMSNorm

**Key Tricks:**
- Circular arc embedding (3 learnable params instead of 30 for lookup table)
- Tied K=V (key and value projections shared)
- Tied O=Q^T (output projection = transpose of query projection)
- Tied lm_head to dynamically-generated embedding table
- RoPE positional encoding (zero trainable params, theta=3)
- QK norms (RMSNorm on Q and K)
- 4-stage training: cosine → AdamW constant → AdamW constant → Adam (no wd) cosine

**Link to Code:** https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_62p.py

**Verification Output:**
```
Model: M10S-62p
Author: Tom Bukic
Parameters (unique): 62
Architecture: 1L decoder-only transformer + circular arc embedding, d=3, 1h/1kv, hd=4, ff=2, tieKV+tieQO, RoPE theta=3, SwiGLU
Tricks: Circular arc embedding (3 params instead of 30), Tied K=V projections (share key/value weights), Tied Q=O projections (output = Q transposed), Tied lm_head to dynamic embedding table, RoPE (zero params), QK norms, 4-stage training: cosine→AdamW→AdamW→Adam(no wd) cosine

Results: 10010/10010 correct (100.00%)
Time: 33.0s (303 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

**Additional Notes:**
- Circular arc embedding: token i is embedded as `[A*cos(i*t + p), A*sin(i*t + p), 1]` with 3 learnable params (A, t, p) — replaces the 30-param embedding lookup table
- The lm_head is dynamically generated from the arc embedding at each forward pass (not a separate parameter)
- Inspired by evindor/MicroAdder's parametric embedding approach
- Verified on independent 50K held-out set (seed=99)
- Paper: https://github.com/tbukic/M10S-Transformer/blob/main/paper/main.pdf
