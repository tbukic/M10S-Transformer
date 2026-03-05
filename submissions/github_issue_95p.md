# AdderBoard Submission: 95 params

## Issue Title
[Submission] 95 params - Tom Bukic

## Fields

**Author:** Tom Bukic

**Unique Parameter Count:** 95

**Accuracy:** 99.98% on 10,010 tests (verify.py seed=2025: 10K random + 10 edge cases)

**Method:** Cosine LR from random init + fine-tuning with circular arc embedding

**Architecture:** 1L decoder-only transformer decoder + circular arc embedding, d=3, 1h/1kv, hd=4, ff=3, RoPE theta=3, SwiGLU, RMSNorm

**Key Tricks:**
- Circular arc embedding (3 learnable params instead of 30 for lookup table)
- Tied lm_head to dynamically-generated embedding table
- RoPE positional encoding (zero trainable params, theta=3)
- QK norms (RMSNorm on Q and K)
- Cosine LR + fine-tuning

**Link to Code:** https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_95p.py

**Verification Output:**
```
Model: M10S-95p
Author: Tom Bukic
Parameters (unique): 95
Architecture: 1L decoder-only transformer + circular arc embedding, d=3, 1h/1kv, hd=4, ff=3, RoPE theta=3, SwiGLU
Tricks: Circular arc embedding (3 params instead of 30), Tied lm_head to dynamic embedding table, RoPE (zero params), QK norms, Cosine LR + fine-tuning

Results: 10008/10010 correct (99.98%)
Time: 38.5s (260 additions/sec)
Status: QUALIFIED (threshold: 99%)

Failures (2):
  9999999999 + 1 = 10000000000, got 9000000000
  6009530873 + 9292951182 = 15302482055, got 15292482055
```

**Additional Notes:**
- Circular arc embedding: token i is embedded as `[A*cos(i*t + p), A*sin(i*t + p), 1]` with 3 learnable params
- 2 remaining errors involve MSB carry (sums exceeding 10 billion)
- Inspired by evindor/MicroAdder's parametric embedding approach
- Paper: https://github.com/tbukic/M10S-Transformer/blob/main/paper/main.pdf
