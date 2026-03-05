# AdderBoard Submission: 96 params

## Issue Title
[Submission] 96 params - Tom Bukic

## Fields

**Author:** Tom Bukic

**Unique Parameter Count:** 96

**Accuracy:** 100.00% on 10,010 tests (verify.py seed=2025: 10K random + 10 edge cases)

**Method:** Cosine LR from random init + fine-tuning

**Architecture:** 1L decoder-only transformer decoder + rank-1 output projection, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU, RMSNorm

**Key Tricks:**
- Rank-1 output projection (7 params instead of 12 for full projection)
- Tied embeddings (input = output)
- Tied K=V (key and value projections shared)
- RoPE positional encoding (zero trainable params, theta=3)
- QK norms (RMSNorm on Q and K)
- Cosine LR schedule + fine-tuning

**Link to Code:** https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_96p.py

**Verification Output:**
```
Model: M10S-96p
Author: Tom Bukic
Parameters (unique): 96
Architecture: 1L decoder-only transformer + rank-1 output proj, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU
Tricks: Rank-1 output projection (7 params instead of 12), Tied embeddings, Tied K=V, RoPE (zero params), QK norms, Cosine LR schedule

Results: 10010/10010 correct (100.00%)
Time: 35.9s (279 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

**Additional Notes:**
- Rank-1 factorization: output projection W_o (4x3) decomposed as outer product of two vectors (4+3=7 params vs 12)
- Inspired by evindor/MicroAdder's rank-1 output approach
- Verified on independent 50K held-out set (seed=99)
- Paper: https://github.com/tbukic/M10S-Transformer/blob/main/paper/main.pdf
