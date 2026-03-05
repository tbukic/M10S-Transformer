# AdderBoard Submission: 101 params

## Issue Title
[Submission] 101 params - Tom Bukic

## Fields

**Author:** Tom Bukic

**Unique Parameter Count:** 101

**Accuracy:** 100.00% on 10,010 tests (verify.py seed=2025: 10K random + 10 edge cases)

**Method:** Cosine LR training + targeted fine-tuning (2 error pairs)

**Architecture:** 1L decoder-only transformer decoder, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU, RMSNorm

**Key Tricks:**
- Tied embeddings (input = output)
- Tied O=Q^T (output projection = transpose of query projection)
- RoPE positional encoding (zero trainable params, theta=3)
- QK norms (RMSNorm on Q and K)
- Targeted fine-tuning (only 2 error pairs needed)

**Link to Code:** https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_101p.py

**Verification Output:**
```
Model: M10S-101p
Author: Tom Bukic
Parameters (unique): 101
Architecture: 1L decoder-only transformer, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU
Tricks: Tied embeddings, Tied O=Q^T, RoPE (zero params), QK norms, Targeted fine-tuning

Results: 10010/10010 correct (100.00%)
Time: 40.0s (250 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

**Additional Notes:**
- Parameter formula: P = 95 + 9*ff - 12*tieQO = 95 + 18 - 12 = 101
- Also verified on independent 50K held-out set (seed=99): 0 errors
- Paper: https://github.com/tbukic/M10S-Transformer/blob/main/paper/main.pdf
