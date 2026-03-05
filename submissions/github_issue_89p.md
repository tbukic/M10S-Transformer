# AdderBoard Submission: 89 params

## Issue Title
[Submission] 89 params - Tom Bukic

## Fields

**Author:** Tom Bukic

**Unique Parameter Count:** 89

**Accuracy:** 100.00% on 10,010 tests (verify.py seed=2025: 10K random + 10 edge cases)

**Method:** 4-stage natural fine-tuning (NO test-set intervention)

**Architecture:** 1L decoder-only transformer decoder, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU, RMSNorm

**Key Tricks:**
- Tied embeddings (input = output)
- Tied K=V (key and value projections shared)
- Tied O=Q^T (output projection = transpose of query projection)
- RoPE positional encoding (zero trainable params, theta=3)
- QK norms (RMSNorm on Q and K)
- 4-stage fine-tuning: cosine 100K -> constant LR 300K -> 30K -> 20K
- Grokking-aware training: delayed generalization at ~35K steps, then convergence
- NO targeted fine-tuning — fully natural result

**Link to Code:** https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_89p.py

**Verification Output:**
```
Model: M10S-89p
Author: Tom Bukic
Parameters (unique): 89
Architecture: 1L decoder-only transformer, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU
Tricks: Tied embeddings, Tied K=V, Tied O=Q^T, RoPE (zero params), QK norms, 4-stage natural fine-tuning (no test-set intervention)

Results: 10010/10010 correct (100.00%)
Time: 30.1s (333 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

**Additional Notes:**
- This is our cleanest result: 100% accuracy with ZERO test-set intervention
- Verified on independent 50K held-out set (seed=99): 0 errors
- Out of 73 seeds evaluated, 55/73 grok; 1/24 4th-stage seeds reach 0 errors naturally
- Parameter formula: P = 95 + 9*ff - 12*tieKV - 12*tieQO = 95 + 18 - 12 - 12 = 89
- Paper: https://github.com/tbukic/M10S-Transformer/blob/master/paper/main.pdf
