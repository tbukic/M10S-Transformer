# AdderBoard Submission: 86 params

## Issue Title
[Submission] 86 params - Tom Bukic

## Fields

**Author:** Tom Bukic

**Unique Parameter Count:** 86

**Accuracy:** 100.00% on 10,010 tests (verify.py seed=2025: 10K random + 10 edge cases)

**Method:** L-BFGS refinement + single-shot targeted fine-tuning (29 error pairs)

**Architecture:** 1L Qwen3 decoder, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU, RMSNorm

**Key Tricks:**
- Tied embeddings (input = output)
- Tied K=V (key and value projections shared)
- Tied O=Q^T (output projection = transpose of query projection)
- Shared block RMSNorms (ln1 = ln2, saves 3 params)
- RoPE positional encoding (zero trainable params, theta=3)
- QK norms (RMSNorm on Q and K)
- L-BFGS second-order optimization after AdamW
- Single-shot targeted fine-tuning (29 error pairs, <100 steps)

**Link to Code:** https://github.com/tbukic/M10S-Transformer

**Verification Output:**
```
Model: M10S-86p
Author: Tom Bukic
Parameters (unique): 86
Architecture: 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU
Tricks: Tied embeddings, Tied K=V, Tied O=Q^T, Shared block RMSNorms, RoPE (zero params), QK norms, L-BFGS + targeted fine-tuning

Results: 10010/10010 correct (100.00%)
Time: 30.4s (329 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

**Additional Notes:**
- Parameter formula: P = 95 + 9*ff - 12*tieKV - 12*tieQO - 3*shbnorm = 95 + 18 - 12 - 12 - 3 = 86
- Also verified on independent 50K held-out set (seed=99): 0 errors
- Paper: https://github.com/tbukic/M10S-Transformer/blob/master/paper/main.pdf
