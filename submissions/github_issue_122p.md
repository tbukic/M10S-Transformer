# AdderBoard Submission: 122 params

## Issue Title
[Submission] 122 params - Tom Bukic

## Fields

**Author:** Tom Bukic

**Unique Parameter Count:** 122

**Accuracy:** 100.00% on 10,010 tests (verify.py seed=2025: 10K random + 10 edge cases)

**Method:** 200K-step cosine LR schedule with grokking

**Architecture:** 1L Qwen3 decoder, d=3, 1h/1kv, hd=4, ff=3, RoPE theta=3, SwiGLU, RMSNorm

**Key Tricks:**
- Tied embeddings (input = output)
- RoPE positional encoding (zero trainable params, theta=3)
- QK norms (RMSNorm on Q and K)
- SwiGLU MLP (ff=3)
- Cosine LR with grokking (delayed generalization at ~25K steps)

**Link to Code:** https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_122p.py

**Verification Output:**
```
Model: M10S-122p
Author: Tom Bukic
Parameters (unique): 122
Architecture: 1L Qwen3, d=3, 1h/1kv, hd=4, ff=3, RoPE theta=3, SwiGLU
Tricks: Tied embeddings, RoPE (zero params), QK norms, SwiGLU, Cosine LR with grokking

Results: 10010/10010 correct (100.00%)
Time: 30.2s (332 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

**Additional Notes:**
- Parameter formula: P = 95 + 9*ff = 95 + 27 = 122
- This is our baseline model with no weight tying beyond standard tied embeddings
- 4/10 seeds grok from scratch; s6 reaches 100% at step 30K
- Paper: https://github.com/tbukic/M10S-Transformer/blob/master/paper/main.pdf
