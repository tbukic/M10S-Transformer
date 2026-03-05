# Training Paths for Sub-60p Models

## 6-Phase Training Pipeline

Our models use up to 6 training phases. The standard pipeline is:

1. **Phase 1: Cosine LR** (from random init)
   - LR: 0.01 → 0.001 cosine decay
   - WD: 0.01
   - Steps: 100K-300K
   - Purpose: Initial grokking phase. Most models reach 80-95% here.

2. **Phase 2: Grokfast-EMA** (breakthrough phase)
   - LR: 0.001 → 0.0001 cosine decay
   - WD: 0.01
   - Grokfast: alpha=0.98, lambda=2.0
   - Steps: 50K-325K
   - Purpose: Amplifies slow-varying gradient components, breaks through plateaus.
   - Key insight: This was the single most effective technique for pushing past ~90%.

3. **Phase 3: Constant LR** (optional)
   - LR: 0.0003
   - WD: 0.01
   - Steps: 30K-200K
   - Purpose: Steady refinement when cosine schedule decays too fast.

4. **Phase 4: L-BFGS** (optional, for larger models)
   - Second-order optimization on error pairs
   - Very effective for 86p+ models, less so for smaller ones

5. **Phase 5: Cosine no-WD** (optional)
   - LR: 0.0005 → 0.00005 cosine decay
   - WD: 0.0
   - Purpose: Remove regularization pressure for final push

6. **Phase 6: Iterated Targeted Fine-Tuning**
   - Find error pairs via autoregressive eval on ~2K-5K samples
   - Train on batch of error pairs + random padding (128 batch size)
   - LR: 0.0003, WD: 0.0
   - Repeat until 0 errors found
   - Purpose: Fix remaining ~1-5% errors

---

## 58p Model (WORLD RECORD - 100% on verify.py)

**Config:** CircularArcQwen3 + K=alpha*Q + gate=alpha*up + tieQO
- d=3, ff=2, 1h/1kv, hd=4, RoPE theta=3, SwiGLU
- Parameters: 58 (base 74p - 11(K=aQ) - 5(gate=a*up) = 58)

**Seed:** 1

### Training Path (58p s1):
```
Phase 1: Cosine LR (0.01→0.001), 290K steps → 83.8%
Phase 2: Grokfast-EMA (LR 0.001→0.0001, alpha=0.98, lambda=2.0), 60K steps → 99.4%
Phase 6: Targeted FT (1 iteration, 9 error pairs, 3000 steps) → 100.0%
```

**Total: ~350K steps, ~10 min on CPU**

Checkpoint: `checkpoints/qwen3_arc_58p_tying_s1_targeted/best.pt`
verify.py: 10010/10010 (100.00%)

### Alternative path (58p s1 natural grokking):
```
Phase 1: Cosine LR, 290K steps → 83.8%
Phase 2: Grokfast-EMA, 80K steps → 100.0% (natural, no targeted FT!)
```
verify.py: 10009/10010 (99.99%) — 1 error

---

## 55p Model (99.99% on verify.py, QUALIFIED)

**Config:** CircularArcQwen3 + K=alpha*Q + gate=alpha*up + tieQO + shared block norms
- d=3, ff=2, 1h/1kv, hd=4, RoPE theta=3, SwiGLU
- Parameters: 55 (base 74p - 11(K=aQ) - 5(gate=a*up) - 3(shbnorm) = 55)

**Seed:** 1

### Training Path (55p s1):
```
Phase 1: Cosine LR (0.01→0.001), 295K steps → 81.2%
Phase 2: Grokfast-EMA (LR 0.001→0.0001, alpha=0.98, lambda=2.0), 60K steps → 94.8%
          (continued to 325K steps → 100.0% on 500-sample training eval)
Phase 6: Targeted FT (2 iterations, 107 cumulative error pairs, 5000 steps each) → 100.0%
```

**Total: ~620K steps + 10K targeted, ~40 min on CPU**

Checkpoint: `checkpoints/qwen3_arc_55p_tying_s1_targeted/best.pt`
verify.py: 10009/10010 (99.99%) — 1 error, QUALIFIED
10K eval (seed=42): 10000/10000 (100.00%)

---

## 56p Model (best: 93.8%, still training)

**Config:** CircularArcQwen3 + tieKV + tieQO + shared norms (standard, no monkey-patching)
- d=3, ff=2, 1h/1kv, hd=4, RoPE theta=3, SwiGLU
- Parameters: 56 (base 62p - 6(shnorm) = 56)

**Best branch:** s8, const LR 0.0001 + Grokfast @ step 1.5M → 93.8%

Multiple continuation branches tried (cosine, const, grokfast, nowd), all plateau around 92-94%.

---

## Killed Branches (potential for later use / GPU)

| Branch | Peak Acc | Steps | Notes |
|--------|----------|-------|-------|
| 55p s1 cosine (lr=0.0005) | 86.4% | 65K phase | Slow convergence |
| 55p s1 const (lr=0.0003) | 92.2% | 160K phase | Plateaued |
| 55p s1 nowd | 84.6% | ~50K phase | Did not converge |
| 58p s1 cosine (lr=0.001) | 84.6% | 40K phase | Slow |
| 58p s8 cosine (lr=0.0005) | 98.4% | 385K phase | Nearly there |
| 55p s22 cosine (lr=0.001) | 53.4% | 65K phase | Very slow seed |
| 56p s8 GF lambda=5 | 82.8% (crashed from 92%) | crashed | Lambda=5 too aggressive |
