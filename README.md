# Minimal Transformers for 10-Digit Addition

A single-layer transformer with **83 trained parameters** achieves **100% accuracy** on 10-digit addition (numbers up to 9,999,999,999). All models are trained from random initialization via standard gradient descent.

**Paper:** [paper/main.pdf](paper/main.pdf)

## Key Results

| Model | Params | verify.py | 50K Holdout | Method |
|-------|--------|-----------|------------|--------|
| 83p (tieKV+tieQO+shnorm) | **83** | **10,010/10,010 (100%)** | 0 err | Iterated targeted FT |
| 86p (tieKV+tieQO+shbnorm) | 86 | **10,010/10,010 (100%)** | 0 err | L-BFGS + targeted FT |
| 89p (tieKV+tieQO) | **89** | **10,010/10,010 (100%)** | 0 err | Multi-stage FT (natural, no targeting) |
| 101p (tieQO) | 101 | **10,010/10,010 (100%)** | 0 err | Targeted FT |
| 122p (base) | 122 | **10,010/10,010 (100%)** | 1 err | Cosine LR |

All 5 models achieve **QUALIFIED** status on the official [AdderBoard](https://github.com/anadim/AdderBoard) `verify.py` (seed=2025, 10K random + 10 edge cases). The 83p model would rank **#1 on the trained-weights leaderboard** (current leader: 311 params). Results also verified on independent 50K held-out test set (seed=99, zero overlap with training data).

## Architecture

All models use a **1-layer Qwen3-style transformer**: d_model=3, 1 attention head, head_dim=4, RoPE (theta=3), SwiGLU MLP, RMSNorm, tied embeddings. Input is LSB-first reversed digits.

```
Params = 95 + 9*ff - 12*tieKV - 12*tieQO - 6*shnorm - 3*shbnorm
```

## Quick Start: Validate Included Checkpoints

The repository includes best-per-param checkpoints (5 models, ~30KB total). To verify them:

```bash
uv sync --extra dev

# Validate all checkpoints: count parameters + evaluate on 10K test set
python scripts/validate_checkpoints.py

# Detailed evaluation (per-position accuracy, carry analysis)
python scripts/validate_checkpoints.py --detailed

# Evaluate on 50K test set
python scripts/validate_checkpoints.py --test-set data/test_50k.json

# Evaluate a single model
python scripts/validate_checkpoints.py --model 89p
```

Expected output: all 5 models show correct parameter counts and 0 errors on 10K.

### Official AdderBoard Verification

```bash
# Run the official AdderBoard verify.py on any submission
python verify.py submissions/submission_83p.py
python verify.py submissions/submission_89p.py
# All 5 models: 10,010/10,010 correct (100.00%) QUALIFIED
```

## Reproduction: Training from Scratch

### Prerequisites

```bash
uv sync --extra dev
```

All training runs on CPU (no GPU required). A single 122p run takes ~10 minutes; 89p with multi-stage fine-tuning takes ~30 minutes.

### Train from scratch

```bash
# 122-parameter model (base, ff=3)
python experiments/qwen3_train.py --d-model 3 --ff 3 --n-heads 1 --n-kv-heads 1 \
    --lr 0.01 --cosine-lr --steps 50000 --seed 42

# 89-parameter model (tieKV+tieQO, ff=2)
python experiments/qwen3_train.py --d-model 3 --ff 2 --n-heads 1 --n-kv-heads 1 \
    --tie-kv --tie-qo --lr 0.01 --cosine-lr --steps 100000 --seed 1

# 83-parameter model (tieKV+tieQO+shnorm, ff=2)
python experiments/qwen3_train.py --d-model 3 --ff 2 --n-heads 1 --n-kv-heads 1 \
    --tie-kv --tie-qo --share-norms --lr 0.01 --cosine-lr --steps 100000 --seed 905
```

**Important:** Always use `--n-heads 1 --n-kv-heads 1`. The default is 2 heads which gives different parameter counts.

### Use reproduce.py for full pipeline reproduction

```bash
# List all configs with phase details
python experiments/reproduce.py --list

# Reproduce 122p (single-phase, simple)
python experiments/reproduce.py --config 122p --seed 6 --device cuda

# Reproduce 89p (4-phase pipeline including FT)
python experiments/reproduce.py --config 89p --seed 11127 --device cuda

# Reproduce 83p (base + FT + iterated targeted FT)
python experiments/reproduce.py --config 83p --seed 905 --train-eval-seed 888 --device cuda

# Smoke test (short run to verify setup)
python experiments/reproduce.py --config 122p --seed 42 --steps-override 200
```

### Multi-stage fine-tuning (for sub-100p models)

Sub-100p models (89p, 86p, 83p) require multi-stage fine-tuning. **reproduce.py automates the full pipeline**:

```bash
# Full automated pipeline (recommended):
python experiments/reproduce.py --config 89p --seed 11127 --device cuda

# Or run stages manually:
# Stage 1: Cosine schedule from scratch
python experiments/qwen3_train.py --d-model 3 --ff 2 --n-heads 1 --n-kv-heads 1 \
    --tie-kv --tie-qo --lr 0.01 --cosine-lr --steps 100000 --seed 1

# Stage 2: Fine-tune from best checkpoint
python experiments/qwen3_train.py --resume checkpoints/.../best.pt \
    --lr 0.001 --batch-size 256 --steps 30000 --seed 118

# Stage 3 (targeted FT for 83p/86p):
python experiments/targeted_finetune.py \
    --checkpoint checkpoints/.../best.pt \
    --test-set data/test_10k.json --iterated --max-iters 10 --lr 0.001 --steps 5000
```

### Run parallel multi-seed experiments

```bash
# Runs Stage 1 + Stage 2 + Eval for multiple seeds in parallel
python experiments/run_all.py --max-parallel 8
```

### Evaluate any checkpoint

```bash
# Basic evaluation
python experiments/qwen3_eval.py checkpoints/.../best.pt --test-set data/test_10k.json

# Detailed evaluation (per-position accuracy, carry analysis)
python experiments/qwen3_eval.py checkpoints/.../best.pt --test-set data/test_10k.json --detailed

# Evaluate on independent held-out set
python experiments/qwen3_eval.py checkpoints/.../best.pt --test-set data/test_holdout_10k.json
```

## Test Sets

| File | Samples | Seed | Purpose |
|------|---------|------|---------|
| `data/test_10k.json` | 10,000 | 42 | Primary evaluation |
| `data/test_50k.json` | 50,000 | 42 | Large-scale verification |
| `data/test_holdout_10k.json` | 10,000 | 123 | Independent held-out (no overlap) |
| `data/test_50k_independent.json` | 50,000 | 99 | Independent held-out (no overlap) |

## Included Checkpoints

Best-per-param checkpoints are committed to the repo (~30KB total):

| Model | Checkpoint | Params | 10K Errors | Method |
|-------|-----------|--------|-----------|--------|
| 83p | `checkpoints/qwen3_d3_ff2_83p_tiekv_tieqo_shnorm_s905_targeted/` | 83 | 0 | Iterated targeted FT |
| 86p | `checkpoints/qwen3_d3_ff2_86p_tiekv_tieqo_shbnorm_s1_targeted/` | 86 | 0 | Targeted FT |
| 89p | `checkpoints/qwen3_d3_ff2_89p_tiekv_tieqo_s11127/` | 89 | 0 | Natural 4-stage FT |
| 101p | `checkpoints/qwen3_d3_ff2_101p_tieqo_s13_targeted/` | 101 | 0 | Targeted FT |
| 122p | `checkpoints/qwen3_d3_ff3_122p_s6/` | 122 | 0 | 200K cosine |

Each checkpoint contains `best.pt` (model weights + full config) and evaluation JSONs.

## Repository Structure

```
src/minimal10digittransformer/
  model/qwen3.py           # Canonical model definition
  data/addition.py          # Data generation, encoding, test sets
  evaluation/metrics.py     # Evaluation (basic + detailed carry analysis)
experiments/
  qwen3_train.py            # Training script
  qwen3_eval.py             # Standalone evaluation
  reproduce.py              # Unified reproduction pipeline (all phases, 1 command)
  targeted_finetune.py      # Targeted fine-tuning pipeline
  run_all.py                # Parallel multi-seed runner
  plot_training.py           # Training curve plots
  archive/                   # Legacy experiment scripts
scripts/
  validate_checkpoints.py   # Validate tracked checkpoints (param count + eval)
data/
  test_10k.json             # Fixed 10K test set (seed=42)
  test_50k.json             # Fixed 50K test set (seed=42)
  test_holdout_10k.json     # Independent held-out 10K (seed=123)
  test_50k_independent.json # Independent held-out 50K (seed=99)
checkpoints/                # Best-per-param model checkpoints
submissions/                # AdderBoard submission files (verify.py compatible)
verify.py                   # Official AdderBoard verification script
paper/
  main.tex                  # LaTeX paper
  main.pdf                  # Compiled PDF
reports/
  main_report.md            # Detailed research report
```

## Requirements

- Python 3.13+ with PyTorch
- CPU-only training (all models train in minutes on CPU)
- `uv` for dependency management
- LaTeX (texlive) for paper compilation (optional)

## Acknowledgments

This work builds on and was inspired by the [AdderBoard](https://github.com/anadim/AdderBoard) community. Key influences:

- **[staghado](https://github.com/staghado/minimal-ten-digit-addition-transformer)** (Said Taghadouini) — Independently developed the same Qwen3 d=3 architecture (122p, 99.95%). We adopted their tiny RoPE theta=3 insight and their L-BFGS fine-tuning finding that AdamW converges to saddle points.
- **[evindor/MicroAdder](https://github.com/evindor/MicroAdder)** (Arseniy Zarechnev) — 67p with parametric circular arc embeddings, rank-1 output projection, and carry-mix curriculum. Directly inspired our CircularArcQwen3 (62p) and Rank1OutModel (96p) experiments, and the metric-triggered weight decay investigation.
- **[rezabyt](https://github.com/rezabyt/digit-addition-311p)** — 311p via rank-3 factorization with curriculum learning. Established the compression paradigm and revealed that position embeddings consume 36% of parameters (motivating our RoPE choice).
- **[yinglunz](https://github.com/yinglunz/A-456-Parameter-Transformer-Solves-10-Digit-Addition)** (Yinglun Zhu) — 456p with mixed-rank factorization showing rank-2 on attention output suffices.
- **[h3nock](https://github.com/h3nock/tiny-adder-lab)** — 305p/335p with curriculum and multi-round fine-tuning.
- **[yhavinga](https://github.com/yhavinga/gpt-acc-jax)** (Yeb Havinga) — 777p in JAX, discovered grokking dynamics in tiny addition models and that learned positions are essential.
- **[JackCai1206](https://github.com/JackCai1206/smallest-addition-transformer)** (Jack Cai) — 234p with spiral positional embeddings, showing structured initialization eliminates grokking.
- **[sanyalsunny111](https://github.com/sanyalsunny111/Smallest-GPT-for-addition)** (Sunny Sanyal) — 296p standard GPT with curriculum and LAWA, demonstrating training recipe importance.
- **Hand-coded solutions** (Lokimorty, yieldthought, SeuperHakkerJa, matanabudy, JagNL, alexlitz, Wonderfall, cosminscn) revealed universal patterns: parabolic embeddings, RoPE period-19 geometry, sparse projections, and two-hinge carry detection.
- **[Dimitris Papailiopoulos](https://github.com/anadim/AdderBoard)** — Founded the AdderBoard competition.

## Citation

```bibtex
@misc{bukic2026minimal,
  author       = {Tom Bukic},
  title        = {Minimal Transformers for 10-Digit Addition},
  year         = {2026},
  url          = {https://github.com/tbukic/M10S-Transformer}
}
```

## License

See [LICENSE](LICENSE).
