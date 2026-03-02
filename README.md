# Minimal Transformers for 10-Digit Addition

A single-layer transformer with **83 trained parameters** achieves **100% accuracy** on 10-digit addition (numbers up to 9,999,999,999). All models are trained from random initialization via standard gradient descent.

## Key Results

| Model | Params | 10K Accuracy | 50K Accuracy | Method |
|-------|--------|-------------|-------------|--------|
| 83p (tieKV+tieQO+shnorm) | **83** | 100.00% | 100.00% | Iterated targeted FT |
| 86p (tieKV+tieQO+shbnorm) | 86 | 100.00% | 100.00% | Single-shot targeted FT |
| 89p (tieKV+tieQO) | **89** | 100.00% | 100.00% | Multi-stage FT (natural, no targeting) |
| 101p (tieQO) | 101 | 100.00% | 100.00% | Cosine LR |
| 122p (base) | 122 | 100.00% | 99.998% | Cosine LR |

All results verified on independent held-out test sets (zero overlap with training data).

## Architecture

All models use a **1-layer Qwen3-style transformer**: d_model=3, 1 attention head, head_dim=4, RoPE (theta=3), SwiGLU MLP, RMSNorm, tied embeddings. Input is LSB-first reversed digits.

```
Params = 95 + 9*ff - 12*tieKV - 12*tieQO - 6*shnorm - 3*shbnorm
```

## Reproduction

### Prerequisites

```bash
uv sync --extra dev
```

All training runs on CPU (no GPU required). A single 122p run takes ~10 minutes; 89p with multi-stage fine-tuning takes ~30 minutes.

### Train from scratch

```bash
# 122-parameter model (base, ff=3)
python experiments/qwen3_train.py --d-model 3 --ff 3 --n-heads 1 --n-kv-heads 1 \
    --lr 0.01 --schedule cosine --steps 50000 --seed 42

# 89-parameter model (tieKV+tieQO, ff=2)
python experiments/qwen3_train.py --d-model 3 --ff 2 --n-heads 1 --n-kv-heads 1 \
    --tie-kv --tie-qo --lr 0.01 --schedule cosine --steps 100000 --seed 1

# 83-parameter model (tieKV+tieQO+shnorm, ff=2)
python experiments/qwen3_train.py --d-model 3 --ff 2 --n-heads 1 --n-kv-heads 1 \
    --tie-kv --tie-qo --share-norms --lr 0.01 --schedule cosine --steps 100000 --seed 905
```

**Important:** Always use `--n-heads 1 --n-kv-heads 1`. The default is 2 heads which gives different parameter counts.

### Use reproduce.py for preset configurations

```bash
# Available configs: 122p, 113p, 101p, 89p, 86p, 83p
python experiments/reproduce.py --config 89p --seeds 0,1,2 --steps 50000

# List all configurations
python experiments/reproduce.py --list
```

### Multi-stage fine-tuning (for sub-100p models)

Sub-100p models (89p, 86p, 83p) require multi-stage fine-tuning:

```bash
# Stage 1: Cosine schedule from scratch
python experiments/qwen3_train.py --d-model 3 --ff 2 --n-heads 1 --n-kv-heads 1 \
    --tie-kv --tie-qo --lr 0.01 --schedule cosine --steps 100000 --seed 1

# Stage 2: Fine-tune from best checkpoint
python experiments/qwen3_train.py --d-model 3 --ff 2 --n-heads 1 --n-kv-heads 1 \
    --tie-kv --tie-qo --lr 0.001 --batch-size 256 --steps 30000 \
    --resume checkpoints/.../best.pt --seed 118
```

### Targeted fine-tuning (for 83p/86p 100% results)

```bash
# Single-shot (for 86p with few errors)
python experiments/targeted_finetune.py \
    --checkpoint checkpoints/.../best.pt \
    --test-set data/test_10k.json --lr 0.001 --steps 5000

# Iterated (for 83p — finds errors, fixes, repeats)
python experiments/targeted_finetune.py \
    --checkpoint checkpoints/.../best.pt \
    --test-set data/test_10k.json --iterated --max-iters 10 --lr 0.001 --steps 5000
```

### Evaluate a checkpoint

```bash
# Basic evaluation
python experiments/qwen3_eval.py checkpoints/.../best.pt --test-set data/test_10k.json

# Detailed evaluation (per-position accuracy, carry analysis)
python experiments/qwen3_eval.py checkpoints/.../best.pt --test-set data/test_10k.json --detailed

# Evaluate on independent held-out set
python experiments/qwen3_eval.py checkpoints/.../best.pt --test-set data/test_holdout_10k.json
```

### Run the full pipeline (parallel multi-seed)

```bash
# Runs Stage 1 + Stage 2 + Eval for multiple seeds in parallel
python experiments/run_all.py --config 89p --seeds 0,1,2,3,4
```

## Test Sets

| File | Samples | Seed | Purpose |
|------|---------|------|---------|
| `data/test_10k.json` | 10,000 | 42 | Primary evaluation |
| `data/test_50k.json` | 50,000 | 42 | Large-scale verification |
| `data/test_holdout_10k.json` | 10,000 | 123 | Independent held-out (no overlap) |
| `data/test_50k_independent.json` | 50,000 | 99 | Independent held-out (no overlap) |

## Checkpoints

Checkpoints are saved to `checkpoints/` (not committed due to size). Each checkpoint contains:
- `state_dict`: Model weights
- `config`: Full model configuration (d_model, ff, n_heads, tie_kv, etc.)
- `step`: Training step
- `accuracy`: Training accuracy at checkpoint time

To evaluate a checkpoint, you only need the `.pt` file — all architecture parameters are stored in `config`.

## Repository Structure

```
src/minimal10digittransformer/
  model/qwen3.py           # Canonical model definition
  data/addition.py          # Data generation, encoding, test sets
  evaluation/metrics.py     # Evaluation (basic + detailed carry analysis)
experiments/
  qwen3_train.py            # Training script
  qwen3_eval.py             # Standalone evaluation
  reproduce.py              # Reproduction configs for all models
  targeted_finetune.py      # Targeted fine-tuning pipeline
  run_all.py                # Parallel multi-stage pipeline
  competitor_train.py       # Comparison architecture (staghado-style)
  lbfgs_finetune.py         # L-BFGS second-order optimization
  swa_ema.py                # Stochastic weight averaging / EMA
  grokking_ablation.py      # Grokking dynamics tracking
data/
  test_10k.json             # Fixed 10K test set (seed=42)
  test_50k.json             # Fixed 50K test set (seed=42)
  test_holdout_10k.json     # Independent held-out 10K (seed=123)
  test_50k_independent.json # Independent held-out 50K (seed=99)
reports/
  main_report.md            # Detailed research report
paper/
  main.tex                  # LaTeX paper
  main.pdf                  # Compiled PDF
```

## Requirements

- Python 3.13+ with PyTorch
- CPU-only training (all models train in minutes on CPU)
- `uv` for dependency management
- LaTeX (texlive) for paper compilation (optional)

## License

See [LICENSE](LICENSE).
