Supercharge-AI: @/home/tom/.local/share/uv/tools/supercharge-ai/lib/python3.13/site-packages/supercharge/data/prompts/claude-md.md

# Project Standards — Minimal 10-Digit Transformer

## Reproduction Pipeline (MANDATORY before any claim or publication)

### Every claimed result MUST be:

1. **Reproducible from scratch in 1 command**: `python reproduce.py --config <name> --seed <N>`
   - From random initialization to claimed accuracy
   - No manual intervention, no multi-step pipelines requiring human judgment
   - If multi-stage training is needed, the script must automate ALL stages
   - Document which seeds work and the overall success rate

2. **Validated on official verify.py**: `python verify.py submissions/submission_Xp.py`
   - Uses seed=2025 (10,010 samples) — completely independent from training
   - Must show exact output including score and QUALIFIED status
   - Save output to `submissions/verify_output_Xp.txt`
   - **200-sample eval is NOT sufficient** — always verify on full 10K+

3. **Independently validated on holdout sets**:
   - Holdout 10K (seed=123) and holdout 50K (seed=99)
   - These must NEVER appear in any training code
   - Results must be consistent with verify.py results

### NO Data Leakage — Zero Tolerance

- Training ONLY uses `generate_batch()` (random pairs from 10^20 space)
- NEVER train on test_10k.json, test_50k.json, or holdout sets
- verify.py data (seed=2025) must NEVER appear anywhere in training code
- If targeted fine-tuning uses error pairs from eval set, MUST disclose this
- When seeding training, avoid seed=42 (collides with test_10k.json generation)
- Document the probabilistic argument for train/test separation

### Documentation Consistency

Before publishing, verify consistency across ALL documents:
- `paper/main.tex` — accuracy numbers, parameter counts, training details
- `reports/main_report.md` — same numbers, more detail
- `README.md` — summary results
- `submissions/` — submission files match claimed results
- `MEMORY.md` — auto-memory matches reality
- All param counts must match the formula: base 95 + 9*ff, minus tying savings

### What to Deliver for Each Model

1. **Submission file**: `submissions/submission_Xp.py` with `build_model()` and `add()`
2. **Verify output**: `submissions/verify_output_Xp.txt` showing QUALIFIED
3. **Checkpoint**: in `checkpoints/` directory
4. **Reproduction script**: single command that retrains from scratch
5. **Training log**: showing the full training trajectory
6. **Grok rate**: documented success rate across seeds (e.g., "3/10 seeds reach 100%")
7. **Provenance chain**: exact sequence of checkpoints and commands

### Review Checklist (before claiming any result)

- [ ] Param count verified (count unique parameters, not total)
- [ ] verify.py passes at 100% (or stated threshold)
- [ ] Holdout eval matches (no overfitting to eval set)
- [ ] Training code reviewed for data leakage
- [ ] Result reproducible from scratch (with documented seed)
- [ ] All documents updated consistently
- [ ] Grok rate / success rate documented
- [ ] Error patterns analyzed (per-digit, per-carry)

### Evaluation Standards

- **Quick eval during training**: 500 samples (for progress tracking only)
- **Intermediate eval**: 2,000 samples (for checkpoint selection)
- **Full eval for claims**: 10,000+ samples (test_10k.json or verify.py)
- **NEVER claim "100%" based on 200-sample eval** — this has repeatedly been wrong

### Training Pipeline Architecture

```
Phase 1 (Base): cosine LR from random init → grokking
Phase 2 (FT):   constant LR from best checkpoint → refinement
Phase 3 (Push): Adam no-wd / targeted FT → push to 100%
```

All phases must be automated in a single script per configuration.
The script must accept `--seed` and produce deterministic results.

**Multi-stage / targeted FT pipelines**: If the final model was reached via
targeted fine-tuning on error pairs, the reproduce script must automate the
full cycle: base training → checkpoint selection → iterated error-finding →
targeted FT. The error pairs come from the model's own eval (not leaked test
data), so different seeds will find different errors — this is expected and
must be documented. The full pipeline still counts as "1 command" if scripted.

### FROZEN FILES — DO NOT MODIFY

The following files are **permanently frozen** and must NEVER be modified, overwritten, or deleted by any agent or script. They represent the official submitted results and their corresponding checkpoints:

**Submission files** (all files in `submissions/`):
- `submissions/submission_*.py` — model definitions + checkpoint paths
- `submissions/verify_output_*.txt` — official verify.py outputs
- `submissions/github_issue_*.md` — submission texts

**Submission checkpoints** (the specific checkpoint directories referenced by submission files):
- `checkpoints/qwen3_d3_ff2_83p_tiekv_tieqo_shnorm_s905_targeted/`
- `checkpoints/qwen3_d3_ff2_86p_tiekv_tieqo_shbnorm_s1_targeted/`
- `checkpoints/qwen3_d3_ff2_89p_tiekv_tieqo_s11127/`
- `checkpoints/qwen3_d3_ff2_101p_tieqo_s13_targeted/`
- `checkpoints/qwen3_d3_ff3_122p_s6/`
- `checkpoints/qwen3_arc_62p_tiekv_tieqo_adam_nowd/`
- `checkpoints/qwen3_arc_95p_ft_s9999/`
- `checkpoints/qwen3_rank1_96p_tiekv_s9999/`

These files are also chmod read-only as a filesystem safeguard. New reproduction runs write to `checkpoints/reproduce_*` directories, which are separate.

### Key References

- **Auto-memory**: `MEMORY.md` (auto-memory dir) — results, architecture, API signatures, file map
- **Data leakage review**: `.claude/SuperchargeAI/tasks/review/.../result.md`
- **Doc update plan**: `.claude/SuperchargeAI/tasks/plan/.../result.md`
- **Uncommitted local notes**: `.claude/CLAUDE.local.md` (if present)
