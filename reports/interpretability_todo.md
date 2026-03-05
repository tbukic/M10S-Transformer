# Interpretability TODO

## Completed
- [x] Analysis 1: Embedding geometry (3D scatter)
- [x] Analysis 2: Complete weight atlas
- [x] Analysis 3: Attention heatmap (double staircase)
- [x] Analysis 4: Logit lens (progressive refinement)
- [x] Analysis 5: Carry vs no-carry attention comparison
- [x] Analysis 6: Training dynamics (best vs final weight change)
- [x] Analysis 7: Residual stream trajectories
- [x] Cross-model embedding comparison (5 models)
- [x] Report: reports/07_interpretability.md
- [x] Ablation study tool: experiments/ablation.py
- [x] Per-weight ablation on ALL 8 submitted models (62p-122p)
- [x] Fixed load_model to handle rank-1 output (96p) architecture

## Ablation Key Findings (all 8 models)
- Component-level: zeroing ANY component → 0% in ALL models (total criticality)
- Effective params converge to ~80-83 across non-arc models (83p-122p)
- 62p arc: ~50 effective params (dim-2 entirely dead due to arc embedding)
- 95p arc: ZERO slack — every weight critical (unique among all models)
- k_proj universally most redundant; q_proj universally most critical
- 122p has 20 slack weights; 83p has only 5 — near theoretical minimum

## In Progress

## Next Up

### High Priority (publishable findings)
- [ ] **Paper section on interpretability** — The double-staircase + carry-cascade in 89 params is a strong result. Write a paper section showing our 89p model discovers the same carry-cascade circuit (Quirke & Barez 2024) as models 19,000x larger. Include attention heatmap, carry comparison, and ablation figures.
- [ ] **OV circuit analysis** — With tied K=V and Q=O^T, the OV circuit is a 3x3 matrix. Compute exactly what "attending to position j" copies into the residual stream. This would give complete mechanistic understanding of what the single head does.
- [ ] **Fourier analysis of arc embeddings** — Run DFT on the 62p/95p arc model embeddings + weights. The circular structure is built in — do Fourier features emerge despite being decimal addition?

### Medium Priority
- [ ] **Grokking weight trajectories** — Re-run 89p s5 (grokks at ~10K, 13 min) saving checkpoints every 500 steps. Plot all 89 weight values through the phase transition. Direct contribution to grokking literature.
- [ ] **Error analysis for 62p arc** — It's stuck at 98.78% (122 errors on 10K). Use interpretability tools to compare activations for correct vs incorrect predictions. Is it always long carry chains?
- [ ] **QK circuit with RoPE** — Compute the effective QK dot product for each pair of positions (accounting for RoPE theta=3). Visualize the "wiring diagram" showing which positions can attend to which.

### Stretch Goals
- [ ] **Complete reverse-engineering** — With 89 params, write out the exact algorithm as a closed-form formula. No other transformer this small has been fully reverse-engineered on a real task.
- [ ] **MLP function visualization** — Sample a grid in R3, plot MLP output as a vector field. Is it computing carry-related nonlinearities?
- [ ] **Causal interventions with pyvene** — Activation patching to confirm which components are necessary vs sufficient for carry propagation.
- [ ] **Training dynamics animation** — Animate the embedding geometry evolving during training (requires step checkpoints).
