# Experiment Plan — Sub-60p Push

## System Resources

| Resource | Total | In Use | Available |
|----------|-------|--------|-----------|
| CPU | 64 cores | ~8 cores | **56 cores** |
| GPU | RTX A6000 49GB | 0% | **100%** |
| RAM | 503 GB | 10 GB | ~490 GB |

GPU gives only 1.4x speedup on these tiny models (kernel overhead).
Better to run many CPU jobs in parallel: 12 jobs × 4 cores = 48 cores.

---

## Currently Running

| Experiment | Script | Progress | ETA |
|-----------|--------|----------|-----|
| d=3 param search | `param_search.py` | 112/280 (40%) | ~24h |
| d=2 SwiGLU search | `d2_search.py` | 4/60 (7%) | ~8h |

### d=3 param search highlights so far (112 runs)

| Params | Config | Best | Grok rate |
|--------|--------|------|-----------|
| **59p** | arc ff=2 tieKV+tieQO+shbnorm | 48% | 1/5 |
| **68p** | arc ff=2 tieQO+shnorm | 97.2% | 3/5 |
| 71p | arc ff=2 tieQO+shbnorm | 97.2% | 2/5 |
| 71p | arc ff=2 tieKV+shbnorm | 91.2% | 1/5 |

### d=2 SwiGLU results so far (3 complete)

| Params | Config | Best | Notes |
|--------|--------|------|-------|
| 41p | arc d=2 ff=2 tieKV+tieQO+shnorm | 0% | Loss→0 but no accuracy. 3/5 seeds done, all dead. |

---

## Wave 2 — Ready to Launch

Script: `experiments/d2_wave2_search.py`

### Param savings from dropping gate (SwiGLU → ReLU)

```
SwiGLU:  gate(ff×d) + up(ff×d) + down(d×ff) = 3 × ff × d params
ReLU:    W1(ff×d)   + W2(d×ff)              = 2 × ff × d params
Savings: ff × d params
```

| d_model | ff | SwiGLU MLP | ReLU MLP | Saved |
|---------|----|-----------:|--------:|------:|
| 3 | 2 | 18p | 12p | **6p** |
| 3 | 3 | 27p | 18p | **9p** |
| 2 | 2 | 12p | 8p | **4p** |

---

### Group A — d=2 ReLU (no gate, 2 matrices)

`python experiments/d2_wave2_search.py --group relu_only`

12 configs × 5 seeds = 60 runs

| Params | Config |
|-------:|--------|
| **37p** | arc d=2 ff=2 relu tieKV+tieQO+shnorm |
| 39p | arc d=2 ff=2 relu tieKV+tieQO+shbnorm |
| 41p | arc d=2 ff=2 relu tieKV+tieQO |
| 45p | arc d=2 ff=2 relu tieKV+shnorm |
| 45p | arc d=2 ff=2 relu tieQO+shnorm |
| 47p | arc d=2 ff=2 relu tieKV+shbnorm |
| 47p | arc d=2 ff=2 relu tieQO+shbnorm |
| 49p | arc d=2 ff=2 relu tieKV |
| 49p | arc d=2 ff=2 relu tieQO |
| 53p | arc d=2 ff=2 relu shnorm |
| 55p | arc d=2 ff=2 relu shbnorm |
| 57p | arc d=2 ff=2 relu none |

---

### Group B — d=2 SwiGLU with ReLU gate (3 matrices, ReLU instead of SiLU)

`python experiments/d2_wave2_search.py --group swiglu_relu`

12 configs × 5 seeds = 60 runs

| Params | Config |
|-------:|--------|
| 41p | arc d=2 ff=2 swirelu tieKV+tieQO+shnorm |
| 43p | arc d=2 ff=2 swirelu tieKV+tieQO+shbnorm |
| 45p | arc d=2 ff=2 swirelu tieKV+tieQO |
| 49p | arc d=2 ff=2 swirelu tieKV+shnorm |
| 49p | arc d=2 ff=2 swirelu tieQO+shnorm |
| 51p | arc d=2 ff=2 swirelu tieKV+shbnorm |
| 51p | arc d=2 ff=2 swirelu tieQO+shbnorm |
| 53p | arc d=2 ff=2 swirelu tieKV |
| 53p | arc d=2 ff=2 swirelu tieQO |
| 57p | arc d=2 ff=2 swirelu shnorm |
| 59p | arc d=2 ff=2 swirelu shbnorm |
| 61p | arc d=2 ff=2 swirelu none |

---

### Group C — d=3 ReLU (no gate, 2 matrices)

`python experiments/d2_wave2_search.py --group relu_d3`

12 configs × 5 seeds = 60 runs. Direct comparison with d=3 SwiGLU results from param search.

| Params | Config | SwiGLU equivalent |
|-------:|--------|-------------------|
| **50p** | arc d=3 ff=2 relu tieKV+tieQO+shnorm | 56p SwiGLU (0%) |
| 53p | arc d=3 ff=2 relu tieKV+tieQO+shbnorm | 59p SwiGLU (48%) |
| 56p | arc d=3 ff=2 relu tieKV+tieQO | 62p SwiGLU (3%) |
| 62p | arc d=3 ff=2 relu tieKV+shnorm | 68p SwiGLU (0%) |
| **62p** | arc d=3 ff=2 relu tieQO+shnorm | **68p SwiGLU (97.2%!)** |
| 65p | arc d=3 ff=2 relu tieKV+shbnorm | 71p SwiGLU (91.2%) |
| 65p | arc d=3 ff=2 relu tieQO+shbnorm | 71p SwiGLU (97.2%) |
| 68p | arc d=3 ff=2 relu tieKV | 74p SwiGLU (pending) |
| 68p | arc d=3 ff=2 relu tieQO | 74p SwiGLU (69.2%) |
| 74p | arc d=3 ff=2 relu shnorm | 80p SwiGLU (pending) |
| 77p | arc d=3 ff=2 relu shbnorm | 83p SwiGLU (pending) |
| 80p | arc d=3 ff=2 relu none | 86p SwiGLU (pending) |

Key comparison: **62p ReLU tieQO+shnorm** vs **68p SwiGLU tieQO+shnorm** (97.2%).
Same param count as our current 62p submission but totally different architecture trade-off.

---

### Group D — Multi-layer shared weights (repeats=2,4,8)

`python experiments/d2_wave2_search.py --group multilayer`

Same params as 1-layer (weights are shared). Tests if running the block multiple
times helps the model converge. 9 configs × 5 seeds = 45 runs.

| Params | Config | Layers | 1-layer baseline |
|-------:|--------|-------:|------------------|
| 62p | arc d=3 ff=2 tieKV+tieQO | 2 | 3% (SwiGLU) |
| 68p | arc d=3 ff=2 tieQO+shnorm | 2 | 97.2% (SwiGLU) |
| 71p | arc d=3 ff=2 tieQO+shbnorm | 2 | 97.2% (SwiGLU) |
| 62p | arc d=3 ff=2 tieKV+tieQO | 4 | 3% |
| 68p | arc d=3 ff=2 tieQO+shnorm | 4 | 97.2% |
| 71p | arc d=3 ff=2 tieQO+shbnorm | 4 | 97.2% |
| 62p | arc d=3 ff=2 tieKV+tieQO | 8 | 3% |
| 68p | arc d=3 ff=2 tieQO+shnorm | 8 | 97.2% |
| 71p | arc d=3 ff=2 tieQO+shbnorm | 8 | 97.2% |

Note: Previous experiment with shared layers rep=2 was "barely alive (15.5% at 400K)"
and rep=3 was "dead after 400K". Those were on different configs — worth retesting on
configs that grok well at 1 layer.

---

## Summary

| Group | Runs | Param range | Question answered |
|-------|-----:|-------------|-------------------|
| A: d=2 ReLU | 60 | 37-57p | Can d=2 work without gating? |
| B: d=2 SwiGLU+ReLU | 60 | 41-61p | Does ReLU gate help d=2? |
| C: d=3 ReLU | 60 | 50-80p | Does ReLU match SwiGLU at d=3? |
| D: Multi-layer | 45 | 62-71p | Do shared-weight layers help? |
| **Total** | **225** | **37-80p** | |

At ~8 min/run, 12 parallel jobs (48 cores): **~2.5 hours for all 225 runs.**

---

## Launch Plan

```bash
# Run all 4 groups in parallel (12 processes total, 4 cores each = 48 cores)
OMP_NUM_THREADS=4 python experiments/d2_wave2_search.py --group relu_only &
OMP_NUM_THREADS=4 python experiments/d2_wave2_search.py --group swiglu_relu &
OMP_NUM_THREADS=4 python experiments/d2_wave2_search.py --group relu_d3 &
OMP_NUM_THREADS=4 python experiments/d2_wave2_search.py --group multilayer &
```

Note: Each search script runs configs sequentially (1 at a time), but the 4 groups
run in parallel. With existing param_search and d2_search still running, total load
would be ~6 processes × 4 cores = 24 cores. Still well within 64-core budget.

Wait for d2_search to finish first? Or launch now alongside it?
