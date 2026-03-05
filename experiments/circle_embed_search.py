"""Circle embedding experiments: fixed/semi-fixed digit placements on a unit circle.

Embedding variants (all use d_model=3):
  Placement strategies:
    uniform:    digits equally spaced at 2πi/10
    random_gaps: random gap sizes (frozen), digits in order 0-9
    random_gaps_shuffled: random gaps, digit order randomized

  Rotation strategies (tilt circle out of XY plane into full 3D):
    none:       stay in XY plane (z=0)
    fixed_45:   rotate π/4 around X then Y axes
    random:     random frozen φ
    learned:    φ is a learnable parameter

12 combos total, tested on reliable grokking configs with many seeds.
"""

import argparse
import csv
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal10digittransformer.model.qwen3 import (
    Qwen3Block, RMSNorm, create_causal_mask, precompute_rope_freqs,
    VOCAB_SIZE, TOTAL_LEN,
)
from minimal10digittransformer.data.addition import generate_batch, generate_test_set
from minimal10digittransformer.evaluation.metrics import evaluate


# ---------------------------------------------------------------------------
# Circle Embedding Model
# ---------------------------------------------------------------------------

def make_rotation_matrix(phi):
    """Compose R_y(phi) · R_x(phi).

    R_x rotates around X-axis, R_y around Y-axis.
    Returns a (3, 3) tensor.
    """
    c, s = torch.cos(phi), torch.sin(phi)
    Rx = torch.stack([
        torch.stack([torch.ones_like(c), torch.zeros_like(c), torch.zeros_like(c)]),
        torch.stack([torch.zeros_like(c), c, -s]),
        torch.stack([torch.zeros_like(c), s, c]),
    ])
    Ry = torch.stack([
        torch.stack([c, torch.zeros_like(c), s]),
        torch.stack([torch.zeros_like(c), torch.ones_like(c), torch.zeros_like(c)]),
        torch.stack([-s, torch.zeros_like(c), c]),
    ])
    return Ry @ Rx  # (3, 3)


class CircleEmbedQwen3(nn.Module):
    """Qwen3 with circle-based token embedding.

    placement: 'uniform' | 'random_gaps' | 'random_gaps_shuffled'
    rotation:  'none' | 'fixed_45' | 'random' | 'learned'
    """

    def __init__(self, d_model: int, n_heads: int = 1, n_kv_heads: int = 1,
                 head_dim: int = 4, ff: int = 3, rope_theta: float = 3.0,
                 max_len: int = TOTAL_LEN + 1, qk_norm: bool = True,
                 use_swiglu: bool = True, tie_kv: bool = False,
                 tie_qo: bool = False, tie_gate: bool = False, repeats: int = 1,
                 share_norms: bool = False, share_block_norms: bool = False,
                 activation: str = "default", window_size: int = 0,
                 placement: str = "uniform", rotation: str = "none",
                 embed_seed: int = 42):
        super().__init__()
        self.d_model = d_model
        self.repeats = repeats
        self.rotation_mode = rotation

        # --- Build the base embedding for 10 digits ---
        rng = random.Random(embed_seed)
        self._placement = placement

        if placement in ("random_uniform_frozen", "random_uniform_learned",
                         "random_gaussian_frozen", "random_gaussian_learned"):
            # Random embeddings (not circle-based)
            torch_rng = torch.Generator().manual_seed(embed_seed)
            if "uniform" in placement:
                base_emb = torch.rand(VOCAB_SIZE, d_model, generator=torch_rng) * 2 - 1
            else:  # gaussian
                base_emb = torch.randn(VOCAB_SIZE, d_model, generator=torch_rng)

            if "learned" in placement:
                self.learned_emb = nn.Parameter(base_emb.clone())
            else:
                self.register_buffer("embedding_table", base_emb)
            # Skip rotation for random embeddings
            rotation = "none"
            self.rotation_mode = rotation

        else:
            # Circle-based placements
            if placement == "uniform":
                # Equal spacing: 2πi/10
                angles = [2.0 * math.pi * i / VOCAB_SIZE for i in range(VOCAB_SIZE)]
            elif placement == "random_gaps":
                # Random gaps, normalized to sum to 2π, digits in order 0-9
                gaps = [rng.random() for _ in range(VOCAB_SIZE)]
                total = sum(gaps)
                gaps = [g / total * 2.0 * math.pi for g in gaps]
                angles = [0.0]
                for g in gaps[:-1]:
                    angles.append(angles[-1] + g)
            elif placement == "random_gaps_shuffled":
                # Random gaps, normalized, digit ORDER also randomized
                gaps = [rng.random() for _ in range(VOCAB_SIZE)]
                total = sum(gaps)
                gaps = [g / total * 2.0 * math.pi for g in gaps]
                cumulative = [0.0]
                for g in gaps[:-1]:
                    cumulative.append(cumulative[-1] + g)
                order = list(range(VOCAB_SIZE))
                rng.shuffle(order)
                angles = [0.0] * VOCAB_SIZE
                for digit, pos_idx in enumerate(order):
                    angles[digit] = cumulative[pos_idx]
            else:
                raise ValueError(f"Unknown placement: {placement}")

            # Build 2D points on unit circle, embed in 3D (z=0)
            base_emb = torch.zeros(VOCAB_SIZE, 3)
            for i in range(VOCAB_SIZE):
                base_emb[i, 0] = math.cos(angles[i])
                base_emb[i, 1] = math.sin(angles[i])

            # --- Apply rotation ---
            if rotation == "none":
                self.register_buffer("embedding_table", base_emb)
            elif rotation == "fixed_45":
                phi = torch.tensor(math.pi / 4.0)
                R = make_rotation_matrix(phi)
                rotated = (R @ base_emb.T).T
                self.register_buffer("embedding_table", rotated)
            elif rotation == "random":
                phi = torch.tensor(rng.random() * 2.0 * math.pi)
                R = make_rotation_matrix(phi)
                rotated = (R @ base_emb.T).T
                self.register_buffer("embedding_table", rotated)
            elif rotation == "learned":
                self.register_buffer("base_emb", base_emb)
                self.phi = nn.Parameter(torch.tensor(math.pi / 4.0))
            else:
                raise ValueError(f"Unknown rotation: {rotation}")

        # RoPE
        rope_cos, rope_sin = precompute_rope_freqs(head_dim, max_len, rope_theta)

        # Shared norm (optional)
        if share_norms:
            shared_norm = RMSNorm(d_model)
        elif share_block_norms:
            shared_norm = RMSNorm(d_model)
        else:
            shared_norm = None

        # Transformer block
        self.block = Qwen3Block(d_model, n_heads, n_kv_heads, head_dim, ff,
                                rope_cos, rope_sin, qk_norm=qk_norm,
                                use_swiglu=use_swiglu, tie_kv=tie_kv,
                                tie_qo=tie_qo, tie_gate=tie_gate,
                                shared_norm=shared_norm, activation=activation)
        self.final_norm = shared_norm if share_norms else RMSNorm(d_model)

        # Causal mask
        mask = create_causal_mask(max_len, window_size)
        self.register_buffer("causal_mask", mask, persistent=False)

        self.apply(self._init_weights)

    def _compute_embedding_table(self):
        if self._placement in ("random_uniform_learned", "random_gaussian_learned"):
            return self.learned_emb
        elif self.rotation_mode == "learned":
            R = make_rotation_matrix(self.phi)
            return (R @ self.base_emb.T).T  # (10, 3)
        else:
            return self.embedding_table

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids):
        emb_table = self._compute_embedding_table()
        x = emb_table[input_ids]
        for _ in range(self.repeats):
            x = self.block(x, self.causal_mask)
        x = self.final_norm(x)
        logits = F.linear(x, emb_table)
        return logits


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def count_params(model):
    seen = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


# Mix of PROVEN grokking configs (for comparison) and small configs (for exploration)
BASE_MODELS = {
    # --- PROVEN: we know these grok, so we can isolate embedding impact ---
    # 122p: d=3 ff=3 — groks reliably (s6: 100% in 200K)
    "122p": dict(
        d_model=3, n_heads=1, n_kv_heads=1, head_dim=4, ff=3,
        rope_theta=3.0, qk_norm=True, use_swiglu=True,
    ),
    # 89p: d=3 ff=2 tieKV+tieQO — groks fast (s5 in ~10K steps!)
    "89p": dict(
        d_model=3, n_heads=1, n_kv_heads=1, head_dim=4, ff=2,
        rope_theta=3.0, qk_norm=True, use_swiglu=True,
        tie_kv=True, tie_qo=True,
    ),
    # --- SMALL: interesting for pushing param boundary ---
    # 68p arc base: tieQO+shnorm, 3/5 seeds grok
    "68p": dict(
        d_model=3, n_heads=1, n_kv_heads=1, head_dim=4, ff=2,
        rope_theta=3.0, qk_norm=True, use_swiglu=True,
        tie_kv=False, tie_qo=True, share_norms=True,
    ),
    # 74p arc base: tieQO, 4/5 showing signal
    "74p": dict(
        d_model=3, n_heads=1, n_kv_heads=1, head_dim=4, ff=2,
        rope_theta=3.0, qk_norm=True, use_swiglu=True,
        tie_kv=False, tie_qo=True,
    ),
}

PLACEMENTS = ["uniform", "random_gaps", "random_gaps_shuffled"]
ROTATIONS = ["none", "fixed_45", "random", "learned"]


def generate_configs():
    configs = []
    for base_name, base_kwargs in BASE_MODELS.items():
        for placement in PLACEMENTS:
            for rotation in ROTATIONS:
                # Build a test model to count params
                test_model = CircleEmbedQwen3(
                    **base_kwargs, placement=placement, rotation=rotation)
                n_params = count_params(test_model)

                p_tag = placement[0]  # u/r/s (uniform/random_gaps/random_gaps_shuffled)
                if placement == "random_gaps_shuffled":
                    p_tag = "rs"
                elif placement == "random_gaps":
                    p_tag = "rg"
                else:
                    p_tag = "u"

                r_tag = rotation[0] if rotation != "fixed_45" else "f45"
                if rotation == "learned":
                    r_tag = "lrn"

                name = f"{base_name}_{p_tag}_{r_tag}"
                desc = f"{base_name} placement={placement} rotation={rotation}"

                configs.append({
                    "name": name,
                    "desc": desc,
                    "base_name": base_name,
                    "base_kwargs": base_kwargs,
                    "placement": placement,
                    "rotation": rotation,
                    "n_params": n_params,
                })

    # Random embedding variants (frozen and learned)
    for base_name, base_kwargs in BASE_MODELS.items():
        for rnd_type in ["random_uniform_frozen", "random_gaussian_frozen",
                         "random_uniform_learned", "random_gaussian_learned"]:
            test_model = CircleEmbedQwen3(
                **base_kwargs, placement=rnd_type, rotation="none")
            n_params = count_params(test_model)

            short = rnd_type.replace("random_", "rnd_")
            name = f"{base_name}_{short}"
            desc = f"{base_name} {rnd_type.replace('_', ' ')}"
            configs.append({
                "name": name,
                "desc": desc,
                "base_name": base_name,
                "base_kwargs": base_kwargs,
                "placement": rnd_type,
                "rotation": "none",
                "n_params": n_params,
            })

    # Arc variants: baseline, start=0 (2p), stride_only (1p: just stride, A=1, start=0)
    from minimal10digittransformer.model.circular_arc import CircularArcQwen3
    for base_name, base_kwargs in BASE_MODELS.items():
        # Standard arc (3p embed: A, start, stride)
        test_model = CircularArcQwen3(**base_kwargs)
        n_params = count_params(test_model)
        configs.append({
            "name": f"{base_name}_arc_baseline",
            "desc": f"{base_name} circular arc (3p embed, baseline)",
            "base_name": base_name,
            "base_kwargs": base_kwargs,
            "placement": "arc",
            "rotation": "arc",
            "n_params": n_params,
        })
        # Arc start=0 (2p embed: A, stride — start frozen at 0)
        configs.append({
            "name": f"{base_name}_arc_start0",
            "desc": f"{base_name} arc with start=0 frozen (2p embed: A, stride)",
            "base_name": base_name,
            "base_kwargs": base_kwargs,
            "placement": "arc_start0",
            "rotation": "arc",
            "n_params": n_params - 1,  # one fewer param
        })
        # Arc stride_only (1p embed: just stride, A=1, start=0)
        configs.append({
            "name": f"{base_name}_arc_stride_only",
            "desc": f"{base_name} arc stride only (1p embed: A=1, start=0 frozen)",
            "base_name": base_name,
            "base_kwargs": base_kwargs,
            "placement": "arc_stride_only",
            "rotation": "arc",
            "n_params": n_params - 2,  # two fewer params
        })

    configs.sort(key=lambda c: (c["n_params"], c["name"]))
    return configs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(cfg, seed, max_steps, eval_interval, device):
    torch.manual_seed(seed)
    random.seed(seed)

    if cfg["placement"] == "arc":
        from minimal10digittransformer.model.circular_arc import CircularArcQwen3
        model = CircularArcQwen3(**cfg["base_kwargs"]).to(device)
    elif cfg["placement"] == "arc_start0":
        from minimal10digittransformer.model.circular_arc import CircularArcQwen3
        model = CircularArcQwen3(**cfg["base_kwargs"]).to(device)
        # Freeze start at 0
        model.arc_start.data.fill_(0.0)
        model.arc_start.requires_grad_(False)
    elif cfg["placement"] == "arc_stride_only":
        from minimal10digittransformer.model.circular_arc import CircularArcQwen3
        model = CircularArcQwen3(**cfg["base_kwargs"]).to(device)
        # Freeze A=1 and start=0, only stride is learned
        model.arc_A.data.fill_(1.0)
        model.arc_A.requires_grad_(False)
        model.arc_start.data.fill_(0.0)
        model.arc_start.requires_grad_(False)
    else:
        model = CircleEmbedQwen3(
            **cfg["base_kwargs"],
            placement=cfg["placement"],
            rotation=cfg["rotation"],
            embed_seed=seed,  # different embed per seed for random variants
        ).to(device)

    n_params = count_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    eval_pairs = generate_test_set(500, seed=12345)

    best_acc = 0.0
    best_step = 0
    grok_step = None
    start_time = time.time()

    for step in range(1, max_steps + 1):
        # Cosine LR schedule
        progress = step / max_steps
        lr = 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        batch, labels = generate_batch(128, device)

        # Shifted loss
        logits = model(batch)
        shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
        shift_labels = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                acc, _ = evaluate(model, device=device, test_pairs=eval_pairs)

            elapsed = time.time() - start_time
            print(f"  [{cfg['name']} s{seed}] step {step}/{max_steps} "
                  f"loss={loss.item():.4f} acc={acc:.1%} best={best_acc:.1%} "
                  f"lr={lr:.6f} [{elapsed:.0f}s]", flush=True)

            if acc > best_acc:
                best_acc = acc
                best_step = step

            if grok_step is None and acc > 0.5:
                grok_step = step

            if acc >= 0.999:
                print(f"  GROKKED at step {step}!")
                break

    elapsed = time.time() - start_time
    return {
        "name": cfg["name"],
        "desc": cfg["desc"],
        "params": n_params,
        "placement": cfg["placement"],
        "rotation": cfg["rotation"],
        "base": cfg["base_name"],
        "seed": seed,
        "best_acc": best_acc,
        "best_step": best_step,
        "grok_step": grok_step or "",
        "final_loss": loss.item(),
        "elapsed": elapsed,
        "steps_done": step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (1,8,15,42,99)")
    parser.add_argument("--config", default=None, help="Run only this config name")
    parser.add_argument("--output", default="experiments/circle_embed_results.csv")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    seed_list = [1, 8, 15, 42, 99, 7, 23, 31, 55, 77][:args.seeds]

    configs = generate_configs()
    if args.config:
        configs = [c for c in configs if c["name"] == args.config]
        if not configs:
            print(f"Config '{args.config}' not found. Available:")
            for c in generate_configs():
                print(f"  {c['name']}")
            return

    print(f"Circle embedding search: {len(configs)} configs × {len(seed_list)} seeds")

    # Load existing results to skip completed runs
    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for row in csv.DictReader(f):
                done.add((row["name"], int(row["seed"])))

    remaining = [(c, s) for c in configs for s in seed_list if (c["name"], s) not in done]
    print(f"Remaining: {len(remaining)}, {len(done)} done\n")

    # Print config table
    seen_names = set()
    for c in configs:
        if c["name"] not in seen_names:
            seen_names.add(c["name"])
            print(f"  {c['n_params']:>4}p  {c['name']:<40} {c['desc']}")
    print()

    # Open CSV
    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    csvf = open(args.output, "a", newline="")
    writer = csv.DictWriter(csvf, fieldnames=[
        "name", "desc", "params", "placement", "rotation", "base",
        "seed", "best_acc", "best_step", "grok_step", "final_loss",
        "elapsed", "steps_done",
    ])
    if write_header:
        writer.writeheader()

    for i, (cfg, seed) in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] {cfg['name']} ({cfg['n_params']}p) seed={seed}")
        print(f"  {cfg['desc']}")

        result = train_one(cfg, seed, args.max_steps, args.eval_interval, args.device)
        writer.writerow(result)
        csvf.flush()

        grok = f"grok@{result['grok_step']}" if result["grok_step"] else "no grok"
        print(f"  -> {result['best_acc']:.1%} (step {result['best_step']}) {grok}\n")

    csvf.close()
    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
