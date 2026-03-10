"""General-purpose weight tying via module replacement.

Instead of monkey-patching forward methods, this module replaces sub-modules
with lightweight tied variants. The parent module's forward (e.g.
``self.k_proj(x)``) transparently delegates to the replacement module, which
in turn references the master module's weights.

Tie Group Format
----------------
A tie group is a list.  The **first element** is the master path (keeps its
weights).  Remaining elements are follower specs with an optional transform
prefix::

    # Identity sharing (follower delegates to master)
    ["block.attn.k_proj", "block.attn.v_proj"]

    # Scalar: follower = alpha * master(x), 1 learnable param
    ["block.attn.q_proj", "scalar:block.attn.k_proj"]

    # Transpose: follower uses master.weight^T, 0 extra params
    ["block.attn.q_proj", "transpose:block.attn.o_proj"]

    # Negate: follower uses -master.weight, 0 extra params
    ["block.mlp.up_proj", "negate:block.mlp.down_proj"]

Supported transforms: (none) = identity, ``scalar``, ``transpose``, ``negate``,
``rotation``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Tied module classes
# ============================================================================

class ScalarTiedLinear(nn.Module):
    """Linear layer that computes ``alpha * source(x)``.

    Replaces a full ``nn.Linear`` and adds 1 learnable scalar parameter.
    The *source* is stored in a plain list to avoid ``nn.Module``
    auto-registration (which would duplicate parameters in the state dict).
    """

    def __init__(self, source: nn.Linear):
        super().__init__()
        self._source = [source]  # list wrapper prevents submodule registration
        self.alpha = nn.Parameter(torch.tensor(1.0))

    @property
    def source(self) -> nn.Linear:
        return self._source[0]

    def forward(self, x):
        return self.alpha * self.source(x)


class TransposeTiedLinear(nn.Module):
    """Linear layer that computes ``F.linear(x, source.weight.t())``.

    Used for O = Q^T tying.  No learnable parameters of its own.
    """

    def __init__(self, source: nn.Linear):
        super().__init__()
        self._source = [source]

    @property
    def source(self) -> nn.Linear:
        return self._source[0]

    def forward(self, x):
        return F.linear(x, self.source.weight.t())


class IdentityTiedLinear(nn.Module):
    """Linear layer that delegates ``source(x)``.

    Used for K = V style sharing.  No learnable parameters of its own.
    """

    def __init__(self, source: nn.Linear):
        super().__init__()
        self._source = [source]

    @property
    def source(self) -> nn.Linear:
        return self._source[0]

    def forward(self, x):
        return self.source(x)


class IdentityTiedNorm(nn.Module):
    """RMSNorm (or any norm) that delegates ``source(x)``.

    Used for shared-norm configurations.  No learnable parameters of its own.
    """

    def __init__(self, source: nn.Module):
        super().__init__()
        self._source = [source]

    @property
    def source(self) -> nn.Module:
        return self._source[0]

    def forward(self, x):
        return self.source(x)


class RotationTiedLinear(nn.Module):
    """Linear layer that computes ``rotate_2d(source(x), theta)``.

    Applies a learnable 2D rotation to pairs of output dimensions.
    Used for K = rotation(Q) tying.  1 learnable parameter (the angle).
    Requires source output dimension to be even.
    """

    def __init__(self, source: nn.Linear):
        super().__init__()
        self._source = [source]
        self.theta = nn.Parameter(torch.tensor(0.1))

    @property
    def source(self) -> nn.Linear:
        return self._source[0]

    def forward(self, x):
        out = self.source(x)
        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)
        x0, x1 = out[..., 0::2], out[..., 1::2]
        rotated = torch.stack([
            x0 * cos_t - x1 * sin_t,
            x0 * sin_t + x1 * cos_t,
        ], dim=-1).flatten(-2)
        return rotated


class LogScalarTiedLinear(nn.Module):
    """Linear layer that computes ``exp(log_alpha) * source(x)``.

    Like :class:`ScalarTiedLinear` but parameterises the scalar in log space.
    This prevents the alpha from collapsing toward zero (a common failure mode
    in very small models like 30p where the gate scalar decays to ~0.04).
    1 learnable parameter.  Initialised so that ``exp(log_alpha) = 1.0``.
    """

    def __init__(self, source: nn.Linear):
        super().__init__()
        self._source = [source]
        self.log_alpha = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0

    @property
    def source(self) -> nn.Linear:
        return self._source[0]

    def forward(self, x):
        return torch.exp(self.log_alpha) * self.source(x)


class VectorScalarTiedLinear(nn.Module):
    """Linear layer that computes ``alpha_vec * source(x)`` element-wise.

    Instead of a single scalar, uses a learnable vector with one element per
    output feature.  This resolves gradient conflicts when different output
    dimensions need different scales (e.g. gate_proj with ff=2 has 2 features
    that may need independent scaling).  Adds ``out_features`` learnable
    parameters (vs 1 for ScalarTiedLinear).
    """

    def __init__(self, source: nn.Linear):
        super().__init__()
        self._source = [source]
        out_features = source.weight.shape[0]
        self.alpha_vec = nn.Parameter(torch.ones(out_features))

    @property
    def source(self) -> nn.Linear:
        return self._source[0]

    def forward(self, x):
        return self.alpha_vec * self.source(x)


class RotationTransposeTiedLinear(nn.Module):
    """Linear layer: ``F.linear(rotate_2d(x, theta), source.weight.t())``.

    Applies a learnable 2D rotation to pairs of *input* dimensions, then
    computes the transpose-tied projection.  Used for
    ``down = up^T ∘ R(θ)`` tying.  1 learnable parameter (the angle).
    Requires the *input* dimension (source's ``out_features``) to be even.
    """

    def __init__(self, source: nn.Linear):
        super().__init__()
        self._source = [source]
        self.theta = nn.Parameter(torch.tensor(0.1))

    @property
    def source(self) -> nn.Linear:
        return self._source[0]

    def forward(self, x):
        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)
        x0, x1 = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack([
            x0 * cos_t - x1 * sin_t,
            x0 * sin_t + x1 * cos_t,
        ], dim=-1).flatten(-2)
        return F.linear(rotated, self.source.weight.t())


class NegateTiedLinear(nn.Module):
    """Linear layer that computes ``F.linear(x, -source.weight)``.

    Used for ``down = -up^T`` style experiments.  No learnable parameters.
    """

    def __init__(self, source: nn.Linear):
        super().__init__()
        self._source = [source]

    @property
    def source(self) -> nn.Linear:
        return self._source[0]

    def forward(self, x):
        return F.linear(x, -self.source.weight)


# ============================================================================
# Helpers
# ============================================================================

def parse_spec(spec: str) -> tuple[str, str]:
    """Parse a follower spec into ``(transform, path)``.

    ``"scalar:block.attn.k_proj"`` -> ``("scalar", "block.attn.k_proj")``
    ``"block.attn.v_proj"``        -> ``("identity", "block.attn.v_proj")``
    """
    if ":" in spec:
        transform, path = spec.split(":", 1)
        return transform, path
    return "identity", spec


def _split_path(path: str) -> tuple[str, str]:
    """Split ``'block.attn.k_proj'`` into ``('block.attn', 'k_proj')``."""
    parts = path.rsplit(".", 1)
    if len(parts) == 1:
        return "", parts[0]
    return parts[0], parts[1]


def count_unique_params(model: nn.Module) -> int:
    """Count unique parameters by ``data_ptr`` deduplication.

    Correctly handles weight tying and shared parameters.
    """
    seen = set()
    total = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += p.numel()
    return total


# ============================================================================
# Core function
# ============================================================================

_TIED_CLASSES = {
    "scalar": ScalarTiedLinear,
    "logscalar": LogScalarTiedLinear,
    "vecscalar": VectorScalarTiedLinear,
    "transpose": TransposeTiedLinear,
    "identity": None,  # decided at runtime based on module type
    "negate": NegateTiedLinear,
    "rotation": RotationTiedLinear,
    "rottranspose": RotationTransposeTiedLinear,
}


def apply_tie_groups(model: nn.Module, tie_groups: list[list[str]]) -> nn.Module:
    """Apply declarative tie groups to a model via module replacement.

    Parameters
    ----------
    model : nn.Module
        Model to modify **in place**.
    tie_groups : list of groups
        Each group is a list of specs.  The first element is the master path.
        Subsequent elements are follower specs, optionally prefixed with a
        transform (``"scalar:path"``, ``"transpose:path"``, ``"negate:path"``).

    Returns
    -------
    nn.Module
        The same model (modified in place).
    """
    for group in tie_groups:
        if len(group) < 2:
            raise ValueError(f"Tie group must have at least 2 members, got: {group}")

        master_path = group[0]
        master_module = model.get_submodule(master_path)

        for spec in group[1:]:
            transform, follower_path = parse_spec(spec)

            if transform not in _TIED_CLASSES:
                raise ValueError(
                    f"Unknown transform '{transform}'. "
                    f"Supported: {list(_TIED_CLASSES.keys())}"
                )

            parent_path, attr_name = _split_path(follower_path)
            if parent_path:
                parent = model.get_submodule(parent_path)
            else:
                parent = model

            if transform == "scalar":
                replacement = ScalarTiedLinear(master_module)
            elif transform == "logscalar":
                replacement = LogScalarTiedLinear(master_module)
            elif transform == "vecscalar":
                replacement = VectorScalarTiedLinear(master_module)
            elif transform == "transpose":
                replacement = TransposeTiedLinear(master_module)
            elif transform == "negate":
                replacement = NegateTiedLinear(master_module)
            elif transform == "rotation":
                replacement = RotationTiedLinear(master_module)
            elif transform == "rottranspose":
                replacement = RotationTransposeTiedLinear(master_module)
            else:  # identity
                if isinstance(master_module, nn.Linear):
                    replacement = IdentityTiedLinear(master_module)
                else:
                    replacement = IdentityTiedNorm(master_module)

            setattr(parent, attr_name, replacement)

    return model
