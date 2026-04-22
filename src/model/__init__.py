"""Model loading and activation caching."""

from .loader import ModelWrapper, ActivationCache, ModelConfig, ForwardResult
from .hooks import (
    zero_ablate,
    mean_ablate,
    patch_with,
    ablate_features,
    boost_features,
    stop_gradient,
    scale_gradient,
    compose_hooks,
    TemporaryHook,
)

__all__ = [
    "ModelWrapper",
    "ActivationCache",
    "ModelConfig",
    "ForwardResult",
    "zero_ablate",
    "mean_ablate",
    "patch_with",
    "ablate_features",
    "boost_features",
    "stop_gradient",
    "scale_gradient",
    "compose_hooks",
    "TemporaryHook",
]