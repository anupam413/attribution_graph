"""
src/model/hooks.py
------------------
Reusable hook utilities for model interventions.

Provides clean abstractions for common intervention patterns:
  - Zero ablation
  - Mean ablation  
  - Activation patching
  - Gradient interception

Usage:
    from src.model.hooks import zero_ablate, mean_ablate, patch_with
    
    # Zero ablate MLP at layer 5
    result = model.run_with_patch(
        prompt, layer=5, patch_fn=zero_ablate
    )
    
    # Mean ablate (replace with dataset mean)
    result = model.run_with_patch(
        prompt, layer=5, patch_fn=mean_ablate(dataset_mean)
    )
"""

import torch
import torch.nn as nn
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Basic ablation functions
# ---------------------------------------------------------------------------

def zero_ablate(activation: torch.Tensor, hook) -> torch.Tensor:
    """Replace activation with zeros."""
    return torch.zeros_like(activation)


def mean_ablate(mean_activation: torch.Tensor) -> Callable:
    """
    Return a hook function that replaces activation with a fixed mean.
    
    Args:
        mean_activation: The mean activation to use (shape must match)
        
    Returns:
        Hook function that replaces activations with the mean
    """
    def _mean_ablate_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        # Broadcast mean to batch dimension
        return mean_activation.expand_as(activation)
    return _mean_ablate_fn


def patch_with(new_activation: torch.Tensor) -> Callable:
    """
    Return a hook function that replaces activation with a specific tensor.
    
    Args:
        new_activation: Tensor to replace with
        
    Returns:
        Hook function that replaces activations
    """
    def _patch_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        return new_activation.expand_as(activation)
    return _patch_fn


# ---------------------------------------------------------------------------
# Feature-level interventions
# ---------------------------------------------------------------------------

def ablate_features(feature_indices: list[int], transcoder) -> Callable:
    """
    Ablate specific transcoder features by setting them to zero.
    
    Args:
        feature_indices: List of feature indices to ablate
        transcoder: Transcoder instance
        
    Returns:
        Hook function that ablates specified features
    """
    def _ablate_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        # Encode to features
        pre_acts = activation @ transcoder.W_enc.T + transcoder.b_enc
        features = torch.nn.functional.relu(pre_acts - transcoder.theta)
        
        # Zero out specified features
        features[:, :, feature_indices] = 0
        
        # Decode back
        reconstruction = features @ transcoder.W_dec.T + transcoder.b_dec
        return reconstruction
    return _ablate_fn


def boost_features(feature_indices: list[int], scale: float, transcoder) -> Callable:
    """
    Scale specific transcoder features up or down.
    
    Args:
        feature_indices: List of feature indices to scale
        scale: Scaling factor (2.0 = double, 0.5 = halve)
        transcoder: Transcoder instance
        
    Returns:
        Hook function that scales specified features
    """
    def _boost_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        # Encode to features
        pre_acts = activation @ transcoder.W_enc.T + transcoder.b_enc
        features = torch.nn.functional.relu(pre_acts - transcoder.theta)
        
        # Scale specified features
        features[:, :, feature_indices] *= scale
        
        # Decode back
        reconstruction = features @ transcoder.W_dec.T + transcoder.b_dec
        return reconstruction
    return _boost_fn


# ---------------------------------------------------------------------------
# Gradient interception
# ---------------------------------------------------------------------------

class StopGradient(torch.autograd.Function):
    """
    Stop gradients from flowing backward through this point.
    
    Useful for computing "virtual weights" in attribution graphs.
    """
    @staticmethod
    def forward(ctx, x):
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output)


def stop_gradient(activation: torch.Tensor, hook) -> torch.Tensor:
    """Hook function that stops gradients."""
    return StopGradient.apply(activation)


def scale_gradient(scale: float) -> Callable:
    """
    Return a hook function that scales gradients by a factor.
    
    Args:
        scale: Gradient scaling factor
        
    Returns:
        Hook function that scales gradients
    """
    class ScaleGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x
        
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output * scale
    
    def _scale_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        return ScaleGradient.apply(activation)
    return _scale_fn


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------

def compose_hooks(*hook_fns: Callable) -> Callable:
    """
    Compose multiple hook functions into a single hook.
    
    Args:
        *hook_fns: Hook functions to compose (applied left to right)
        
    Returns:
        Composed hook function
    """
    def _composed_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        result = activation
        for fn in hook_fns:
            result = fn(result, hook)
        return result
    return _composed_fn


# ---------------------------------------------------------------------------
# Context manager for temporary hooks
# ---------------------------------------------------------------------------

class TemporaryHook:
    """
    Context manager for temporarily adding hooks to a model.
    
    Usage:
        with TemporaryHook(model, hook_name, hook_fn):
            output = model(input)
        # Hook is automatically removed
    """
    def __init__(self, model, hook_name: str, hook_fn: Callable):
        self.model = model
        self.hook_name = hook_name
        self.hook_fn = hook_fn
        self.handle = None
    
    def __enter__(self):
        self.handle = self.model.add_hook(self.hook_name, self.hook_fn)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()