"""
src/model/loader.py
-------------------
Phase 2: Base model setup using TransformerLens.

Provides:
  - ModelWrapper       : loads a HookedTransformer, exposes clean hook API
  - ActivationCache    : runs forward passes and saves MLP in/out tensors
                         needed to train the transcoder in Phase 3

Supported models (all work identically via TransformerLens):
  - gpt2              (117M)  ← recommended starting point
  - gpt2-medium       (345M)
  - EleutherAI/pythia-70m
  - EleutherAI/pythia-160m

Usage:
    from src.model.loader import ModelWrapper, ActivationCache

    model = ModelWrapper.load("gpt2")
    out   = model.run("The Eiffel Tower is in")
    print(out.top_token)          # → " Paris"

    cache = ActivationCache(model)
    cache.collect(prompts, save_path="data/activation_cache/gpt2.pt")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """All tunable settings for the base model."""
    model_name: str = "gpt2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float32 for training stability; bfloat16 for faster inference
    dtype: torch.dtype = torch.float32
    # Keep the model in eval mode (no dropout) during all experiments
    eval_mode: bool = True
    # Centre the writing subspace (recommended for interpretability work)
    center_writing_weights: bool = True
    center_unembed_weights: bool = True
    fold_ln: bool = False        # absorb LayerNorm into adjacent weights


# ---------------------------------------------------------------------------
# Forward pass result
# ---------------------------------------------------------------------------

@dataclass
class ForwardResult:
    """Everything you might want from a single forward pass."""
    logits: torch.Tensor              # (seq_len, vocab_size)
    top_token: str                    # decoded top-1 prediction at last position
    top_token_id: int
    top_prob: float
    # Hook caches (populated only when requested)
    residual_stream: dict[str, torch.Tensor] = field(default_factory=dict)
    mlp_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    mlp_outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    attention_patterns: dict[str, torch.Tensor] = field(default_factory=dict)
    # LayerNorm denominators — populated when cache_attn=True.
    # Kept separate from attention_patterns for clarity.
    # These are frozen in the Local Replacement Model (Phase 3c).
    ln_scales: dict[str, torch.Tensor] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class ModelWrapper:
    """
    Thin wrapper around a TransformerLens HookedTransformer.

    The main job of this class is to:
      1. Load the model with interpretability-friendly settings.
      2. Provide a clean run() method that returns a ForwardResult.
      3. Expose hook names in a consistent format used by downstream modules.
    """

    def __init__(self, model: HookedTransformer, config: ModelConfig):
        self.model = model
        self.cfg = config
        self.n_layers: int = model.cfg.n_layers
        self.d_model: int = model.cfg.d_model
        self.d_mlp: int = model.cfg.d_mlp
        self.vocab_size: int = model.cfg.d_vocab

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, model_name: str = "gpt2", **kwargs) -> "ModelWrapper":
        """
        Load a model by name and return a ModelWrapper.

        Args:
            model_name: TransformerLens model string, e.g. "gpt2",
                        "gpt2-medium", "EleutherAI/pythia-160m"
            **kwargs:   Overrides for ModelConfig fields.

        Example:
            model = ModelWrapper.load("gpt2")
            model = ModelWrapper.load("EleutherAI/pythia-160m", device="cpu")
        """
        cfg = ModelConfig(model_name=model_name, **kwargs)

        print(f"Loading {model_name} on {cfg.device} ({cfg.dtype}) ...")
        model = HookedTransformer.from_pretrained(
            model_name,
            center_writing_weights=cfg.center_writing_weights,
            # center_unembed_weights=cfg.center_unembed_weights,
            fold_ln=cfg.fold_ln,
        )
        model = model.to(cfg.device).to(cfg.dtype)

        if cfg.eval_mode:
            model.eval()

        wrapper = cls(model, cfg)
        print(f"  Layers : {wrapper.n_layers}")
        print(f"  d_model: {wrapper.d_model}")
        print(f"  d_mlp  : {wrapper.d_mlp}")
        return wrapper

    # ------------------------------------------------------------------
    # Hook name helpers (used throughout the codebase)
    # ------------------------------------------------------------------

    def hook_resid_pre(self, layer: int) -> str:
        """Residual stream BEFORE layer `layer`."""
        return f"blocks.{layer}.hook_resid_pre"

    def hook_resid_post(self, layer: int) -> str:
        """Residual stream AFTER layer `layer` (after MLP)."""
        return f"blocks.{layer}.hook_resid_post"

    def hook_mlp_in(self, layer: int) -> str:
        # for gpt-2 in transformerlens the input to the MLP block after attention is added is called hook_resid_mid
        """Input to the MLP in layer `layer` (= residual stream post-attn)."""
        return f"blocks.{layer}.hook_resid_mid"

    def hook_mlp_out(self, layer: int) -> str:
        """Output of the MLP in layer `layer`."""
        return f"blocks.{layer}.hook_mlp_out"
    
    def hook_attn_out(self, layer: int) -> str:
        """Output of the attention in layer `layer`."""
        return f"blocks.{layer}.hook_attn_out"

    def hook_attn_pattern(self, layer: int) -> str:
        """Attention pattern (softmax output) in layer `layer`."""
        return f"blocks.{layer}.attn.hook_pattern"

    def hook_ln_scale(self, layer: int) -> str:
        """LayerNorm scale (denominator) before the MLP in layer `layer`.
        
        Important: this is what we freeze in the local replacement model.
        """
        return f"blocks.{layer}.ln2.hook_scale"

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def tokenize(self, prompt: str) -> torch.Tensor:
        """Return (1, seq_len) token id tensor on the model's device."""
        return self.model.to_tokens(prompt)

    def decode(self, token_id: int) -> str:
        return self.model.tokenizer.decode([token_id])

    def decode_tokens(self, token_ids: torch.Tensor) -> list[str]:
        return [self.decode(t.item()) for t in token_ids.squeeze()]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        cache_resid: bool = False,
        cache_mlp: bool = False,
        cache_attn: bool = False,
    ) -> ForwardResult:
        """
        Run a forward pass and return a ForwardResult.

        Args:
            prompt      : The input string.
            cache_resid : If True, store residual stream at every layer.
            cache_mlp   : If True, store MLP inputs and outputs at every layer.
            cache_attn  : If True, store attention patterns at every layer.

        Returns:
            ForwardResult with logits, top predicted token, and requested caches.

        Example:
            result = model.run("The Eiffel Tower is in", cache_mlp=True)
            print(result.top_token)
            mlp_out_layer0 = result.mlp_outputs["layer_0"]
        """
        tokens = self.tokenize(prompt)

        # Decide which hook names to capture
        names_to_cache: list[str] = []
        if cache_resid:
            names_to_cache += [self.hook_resid_pre(l) for l in range(self.n_layers)]
            names_to_cache += [self.hook_resid_post(l) for l in range(self.n_layers)]
        if cache_mlp:
            names_to_cache += [self.hook_mlp_in(l)  for l in range(self.n_layers)]
            names_to_cache += [self.hook_mlp_out(l) for l in range(self.n_layers)]
        # if cache_mlp:
        #     names_to_cache += [self.hook_resid_pre(l) for l in range(self.n_layers)]
        #     names_to_cache += [self.hook_attn_out(l)  for l in range(self.n_layers)]
        #     names_to_cache += [self.hook_mlp_out(l)   for l in range(self.n_layers)]
        if cache_attn:
            names_to_cache += [self.hook_attn_pattern(l) for l in range(self.n_layers)]
            names_to_cache += [self.hook_ln_scale(l)     for l in range(self.n_layers)]

        with torch.no_grad():
            logits, cache = self.model.run_with_cache(
                tokens,
                names_filter=names_to_cache if names_to_cache else None,
                return_type="logits",
            )

        # logits shape: (1, seq_len, vocab_size) → take last position
        last_logits = logits[0, -1]                          # (vocab_size,)
        probs       = torch.softmax(last_logits, dim=-1)
        top_id      = last_logits.argmax().item()

        # Build output caches with human-friendly keys
        resid_cache = {}
        mlp_in_cache = {}
        mlp_out_cache = {}
        attn_cache = {}
        ln_scale_cache = {}

        for layer in range(self.n_layers):
            key = f"layer_{layer}"
            if cache_resid:
                resid_cache[f"pre_{layer}"]  = cache[self.hook_resid_pre(layer)]
                resid_cache[f"post_{layer}"] = cache[self.hook_resid_post(layer)]
            if cache_mlp:
                mlp_in_cache[key]  = cache[self.hook_mlp_in(layer)]
                mlp_out_cache[key] = cache[self.hook_mlp_out(layer)]
            # if cache_mlp:
            #     mlp_in_cache[key] = (
            #         cache[self.hook_resid_pre(layer)] +
            #         cache[self.hook_attn_out(layer)]
            #     )
            #     mlp_out_cache[key] = cache[self.hook_mlp_out(layer)]
            if cache_attn:
                attn_cache[key]     = cache[self.hook_attn_pattern(layer)]
                ln_scale_cache[key] = cache[self.hook_ln_scale(layer)]

        return ForwardResult(
            logits=last_logits.cpu(),
            top_token=self.decode(top_id),
            top_token_id=top_id,
            top_prob=probs[top_id].item(),
            residual_stream=resid_cache,
            mlp_inputs=mlp_in_cache,
            mlp_outputs=mlp_out_cache,
            attention_patterns=attn_cache,
            ln_scales=ln_scale_cache,
        )

    def run_batch(
        self,
        prompts: list[str],
        cache_mlp: bool = False,
    ) -> list[ForwardResult]:
        """Run multiple prompts. Useful for collecting training data."""
        return [self.run(p, cache_mlp=cache_mlp) for p in prompts]

    # ------------------------------------------------------------------
    # Intervention helpers (used in Phase 4 validation)
    # ------------------------------------------------------------------

    def run_with_patch(
        self,
        prompt: str,
        layer: int,
        patch_fn,
    ) -> ForwardResult:
        """
        Run the model with a custom hook patching MLP output at `layer`.

        Args:
            prompt  : Input string.
            layer   : Which layer's MLP output to patch.
            patch_fn: Callable(tensor, hook) → tensor.
                      Receives the MLP output tensor and returns the patched version.

        Example (zero-ablate MLP at layer 5):
            result = model.run_with_patch(
                "The capital of France is",
                layer=5,
                patch_fn=lambda x, hook: torch.zeros_like(x),
            )
        """
        tokens = self.tokenize(prompt)
        hook_name = self.hook_mlp_out(layer)

        with torch.no_grad():
            logits = self.model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, patch_fn)],
                return_type="logits",
            )

        last_logits = logits[0, -1]
        top_id = last_logits.argmax().item()
        probs = torch.softmax(last_logits, dim=-1)

        return ForwardResult(
            logits=last_logits.cpu(),
            top_token=self.decode(top_id),
            top_token_id=top_id,
            top_prob=probs[top_id].item(),
        )

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------

    def check_hooks(self) -> None:
        """
        Print available hook names for inspection.
        Useful for verifying hook names match TransformerLens internals.
        """
        print(f"\nHook names for {self.cfg.model_name}:")
        for l in range(min(2, self.n_layers)):
            print(f"  layer {l}: {self.hook_mlp_in(l)}")
            print(f"  layer {l}: {self.hook_mlp_out(l)}")
            print(f"  layer {l}: {self.hook_attn_pattern(l)}")
            print(f"  layer {l}: {self.hook_ln_scale(l)}")
        print("  ...")


# ---------------------------------------------------------------------------
# Activation Cache  (training data for Phase 3 transcoder)
# ---------------------------------------------------------------------------

class ActivationCache:
    """
    Collect and save (mlp_input, mlp_output) pairs from a corpus of prompts.

    These pairs are the training data for the transcoder in Phase 3.
    Each pair is a tensor of shape (seq_len, d_model).

    Usage:
        cache = ActivationCache(model)
        cache.collect(prompts, save_path="data/activation_cache/gpt2.pt")

        # Later, load and iterate:
        data = torch.load("data/activation_cache/gpt2.pt")
        for layer_idx, (inp, out) in enumerate(zip(data["mlp_in"], data["mlp_out"])):
            ...
    """

    def __init__(self, model: ModelWrapper):
        self.model = model
        self.n_layers = model.n_layers

    def collect(
        self,
        prompts: list[str],
        save_path: Optional[str] = None,
        max_seq_len: int = 128,
        verbose: bool = True,
    ) -> dict[str, list[torch.Tensor]]:
        """
        Run each prompt through the model and collect MLP inputs/outputs.

        Args:
            prompts     : List of text strings.
            save_path   : If given, save the result to this .pt file.
            max_seq_len : Truncate prompts longer than this (saves memory).
            verbose     : Print progress.

        Returns:
            dict with keys:
              "mlp_in"  : list of length n_layers, each a (N_tokens, d_model) tensor
              "mlp_out" : list of length n_layers, each a (N_tokens, d_model) tensor
              "prompts" : the original prompts (for debugging)
        """
        # Accumulators: one list per layer
        all_mlp_in  = [[] for _ in range(self.n_layers)]
        all_mlp_out = [[] for _ in range(self.n_layers)]

        for i, prompt in enumerate(prompts):
            if verbose and i % 50 == 0:
                print(f"  Caching activations: {i}/{len(prompts)}", end="\r")

            # Truncate to keep memory manageable
            tokens = self.model.tokenize(prompt)
            if tokens.shape[1] > max_seq_len:
                tokens = tokens[:, :max_seq_len]
            prompt_truncated = self.model.model.tokenizer.decode(tokens[0])

            result = self.model.run(prompt_truncated, cache_mlp=True)

            for layer in range(self.n_layers):
                key = f"layer_{layer}"
                # Squeeze batch dim, keep (seq_len, d_model), move to CPU
                inp = result.mlp_inputs[key].squeeze(0).cpu()
                out = result.mlp_outputs[key].squeeze(0).cpu()
                all_mlp_in[layer].append(inp)
                all_mlp_out[layer].append(out)

        if verbose:
            print(f"\n  Done. Collected {len(prompts)} prompts.")

        # Concatenate along the token dimension for each layer
        data = {
            "mlp_in":  [torch.cat(all_mlp_in[l],  dim=0) for l in range(self.n_layers)],
            "mlp_out": [torch.cat(all_mlp_out[l], dim=0) for l in range(self.n_layers)],
            "prompts": prompts,
            "model_name": self.model.cfg.model_name,
        }

        if verbose:
            n_tokens = data["mlp_in"][0].shape[0]
            print(f"  Total tokens per layer: {n_tokens:,}")
            print(f"  Tensor shape: {data['mlp_in'][0].shape}")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, save_path)
            print(f"  Saved to {save_path}")

        return data

    @staticmethod
    def load(path: str) -> dict:
        """Load a previously saved activation cache."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        print(f"Loaded activation cache from {path}")
        print(f"  Model     : {data.get('model_name', 'unknown')}")
        print(f"  Layers    : {len(data['mlp_in'])}")
        print(f"  Tokens    : {data['mlp_in'][0].shape[0]:,}")
        return data


# ---------------------------------------------------------------------------
# Quick validation script
# ---------------------------------------------------------------------------

def _validate(model_name: str = "gpt2"):
    """
    Run a basic sanity check to confirm the model and hooks work correctly.
    Called when this file is run directly: python -m src.model.loader
    """
    print("=" * 60)
    print(f"Validating ModelWrapper with {model_name}")
    print("=" * 60)

    model = ModelWrapper.load(model_name)

    # 1. Basic forward pass
    print("\n[1] Basic forward pass")
    result = model.run("The Eiffel Tower is in")
    print(f"  Prompt : 'The Eiffel Tower is in'")
    print(f"  Top-1  : '{result.top_token}' ({result.top_prob:.1%})")

    # 2. Forward pass with caches
    print("\n[2] Forward pass with MLP cache")
    result = model.run("The capital of Germany is", cache_mlp=True)
    print(f"  Top-1  : '{result.top_token}' ({result.top_prob:.1%})")
    for layer in range(min(3, model.n_layers)):
        key = f"layer_{layer}"
        inp_shape = result.mlp_inputs[key].shape
        out_shape = result.mlp_outputs[key].shape
        print(f"  Layer {layer}: MLP in {inp_shape}, out {out_shape}")

    # 3. Intervention test: zero-ablate MLP at layer 0
    print("\n[3] Zero-ablation at layer 0")
    baseline = model.run("Michael Jordan plays the sport of")
    ablated  = model.run_with_patch(
        "Michael Jordan plays the sport of",
        layer=0,
        patch_fn=lambda x, hook: torch.zeros_like(x),
    )
    print(f"  Baseline : '{baseline.top_token}' ({baseline.top_prob:.1%})")
    print(f"  Ablated  : '{ablated.top_token}'  ({ablated.top_prob:.1%})")

    # 4. Small activation cache collection
    print("\n[4] Activation cache (3 prompts)")
    prompts = [
        "The Eiffel Tower is in",
        "Michael Jordan plays the sport of",
        "The capital of Japan is",
    ]
    cache = ActivationCache(model)
    data = cache.collect(prompts, verbose=True)
    print(f"  mlp_in[0] shape : {data['mlp_in'][0].shape}")
    print(f"  mlp_out[0] shape: {data['mlp_out'][0].shape}")

    # 5. Hook names
    print("\n[5] Hook names")
    model.check_hooks()

    print("\nAll checks passed.")


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    _validate(model_name)
