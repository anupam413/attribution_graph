"""
src/replacement_model/local.py
------------------------------
Phase 3c: Local replacement model with frozen attention and transcoded MLPs.

The replacement model:
  1. Freezes all attention patterns (computed once on the input prompt)
  2. Replaces each MLP with its transcoder approximation
  3. Adds per-layer "error nodes" to account for reconstruction error
  4. Produces the same final output as the original model (approximately)

This enables attribution graphs because:
  - Features (not neurons) become the compute nodes
  - Error nodes capture what the transcoder missed
  - Attention is deterministic, so gradients flow cleanly through features

Architecture:
  For each layer ℓ:
    - Compute attention output (frozen from original forward pass)
    - residual = prev_residual + attn_output
    - mlp_in = residual
    - transcoder_out = transcoder_ℓ(mlp_in)
    - error = true_mlp_out - transcoder_out
    - residual = residual + transcoder_out + error
    
  The error nodes ensure the replacement model's outputs match the original.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

from src.transcoder.per_layer import PerLayerTranscoder
from src.transcoder.cross_layer import CrossLayerTranscoder


@dataclass
class ReplacementModelConfig:
    """Configuration for the local replacement model."""
    use_error_nodes: bool = True  # Include error correction nodes
    detach_errors: bool = True    # Stop gradients through error nodes
    cache_attention: bool = True  # Freeze attention patterns


@dataclass
class ReplacementModelOutput:
    """Output from the replacement model forward pass."""
    logits: torch.Tensor            # (batch, seq_len, vocab_size)
    feature_acts: list[torch.Tensor]  # List of (batch, seq_len, n_features) per layer
    error_acts: list[torch.Tensor]    # List of (batch, seq_len, d_model) per layer
    final_residual: torch.Tensor    # (batch, seq_len, d_model)


class LocalReplacementModel(nn.Module):
    """
    Replacement model that substitutes MLPs with transcoders.
    
    This model:
    - Runs the original model once to cache attention patterns
    - Replaces each MLP with a transcoder
    - Adds error nodes to maintain fidelity
    - Allows attribution graphs to be built on transcoder features
    
    Usage:
        # With per-layer transcoders (PLT)
        transcoders = [tc_layer_0, tc_layer_1, ...]
        replacement = LocalReplacementModel(model, transcoders, is_clt=False)
        
        # With cross-layer transcoder (CLT)
        replacement = LocalReplacementModel(model, clt, is_clt=True)
        
        # Forward pass
        tokens = model.to_tokens("Michael Jordan plays the sport of")
        output = replacement(tokens)
        print(output.logits.shape)  # (1, seq_len, vocab_size)
    """
    
    def __init__(
        self,
        base_model: HookedTransformer,
        transcoders: Union[PerLayerTranscoder, CrossLayerTranscoder, list[PerLayerTranscoder]],
        cfg: Optional[ReplacementModelConfig] = None,
        is_clt: bool = False,
    ):
        super().__init__()
        self.base_model = base_model
        self.cfg = cfg or ReplacementModelConfig()
        self.is_clt = is_clt
        
        # Store transcoders
        if is_clt:
            self.clt = transcoders
            self.transcoders = None
            self.n_layers = transcoders.cfg.n_layers
        else:
            if isinstance(transcoders, list):
                self.transcoders = nn.ModuleList(transcoders)
            else:
                # Single transcoder - assume it's for all layers
                self.transcoders = transcoders
            self.clt = None
            self.n_layers = base_model.cfg.n_layers
        
        # Cached activations from original model
        self.cached_attn_out: Optional[list[torch.Tensor]] = None
        self.cached_mlp_out: Optional[list[torch.Tensor]] = None
        self.cached_resid: Optional[list[torch.Tensor]] = None
        
        # Feature activations (populated during forward pass)
        self.feature_acts: Optional[list[torch.Tensor]] = None  # per-layer features
        self.error_acts: Optional[list[torch.Tensor]] = None     # per-layer errors
    
    @torch.no_grad()
    def cache_base_model_activations(self, tokens: torch.Tensor) -> None:
        """
        Run the base model once and cache all intermediate activations.
        
        This freezes the attention patterns and provides ground truth
        for error node computation.
        
        Args:
            tokens: (batch, seq_len) token IDs
        """
        cache_attn = []
        cache_mlp = []
        cache_resid = []
        
        _, cache = self.base_model.run_with_cache(tokens)
        
        for layer_idx in range(self.n_layers):
            # Attention output: residual delta from attention
            attn_out = cache[f"blocks.{layer_idx}.attn.hook_result"]
            cache_attn.append(attn_out)
            
            # MLP output: residual delta from MLP
            mlp_out = cache[f"blocks.{layer_idx}.hook_mlp_out"]
            cache_mlp.append(mlp_out)
            
            # Residual stream after this layer
            resid = cache[f"blocks.{layer_idx}.hook_resid_post"]
            cache_resid.append(resid)
        
        self.cached_attn_out = cache_attn
        self.cached_mlp_out = cache_mlp
        self.cached_resid = cache_resid
    
    def forward(self, tokens: torch.Tensor) -> ReplacementModelOutput:
        """
        Forward pass through replacement model.
        
        Args:
            tokens: (batch, seq_len) token IDs
            
        Returns:
            ReplacementModelOutput with logits, features, errors
        """
        # Cache base model activations if needed
        if self.cfg.cache_attention and self.cached_attn_out is None:
            self.cache_base_model_activations(tokens)
        
        # Embed tokens
        residual = self.base_model.embed(tokens)  # (batch, seq_len, d_model)
        
        # Storage for features and errors
        all_features = []
        all_errors = []
        
        # Process each layer
        for layer_idx in range(self.n_layers):
            # 1. Add cached attention output (frozen)
            if self.cfg.cache_attention:
                attn_out = self.cached_attn_out[layer_idx]
                if attn_out.ndim ==4: # if has head dimension, sum over heads
                    attn_out = attn_out.sum(dim=2)  # (batch, seq_len, d_model)
            else:
                # Recompute attention (for ablation studies)
                attn_out = self.base_model.blocks[layer_idx].attn(residual)
            
            residual = residual + attn_out
            
            # 2. Apply transcoder to residual (MLP input)
            mlp_in = residual
            
            if self.is_clt:
                # Cross-layer transcoder
                # Need to pass all layer inputs
                batch_size, seq_len, d_model = mlp_in.shape
                
                # Collect all MLP inputs up to this layer
                enc_inputs = []
                temp_resid = self.base_model.embed(tokens)
                for prev_layer in range(layer_idx + 1):
                    if prev_layer > 0:
                        prev_attn = self.cached_attn_out[prev_layer - 1]
                        temp_resid = temp_resid + prev_attn
                    enc_inputs.append(temp_resid)
                    if prev_layer < layer_idx:
                        # Add prev MLP for next iteration
                        if self.cfg.use_error_nodes:
                            tc_out_prev = self.clt.forward_from_layer(
                                torch.stack(enc_inputs, dim=1), prev_layer
                            ).reconstructions[prev_layer]
                            error_prev = self.cached_mlp_out[prev_layer] - tc_out_prev
                            temp_resid = temp_resid + tc_out_prev + error_prev
                        else:
                            temp_resid = temp_resid + self.cached_mlp_out[prev_layer]
                
                # Stack inputs and forward through CLT
                enc_input_tensor = torch.stack(enc_inputs, dim=1)  # (batch, layer_idx+1, d_model)
                tc_output = self.clt.forward_from_layer(enc_input_tensor, layer_idx)
                transcoder_out = tc_output.reconstructions[layer_idx]
                features = tc_output.feature_acts[layer_idx]
                
            else:
                # Per-layer transcoder
                tc_output = self.transcoders[layer_idx](mlp_in)
                transcoder_out = tc_output.reconstruction
                features = tc_output.feature_acts
            
            all_features.append(features)
            
            # 3. Compute error node
            if self.cfg.use_error_nodes:
                true_mlp_out = self.cached_mlp_out[layer_idx]
                error = true_mlp_out - transcoder_out
                
                if self.cfg.detach_errors:
                    error = error.detach()  # Stop gradients through error
                
                all_errors.append(error)
            else:
                error = torch.zeros_like(transcoder_out)
                all_errors.append(error)
            
            # 4. Update residual stream
            residual = residual + transcoder_out + error
        
        # Final layer norm and unembed
        residual = self.base_model.ln_final(residual)
        logits = self.base_model.unembed(residual)
        
        # Store for later attribution
        self.feature_acts = all_features
        self.error_acts = all_errors
        
        return ReplacementModelOutput(
            logits=logits,
            feature_acts=all_features,
            error_acts=all_errors,
            final_residual=residual,
        )
    
    def get_feature_activations(self, layer_idx: int) -> torch.Tensor:
        """Get feature activations for a specific layer."""
        if self.feature_acts is None:
            raise RuntimeError("Run forward() first to populate feature activations")
        return self.feature_acts[layer_idx]
    
    def get_error_activations(self, layer_idx: int) -> torch.Tensor:
        """Get error node activations for a specific layer."""
        if self.error_acts is None:
            raise RuntimeError("Run forward() first to populate error activations")
        return self.error_acts[layer_idx]
    
    @torch.no_grad()
    def compare_with_base(self, tokens: torch.Tensor) -> dict:
        """
        Compare replacement model output with base model.
        
        Returns:
            dict with KL divergence, top-1 accuracy, etc.
        """
        # Base model
        base_logits = self.base_model(tokens)
        
        # Replacement model
        repl_output = self(tokens)
        repl_logits = repl_output.logits
        
        # Compute metrics
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(repl_logits, dim=-1),
            torch.nn.functional.softmax(base_logits, dim=-1),
            reduction='batchmean',
        )
        
        # Top-1 accuracy
        base_top1 = base_logits.argmax(dim=-1)
        repl_top1 = repl_logits.argmax(dim=-1)
        top1_acc = (base_top1 == repl_top1).float().mean()
        
        # Logit MSE
        logit_mse = torch.nn.functional.mse_loss(repl_logits, base_logits)
        
        return {
            'kl_divergence': kl_div.item(),
            'top1_accuracy': top1_acc.item(),
            'logit_mse': logit_mse.item(),
        }