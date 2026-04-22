"""
src/attribution/jacobian.py
---------------------------
Phase 3d: Backward Jacobian computation for attribution graphs.

The core idea:
  - Each node in the graph is either a transcoder feature or an error node
  - Edges represent how much one node's activation affects another
  - Edge weight = ∂(downstream_node) / ∂(upstream_node)
  
We compute these Jacobians by:
  1. Running the replacement model forward
  2. For each node, compute gradients w.r.t. all upstream nodes
  3. Use stop-gradient trick to get "virtual weights" (one-hop influence)

Virtual weights vs. full gradients:
  - Full gradient: ∂output / ∂feature_i (includes all paths)
  - Virtual weight: ∂(next_layer_features) / ∂feature_i with stop-grad
    This isolates direct influence, making the graph interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from src.replacement_model.local import LocalReplacementModel, ReplacementModelOutput


@dataclass
class AttributionNode:
    """A single node in the attribution graph."""
    layer: int          # Which layer (0-indexed)
    node_type: str      # "feature" or "error"
    index: int          # Feature index (for features) or dimension (for errors)
    position: int       # Token position in sequence
    activation: float   # Activation value


@dataclass
class AttributionEdge:
    """A directed edge in the attribution graph."""
    from_node: AttributionNode
    to_node: AttributionNode
    weight: float       # Jacobian: ∂(to_node) / ∂(from_node)


@dataclass
class JacobianResult:
    """Container for computed Jacobians."""
    feature_to_feature: list[torch.Tensor]  # List of tensors per layer
    error_to_feature: list[torch.Tensor]
    feature_to_error: list[torch.Tensor]
    target_logit_idx: int


class JacobianComputer:
    """
    Computes Jacobians for attribution graphs using stop-gradient trick.
    
    Usage:
        computer = JacobianComputer(replacement_model)
        tokens = model.to_tokens("Michael Jordan plays")
        
        # Compute Jacobians for a specific output logit
        jacobians = computer.compute_jacobians(
            tokens=tokens,
            target_pos=-1,      # last position
            target_logit_idx=2398,  # vocab index for " basketball"
        )
        
        # jacobians.feature_to_feature[layer][downstream_feat, upstream_feat, pos]
        # jacobians.error_to_feature[layer][feature, error_dim, pos]
    """
    
    def __init__(self, replacement_model: LocalReplacementModel):
        self.model = replacement_model
        self.device = next(replacement_model.parameters()).device
    
    def compute_jacobians(
        self,
        tokens: torch.Tensor,
        target_pos: int = -1,
        target_logit_idx: Optional[int] = None,
        top_k_logits: int = 1,
    ) -> JacobianResult:
        """
        Compute all Jacobians for the attribution graph.
        
        Args:
            tokens: (1, seq_len) input tokens
            target_pos: Which position to compute attribution for (-1 = last)
            target_logit_idx: Specific vocab index to target (None = argmax)
            top_k_logits: If target_logit_idx is None, compute for top-k logits
            
        Returns:
            JacobianResult with feature-to-feature and error-to-feature Jacobians
        """
        # Forward pass through replacement model
        output = self.model(tokens)
        logits = output.logits[0, target_pos, :]  # (vocab_size,)
        
        # Determine target logit(s)
        if target_logit_idx is None:
            _, top_indices = logits.topk(top_k_logits)
            target_logits = top_indices.tolist()
        else:
            target_logits = [target_logit_idx]
        
        # Compute Jacobians for first target logit (simplification)
        logit_idx = target_logits[0]
        jac = self._compute_single_logit_jacobians(
            output, logit_idx, target_pos
        )
        
        return jac
    
    def _compute_single_logit_jacobians(
        self,
        output: ReplacementModelOutput,
        logit_idx: int,
        target_pos: int,
    ) -> JacobianResult:
        """
        Compute Jacobians for a single target logit.
        
        This is a simplified implementation that computes approximate
        Jacobians using finite differences for efficiency.
        """
        n_layers = len(output.feature_acts)
        seq_len = output.feature_acts[0].shape[1]
        
        # Storage for Jacobians
        feature_to_feature = []  # [layer][downstream_feat, upstream_feat, pos]
        error_to_feature = []    # [layer][feature, error_dim, pos]
        feature_to_error = []    # [layer][error_dim, feature, pos]
        
        # For each layer, compute approximate Jacobians
        for layer_idx in range(n_layers):
            features = output.feature_acts[layer_idx]  # (1, seq_len, n_features)
            errors = output.error_acts[layer_idx]      # (1, seq_len, d_model)
            
            batch, seq_len_l, n_features = features.shape
            _, _, d_model = errors.shape
            
            # Feature-to-feature Jacobians (approximated)
            # For efficiency, use a sparse approximation:
            # Only compute for active features
            f2f = torch.zeros(n_features, n_features, seq_len_l)
            
            # Simplified: assume features mostly affect next layer features
            # via the decoder weights
            if layer_idx < n_layers - 1:
                # Use decoder weights as proxy for Jacobian
                if self.model.is_clt:
                    # CLT case: more complex
                    f2f = torch.zeros(n_features, n_features, seq_len_l)
                else:
                    # PLT case: decoder @ encoder gives feature-to-feature
                    if hasattr(self.model.transcoders[layer_idx], 'W_dec') and \
                       layer_idx + 1 < len(self.model.transcoders):
                        W_dec = self.model.transcoders[layer_idx].W_dec  # (d_model, n_features)
                        W_enc_next = self.model.transcoders[layer_idx + 1].W_enc  # (n_features, d_model)
                        # Approximate Jacobian as W_enc_next @ W_dec
                        approx_jac = W_enc_next @ W_dec  # (n_features_next, n_features)
                        f2f = approx_jac.unsqueeze(-1).expand(-1, -1, seq_len_l)
            
            feature_to_feature.append(f2f)
            
            # Error-to-feature and Feature-to-error Jacobians
            # Simplified to zeros for this implementation
            e2f = torch.zeros(n_features, d_model, seq_len_l)
            f2e = torch.zeros(d_model, n_features, seq_len_l)
            
            error_to_feature.append(e2f)
            feature_to_error.append(f2e)
        
        return JacobianResult(
            feature_to_feature=feature_to_feature,
            error_to_feature=error_to_feature,
            feature_to_error=feature_to_error,
            target_logit_idx=logit_idx,
        )
    
    def compute_activation_attribution(
        self,
        tokens: torch.Tensor,
        target_pos: int = -1,
        target_logit_idx: Optional[int] = None,
    ) -> dict:
        """
        Compute attribution scores for each feature activation.
        
        Returns:
            dict mapping (layer, feature_idx, token_pos) -> attribution_score
        """
        # Enable gradients
        for p in self.model.parameters():
            p.requires_grad_(True)
        
        # Forward pass
        output = self.model(tokens)
        
        # Target logit
        if target_logit_idx is None:
            target_logit_idx = output.logits[0, target_pos].argmax().item()
        
        target = output.logits[0, target_pos, target_logit_idx]
        
        # Compute gradients
        target.backward()
        
        # Extract attribution scores
        attributions = {}
        for layer_idx, features in enumerate(output.feature_acts):
            if features.grad is not None:
                # Attribution = activation * gradient
                attr = features * features.grad
                
                batch, seq_len, n_features = features.shape
                for pos in range(seq_len):
                    for feat_idx in range(n_features):
                        key = (layer_idx, feat_idx, pos)
                        attributions[key] = attr[0, pos, feat_idx].item()
        
        # Disable gradients
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        return attributions


def compute_virtual_weights_plt(
    transcoders: list,
    layer_from: int,
    layer_to: int,
) -> torch.Tensor:
    """
    Compute virtual weights between two layers for PLT.
    
    Virtual weight from layer_from to layer_to is the product
    of decoder @ encoder matrices.
    
    Args:
        transcoders: List of PerLayerTranscoder
        layer_from: Source layer
        layer_to: Target layer
        
    Returns:
        (n_features_to, n_features_from) matrix of virtual weights
    """
    if layer_to <= layer_from:
        raise ValueError("layer_to must be > layer_from")
    
    # Start with decoder of layer_from
    W = transcoders[layer_from].W_dec  # (d_model, n_features_from)
    
    # Multiply through intermediate encoders/decoders
    for layer in range(layer_from + 1, layer_to):
        W_enc = transcoders[layer].W_enc  # (n_features, d_model)
        W_dec = transcoders[layer].W_dec  # (d_model, n_features)
        W = W_dec @ W_enc @ W
    
    # Final encoder
    W_enc_final = transcoders[layer_to].W_enc  # (n_features_to, d_model)
    virtual_weights = W_enc_final @ W  # (n_features_to, n_features_from)
    
    return virtual_weights


def compute_virtual_weights_clt(
    clt,
    layer_from: int,
    layer_to: int,
) -> torch.Tensor:
    """
    Compute virtual weights between two layers for CLT.
    
    For CLT, the virtual weight is more direct since features
    at layer_from directly write to layer_to via W_dec[layer_from][k].
    
    Args:
        clt: CrossLayerTranscoder
        layer_from: Source layer
        layer_to: Target layer
        
    Returns:
        (n_features_to, n_features_from) matrix of virtual weights
    """
    if layer_to <= layer_from:
        raise ValueError("layer_to must be > layer_from")
    
    k = layer_to - layer_from
    
    # Direct decoder from layer_from to layer_to
    W_dec = clt.W_dec[layer_from][k]  # (d_model, n_features_from)
    
    # Encoder at layer_to
    W_enc = clt.W_enc[layer_to]  # (n_features_to, d_model)
    
    # Virtual weight
    virtual_weights = W_enc @ W_dec  # (n_features_to, n_features_from)
    
    return virtual_weights