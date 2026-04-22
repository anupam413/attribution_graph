"""
src/transcoder/per_layer.py
---------------------------
Phase 3a: Per-layer transcoder with JumpReLU activation.

A per-layer transcoder (PLT) learns to approximate a single MLP layer:
    MLP_output ≈ W_dec @ JumpReLU(W_enc @ residual_stream + b_enc) + b_dec

Each layer gets its own independent transcoder.
This is simpler than CLTs (Phase 3b) and still produces useful attribution graphs.

The paper uses JumpReLU rather than standard ReLU because:
  - Features have a learned per-feature activation threshold (θ)
  - Activations below θ are hard-zeroed → cleaner sparsity than ReLU
  - The threshold is trained via straight-through gradient estimation

Architecture for one transcoder:
    x  ∈ R^{d_model}   (residual stream input)
    pre = W_enc @ x + b_enc          shape: (n_features,)
    a   = JumpReLU(pre, θ)           shape: (n_features,)   ← sparse
    ŷ   = W_dec @ a  + b_dec         shape: (d_mlp,)        ← MLP output approx

Note: d_mlp is the MLP output dimension. For GPT-2 small, d_model=768, d_mlp=768.
For GPT-2 medium, d_model=1024, d_mlp=1024. (Transformer_lens uses same dim for MLP out.)

Usage:
    from src.transcoder.per_layer import PerLayerTranscoder, TranscoderConfig

    cfg = TranscoderConfig(d_model=768, d_mlp=768, n_features=4096)
    tc  = PerLayerTranscoder(cfg)

    # Forward pass
    mlp_in  = torch.randn(32, 768)   # (batch, d_model)
    out     = tc(mlp_in)
    print(out.reconstruction.shape)  # (32, 768)
    print(out.feature_acts.shape)    # (32, 4096)
    print(out.l0.item())             # avg features active per token
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TranscoderConfig:
    """
    Hyperparameters for a single per-layer transcoder.

    Key decisions:
      n_features   : Dictionary size. Paper uses 4K–10M. Start with 4096 for GPT-2 small.
                     Rule of thumb: 4x–16x d_model is a good starting range.
      sparsity_coef: λ in the paper. Controls sparsity vs reconstruction tradeoff.
                     Too high → features collapse to zero. Too low → polysemantic features.
                     Start with 1e-3 and tune.
      sparsity_c   : c in the tanh(c·‖W_dec_i‖·a_i) sparsity term.
                     Controls how steeply the penalty ramps up. Default 1.0 works well.
      jump_thresh  : Initial value of θ (JumpReLU threshold). Gets learned during training.
                     Start at 0.0 (equivalent to ReLU) and let it adapt.
    """
    d_model: int = 768        # residual stream dimension (input to transcoder)
    d_mlp: int = 768          # MLP output dimension (what we're reconstructing)
    n_features: int = 4096    # number of sparse features (dictionary size)
    sparsity_coef: float = 1e-3   # λ: overall sparsity penalty weight
    sparsity_c: float = 1.0       # c: tanh sharpness in sparsity term
    jump_thresh: float = 0.0      # initial JumpReLU threshold θ (learned)
    normalize_decoder: bool = True # keep decoder columns unit-norm during training
    seed: int = 42


# ---------------------------------------------------------------------------
# JumpReLU activation
# ---------------------------------------------------------------------------

class JumpReLUFunction(torch.autograd.Function):
    """
    JumpReLU with straight-through gradient estimator for the threshold.

    Forward:  f(x, θ) = x  if x > θ
                         0  otherwise

    Backward: We need gradients w.r.t. both x and θ.
      - ∂f/∂x  : straight-through (treat step function as 1 everywhere)
      - ∂f/∂θ  : −δ(x − θ) approximated as a narrow Gaussian bump
                 This is the "STE" trick that lets the threshold θ be learned.

    The bandwidth parameter controls how wide the Gaussian approximation is.
    Smaller bandwidth → more faithful approximation, noisier gradients.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, theta: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(x, theta)
        ctx.bandwidth = bandwidth
        return x * (x > theta).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, theta = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        # Straight-through for x: pass gradients through as if activation = x
        grad_x = grad_output * (x > theta).float()

        # Gaussian approximation for theta gradient
        # ∂L/∂θ = ∂L/∂f · (−δ(x−θ)) ≈ ∂L/∂f · (−Gauss(x−θ, bandwidth))
        gauss = torch.exp(-0.5 * ((x - theta) / bandwidth) ** 2)
        gauss = gauss / (bandwidth * math.sqrt(2 * math.pi))
        grad_theta = (-grad_output * x * gauss).sum(0)  # sum over batch dim

        return grad_x, grad_theta, None  # no grad for bandwidth


def jumprelu(x: torch.Tensor, theta: torch.Tensor, bandwidth: float = 0.1) -> torch.Tensor:
    """Apply JumpReLU with learnable threshold. Functional interface."""
    return JumpReLUFunction.apply(x, theta, bandwidth)


# ---------------------------------------------------------------------------
# Transcoder output (named tuple for clarity)
# ---------------------------------------------------------------------------

class TranscoderOutput(NamedTuple):
    reconstruction: torch.Tensor   # (batch, d_mlp)   — approximated MLP output
    feature_acts: torch.Tensor     # (batch, n_features) — sparse feature activations
    pre_acts: torch.Tensor         # (batch, n_features) — pre-JumpReLU activations
    l0: torch.Tensor               # scalar — avg features active per token


# ---------------------------------------------------------------------------
# Per-layer transcoder module
# ---------------------------------------------------------------------------

class PerLayerTranscoder(nn.Module):
    """
    A single per-layer transcoder approximating one MLP layer.

    Maps: residual_stream → sparse features → approx. MLP output

    Parameters:
      W_enc : (n_features, d_model)   encoder weight matrix
      b_enc : (n_features,)           encoder bias
      W_dec : (d_mlp, n_features)     decoder weight matrix
      b_dec : (d_mlp,)                decoder bias (= mean MLP output)
      theta : (n_features,)           per-feature JumpReLU threshold

    The paper recommends initializing:
      - W_enc columns as unit vectors (random)
      - W_dec rows as unit vectors that are transposes of W_enc columns
        (tied initialization, helps early training)
      - b_dec as zeros (or mean of MLP outputs if you have a warm-up pass)
      - theta as 0.0 everywhere
    """

    def __init__(self, cfg: TranscoderConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        # Encoder
        self.W_enc = nn.Parameter(
            self._init_encoder(cfg.d_model, cfg.n_features)
        )
        self.b_enc = nn.Parameter(torch.zeros(cfg.n_features))

        # JumpReLU threshold (one per feature)
        self.theta = nn.Parameter(
            torch.full((cfg.n_features,), cfg.jump_thresh)
        )

        # Decoder
        # Initialize as transpose of encoder for tied init (common in SAE literature)
        self.W_dec = nn.Parameter(
            self._init_decoder(cfg.d_mlp, cfg.n_features)
        )
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp))

        # Bandwidth for JumpReLU STE (not learned, just a hyperparameter)
        self.bandwidth: float = 0.1

    @staticmethod
    def _init_encoder(d_model: int, n_features: int) -> torch.Tensor:
        """Kaiming uniform init, then normalize rows to unit norm."""
        W = torch.empty(n_features, d_model)
        nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        W = F.normalize(W, dim=1)  # unit-norm rows
        return W

    @staticmethod
    def _init_decoder(d_mlp: int, n_features: int) -> torch.Tensor:
        """Kaiming uniform init, then normalize columns to unit norm."""
        W = torch.empty(d_mlp, n_features)
        nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        W = F.normalize(W, dim=0)  # unit-norm columns
        return W

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> TranscoderOutput:
        """
        Args:
            x: (batch_size, d_model) — residual stream at this layer's MLP input

        Returns:
            TranscoderOutput with reconstruction, feature_acts, pre_acts, l0
        """
        # Encode: (batch, d_model) → (batch, n_features)
        pre_acts = x @ self.W_enc.T + self.b_enc   # (batch, n_features)

        # Apply JumpReLU with per-feature threshold
        feature_acts = jumprelu(pre_acts, self.theta, self.bandwidth)

        # Decode: (batch, n_features) → (batch, d_mlp)
        reconstruction = feature_acts @ self.W_dec.T + self.b_dec

        # L0: average number of active features per token
        l0 = (feature_acts > 0).float().sum(dim=-1).mean()

        return TranscoderOutput(
            reconstruction=reconstruction,
            feature_acts=feature_acts,
            pre_acts=pre_acts,
            l0=l0,
        )

    # ------------------------------------------------------------------
    # Decoder normalization (called after each optimizer step)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """
        Project decoder columns back to unit norm.

        This is a constrained optimization technique from the SAE literature.
        Without it, the model can reduce the sparsity penalty by shrinking
        feature activations while growing decoder norms — they cancel out
        in the reconstruction but the penalty only sees the activation.

        Call this after every optimizer.step() when cfg.normalize_decoder=True.
        """
        if self.cfg.normalize_decoder:
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def loss(
        self,
        x: torch.Tensor,
        mlp_true_output: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the full training loss for one batch.

        Loss = MSE(reconstruction, mlp_true_output)
             + λ · Σ_i tanh(c · ‖W_dec_i‖ · a_i)

        The sparsity term in the paper:
          L_sparsity = λ · Σ_tokens Σ_i tanh(c · ‖W_dec_i‖ · a_i)

        ‖W_dec_i‖ is the L2 norm of the i-th decoder column.
        This couples sparsity penalty to decoder magnitude so the model
        can't cheat by making activations large and decoder small.

        Args:
            x               : (batch, d_model)  — MLP input (residual stream)
            mlp_true_output : (batch, d_mlp)    — true MLP output to reconstruct

        Returns:
            (total_loss, metrics_dict)
        """
        out = self(x)

        # --- Reconstruction loss ---
        mse = F.mse_loss(out.reconstruction, mlp_true_output)

        # Normalized MSE (useful for tracking across layers of different scale)
        with torch.no_grad():
            target_var = mlp_true_output.var(dim=0).mean()
            nmse = mse / (target_var + 1e-8)

        # --- Sparsity loss ---
        # ‖W_dec_i‖ for each feature i: shape (n_features,)
        dec_norms = self.W_dec.norm(dim=0)  # (n_features,)

        # tanh(c · ‖W_dec_i‖ · a_i): shape (batch, n_features)
        sparsity_per_feature = torch.tanh(
            self.cfg.sparsity_c * dec_norms.unsqueeze(0) * out.feature_acts
        )
        sparsity = sparsity_per_feature.sum(dim=-1).mean()  # avg over batch

        total_loss = mse + self.cfg.sparsity_coef * sparsity

        metrics = {
            "loss":       total_loss.item(),
            "mse":        mse.item(),
            "nmse":       nmse.item(),
            "sparsity":   sparsity.item(),
            "l0":         out.l0.item(),
        }

        return total_loss, metrics

    # ------------------------------------------------------------------
    # Feature analysis helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_active_features(
        self,
        x: torch.Tensor,
        top_k: int = 20,
    ) -> list[dict]:
        """
        For each token in x, return the top-k most active features.

        Useful for inspecting what features fire on a specific prompt.

        Args:
            x     : (seq_len, d_model)
            top_k : How many top features to return per token

        Returns:
            List of dicts, one per token:
            [{"token_pos": 0, "feature_ids": [42, 7, ...], "activations": [0.8, 0.6, ...]}, ...]
        """
        out = self(x)
        results = []
        for pos in range(x.shape[0]):
            acts = out.feature_acts[pos]         # (n_features,)
            active_mask = acts > 0
            active_ids = active_mask.nonzero(as_tuple=True)[0]

            if len(active_ids) == 0:
                results.append({"token_pos": pos, "feature_ids": [], "activations": []})
                continue

            active_acts = acts[active_ids]
            sorted_idx = active_acts.argsort(descending=True)[:top_k]

            results.append({
                "token_pos":    pos,
                "feature_ids":  active_ids[sorted_idx].tolist(),
                "activations":  active_acts[sorted_idx].tolist(),
            })
        return results

    @torch.no_grad()
    def decoder_cosine_similarity(self, feat_i: int, feat_j: int) -> float:
        """
        Cosine similarity between two decoder columns.
        High similarity → features may be redundant (feature splitting).
        """
        d_i = self.W_dec[:, feat_i]
        d_j = self.W_dec[:, feat_j]
        return F.cosine_similarity(d_i.unsqueeze(0), d_j.unsqueeze(0)).item()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        cfg = self.cfg
        return (
            f"PerLayerTranscoder("
            f"d_model={cfg.d_model}, d_mlp={cfg.d_mlp}, "
            f"n_features={cfg.n_features}, "
            f"λ={cfg.sparsity_coef}, params={self.num_parameters():,})"
        )
