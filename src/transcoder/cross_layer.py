"""
src/transcoder/cross_layer.py
------------------------------
Phase 3b: Cross-Layer Transcoder (CLT).

The critical difference from a per-layer transcoder (Phase 3a):
  - A feature at SOURCE layer ℓ reads from the residual stream at layer ℓ
  - BUT writes to the MLP output reconstruction at ALL layers ℓ' >= ℓ
  - All features across all layers are trained JOINTLY with one optimizer

This means:
  - A single layer-0 feature can influence reconstructions at ALL 12 layers.
  - A layer-11 feature only influences the reconstruction at layer 11.
  - The reconstruction at layer ℓ' is the SUM of contributions from features
    at layers 0, 1, ..., ℓ'.

Why this matters for attribution graphs:
  - CLTs collapse "amplification chains" (the same concept being passed through
    many consecutive MLP layers) into a single early-layer feature.
  - This dramatically reduces average path length in attribution graphs,
    making circuits much easier to read.
  - The paper shows CLTs reduce mean path length from ~3.7 (PLT) to ~2.3 (CLT).

Architecture:
  Encoder at layer ℓ:
    a[ℓ] = JumpReLU(W_enc[ℓ] @ x[ℓ] + b_enc[ℓ], theta[ℓ])
    x[ℓ] = residual stream at layer ℓ (= hook_mlp_in in TransformerLens)

  Reconstruction at output layer ℓ':
    ŷ[ℓ'] = b_dec[ℓ'] + Σ_{ℓ=0}^{ℓ'} W_dec[ℓ→ℓ'] @ a[ℓ]

  Loss:
    L = Σ_ℓ' MSE(ŷ[ℓ'], y[ℓ'])
      + λ · Σ_ℓ Σ_i tanh(c · ‖W_dec_i^ℓ‖_concat · a_i[ℓ])

  where ‖W_dec_i^ℓ‖_concat = L2 norm of ALL decoder vectors for feature i
  at source layer ℓ, concatenated across all output layers it writes to.

Parameter count for GPT-2 small (L=12, N=2048, d_model=768):
  Encoders : 12 × (2048 × 768)  ≈  19M params
  Decoders : 78 × (768  × 2048) ≈  123M params   [78 = 12+11+...+1]
  Biases   : negligible
  Total    ≈  142M params  (~570MB float32)

Usage:
    from src.transcoder.cross_layer import CrossLayerTranscoder, CLTConfig

    cfg = CLTConfig(d_model=768, d_mlp=768, n_layers=12, n_features=2048)
    clt = CrossLayerTranscoder(cfg)

    # Forward: enc_inputs is (batch, n_layers, d_model)
    enc_inputs = torch.randn(32, 12, 768)
    out = clt(enc_inputs)
    print(out.reconstructions[0].shape)   # (32, 768)  — layer 0 reconstruction
    print(out.feature_acts[5].shape)      # (32, 2048) — features at layer 5
    print(out.l0_mean.item())             # mean active features per token, per layer

    # Training loss
    mlp_outputs = torch.randn(32, 12, 768)
    loss, metrics = clt.loss(enc_inputs, mlp_outputs)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use JumpReLU from Phase 3a — no need to redefine it
from src.transcoder.per_layer import jumprelu


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CLTConfig:
    """
    Hyperparameters for the Cross-Layer Transcoder.

    Key differences from TranscoderConfig:
      n_layers      : How many MLP layers the CLT spans (= model's n_layers).
      n_features    : Features PER LAYER. Total features = n_layers × n_features.

    Memory guidance for GPT-2 small (d_model=768, n_layers=12):
      n_features=1024 → ~285MB  (good for development/debugging)
      n_features=2048 → ~570MB  (recommended starting point)
      n_features=4096 → ~1.1GB  (paper-scale, needs 8GB+ VRAM)

    Tuning:
      sparsity_coef : Same role as in PLT. Start at 1e-3.
                      CLT features tend to be slightly less sparse than PLT
                      features at the same λ, so you may need to increase it.
      bandwidth     : JumpReLU STE bandwidth. 0.1 is stable; lower = noisier grads.
    """
    d_model: int = 768
    d_mlp: int = 768
    n_layers: int = 12
    n_features: int = 2048       # per layer — see memory guidance above
    sparsity_coef: float = 1e-3
    sparsity_c: float = 1.0
    jump_thresh: float = 0.0
    normalize_decoder: bool = True
    bandwidth: float = 0.1
    seed: int = 42


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class CLTOutput(NamedTuple):
    """
    Output of a single CLT forward pass.

    All lists have length = n_layers, indexed by layer number.
    """
    reconstructions: list    # [layer] → (batch, d_mlp)  approximated MLP output
    feature_acts:    list    # [layer] → (batch, n_features)  sparse activations
    pre_acts:        list    # [layer] → (batch, n_features)  pre-JumpReLU values
    l0_per_layer:    list    # [layer] → scalar  avg active features per token
    l0_mean:         torch.Tensor   # mean L0 across all layers


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class CrossLayerTranscoder(nn.Module):
    """
    Cross-Layer Transcoder (CLT).

    Parameter layout:
      W_enc  [L]         : ParameterList of (n_features, d_model) encoder matrices
      b_enc  [L]         : ParameterList of (n_features,) encoder biases
      theta  [L]         : ParameterList of (n_features,) JumpReLU thresholds
      W_dec  [L][K]      : Nested ModuleList/ParameterList of (d_mlp, n_features)
                           W_dec[src][k] writes to output layer (src + k)
                           k = 0, 1, ..., (L - 1 - src)
      b_dec  [L]         : ParameterList of (d_mlp,) output biases, one per OUTPUT layer

    Forward pass computes:
      1. Feature activations at each layer using that layer's encoder
      2. Reconstruction at each output layer as sum over all source layer contributions
    """

    def __init__(self, cfg: CLTConfig):
        super().__init__()
        self.cfg = cfg
        L = cfg.n_layers
        torch.manual_seed(cfg.seed)

        # ---- Encoders (one per source layer) ----
        self.W_enc = nn.ParameterList([
            nn.Parameter(self._init_encoder(cfg.d_model, cfg.n_features))
            for _ in range(L)
        ])
        self.b_enc = nn.ParameterList([
            nn.Parameter(torch.zeros(cfg.n_features))
            for _ in range(L)
        ])
        self.theta = nn.ParameterList([
            nn.Parameter(torch.full((cfg.n_features,), cfg.jump_thresh))
            for _ in range(L)
        ])

        # ---- Decoders (cross-layer: src writes to all tgt >= src) ----
        # W_dec[src] is a ParameterList with (L - src) matrices.
        # W_dec[src][k] has shape (d_mlp, n_features) and writes to output layer (src+k).
        #
        # We scale decoder init by 1/sqrt(n_output_layers) per source layer so that
        # the total decoder contribution to each output layer has similar scale
        # regardless of how many source layers contribute.
        self.W_dec = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(self._init_decoder(
                    cfg.d_mlp, cfg.n_features,
                    scale=1.0 / math.sqrt(L - src)
                ))
                for _ in range(L - src)      # k = 0, 1, ..., L-1-src
            ])
            for src in range(L)
        ])

        # ---- Output biases (one per OUTPUT layer) ----
        # These absorb the mean MLP output at each layer.
        # Call init_biases_from_cache() after construction for better initialization.
        self.b_dec = nn.ParameterList([
            nn.Parameter(torch.zeros(cfg.d_mlp))
            for _ in range(L)
        ])

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_encoder(d_model: int, n_features: int) -> torch.Tensor:
        W = torch.empty(n_features, d_model)
        nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        return F.normalize(W, dim=1)

    @staticmethod
    def _init_decoder(d_mlp: int, n_features: int, scale: float = 1.0) -> torch.Tensor:
        W = torch.empty(d_mlp, n_features)
        nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        W = F.normalize(W, dim=0) * scale
        return W

    @torch.no_grad()
    def init_biases_from_cache(self, mlp_outputs: list[torch.Tensor]) -> None:
        """
        Initialize b_dec[ℓ] to the mean MLP output at layer ℓ.

        This gives the decoder a head start: even with zero feature activations,
        the reconstruction will equal the average output instead of zero.
        Often reduces early training loss by ~20-30%.

        Args:
            mlp_outputs: list of (N, d_mlp) tensors, one per layer.
                         Typically from ActivationCache: data["mlp_out"]
        """
        for layer, y in enumerate(mlp_outputs):
            if layer >= self.cfg.n_layers:
                break
            mean_out = y.mean(dim=0)
            self.b_dec[layer].data.copy_(mean_out.to(self.b_dec[layer].device))
        print(f"Initialized b_dec for {min(len(mlp_outputs), self.cfg.n_layers)} layers "
              f"from cache means.")

    # ------------------------------------------------------------------
    # Decoder column stack helper (used in loss, normalize, and attribution)
    # ------------------------------------------------------------------

    def get_decoder_stack(self, src: int) -> torch.Tensor:
        """
        Return all decoder matrices for source layer `src` as a stacked tensor.

        Returns:
            (K, d_mlp, n_features) where K = n_layers - src
            Stack[k] = W_dec[src][k] = decoder writing to output layer (src + k)

        Used for:
          - Sparsity penalty computation (concatenated decoder norm per feature)
          - Decoder normalization
          - Computing virtual weights in Phase 3d (attribution graphs)
        """
        return torch.stack([self.W_dec[src][k] for k in range(self.cfg.n_layers - src)])

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, enc_inputs: torch.Tensor) -> CLTOutput:
        """
        Args:
            enc_inputs: (batch, n_layers, d_model)
                        Residual stream at each layer's MLP input position.
                        Equivalent to hook_mlp_in at each layer.

        Returns:
            CLTOutput with reconstructions, feature_acts, pre_acts, l0 stats.
        """
        B = enc_inputs.shape[0]
        L = self.cfg.n_layers

        # ---- Step 1: Encode — compute feature activations at each layer ----
        feature_acts = []
        pre_acts_list = []

        for src in range(L):
            x = enc_inputs[:, src, :]                              # (B, d_model)
            pre = x @ self.W_enc[src].T + self.b_enc[src]         # (B, n_features)
            acts = jumprelu(pre, self.theta[src], self.cfg.bandwidth)  # (B, n_features)
            feature_acts.append(acts)
            pre_acts_list.append(pre)

        # ---- Step 2: Decode — reconstruct MLP output at each target layer ----
        #
        # ŷ[tgt] = b_dec[tgt] + Σ_{src=0}^{tgt} W_dec[src][tgt-src] @ a[src]
        #
        # We build reconstructions layer by layer. Each new target layer adds
        # contributions from all source layers up to and including itself.
        #
        # Implementation note: we accumulate contributions as we go, keeping
        # a running sum of "src → tgt" contributions to avoid recomputing.
        # This is O(L²) matmuls but each is small (batch × n_features → d_mlp).

        reconstructions = []

        for tgt in range(L):
            # Start with the output bias for this layer
            recon = self.b_dec[tgt].unsqueeze(0).expand(B, -1).clone()  # (B, d_mlp)

            # Add contribution from every source layer src <= tgt
            for src in range(tgt + 1):
                k = tgt - src         # index into W_dec[src]
                # (B, n_features) @ (n_features, d_mlp) = (B, d_mlp)
                recon = recon + feature_acts[src] @ self.W_dec[src][k].T

            reconstructions.append(recon)

        # ---- L0 stats ----
        l0_per_layer = [
            (feature_acts[ℓ] > 0).float().sum(dim=-1).mean()
            for ℓ in range(L)
        ]
        l0_mean = torch.stack(l0_per_layer).mean()

        return CLTOutput(
            reconstructions=reconstructions,
            feature_acts=feature_acts,
            pre_acts=pre_acts_list,
            l0_per_layer=l0_per_layer,
            l0_mean=l0_mean,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        enc_inputs: torch.Tensor,
        mlp_true_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute training loss for one batch.

        Loss = Σ_ℓ' MSE(ŷ[ℓ'], y[ℓ'])
             + λ · Σ_ℓ Σ_i tanh(c · ‖W_dec_i^ℓ‖_concat · a_i[ℓ])

        The sparsity term uses the CONCATENATED decoder norm:
          ‖W_dec_i^ℓ‖_concat = ‖[W_dec[ℓ][0][:,i]; W_dec[ℓ][1][:,i]; ...]‖_2

        This is key: the penalty for activating feature i at layer ℓ scales
        with the TOTAL magnitude of all its effects across all output layers.
        A feature that writes strongly to many layers gets penalized more.

        Args:
            enc_inputs      : (batch, n_layers, d_model)
            mlp_true_outputs: (batch, n_layers, d_mlp)

        Returns:
            (total_loss, metrics_dict)
            metrics_dict keys: loss, mse_total, nmse_mean, sparsity, l0_mean,
                               plus nmse_layer_{i} for each layer
        """
        out = self(enc_inputs)
        L = self.cfg.n_layers

        # ---- Reconstruction losses (summed across all output layers) ----
        total_mse = torch.tensor(0.0, device=enc_inputs.device)
        nmse_per_layer = []

        for tgt in range(L):
            y_true = mlp_true_outputs[:, tgt, :]   # (B, d_mlp)
            mse_tgt = F.mse_loss(out.reconstructions[tgt], y_true)
            total_mse = total_mse + mse_tgt

            with torch.no_grad():
                var_tgt = y_true.var(dim=0).mean().clamp(min=1e-8)
                nmse_per_layer.append((mse_tgt / var_tgt).item())

        nmse_mean = sum(nmse_per_layer) / L

        # ---- Sparsity penalty (summed across source layers) ----
        total_sparsity = torch.tensor(0.0, device=enc_inputs.device)

        for src in range(L):
            # Concatenated decoder norm for each feature at source layer src
            # dec_stack shape: (K, d_mlp, n_features), K = L - src
            dec_stack = self.get_decoder_stack(src)   # (K, d_mlp, n_features)

            # L2 norm over the concatenated decoder (dims 0=K, 1=d_mlp) per feature
            # Result shape: (n_features,)
            dec_norms = dec_stack.pow(2).sum(dim=[0, 1]).sqrt()   # (n_features,)

            # tanh(c · ‖W_dec_i‖ · a_i): shape (B, n_features)
            penalty = torch.tanh(
                self.cfg.sparsity_c * dec_norms.unsqueeze(0) * out.feature_acts[src]
            )
            total_sparsity = total_sparsity + penalty.sum(dim=-1).mean()

        total_loss = total_mse + self.cfg.sparsity_coef * total_sparsity

        metrics = {
            "loss":       total_loss.item(),
            "mse_total":  total_mse.item(),
            "nmse_mean":  nmse_mean,
            "sparsity":   total_sparsity.item(),
            "l0_mean":    out.l0_mean.item(),
        }
        for i, nmse in enumerate(nmse_per_layer):
            metrics[f"nmse_layer_{i}"] = nmse

        return total_loss, metrics

    # ------------------------------------------------------------------
    # Decoder normalization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """
        Project each feature's concatenated decoder back to unit norm.

        For feature i at source layer src, the "concatenated decoder" is:
          concat([W_dec[src][k][:, i] for k in range(L - src)])

        We want ‖concat(...)‖ = 1, so we divide ALL of feature i's
        decoder matrices at this source layer by the same norm scalar.

        Call this after every optimizer.step() when cfg.normalize_decoder=True.
        """
        if not self.cfg.normalize_decoder:
            return

        L = self.cfg.n_layers
        for src in range(L):
            K = L - src
            dec_stack = torch.stack([self.W_dec[src][k].data for k in range(K)])
            # (K, d_mlp, n_features) → norm per feature: (n_features,)
            dec_norms = dec_stack.pow(2).sum(dim=[0, 1]).sqrt().clamp(min=1e-8)

            for k in range(K):
                # Divide each column (feature) by its norm
                self.W_dec[src][k].data.div_(dec_norms.unsqueeze(0))

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_active_features(
        self,
        enc_inputs: torch.Tensor,
        layer: int,
        top_k: int = 20,
    ) -> list[dict]:
        """
        Return the top-k active features at a specific source layer for
        each token position.

        Args:
            enc_inputs : (seq_len, n_layers, d_model) — full prompt activations
            layer      : Which source layer to inspect
            top_k      : Max features to return per token position

        Returns:
            List of dicts, one per token:
            [{"token_pos": 0, "feature_ids": [...], "activations": [...]}, ...]
        """
        out = self(enc_inputs)
        acts = out.feature_acts[layer]   # (seq_len, n_features)

        results = []
        for pos in range(acts.shape[0]):
            a = acts[pos]
            active_ids = (a > 0).nonzero(as_tuple=True)[0]

            if len(active_ids) == 0:
                results.append({"token_pos": pos, "layer": layer,
                                 "feature_ids": [], "activations": []})
                continue

            active_vals = a[active_ids]
            sorted_idx = active_vals.argsort(descending=True)[:top_k]
            results.append({
                "token_pos":   pos,
                "layer":       layer,
                "feature_ids": active_ids[sorted_idx].tolist(),
                "activations": active_vals[sorted_idx].tolist(),
            })
        return results

    @torch.no_grad()
    def virtual_weight(self, src: int, tgt: int, feat_i: int) -> torch.Tensor:
        """
        Compute the virtual weight from feature feat_i at source layer src
        to the residual stream at output layer tgt (direct, no attention paths).

        Virtual weight = sum_{k} W_dec[src][k][:, feat_i] for k in [0, tgt-src]
                       = sum of all decoder outputs from feat_i that have been
                         added to the residual stream by layer tgt.

        This is the context-independent part of the edge weight in attribution
        graphs. Used in Phase 3d/Global Weights analysis.

        Args:
            src    : Source layer (where the feature lives)
            tgt    : Target output layer (where we measure the virtual weight)
            feat_i : Feature index

        Returns:
            (d_mlp,) vector — the virtual weight in the residual stream space
        """
        assert tgt >= src, f"tgt ({tgt}) must be >= src ({src})"
        k = tgt - src
        # Sum decoder columns from src up to tgt
        vw = torch.zeros(self.cfg.d_mlp, device=self.W_dec[src][0].device)
        for kk in range(k + 1):
            vw = vw + self.W_dec[src][kk][:, feat_i]
        return vw

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_decoder_parameters(self) -> int:
        total = 0
        for src in range(self.cfg.n_layers):
            for k in range(self.cfg.n_layers - src):
                total += self.W_dec[src][k].numel()
        return total

    def __repr__(self) -> str:
        cfg = self.cfg
        n_dec_matrices = cfg.n_layers * (cfg.n_layers + 1) // 2
        return (
            f"CrossLayerTranscoder("
            f"d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
            f"n_features={cfg.n_features}/layer, "
            f"decoder_matrices={n_dec_matrices}, "
            f"total_params={self.num_parameters():,})"
        )
