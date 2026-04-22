"""
src/transcoder/train.py
-----------------------
Phase 3a + 3b: Training loops for per-layer and cross-layer transcoders.

Phase 3a classes (unchanged):
  TranscoderTrainer   — trains one PLT per layer independently
  MLPActivationDataset — per-layer DataLoader wrapper
  TrainingConfig       — PLT hyperparameters

Phase 3b additions:
  CLTTrainingConfig    — CLT-specific hyperparameters
  CrossLayerDataset    — stacks ALL layers per token for joint training
  CLTTrainer           — trains a single CLT across all layers jointly
  evaluate_clt         — per-layer NMSE/L0/dead-feature metrics for a CLT

Key difference between PLT and CLT training:
  PLT: each layer trained independently, tokens shuffled per layer separately.
  CLT: ALL layers trained jointly. Each batch contains the SAME tokens' activations
       at every layer simultaneously, so cross-layer features can learn to write
       consistently across layers.

Usage (CLT):
    from src.transcoder.train import CLTTrainer, CLTTrainingConfig

    trainer = CLTTrainer.from_cache(
        cache_path="data/activation_cache/gpt2.pt",
        train_cfg=CLTTrainingConfig(n_features=2048, n_steps=30_000),
    )
    clt = trainer.train()
    # Evaluate
    from src.transcoder.train import evaluate_clt
    metrics = evaluate_clt(clt, cache_data)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import torch.nn.functional as F


from src.transcoder.per_layer import PerLayerTranscoder, TranscoderConfig
from src.transcoder.cross_layer import CrossLayerTranscoder, CLTConfig


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Hyperparameters for the training loop.

    Recommended starting values for GPT-2 small (d_model=768):
      n_features    : 4096   (≈5× d_model)
      batch_size    : 2048   (tokens per step — large batches help stability)
      lr            : 2e-4   (Adam, no weight decay)
      n_steps       : 20_000 (adjust based on dataset size)
      sparsity_coef : 1e-3   (start here, tune if features are too dense/sparse)

    Signs the model needs tuning:
      l0 > 100      → sparsity_coef too low, increase λ
      l0 < 5        → sparsity_coef too high, decrease λ or lower jump_thresh
      nmse > 0.5    → n_features too low or training too short
      nmse < 0.05   → good reconstruction, check if features are interpretable
    """
    # Architecture
    n_features: int = 4096
    sparsity_coef: float = 1e-3
    sparsity_c: float = 1.0
    jump_thresh: float = 0.0
    normalize_decoder: bool = True

    # Optimization
    batch_size: int = 2048
    lr: float = 2e-4
    betas: tuple = (0.9, 0.999)
    n_steps: int = 20_000
    warmup_steps: int = 200      # linear LR warmup

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 2000
    log_every_n_steps: int = 100

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Which layers to train (None = all layers)
    layers_to_train: Optional[list[int]] = None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MLPActivationDataset:
    """
    Wraps the cached (mlp_input, mlp_output) tensors from ActivationCache
    and provides DataLoader-ready TensorDatasets per layer.

    The tensors can be large (millions of tokens × d_model). We keep them
    on CPU and move batches to the target device on-the-fly.
    """

    def __init__(self, cache_data: dict):
        self.mlp_in  = cache_data["mlp_in"]   # list[tensor], one per layer
        self.mlp_out = cache_data["mlp_out"]  # list[tensor], one per layer
        self.n_layers = len(self.mlp_in)
        self.model_name = cache_data.get("model_name", "unknown")

    @classmethod
    def from_file(cls, path: str) -> "MLPActivationDataset":
        data = torch.load(path, map_location="cpu", weights_only=False)
        print(f"Loaded activation cache: {data['mlp_in'][0].shape[0]:,} tokens, "
              f"{len(data['mlp_in'])} layers")
        return cls(data)

    def get_loader(self, layer: int, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Return a DataLoader for a single layer's (input, output) pairs."""
        ds = TensorDataset(self.mlp_in[layer], self.mlp_out[layer])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available(), num_workers=0)

    def d_model(self, layer: int = 0) -> int:
        return self.mlp_in[layer].shape[-1]

    def d_mlp(self, layer: int = 0) -> int:
        return self.mlp_out[layer].shape[-1]

    def n_tokens(self, layer: int = 0) -> int:
        return self.mlp_in[layer].shape[0]


# ---------------------------------------------------------------------------
# Warmup scheduler
# ---------------------------------------------------------------------------

class LinearWarmupScheduler:
    """Simple linear learning rate warmup, then constant."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / self.warmup_steps
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg["lr"] = base_lr * scale

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class TranscoderTrainer:
    """
    Trains per-layer transcoders from a precomputed activation cache.

    One transcoder is trained independently per MLP layer.
    Each transcoder gets its own optimizer and scheduler.
    """

    def __init__(
        self,
        dataset: MLPActivationDataset,
        train_cfg: TrainingConfig,
    ):
        self.dataset = dataset
        self.train_cfg = train_cfg
        self.device = torch.device(train_cfg.device)
        Path(train_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_cache(
        cls,
        cache_path: str,
        train_cfg: Optional[TrainingConfig] = None,
    ) -> "TranscoderTrainer":
        dataset = MLPActivationDataset.from_file(cache_path)
        cfg = train_cfg or TrainingConfig()
        return cls(dataset, cfg)

    # ------------------------------------------------------------------
    # Train all layers
    # ------------------------------------------------------------------

    def train_all_layers(self) -> list[PerLayerTranscoder]:
        """
        Train one transcoder for each layer and return them as a list.
        Layers are trained sequentially (independent, no shared state).
        """
        layers = self.train_cfg.layers_to_train or list(range(self.dataset.n_layers))
        transcoders = []

        for layer in layers:
            print(f"\n{'='*60}")
            print(f"Training transcoder for layer {layer}/{self.dataset.n_layers - 1}")
            print(f"{'='*60}")
            tc = self.train_layer(layer)
            transcoders.append(tc)

        return transcoders

    # ------------------------------------------------------------------
    # Train single layer
    # ------------------------------------------------------------------

    def train_layer(self, layer: int) -> PerLayerTranscoder:
        """
        Train a single transcoder for the given MLP layer.

        Returns the trained PerLayerTranscoder.
        """
        cfg = self.train_cfg

        # Build transcoder
        tc_cfg = TranscoderConfig(
            d_model=self.dataset.d_model(layer),
            d_mlp=self.dataset.d_mlp(layer),
            n_features=cfg.n_features,
            sparsity_coef=cfg.sparsity_coef,
            sparsity_c=cfg.sparsity_c,
            jump_thresh=cfg.jump_thresh,
            normalize_decoder=cfg.normalize_decoder,
        )
        tc = PerLayerTranscoder(tc_cfg).to(self.device)
        print(f"  {tc}")
        print(f"  Training on {self.dataset.n_tokens(layer):,} tokens")

        # Optimizer + scheduler
        optimizer = torch.optim.Adam(tc.parameters(), lr=cfg.lr, betas=cfg.betas)
        scheduler = LinearWarmupScheduler(optimizer, cfg.warmup_steps)

        # DataLoader — cycle it if we have fewer tokens than n_steps × batch_size
        loader = self.dataset.get_loader(layer, cfg.batch_size, shuffle=True)
        loader_iter = _infinite_loader(loader)

        # Tracking
        best_nmse = float("inf")
        best_ckpt_path = None
        step_metrics: list[dict] = []

        t0 = time.time()

        for step in range(1, cfg.n_steps + 1):
            # ---- Forward + loss ----
            mlp_in, mlp_out = next(loader_iter)
            mlp_in  = mlp_in.to(self.device, non_blocking=True)
            mlp_out = mlp_out.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            loss, metrics = tc.loss(mlp_in, mlp_out)
            loss.backward()

            # Gradient clipping (stabilizes JumpReLU threshold training)
            nn.utils.clip_grad_norm_(tc.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Project decoder back to unit norm (constrained optimization)
            tc.normalize_decoder_()

            # ---- Logging ----
            if step % cfg.log_every_n_steps == 0:
                elapsed = time.time() - t0
                steps_per_sec = step / elapsed
                eta_min = (cfg.n_steps - step) / steps_per_sec / 60

                print(
                    f"  step {step:>6}/{cfg.n_steps}  "
                    f"loss={metrics['loss']:.4f}  "
                    f"mse={metrics['mse']:.4f}  "
                    f"nmse={metrics['nmse']:.4f}  "
                    f"l0={metrics['l0']:.1f}  "
                    f"lr={scheduler.get_lr():.2e}  "
                    f"eta={eta_min:.1f}m"
                )
                step_metrics.append({**metrics, "step": step})

            # ---- Checkpointing ----
            if step % cfg.save_every_n_steps == 0:
                ckpt_path = self._save_checkpoint(tc, layer, step, metrics)

                if metrics["nmse"] < best_nmse:
                    best_nmse = metrics["nmse"]
                    best_ckpt_path = ckpt_path
                    print(f"  → New best NMSE: {best_nmse:.4f} (saved {ckpt_path})")

        # Save final checkpoint
        final_path = self._save_checkpoint(tc, layer, cfg.n_steps, metrics, tag="final")
        print(f"\n  Training complete.")
        print(f"  Final  NMSE: {metrics['nmse']:.4f}, L0: {metrics['l0']:.1f}")
        print(f"  Best   NMSE: {best_nmse:.4f} (checkpoint: {best_ckpt_path})")
        print(f"  Saved to: {final_path}")

        return tc

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        tc: PerLayerTranscoder,
        layer: int,
        step: int,
        metrics: dict,
        tag: str = "",
    ) -> str:
        filename = f"layer_{layer}_step_{step}"
        if tag:
            filename += f"_{tag}"
        filename += ".pt"
        path = str(Path(self.train_cfg.checkpoint_dir) / filename)

        torch.save({
            "model_state_dict": tc.state_dict(),
            "transcoder_config": tc.cfg,
            "step": step,
            "metrics": metrics,
        }, path)
        return path

    @staticmethod
    def load_checkpoint(path: str, device: str = "cpu") -> PerLayerTranscoder:
        """Load a transcoder from a checkpoint file."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        tc = PerLayerTranscoder(ckpt["transcoder_config"])
        tc.load_state_dict(ckpt["model_state_dict"])
        tc = tc.to(device)
        print(f"Loaded transcoder from {path}")
        print(f"  Step  : {ckpt['step']}")
        print(f"  NMSE  : {ckpt['metrics']['nmse']:.4f}")
        print(f"  L0    : {ckpt['metrics']['l0']:.1f}")
        return tc


# ---------------------------------------------------------------------------
# Utility: infinite dataloader iterator
# ---------------------------------------------------------------------------

def _infinite_loader(loader: DataLoader):
    """Cycle a DataLoader indefinitely."""
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_transcoder(
    tc: PerLayerTranscoder,
    mlp_in: torch.Tensor,
    mlp_out: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 1024,
) -> dict[str, float]:
    """
    Evaluate a trained transcoder on a held-out set.

    Returns dict with nmse, l0, frac_variance_explained, dead_features.

    Args:
        tc      : Trained PerLayerTranscoder.
        mlp_in  : (N, d_model) MLP input activations.
        mlp_out : (N, d_mlp)  MLP output activations.
        device  : Device to run evaluation on.
        batch_size: Batch size for evaluation (no grads needed).

    Usage:
        metrics = evaluate_transcoder(tc, data["mlp_in"][5], data["mlp_out"][5])
        print(metrics)
    """
    tc = tc.to(device).eval()
    all_acts = []
    total_mse = 0.0
    n_batches = 0

    for start in range(0, mlp_in.shape[0], batch_size):
        x = mlp_in[start:start + batch_size].to(device)
        y = mlp_out[start:start + batch_size].to(device)
        out = tc(x)

        total_mse += F.mse_loss(out.reconstruction, y, reduction="sum").item()
        all_acts.append(out.feature_acts.cpu())
        n_batches += 1

    all_acts = torch.cat(all_acts, dim=0)   # (N, n_features)
    mean_mse = total_mse / mlp_in.shape[0]

    # Normalized MSE
    target_var = mlp_out.var(dim=0).mean().item()
    nmse = mean_mse / (target_var + 1e-8)

    # L0: avg active features per token
    l0 = (all_acts > 0).float().sum(dim=-1).mean().item()

    # Fraction of variance explained
    fve = max(0.0, 1.0 - nmse)

    # Dead features: features that never activated across the eval set
    ever_active = (all_acts > 0).any(dim=0)
    dead_frac = (~ever_active).float().mean().item()

    return {
        "nmse":     nmse,
        "l0":       l0,
        "fve":      fve,           # fraction variance explained
        "dead_frac": dead_frac,    # fraction of features never active
        "n_tokens": mlp_in.shape[0],
    }


# ---------------------------------------------------------------------------
# Quick validation / demo
# ---------------------------------------------------------------------------

def _validate():
    """
    Smoke test: generate synthetic MLP-like activations and train a tiny transcoder.
    Does NOT require the real model or activation cache.
    Useful for checking that the training loop runs without errors.
    """
    import torch.nn.functional as F

    print("="*60)
    print("Smoke test: training a tiny transcoder on synthetic data")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model, d_mlp, n_features = 64, 64, 256
    n_tokens = 8000

    # Synthetic "MLP": random linear + GELU (mimics real MLP structure)
    W1 = torch.randn(d_mlp * 4, d_model).to(device)
    W2 = torch.randn(d_mlp, d_mlp * 4).to(device)

    x_all = torch.randn(n_tokens, d_model)
    with torch.no_grad():
        y_all = F.gelu(x_all.to(device) @ W1.T) @ W2.T
        y_all = y_all / (y_all.std(dim=0, keepdim=True) + 1e-6)
        y_all = y_all.cpu()

    # Fake cache data
    cache_data = {
        "mlp_in":  [x_all],    # single layer
        "mlp_out": [y_all],
        "model_name": "synthetic",
    }

    dataset = MLPActivationDataset(cache_data)
    train_cfg = TrainingConfig(
        n_features=n_features,
        n_steps=1000,
        batch_size=256,
        log_every_n_steps=50,
        save_every_n_steps=300,
        checkpoint_dir="checkpoints/smoke_test",
        device=device,
        sparsity_coef=1e-4,
    )

    trainer = TranscoderTrainer(dataset, train_cfg)
    tc = trainer.train_layer(layer=0)

    # Evaluate
    print("\nFinal evaluation:")
    metrics = evaluate_transcoder(tc, x_all, y_all, device=device)
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")

    assert metrics["nmse"] < 0.5, f"NMSE too high: {metrics['nmse']:.4f}"
    assert metrics["l0"] > 0,     "No features active — sparsity_coef too high"
    print("\nSmoke test passed.")


if __name__ == "__main__":
    _validate()


# ===========================================================================
# Phase 3b additions — Cross-Layer Transcoder training
# ===========================================================================

# ---------------------------------------------------------------------------
# CLT training config
# ---------------------------------------------------------------------------

@dataclass
class CLTTrainingConfig:
    """
    Hyperparameters for CLT training.

    Differences from TrainingConfig:
      - batch_size is smaller by default (each sample contains ALL layers,
        so memory per sample is n_layers× larger than for PLT)
      - n_steps is larger (more parameters to train jointly)
      - init_biases is new: pre-initialize b_dec from cache means

    Recommended starting values for GPT-2 small:
      n_features   : 2048   (per layer; 4096 if you have 8GB+ VRAM)
      batch_size   : 256    (tokens; each sample has 12-layer activations)
      lr           : 2e-4
      n_steps      : 30_000 (more steps vs PLT since training is joint)

    Signs the CLT needs tuning (same logic as PLT):
      l0_mean > 150   → increase sparsity_coef
      l0_mean < 5     → decrease sparsity_coef
      nmse_mean > 0.4 → increase n_features or n_steps
      Many dead feats → decrease sparsity_coef or increase n_steps
    """
    n_features: int = 2048
    sparsity_coef: float = 1e-3
    sparsity_c: float = 1.0
    jump_thresh: float = 0.0
    normalize_decoder: bool = True
    bandwidth: float = 0.1

    batch_size: int = 256
    lr: float = 2e-4
    betas: tuple = (0.9, 0.999)
    n_steps: int = 30_000
    warmup_steps: int = 300

    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 5000
    log_every_n_steps: int = 200

    # Initialize b_dec from cache mean outputs before training
    init_biases: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Cross-layer dataset
# ---------------------------------------------------------------------------


class CrossLayerDataset(Dataset):
    """
    Dataset for CLT training. Returns ALL layers' activations for the SAME
    token, so the CLT can learn cross-layer features jointly.

    Unlike PLT training (where each layer's tokens can be shuffled independently),
    CLT training MUST keep layers aligned: the token at index i in layer 0
    must be the same token as index i in layer 1, etc.

    The existing ActivationCache already preserves this ordering — tokens are
    concatenated in the same prompt/position order across all layers.

    Returns per sample:
      enc_inputs  : (n_layers, d_model)  — residual stream at each layer's MLP input
      mlp_outputs : (n_layers, d_mlp)   — true MLP output at each layer
    """

    def __init__(self, cache_data: dict):
        mlp_in_list  = cache_data["mlp_in"]   # list of (N, d_model), one per layer
        mlp_out_list = cache_data["mlp_out"]  # list of (N, d_mlp),   one per layer

        # Stack along a new layer dimension: (N, L, d_model) and (N, L, d_mlp)
        # This is the core change vs PLT: we need all layers together per token.
        self.enc_inputs  = torch.stack(mlp_in_list,  dim=1)   # (N, L, d_model)
        self.mlp_outputs = torch.stack(mlp_out_list, dim=1)   # (N, L, d_mlp)

        self.n_tokens  = self.enc_inputs.shape[0]
        self.n_layers  = self.enc_inputs.shape[1]
        self.d_model   = self.enc_inputs.shape[2]
        self.d_mlp     = self.mlp_outputs.shape[2]

    def __len__(self) -> int:
        return self.n_tokens

    def __getitem__(self, idx: int):
        return self.enc_inputs[idx], self.mlp_outputs[idx]

    @classmethod
    def from_file(cls, path: str) -> "CrossLayerDataset":
        data = torch.load(path, map_location="cpu", weights_only=False)
        ds = cls(data)
        print(f"CrossLayerDataset: {ds.n_tokens:,} tokens × "
              f"{ds.n_layers} layers × d_model={ds.d_model}")
        return ds, data   # also return raw data for bias init


# ---------------------------------------------------------------------------
# CLT Trainer
# ---------------------------------------------------------------------------

class CLTTrainer:
    """
    Trains a CrossLayerTranscoder jointly across all layers.

    Unlike TranscoderTrainer (which trains one layer at a time), CLTTrainer
    trains a single model with one optimizer. Every step sees a batch of
    tokens with activations at ALL layers simultaneously.
    """

    def __init__(
        self,
        dataset: CrossLayerDataset,
        raw_cache: dict,
        train_cfg: CLTTrainingConfig,
        model_n_layers: int,
        model_d_model: int,
        model_d_mlp: int,
    ):
        self.dataset = dataset
        self.raw_cache = raw_cache
        self.cfg = train_cfg
        self.device = torch.device(train_cfg.device)
        self.n_layers = model_n_layers
        self.d_model  = model_d_model
        self.d_mlp    = model_d_mlp
        Path(train_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_cache(
        cls,
        cache_path: str,
        train_cfg: Optional[CLTTrainingConfig] = None,
    ) -> "CLTTrainer":
        cfg = train_cfg or CLTTrainingConfig()
        ds, raw = CrossLayerDataset.from_file(cache_path)
        return cls(
            dataset=ds,
            raw_cache=raw,
            train_cfg=cfg,
            model_n_layers=ds.n_layers,
            model_d_model=ds.d_model,
            model_d_mlp=ds.d_mlp,
        )

    def train(self) -> CrossLayerTranscoder:
        """
        Train the CLT jointly across all layers and return the trained model.

        Training loop:
          1. Sample a batch of tokens (with all layers' activations).
          2. Forward pass: compute CLT feature activations and reconstructions.
          3. Loss = sum of per-layer MSE + sparsity penalty.
          4. Backward, clip gradients, step optimizer.
          5. Normalize decoder columns to unit concatenated norm.
          6. Repeat.
        """
        cfg = self.cfg

        # ---- Build CLT ----
        clt_cfg = CLTConfig(
            d_model=self.d_model,
            d_mlp=self.d_mlp,
            n_layers=self.n_layers,
            n_features=cfg.n_features,
            sparsity_coef=cfg.sparsity_coef,
            sparsity_c=cfg.sparsity_c,
            jump_thresh=cfg.jump_thresh,
            normalize_decoder=cfg.normalize_decoder,
            bandwidth=cfg.bandwidth,
        )
        clt = CrossLayerTranscoder(clt_cfg).to(self.device)
        print(f"\n{clt}")
        print(f"Training on {len(self.dataset):,} tokens × {self.n_layers} layers")
        print(f"Steps: {cfg.n_steps:,}   Batch: {cfg.batch_size}   LR: {cfg.lr}")

        # ---- Optional: initialize b_dec from cache means ----
        if cfg.init_biases:
            mlp_outs_cpu = self.raw_cache["mlp_out"]  # list of (N, d_mlp)
            clt.init_biases_from_cache(mlp_outs_cpu)

        # ---- Optimizer + scheduler ----
        optimizer = torch.optim.Adam(clt.parameters(), lr=cfg.lr, betas=cfg.betas)
        scheduler = LinearWarmupScheduler(optimizer, cfg.warmup_steps)

        # ---- DataLoader ----
        loader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=0,   # keep 0 to avoid multiprocessing issues on CPU
        )
        loader_iter = _infinite_loader(loader)

        # ---- Training loop ----
        best_nmse = float("inf")
        best_ckpt_path = None
        t0 = time.time()
        metrics = {}

        for step in range(1, cfg.n_steps + 1):
            enc_inputs, mlp_outputs = next(loader_iter)
            enc_inputs  = enc_inputs.to(self.device,  non_blocking=True)
            mlp_outputs = mlp_outputs.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            loss, metrics = clt.loss(enc_inputs, mlp_outputs)
            loss.backward()

            nn.utils.clip_grad_norm_(clt.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            clt.normalize_decoder_()

            # ---- Logging ----
            if step % cfg.log_every_n_steps == 0:
                elapsed = time.time() - t0
                eta_min = (cfg.n_steps - step) / (step / elapsed) / 60
                print(
                    f"  step {step:>6}/{cfg.n_steps}  "
                    f"loss={metrics['loss']:.4f}  "
                    f"nmse={metrics['nmse_mean']:.4f}  "
                    f"l0={metrics['l0_mean']:.1f}  "
                    f"sparsity={metrics['sparsity']:.3f}  "
                    f"lr={scheduler.get_lr():.2e}  "
                    f"eta={eta_min:.1f}m"
                )

            # ---- Checkpointing ----
            if step % cfg.save_every_n_steps == 0:
                ckpt_path = self._save_checkpoint(clt, step, metrics)
                if metrics["nmse_mean"] < best_nmse:
                    best_nmse = metrics["nmse_mean"]
                    best_ckpt_path = ckpt_path
                    print(f"  → New best NMSE: {best_nmse:.4f} ({ckpt_path})")

        # ---- Final save ----
        final_path = self._save_checkpoint(clt, cfg.n_steps, metrics, tag="final")
        print(f"\nCLT training complete.")
        print(f"  Final NMSE (mean): {metrics['nmse_mean']:.4f}   L0: {metrics['l0_mean']:.1f}")
        print(f"  Best NMSE:         {best_nmse:.4f} → {best_ckpt_path}")
        print(f"  Saved to:          {final_path}")

        return clt

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        clt: CrossLayerTranscoder,
        step: int,
        metrics: dict,
        tag: str = "",
    ) -> str:
        filename = f"clt_step_{step}"
        if tag:
            filename += f"_{tag}"
        filename += ".pt"
        path = str(Path(self.cfg.checkpoint_dir) / filename)
        torch.save({
            "model_state_dict": clt.state_dict(),
            "clt_config":       clt.cfg,
            "step":             step,
            "metrics":          metrics,
        }, path)
        return path

    @staticmethod
    def load_checkpoint(path: str, device: str = "cpu") -> CrossLayerTranscoder:
        """Load a CLT from a checkpoint file."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        clt = CrossLayerTranscoder(ckpt["clt_config"])
        clt.load_state_dict(ckpt["model_state_dict"])
        clt = clt.to(device)
        print(f"Loaded CLT from {path}")
        print(f"  Step       : {ckpt['step']}")
        print(f"  NMSE (mean): {ckpt['metrics'].get('nmse_mean', 'n/a'):.4f}")
        print(f"  L0 (mean)  : {ckpt['metrics'].get('l0_mean', 'n/a'):.1f}")
        return clt


# ---------------------------------------------------------------------------
# CLT evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_clt(
    clt: CrossLayerTranscoder,
    cache_data: dict,
    device: str = "cpu",
    batch_size: int = 128,
) -> dict:
    """
    Evaluate a trained CLT on the full activation cache.

    Returns per-layer NMSE and L0, plus global dead-feature fraction.

    Args:
        clt        : Trained CrossLayerTranscoder.
        cache_data : Raw dict from torch.load(cache_path).
        device     : Evaluation device.
        batch_size : Tokens per forward pass (no grads, so can use smaller).

    Returns dict with keys:
      nmse_layer_{i}  : NMSE for each layer i
      l0_layer_{i}    : Mean L0 for each layer i
      nmse_mean       : Mean NMSE across layers
      l0_mean         : Mean L0 across layers
      dead_frac       : Fraction of features that never activated (global)
      n_tokens        : Total tokens evaluated

    Usage:
        clt = CLTTrainer.load_checkpoint("checkpoints/clt_step_30000_final.pt")
        data = torch.load("data/activation_cache/gpt2.pt", map_location="cpu")
        metrics = evaluate_clt(clt, data)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    """
    clt = clt.to(device).eval()
    L = clt.cfg.n_layers

    # Stack cache into (N, L, d) tensors
    enc_inputs_all  = torch.stack(cache_data["mlp_in"],  dim=1)   # (N, L, d_model)
    mlp_outputs_all = torch.stack(cache_data["mlp_out"], dim=1)   # (N, L, d_mlp)
    N = enc_inputs_all.shape[0]

    # Accumulators
    total_mse    = [0.0] * L
    total_var    = [0.0] * L
    all_acts     = [[] for _ in range(L)]

    for start in range(0, N, batch_size):
        enc_b  = enc_inputs_all[start:start + batch_size].to(device)
        out_b  = mlp_outputs_all[start:start + batch_size].to(device)

        clt_out = clt(enc_b)

        for tgt in range(L):
            recon = clt_out.reconstructions[tgt]
            y     = out_b[:, tgt, :]
            total_mse[tgt] += F.mse_loss(recon, y, reduction="sum").item()
            total_var[tgt] += y.var(dim=0).sum().item()

        for src in range(L):
            all_acts[src].append(clt_out.feature_acts[src].cpu())

    # Concatenate accumulated activations
    all_acts = [torch.cat(acts, dim=0) for acts in all_acts]   # [(N, n_features)] × L

    # Build metrics
    result = {"n_tokens": N}
    nmse_list = []
    l0_list   = []

    for i in range(L):
        nmse_i = (total_mse[i] / N) / ((total_var[i] / N) + 1e-8)
        l0_i   = (all_acts[i] > 0).float().sum(dim=-1).mean().item()
        result[f"nmse_layer_{i}"] = nmse_i
        result[f"l0_layer_{i}"]   = l0_i
        nmse_list.append(nmse_i)
        l0_list.append(l0_i)

    result["nmse_mean"] = sum(nmse_list) / L
    result["l0_mean"]   = sum(l0_list)   / L

    # Global dead feature fraction (a feature is dead if NEVER active across all layers)
    ever_active = torch.zeros(clt.cfg.n_features, dtype=torch.bool)
    for src in range(L):
        ever_active |= (all_acts[src] > 0).any(dim=0)
    result["dead_frac"] = (~ever_active).float().mean().item()

    return result


# ---------------------------------------------------------------------------
# CLT smoke test
# ---------------------------------------------------------------------------

def _validate_clt():
    """
    Smoke test for CLT training. Uses synthetic multi-layer MLP activations.
    Run with: python -m src.transcoder.train --clt
    """
    import torch.nn.functional as _F

    print("=" * 60)
    print("Smoke test: CLT training on synthetic multi-layer data")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model, d_mlp = 64, 64
    n_layers = 4
    n_features = 128
    n_tokens = 4000

    # Synthetic MLPs for each layer
    mlps = [
        (torch.randn(d_mlp * 4, d_model).to(device),
         torch.randn(d_mlp, d_mlp * 4).to(device))
        for _ in range(n_layers)
    ]

    x_base = torch.randn(n_tokens, d_model)
    mlp_in_list, mlp_out_list = [], []
    x = x_base.to(device)

    with torch.no_grad():
        for W1, W2 in mlps:
            y = _F.gelu(x @ W1.T) @ W2.T
            mlp_in_list.append(x.cpu())
            mlp_out_list.append(y.cpu())
            x = x + y   # residual connection

    cache_data = {
        "mlp_in":  mlp_in_list,
        "mlp_out": mlp_out_list,
        "model_name": "synthetic",
    }

    train_cfg = CLTTrainingConfig(
        n_features=n_features,
        n_steps=400,
        batch_size=128,
        log_every_n_steps=100,
        save_every_n_steps=400,
        checkpoint_dir="checkpoints/clt_smoke_test",
        device=device,
        init_biases=True,
    )

    trainer = CLTTrainer(
        dataset=CrossLayerDataset(cache_data),
        raw_cache=cache_data,
        train_cfg=train_cfg,
        model_n_layers=n_layers,
        model_d_model=d_model,
        model_d_mlp=d_mlp,
    )
    clt = trainer.train()

    print("\nFinal evaluation:")
    metrics = evaluate_clt(clt, cache_data, device=device)
    print(f"  nmse_mean : {metrics['nmse_mean']:.4f}")
    print(f"  l0_mean   : {metrics['l0_mean']:.1f}")
    print(f"  dead_frac : {metrics['dead_frac']:.2%}")
    for i in range(n_layers):
        print(f"  layer {i}: nmse={metrics[f'nmse_layer_{i}']:.4f}  "
              f"l0={metrics[f'l0_layer_{i}']:.1f}")

    assert metrics["nmse_mean"] < 0.8, f"NMSE too high: {metrics['nmse_mean']:.4f}"
    assert metrics["l0_mean"] > 0,     "No features active"
    print("\nCLT smoke test passed.")


if __name__ == "__main__":
    import sys
    if "--clt" in sys.argv:
        _validate_clt()
    else:
        _validate()

