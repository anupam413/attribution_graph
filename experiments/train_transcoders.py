"""
experiments/train_transcoders.py
---------------------------------
End-to-end Phase 3a + 3b experiment script.

Run this after Phase 2 to:
  1. Collect MLP activations from the base model  (if not already cached)
  2. Train transcoders (PLT per-layer OR CLT cross-layer)
  3. Evaluate reconstruction quality
  4. Spot-check feature activations on example prompts

Usage:
    # PLT: train all layers independently (Phase 3a)
    python experiments/train_transcoders.py

    # CLT: train cross-layer transcoder jointly (Phase 3b)
    python experiments/train_transcoders.py --clt

    # Single layer PLT only
    python experiments/train_transcoders.py --layer 5

    # Skip collection (use existing cache)
    python experiments/train_transcoders.py --skip-collection --clt

    # Evaluate a saved CLT checkpoint
    python experiments/train_transcoders.py --eval-only --clt \\
        --checkpoint checkpoints/gpt2/clt_step_30000_final.pt
"""

import argparse
import json
from pathlib import Path

import torch

from src.model.loader import ModelWrapper, ActivationCache
from src.transcoder.per_layer import PerLayerTranscoder
from src.transcoder.cross_layer import CrossLayerTranscoder
from src.transcoder.train import (
    MLPActivationDataset,
    TrainingConfig,
    TranscoderTrainer,
    evaluate_transcoder,
    CLTTrainingConfig,
    CLTTrainer,
    CrossLayerDataset,
    evaluate_clt,
)

# ---------------------------------------------------------------------------
# Prompts used to collect activations (augment with your own)
# ---------------------------------------------------------------------------

COLLECTION_PROMPTS = [
    # Factual recall
    "The Eiffel Tower is located in",
    "The capital of Germany is",
    "The capital of Japan is",
    "Michael Jordan plays the sport of",
    "Albert Einstein was born in",
    "The Amazon river is in",
    "Shakespeare wrote the play",
    "The chemical symbol for gold is",
    # Simple arithmetic context
    "The answer to 3 + 4 is",
    "calc: 12 + 15 =",
    "calc: 36 + 59 =",
    # Continuation
    "Once upon a time there was a",
    "The quick brown fox jumps over the",
    "To be or not to be, that is",
    "It was the best of times, it was",
    # Language variety
    "In Python, to print hello world you write",
    "The boiling point of water is",
    "The speed of light is approximately",
    # Add more prompts here — more is better for transcoder quality
    # Aim for at least 500–1000 diverse prompts for real experiments
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="gpt2",
                        help="TransformerLens model name")
    parser.add_argument("--clt", action="store_true",
                        help="Train cross-layer transcoder (Phase 3b) instead of PLT")
    parser.add_argument("--layer",    type=int, default=None,
                        help="PLT only: train only this layer (default: all layers)")
    parser.add_argument("--n-features", type=int, default=None,
                        help="Transcoder dictionary size (default: 4096 PLT / 2048 CLT)")
    parser.add_argument("--n-steps",  type=int, default=None,
                        help="Training steps (default: 10k PLT / 30k CLT)")
    parser.add_argument("--sparsity-coef", type=float, default=1e-3)
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip activation collection, use existing cache")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate, skip training")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint for --eval-only mode")
    parser.add_argument("--cache-path",
                        default="data/activation_cache/activations.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Step 1: Collect activations
    # ------------------------------------------------------------------
    cache_path = args.cache_path

    if not args.skip_collection and not args.eval_only:
        print("\n[Step 1] Collecting MLP activations ...")
        model = ModelWrapper.load(args.model, device=device)
        collector = ActivationCache(model)
        collector.collect(
            prompts=COLLECTION_PROMPTS,
            save_path=cache_path,
            verbose=True,
        )
        del model   # free GPU memory before training
        torch.cuda.empty_cache()
    else:
        print(f"\n[Step 1] Skipping collection. Using cache at: {cache_path}")
        assert Path(cache_path).exists(), (
            f"Cache file not found: {cache_path}\n"
            f"Run without --skip-collection first."
        )

    # ------------------------------------------------------------------
    # Step 2: Eval-only mode
    # ------------------------------------------------------------------
    if args.eval_only:
        assert args.checkpoint, "Provide --checkpoint for --eval-only"
        data = torch.load(cache_path, map_location="cpu")

        if args.clt:
            print(f"\n[Eval only — CLT] {args.checkpoint}")
            clt = CLTTrainer.load_checkpoint(args.checkpoint, device=device)
            metrics = evaluate_clt(clt, data, device=device)
            print(f"\nCLT evaluation:")
            print(f"  nmse_mean : {metrics['nmse_mean']:.4f}")
            print(f"  l0_mean   : {metrics['l0_mean']:.1f}")
            print(f"  dead_frac : {metrics['dead_frac']:.2%}")
            for i in range(clt.cfg.n_layers):
                print(f"  layer {i:2d}: nmse={metrics[f'nmse_layer_{i}']:.4f}  "
                      f"l0={metrics[f'l0_layer_{i}']:.1f}")
        else:
            assert args.layer is not None, "Provide --layer for PLT --eval-only"
            print(f"\n[Eval only — PLT layer {args.layer}] {args.checkpoint}")
            tc = TranscoderTrainer.load_checkpoint(args.checkpoint, device=device)
            metrics = evaluate_transcoder(
                tc, data["mlp_in"][args.layer], data["mlp_out"][args.layer],
                device=device,
            )
            for k, v in metrics.items():
                print(f"  {k:20s}: {v:.4f}")
        return

    # ------------------------------------------------------------------
    # Step 3: Train
    # ------------------------------------------------------------------
    if args.clt:
        # ---- Phase 3b: Cross-Layer Transcoder ----
        print("\n[Step 2] Training CLT (Phase 3b) ...")
        n_features = args.n_features or 2048
        n_steps    = args.n_steps    or 30_000

        train_cfg = CLTTrainingConfig(
            n_features=n_features,
            sparsity_coef=args.sparsity_coef,
            n_steps=n_steps,
            batch_size=256,
            log_every_n_steps=200,
            save_every_n_steps=5000,
            checkpoint_dir=f"checkpoints/{args.model}",
            device=device,
            init_biases=True,
        )
        trainer = CLTTrainer.from_cache(cache_path, train_cfg)
        clt = trainer.train()

        # ---- Evaluate ----
        print("\n[Step 3] Evaluating CLT ...")
        data = torch.load(cache_path, map_location="cpu")
        metrics = evaluate_clt(clt, data, device=device)

        print(f"\n  SUMMARY")
        print(f"  {'nmse_mean':20s}: {metrics['nmse_mean']:.4f}")
        print(f"  {'l0_mean':20s}: {metrics['l0_mean']:.1f}")
        print(f"  {'dead_frac':20s}: {metrics['dead_frac']:.2%}")
        print(f"  Per-layer NMSE:")
        for i in range(clt.cfg.n_layers):
            print(f"    layer {i:2d}: {metrics[f'nmse_layer_{i}']:.4f}  "
                  f"l0={metrics[f'l0_layer_{i}']:.1f}")

        summary_path = f"checkpoints/{args.model}/clt_eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nEval summary saved to {summary_path}")

    else:
        # ---- Phase 3a: Per-Layer Transcoder ----
        print("\n[Step 2] Training PLTs (Phase 3a) ...")
        n_features = args.n_features or 4096
        n_steps    = args.n_steps    or 10_000

        train_cfg = TrainingConfig(
            n_features=n_features,
            sparsity_coef=args.sparsity_coef,
            n_steps=n_steps,
            batch_size=2048,
            log_every_n_steps=200,
            save_every_n_steps=2000,
            checkpoint_dir=f"checkpoints/{args.model}",
            device=device,
            layers_to_train=[args.layer] if args.layer is not None else None,
        )

        trainer = TranscoderTrainer.from_cache(cache_path, train_cfg)
        transcoders = trainer.train_all_layers()

        # ---- Evaluate ----
        print("\n[Step 3] Evaluating PLTs ...")
        data = torch.load(cache_path, map_location="cpu")
        results = {}
        layers = train_cfg.layers_to_train or list(range(len(data["mlp_in"])))
        for layer_idx, tc in zip(layers, transcoders):
            m = evaluate_transcoder(
                tc, data["mlp_in"][layer_idx], data["mlp_out"][layer_idx],
                device=device,
            )
            results[f"layer_{layer_idx}"] = m
            print(f"  Layer {layer_idx:2d}: NMSE={m['nmse']:.3f}  "
                  f"FVE={m['fve']:.3f}  L0={m['l0']:.1f}  dead={m['dead_frac']:.2%}")

        summary_path = f"checkpoints/{args.model}/plt_eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nEval summary saved to {summary_path}")

        # Spot-check features
        print("\n[Step 4] Spot-checking features on example prompts ...")
        _spot_check(transcoders, layers, data, args.model, device)


def _spot_check(
    transcoders: list[PerLayerTranscoder],
    layers: list[int],
    cache_data: dict,
    model_name: str,
    device: str,
):
    """
    For each trained transcoder, show which features fire on the first
    few tokens of the first cached prompt.

    This is a quick sanity check that features are actually activating
    and not all dead.
    """
    # Use first ~10 tokens from the cache as a demo
    for layer_idx, tc in zip(layers, transcoders):
        demo_x = cache_data["mlp_in"][layer_idx][:10].to(device)
        tc = tc.to(device).eval()

        with torch.no_grad():
            active = tc.get_active_features(demo_x, top_k=5)

        print(f"\n  Layer {layer_idx} — top-5 features per token position:")
        for pos_info in active[:5]:   # show first 5 token positions
            pos  = pos_info["token_pos"]
            ids  = pos_info["feature_ids"]
            acts = pos_info["activations"]
            act_str = ", ".join(f"feat_{i}={a:.3f}" for i, a in zip(ids, acts))
            print(f"    pos {pos}: {act_str if act_str else '(none active)'}")


if __name__ == "__main__":
    main()
