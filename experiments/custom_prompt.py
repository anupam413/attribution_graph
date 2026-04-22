"""
experiments/custom_prompt.py
----------------------------
Template for custom attribution graph experiments.

Use this to analyze any prompt with your trained transcoders.

Usage:
    python experiments/custom_prompt.py \
        --prompt "The capital of France is" \
        --transcoder-path checkpoints/gpt2/plt_final.pt

    python experiments/custom_prompt.py \
        --prompt "Once upon a time" \
        --transcoder-path checkpoints/gpt2/clt_final.pt \
        --clt
"""

import argparse
from pathlib import Path

import torch

from src.model.loader import ModelWrapper
from src.transcoder.train import TranscoderTrainer, CLTTrainer
from src.replacement_model.local import LocalReplacementModel
from src.attribution.graph import GraphBuilder
from src.attribution.prune import GraphPruner
from src.utils.viz import (
    print_graph_summary,
    plot_graph,
    plot_feature_activations,
    create_attribution_visualization,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="Input prompt to analyze")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--transcoder-path", required=True, help="Path to trained transcoder")
    parser.add_argument("--clt", action="store_true", help="Use CLT instead of PLT")
    parser.add_argument("--target-token-idx", type=int, default=None, 
                         help="Specific token to analyze (default: argmax)")
    parser.add_argument("--threshold", type=float, default=0.01, help="Edge weight threshold")
    parser.add_argument("--top-k-edges", type=int, default=100, help="Keep top-k edges")
    parser.add_argument("--output-dir", default="outputs/custom", help="Output directory")
    parser.add_argument("--show-top-k-features", type=int, default=10, 
                         help="Show top-k features per layer")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directory with prompt name
    prompt_name = args.prompt.replace(" ", "_")[:50]  # Shorten for filename
    output_dir = Path(args.output_dir) / prompt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"CUSTOM ATTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # -------------------------------------------------------------------------
    # 1. Load model
    # -------------------------------------------------------------------------
    print(f"[1/6] Loading model: {args.model}")
    model_wrapper = ModelWrapper.load(args.model, device=device)
    base_model = model_wrapper.model

    base_model.set_use_attn_result(True)  # Enable attention weights for attribution
    
    # -------------------------------------------------------------------------
    # 2. Load transcoder
    # -------------------------------------------------------------------------
    print(f"[2/6] Loading transcoder: {args.transcoder_path}")
    
    if args.clt:
        # CLTTrainer.load_checkpoint returns CrossLayerTranscoder directly
        transcoder = CLTTrainer.load_checkpoint(args.transcoder_path, device=device)
        print(f"  CLT with {transcoder.cfg.n_features} features/layer")
    else:
        transcoders = []
        for layer_idx in range(model_wrapper.n_layers):
            tc_path = args.transcoder_path.replace("_final", f"_layer_{layer_idx}_final")
            tc_path = tc_path.replace("layer_0", f"layer_{layer_idx}")  # Handle layer_0 pattern
            
            if Path(tc_path).exists():
                trainer_or_tc = TranscoderTrainer.load_checkpoint(tc_path, device=device)
                # Handle both trainer and direct transcoder
                if hasattr(trainer_or_tc, 'transcoder'):
                    transcoders.append(trainer_or_tc.transcoder)
                else:
                    transcoders.append(trainer_or_tc)
            else:
                # Single file for all layers
                trainer_or_tc = TranscoderTrainer.load_checkpoint(args.transcoder_path, device=device)
                # Handle both trainer and direct transcoder
                if hasattr(trainer_or_tc, 'transcoder'):
                    tc = trainer_or_tc.transcoder
                else:
                    tc = trainer_or_tc
                transcoders = [tc for _ in range(model_wrapper.n_layers)]
                break
        transcoder = transcoders
        print(f"  Loaded {len(transcoders)} PLT transcoders")
    
    # -------------------------------------------------------------------------
    # 3. Tokenize and get predictions
    # -------------------------------------------------------------------------
    print(f"\n[3/6] Running forward pass")
    
    tokens = base_model.to_tokens(args.prompt)
    token_strs = [base_model.tokenizer.decode([t]) for t in tokens[0]]
    
    print(f"Tokens ({len(token_strs)}): {token_strs}")
    
    # Base model predictions
    base_logits = base_model(tokens)
    base_probs = torch.softmax(base_logits[0, -1], dim=-1)
    
    print(f"\nBase model top-5 predictions:")
    top_probs, top_indices = base_probs.topk(5)
    for prob, idx in zip(top_probs, top_indices):
        token = base_model.tokenizer.decode([idx])
        print(f"  {prob:.2%}: '{token}'")
    
    # Determine target token
    if args.target_token_idx is not None:
        target_idx = args.target_token_idx
        target_token = base_model.tokenizer.decode([target_idx])
        print(f"\nAnalyzing specified target: '{target_token}' (idx {target_idx})")
    else:
        target_idx = top_indices[0].item()
        target_token = base_model.tokenizer.decode([target_idx])
        print(f"\nAnalyzing top prediction: '{target_token}' (idx {target_idx})")
    
    # -------------------------------------------------------------------------
    # 4. Replacement model
    # -------------------------------------------------------------------------
    print(f"\n[4/6] Running replacement model")
    
    replacement = LocalReplacementModel(
        base_model,
        transcoder,
        is_clt=args.clt,
    )
    
    output = replacement(tokens)
    
    # Compare with base model
    comparison = replacement.compare_with_base(tokens)
    print(f"Replacement model fidelity:")
    print(f"  KL divergence: {comparison['kl_divergence']:.4f}")
    print(f"  Top-1 accuracy: {comparison['top1_accuracy']:.2%}")
    print(f"  Logit MSE: {comparison['logit_mse']:.4f}")
    
    # -------------------------------------------------------------------------
    # 5. Build attribution graph
    # -------------------------------------------------------------------------
    print(f"\n[5/6] Building attribution graph")
    
    builder = GraphBuilder()
    
    if args.clt:
        graph = builder.build_from_activations(
            output.feature_acts,
            output.error_acts,
            None,
            threshold=args.threshold,
        )
    else:
        graph = builder.build_from_activations(
            output.feature_acts,
            output.error_acts,
            transcoders,
            threshold=args.threshold,
        )
    
    print_graph_summary(graph)
    
    # Prune
    pruner = GraphPruner()
    pruned_graph = pruner.prune_by_top_k(graph, k=args.top_k_edges)
    
    print(f"\nAfter pruning to top-{args.top_k_edges} edges:")
    print_graph_summary(pruned_graph)
    
    # -------------------------------------------------------------------------
    # 6. Analyze and visualize
    # -------------------------------------------------------------------------
    print(f"\n[6/6] Creating visualizations and analysis")
    
    # Show top features at last position
    print(f"\n{'='*70}")
    print(f"TOP FEATURES AT FINAL POSITION ('{token_strs[-1]}')")
    print(f"{'='*70}")
    
    for layer_idx in range(len(output.feature_acts)):
        features = output.feature_acts[layer_idx][0, -1]  # Last position
        top_acts, top_indices = features.topk(args.show_top_k_features)
        
        if top_acts[0] > 0:
            print(f"\nLayer {layer_idx}:")
            for rank, (act, idx) in enumerate(zip(top_acts, top_indices), 1):
                if act > 0:
                    print(f"  {rank}. Feature {idx.item():4d}: {act.item():.4f}")
    
    # Visualizations
    print(f"\nGenerating visualizations...")
    
    # 1. Attribution graph
    plot_graph(
        pruned_graph,
        figsize=(20, 14),
        layout="hierarchical",
        save_path=output_dir / "attribution_graph.png",
        title=f"Attribution: {args.prompt} → {target_token}",
        show_activations=True,
    )
    print(f"  ✓ Attribution graph saved")
    
    # 2. Feature activations
    plot_feature_activations(
        output.feature_acts,
        tokens=token_strs,
        layers_to_plot=list(range(min(6, len(output.feature_acts)))),
        top_k_features=15,
        save_path=output_dir / "feature_activations.png",
    )
    print(f"  ✓ Feature activations saved")
    
    # 3. Comprehensive visualization
    create_attribution_visualization(
        pruned_graph,
        prompt=args.prompt,
        tokens=token_strs,
        target_token=target_token,
        save_path=output_dir / "comprehensive.png",
    )
    print(f"  ✓ Comprehensive visualization saved")
    
    # Save text summary
    summary_path = output_dir / "analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"ATTRIBUTION ANALYSIS SUMMARY\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Target token: {target_token} (idx {target_idx})\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Transcoder: {'CLT' if args.clt else 'PLT'}\n\n")
        
        f.write(f"Model Predictions:\n")
        for prob, idx in zip(top_probs, top_indices):
            token = base_model.tokenizer.decode([idx])
            f.write(f"  {prob:.2%}: '{token}'\n")
        
        f.write(f"\nReplacement Model Fidelity:\n")
        f.write(f"  KL divergence: {comparison['kl_divergence']:.4f}\n")
        f.write(f"  Top-1 accuracy: {comparison['top1_accuracy']:.2%}\n")
        f.write(f"  Logit MSE: {comparison['logit_mse']:.4f}\n")
        
        f.write(f"\nAttribution Graph:\n")
        f.write(f"  Nodes: {len(pruned_graph.nodes)}\n")
        f.write(f"  Edges: {len(pruned_graph.edges)}\n")
        f.write(f"  Features: {len([n for n in pruned_graph.nodes if n.node_type == 'feature'])}\n")
        f.write(f"  Errors: {len([n for n in pruned_graph.nodes if n.node_type == 'error'])}\n")
        
        f.write(f"\nTop Features per Layer (at final position):\n")
        for layer_idx in range(len(output.feature_acts)):
            features = output.feature_acts[layer_idx][0, -1]
            top_acts, top_indices = features.topk(3)
            if top_acts[0] > 0:
                f.write(f"\n  Layer {layer_idx}:\n")
                for act, idx in zip(top_acts, top_indices):
                    if act > 0:
                        f.write(f"    Feature {idx.item()}: {act.item():.4f}\n")
    
    print(f"  ✓ Text summary saved")
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()