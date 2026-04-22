"""
experiments/factual_recall.py
-----------------------------
Reproduce the Michael Jordan factual recall case study from the paper.

This experiment:
  1. Loads a trained transcoder
  2. Builds attribution graph for "Michael Jordan plays the sport of"
  3. Visualizes which features are responsible for predicting "basketball"
  4. Analyzes the circuit from early-layer person features to late-layer sport features

Usage:
    python experiments/factual_recall.py --model gpt2 --transcoder-path checkpoints/gpt2/plt_final.pt
    python experiments/factual_recall.py --model gpt2 --transcoder-path checkpoints/gpt2/clt_final.pt --clt
"""

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import networkx as nx

from src.model.loader import ModelWrapper
from src.transcoder.per_layer import PerLayerTranscoder
from src.transcoder.cross_layer import CrossLayerTranscoder
from src.transcoder.train import TranscoderTrainer, CLTTrainer
from src.replacement_model.local import LocalReplacementModel
from src.attribution.jacobian import JacobianComputer, compute_virtual_weights_plt
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
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--transcoder-path", required=True, help="Path to trained transcoder")
    parser.add_argument("--clt", action="store_true", help="Use CLT instead of PLT")
    parser.add_argument("--threshold", type=float, default=0.01, help="Edge threshold")
    parser.add_argument("--top-k", type=int, default=100, help="Keep top-k edges")
    parser.add_argument("--output-dir", default="outputs/factual_recall", help="Output directory")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. Load model
    # -------------------------------------------------------------------------
    print(f"\n[1/6] Loading model: {args.model}")
    model_wrapper = ModelWrapper.load(args.model, device=device)
    base_model = model_wrapper.model

    base_model.set_use_attn_result(True)  # Enable attention weights for attribution
    
    # -------------------------------------------------------------------------
    # 2. Load transcoder
    # -------------------------------------------------------------------------
    print(f"\n[2/6] Loading transcoder: {args.transcoder_path}")
    
    if args.clt:
        # CLTTrainer.load_checkpoint returns CrossLayerTranscoder directly
        transcoder = CLTTrainer.load_checkpoint(args.transcoder_path, device=device)
        print(f"Loaded CLT with {transcoder.cfg.n_features} features/layer")
    else:
        # Load all PLT transcoders
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
                # Try loading single file for all layers
                trainer_or_tc = TranscoderTrainer.load_checkpoint(args.transcoder_path, device=device)
                # Handle both trainer and direct transcoder
                if hasattr(trainer_or_tc, 'transcoder'):
                    tc = trainer_or_tc.transcoder
                else:
                    tc = trainer_or_tc
                transcoders = [tc for _ in range(model_wrapper.n_layers)]
                break
        
        transcoder = transcoders
        print(f"Loaded {len(transcoders)} PLT transcoders")
    
    # -------------------------------------------------------------------------
    # 3. Create replacement model
    # -------------------------------------------------------------------------
    print(f"\n[3/6] Creating replacement model")
    replacement = LocalReplacementModel(
        base_model,
        transcoder,
        is_clt=args.clt,
    )
    
    # -------------------------------------------------------------------------
    # 4. Run forward pass on prompt
    # -------------------------------------------------------------------------
    print(f"\n[4/6] Running forward pass")
    prompt = "Michael Jordan plays the sport of"
    print(f"Prompt: '{prompt}'")
    
    tokens = base_model.to_tokens(prompt)
    token_strs = [base_model.tokenizer.decode([t]) for t in tokens[0]]
    print(f"Tokens: {token_strs}")
    
    # Base model prediction
    base_logits = base_model(tokens)
    base_probs = torch.softmax(base_logits[0, -1], dim=-1)
    top_k = 5
    top_probs, top_indices = base_probs.topk(top_k)
    
    print(f"\nBase model top-{top_k} predictions:")
    for prob, idx in zip(top_probs, top_indices):
        token = base_model.tokenizer.decode([idx])
        print(f"  '{token}': {prob:.2%}")
    
    target_token = base_model.tokenizer.decode([top_indices[0]])
    target_idx = top_indices[0].item()
    print(f"\nTarget token: '{target_token}' (idx {target_idx})")
    
    # Replacement model
    output = replacement(tokens)
    repl_logits = output.logits
    repl_probs = torch.softmax(repl_logits[0, -1], dim=-1)
    
    print(f"\nReplacement model prediction: '{base_model.tokenizer.decode([repl_logits[0, -1].argmax()])}'")
    print(f"Replacement fidelity: {repl_probs[target_idx]:.2%}")
    
    # -------------------------------------------------------------------------
    # 5. Build attribution graph
    # -------------------------------------------------------------------------
    print(f"\n[5/6] Building attribution graph")
    
    builder = GraphBuilder()
    
    if args.clt:
        # For CLT, use virtual weights
        graph = builder.build_from_activations(
            output.feature_acts,
            output.error_acts,
            None,  # Not needed for CLT
            threshold=args.threshold,
        )
    else:
        # For PLT, use virtual weights
        graph = builder.build_from_activations(
            output.feature_acts,
            output.error_acts,
            transcoders,
            threshold=args.threshold,
        )
    
    print_graph_summary(graph)
    
    # Prune graph
    pruner = GraphPruner()
    pruned_graph = pruner.prune_by_top_k(graph, k=args.top_k)
    
    print(f"\nAfter pruning to top-{args.top_k} edges:")
    print_graph_summary(pruned_graph)
    
    # -------------------------------------------------------------------------
    # 6. Visualize
    # -------------------------------------------------------------------------
    print(f"\n[6/6] Creating visualizations")
    
    # Graph visualization
    plot_graph(
        pruned_graph,
        figsize=(20, 14),
        layout="hierarchical",
        save_path=output_dir / "attribution_graph.png",
        title=f"Attribution Graph: {prompt} → {target_token}",
        show_activations=True,
    )
    
    # Feature activations
    plot_feature_activations(
        output.feature_acts,
        tokens=token_strs,
        layers_to_plot=list(range(min(6, len(output.feature_acts)))),
        top_k_features=15,
        save_path=output_dir / "feature_activations.png",
    )
    
    # Comprehensive visualization
    create_attribution_visualization(
        pruned_graph,
        prompt=prompt,
        tokens=token_strs,
        target_token=target_token,
        save_path=output_dir / "comprehensive_attribution.png",
    )
    
    # -------------------------------------------------------------------------
    # Analysis: Find key features
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"KEY FEATURES ANALYSIS")
    print(f"{'='*60}")
    
    # Find most active features at each layer
    for layer_idx in range(min(len(output.feature_acts), model_wrapper.n_layers)):
        features = output.feature_acts[layer_idx][0]  # (seq_len, n_features)
        
        # Most active features at last position
        last_pos_features = features[-1]
        top_k_feat = 5
        top_acts, top_indices = last_pos_features.topk(top_k_feat)
        
        if top_acts[0] > 0:
            print(f"\nLayer {layer_idx} - Top {top_k_feat} features at position '{token_strs[-1]}':")
            for act, idx in zip(top_acts, top_indices):
                if act > 0:
                    print(f"  Feature {idx.item()}: {act.item():.4f}")
    
    # Find features that bridge early and late layers
    print(f"\n{'='*60}")
    print(f"CROSS-LAYER CONNECTIONS")
    print(f"{'='*60}")
    
    # Analyze paths from layer 0 to final layer
    G = pruned_graph.to_networkx()
    source_layer = 0
    target_layer = max(n.layer for n in pruned_graph.nodes)
    
    source_nodes = [i for i, n in enumerate(pruned_graph.nodes) if n.layer == source_layer]
    target_nodes = [i for i, n in enumerate(pruned_graph.nodes) if n.layer == target_layer]
    
    path_count = 0
    for source in source_nodes[:5]:  # Limit for performance
        for target in target_nodes[:5]:
            try:
                paths = list(nx.all_simple_paths(G, source, target, cutoff=target_layer+1))
                for path in paths[:2]:  # Top 2 paths per pair
                    path_count += 1
                    path_nodes = [pruned_graph.nodes[i] for i in path]
                    path_str = " → ".join([
                        f"L{n.layer}F{n.index}" for n in path_nodes
                    ])
                    print(f"\nPath {path_count}: {path_str}")
                    
                    # Calculate path strength
                    strength = sum(
                        abs(G[path[i]][path[i+1]]['weight'])
                        for i in range(len(path)-1)
                        if G.has_edge(path[i], path[i+1])
                    )
                    print(f"  Strength: {strength:.4f}")
            except:
                continue
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()