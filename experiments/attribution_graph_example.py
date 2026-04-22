"""
experiments/attribution_graph_example.py
---------------------------------------
End-to-end example: Build attribution graph for a prompt.

Usage:
    python experiments/attribution_graph_example.py
"""

import torch
from src.model.loader import ModelWrapper
from src.transcoder.per_layer import PerLayerTranscoder, TranscoderConfig
from src.replacement_model.local import LocalReplacementModel
from src.attribution.jacobian import JacobianComputer
from src.attribution.graph import GraphBuilder
from src.attribution.prune import GraphPruner
from src.utils.viz import print_graph_summary, plot_graph


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    print("Loading model...")
    model = ModelWrapper.load("gpt2", device=device)
    
    # Load or create transcoders (for demo, create dummy ones)
    print("Setting up transcoders...")
    n_layers = model.model.cfg.n_layers
    transcoders = []
    for layer_idx in range(n_layers):
        cfg = TranscoderConfig(
            d_model=model.model.cfg.d_model,
            d_mlp=model.model.cfg.d_mlp,
            n_features=1024,  # Smaller for demo
        )
        tc = PerLayerTranscoder(cfg).to(device)
        transcoders.append(tc)
    
    # Create replacement model
    print("Creating replacement model...")
    replacement = LocalReplacementModel(model.model, transcoders, is_clt=False)
    
    # Forward pass on example prompt
    prompt = "Michael Jordan plays the sport of"
    tokens = model.model.to_tokens(prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Tokens shape: {tokens.shape}")
    
    output = replacement(tokens)
    print(f"Output logits shape: {output.logits.shape}")
    
    # Compute Jacobians
    print("\nComputing Jacobians...")
    jac_computer = JacobianComputer(replacement)
    jacobians = jac_computer.compute_jacobians(
        tokens=tokens,
        target_pos=-1,
        top_k_logits=5,
    )
    
    # Build graph
    print("Building attribution graph...")
    builder = GraphBuilder()
    graph = builder.build_graph(
        jacobian_result=jacobians,
        feature_acts=output.feature_acts,
        error_acts=output.error_acts,
        threshold=0.01,
    )
    
    print_graph_summary(graph)
    
    # Prune graph
    print("\nPruning graph...")
    pruner = GraphPruner()
    pruned_graph = pruner.prune_by_top_k(graph, k=100)
    
    print("\nAfter pruning:")
    print_graph_summary(pruned_graph)
    
    # Visualize (optional)
    # plot_graph(pruned_graph, save_path="attribution_graph.png")


if __name__ == "__main__":
    main()